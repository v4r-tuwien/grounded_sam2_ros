#!/usr/bin/python3.10
import torch
import numpy as np
import cv2
import os
import yaml
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image_numpy, predict
import rospy
from actionlib import SimpleActionServer
from sensor_msgs.msg import RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from object_detector_msgs.msg import BoundingBox, Detection, Detections
from object_detector_msgs.srv import detectron2_service_server

# Default object names
default_object_names = [
    "can", "mug", "cup", "plate", "garbage", "table",
    "chair", "mustard", "bottle", "cable", "box", "fruit"
]

class GS2_ROS_Wrapper():

    def __init__(self):
        """Setting hyperparameters"""
        rospy.loginfo("Initializing Grounded SAM 2 ROS Wrapper")
        BASE_PATH = "/root/grounded_sam2_ros/"
        yaml_file_path = "/root/config/scene_obj_list.yaml"

        try:
            with open(yaml_file_path, 'r') as f:
                data = yaml.safe_load(f)
                object_names = data.get('gsam2_object_names', [])
                if not object_names:
                    rospy.logwarn("YAML file loaded but 'gsam2_object_names' is empty or missing.")
                    object_names = default_object_names
                else:
                    rospy.loginfo("Loaded object names from YAML file.")
        except Exception as e:
            rospy.logwarn(f"Could not load object names from YAML file: {e}")
            object_names = default_object_names

        self.TEXT_PROMPT = '. '.join(object_names) + '.'
        rospy.loginfo(f"Using text prompt: {self.TEXT_PROMPT}")

        SAM2_CHECKPOINT = BASE_PATH + "checkpoints/sam2_hiera_large.pt"
        SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = BASE_PATH + "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = BASE_PATH + "gdino_checkpoints/groundingdino_swint_ogc.pth"

        self.BOX_THRESHOLD = rospy.get_param("/gsam2/box_threshold", 0.3)
        self.TEXT_THRESHOLD = rospy.get_param("/gsam2/text_threshold", 0.25)
        self.AREA_THRESHOLD = rospy.get_param("/gsam2/area_threshold", 80) # minimum area of the bounding boxes in pixel^2
        rospy.loginfo(f"Using thresholds: {self.BOX_THRESHOLD = }, {self.TEXT_THRESHOLD = }, {self.AREA_THRESHOLD = }")
        print(f"{torch.cuda.is_available() = }")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if SAM2 checkpoint exists
        if not os.path.isfile(SAM2_CHECKPOINT):
            rospy.logerr(f"SAM2 checkpoint not found at: {SAM2_CHECKPOINT}")
            raise FileNotFoundError(f"SAM2 checkpoint not found at: {SAM2_CHECKPOINT}")

        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )

        # ROS Stuff
        self.bridge = CvBridge()
        self.service = rospy.Service("/detect_objects", detectron2_service_server, self.service_call)

        rospy.loginfo("Ready to receive images")
    
    def service_call(self, req):
        rgb = req.image
        rospy.loginfo(f"Received image with shape {rgb.width}x{rgb.height}")
        width, height = rgb.width, rgb.height

        try:
            image = self.bridge.imgmsg_to_cv2(rgb, "rgb8")
        except CvBridgeError as e:
            print(e)
        
        ros_detections = self.inference(image, rgb.header)

        return ros_detections

    def inference(self, image, rgb_header):
        rospy.loginfo(f"GSam2 is using starting estimation ...")

        results = self.inference_model_intern(image)  # predict on an image
        boxes, masks, class_names_, confidences_ = results

        height, width, _ = image.shape
        idx = 0
        detections = []

        for box, mask, class_name, confidence in zip(boxes, masks, class_names_, confidences_):
            detection = Detection()
            detection.name = class_name

            bbox_msg = BoundingBox()
            bbox_msg.xmin = int(box[0])
            bbox_msg.ymin = int(box[1])
            bbox_msg.xmax = int(box[2])
            bbox_msg.ymax = int(box[3])
            detection.bbox = bbox_msg

            bbox_height = bbox_msg.ymax - bbox_msg.ymin
            bbox_width = bbox_msg.xmax - bbox_msg.xmin

            if (bbox_width * bbox_height) < self.AREA_THRESHOLD:
                rospy.loginfo(f"Skipping object {class_name} with area {bbox_width * bbox_height} since it is smaller than the threshold of {self.AREA_THRESHOLD}")
                continue
            
            mask_ids = np.argwhere(mask.reshape((height * width)) > 0)
            detection.mask = list(mask_ids.flat)

            detection.score = confidence

            detections.append(detection)
            
            idx += 1

        ros_detections = Detections()
        ros_detections.width, ros_detections.height = width, height
        ros_detections.detections = detections

        return ros_detections    
    
    def inference_model_intern(self, np_img):
        image_source, image = load_image_numpy(np_img)

        self.sam2_predictor.set_image(image_source)

        torch.autocast(device_type="cuda", enabled=False).__enter__()
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
        )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        rospy.loginfo(labels)
        return input_boxes, masks, class_names, confidences

if __name__ == "__main__":

    try:
        rospy.init_node(f'GROUNDED_SAM_2')
        GS2_ROS_Wrapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
