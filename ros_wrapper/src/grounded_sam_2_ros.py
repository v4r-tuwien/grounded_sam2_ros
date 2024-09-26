#!/usr/bin/python3.10
import torch
import numpy as np
import cv2
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image_numpy, predict
import rospy
from actionlib import SimpleActionServer
from sensor_msgs.msg import RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction


class GS2_ROS_Wrapper():

    def __init__(self):
        """Setting hyperparameters"""
        rospy.loginfo("Initializing Grounded SAM 2 ROS Wrapper")
        BASE_PATH = "/root/grounded_sam2/"
        # VERY important: text queries need to be lowercased + end with a dot
        self.TEXT_PROMPT = "apple. banana. orange. lemon. candy. pitcher. can. mug. cup. plate. rubbish. garbage. table. chair. coffee. mustard. tube. bottle. tissues. cable. bowl. pear. banana. box. peach"
        SAM2_CHECKPOINT = BASE_PATH + "./checkpoints/sam2_hiera_large.pt"
        SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = BASE_PATH + "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = BASE_PATH + "gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.BOX_THRESHOLD = 0.3
        self.TEXT_THRESHOLD = 0.25
        print(f"{torch.cuda.is_available() = }")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.bridge = CvBridge()
        self.server = SimpleActionServer(f'/object_detector/grounded_sam2', GenericImgProcAnnotatorAction, self.service_call, False)
        self.server.start()
        rospy.loginfo("Ready to receive images")
    
    def service_call(self, goal):
        rospy.loginfo(f"Received image with shape {goal.rgb.width}x{goal.rgb.height}")
        rgb = goal.rgb
        width, height = rgb.width, rgb.height

        try:
            image = self.bridge.imgmsg_to_cv2(rgb, "rgb8")
        except CvBridgeError as e:
            print(e)
        
        ros_detections = self.inference(image, rgb.header)

        if ros_detections.success:
            self.server.set_succeeded(ros_detections)
        else:
            self.server.set_aborted(ros_detections)

    def inference(self, image, rgb_header):
        scale = rospy.get_param("/gsam2/scale", 1)
        do_sharpen = rospy.get_param("/gsam2/do_sharpen", False)
        rospy.loginfo(f"GSam2 is using: {scale = }, {do_sharpen = }")
        orig_shape = (image.shape[1], image.shape[0])
        image = cv2.resize(image, (image.shape[1]*scale, image.shape[0]*scale))
        if do_sharpen:
            blurred = cv2.GaussianBlur(image, (9, 9), 10)
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            image = sharpened
        results = self.inference_model_intern(image)  # predict on an image
        boxes, masks, class_names, confidences = results
        rospy.loginfo(f"{class_names  = }")
        rospy.loginfo(f"Found {len(results[2])} objects")

        height, width, channels = image.shape
        bboxes_ros = []
        label_image = np.full((height, width), -1, np.int16)
        idx = 0
        for box, mask in zip(boxes, masks):
            bb = RegionOfInterest()
            # bb is in format [x1,y1,x2,y2] with (x1,y1) being the top left corner
            xmin = int(box[0]/scale)
            ymin = int(box[1]/scale)
            xmax = int(box[2]/scale)
            ymax = int(box[3]/scale)
            bb.x_offset = xmin
            bb.y_offset = ymin
            bb.height = ymax - ymin
            bb.width = xmax - xmin
            bb.do_rectify = False
            bboxes_ros.append(bb)
            
            label_image[mask > 0] = idx

            idx += 1

        if len(class_names) > 0:
            rospy.loginfo(f"Found {len(class_names)} objects. Returning results")
            label_image = cv2.resize(label_image, orig_shape)
            mask_image = self.bridge.cv2_to_imgmsg(label_image, encoding="16SC1", header=rgb_header)
            server_result = GenericImgProcAnnotatorResult()
            server_result.success = True
            server_result.bounding_boxes = bboxes_ros
            server_result.class_names = class_names
            server_result.class_confidences = confidences
            server_result.image = mask_image
        else:
            rospy.loginfo("No objects found")
            server_result = GenericImgProcAnnotatorResult()
            server_result.success = False

        return server_result 
    
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

        masks, scores, logits = self.sam2_predictor.predict(
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
        return input_boxes, masks, labels, confidences

if __name__ == "__main__":

    try:
        rospy.init_node(f'GROUNDED_SAM_2')
        GS2_ROS_Wrapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
