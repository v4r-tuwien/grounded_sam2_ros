#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/dev/bus/usb:/dev/bus/usb" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/home/hoenig/temp/ichores_pipeline/src/grounded_sam2_ros:/root/grounded_sam2_ros" \
--name=gsam2_docker_ros_v0 gsam2_docker_ros