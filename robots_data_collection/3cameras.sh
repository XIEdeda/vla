#!/bin/bash

ros2 launch realsense2_camera three_multi_camera_launch.py \
  camera_name3:=camera_head \
  serial_no3:=_213522072131 \
  depth_module.depth_profile3:=1280x720x30 \
  rgb_camera.color_profile3:=1280x720x30 \
  camera_name2:=camera_right \
  serial_no2:=_230422272423 \
  depth_module.depth_profile2:=640x480x30 \
  depth_module.color_profile2:=640x480x30 \
  camera_name1:=camera_left \
  serial_no1:=_230322270825 \
  depth_module.depth_profile1:=640x480x30 \
  depth_module.color_profile1:=640x480x30
