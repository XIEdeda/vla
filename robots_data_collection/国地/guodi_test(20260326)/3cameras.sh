#!/bin/bash

ros2 launch realsense2_camera three_multi_camera_launch.py \
  camera_name3:=camera_head \
  serial_no3:=_213622073652 \
  depth_module.depth_profile3:=640x480x30 \
  rgb_camera.color_profile3:=640x480x30 \
  camera_name2:=camera_right \
  serial_no2:=_315122272235 \
  depth_module.depth_profile2:=640x480x30 \
  depth_module.color_profile2:=640x480x30 \
  camera_name1:=camera_left \
  serial_no1:=_230322272878 \
  depth_module.depth_profile1:=640x480x30 \
  depth_module.color_profile1:=640x480x30
