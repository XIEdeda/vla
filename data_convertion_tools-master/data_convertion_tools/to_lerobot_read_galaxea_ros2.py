#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, CompressedImage

import numpy as np
import cv2, os
import datetime
from pathlib import Path
"""
refer to: https://mcap.dev/docs/python/ros2_example#reading-messages

# sudo apt install ros-humble-ros-base
# sudo apt install ros-humble-ros2bag
# sudo apt install ros-humble-rosbag2-transport
# sudo apt install ros-humble-rosbag2-py

apt-cache policy ros-humble-rosbag2-storage-mcap
sudo apt install ros-humble-rosbag2-storage-mcap


conda create -y -n ros2 python=3.10
conda activate ros2


Problem:
ImportError: /home/vla/miniconda3/envs/ros2/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /opt/ros/humble/lib/librosbag2_compression.so)
Solution:
conda install -c conda-forge gcc=12.1.0
refer to: https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin


pip install opencv-python


# colcon 安装
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lark
pip install empy==3.3.4
pip3 install -U colcon-common-extensions
"""

def to_sec(t):
    return t.sec + t.nanosec * 1e-9

def read_bag(bag_file_path, unix_timestamp, IMAGE_SHAPE, SINGLE_ARM_DOF, **kwargs):
    try:
        # 创建 Bag 文件存储选项
        if bag_file_path.endswith(".db3"):
            storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id='sqlite3')
        elif bag_file_path.endswith(".mcap"):
            storage_options = rosbag2_py.StorageOptions(uri=bag_file_path, storage_id='mcap')
        else:
            raise ValueError(f"❌ Invalid file extension: {bag_file_path}. Expected '.mcap', '.db3'")
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    
        # 打开 Bag 文件 (Reader 模式)
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
    except FileNotFoundError:
        print(f"❌ File does not exist: {bag_file_path}")
        return {}, {}
    
    TopicName2Type = {}
    DATA = {}
    # 读取所有可用的 Topic 和类型信息
    topics = reader.get_all_topics_and_types()
    # print("\nAvailable topics & types:")
    for topic in topics:
        # print("Topic:" + topic.name + ", Type: " + topic.type)      
        DATA[topic.name] = []
        TopicName2Type[topic.name] = topic.type


    # 读取每条消息
    # print("\nReading messages from the bag:")
    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if TopicName2Type[topic] == "sensor_msgs/msg/CompressedImage":
            """
            /hdas/camera_head/left_raw/image_raw_color/compressed
            /hdas/camera_head/right_raw/image_raw_color/compressed
            /hdas/camera_wrist_left/color/image_rect_raw/compressed
            /hdas/camera_wrist_right/color/image_rect_raw/compressed
            /hdas/camera_wrist_left/color/image_raw/compressed
            /hdas/camera_wrist_right/color/image_raw/compressed
            """
            msg = deserialize_message(data, CompressedImage)
            stamp = to_sec(msg.header.stamp)
            if topic not in DATA:
                DATA[topic] = []

            # 将字节流转换为 NumPy 数组
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            # 使用 OpenCV 解码 JPEG 图像, 解码为 BGR 图像
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if stamp > unix_timestamp:
                DATA[topic].append({
                    "timestamp": stamp, "image": cv2.resize(img, IMAGE_SHAPE, interpolation=cv2.INTER_AREA),})
            else:
                DATA[topic].append({
                    "timestamp": timestamp * 1e-9, "image": cv2.resize(img, IMAGE_SHAPE, interpolation=cv2.INTER_AREA),})


        elif TopicName2Type[topic] == "sensor_msgs/msg/JointState":
            """
            /motion_target/target_joint_state_arm_left
            /motion_target/target_joint_state_arm_right
            /motion_target/target_position_gripper_left
            /motion_target/target_position_gripper_right
            /hdas/feedback_arm_left
            /hdas/feedback_arm_right
            /hdas/feedback_gripper_left
            /hdas/feedback_gripper_right
            /hdas/feedback_torso
            /hdas/feedback_chassis
            """
            msg = deserialize_message(data, JointState)
            stamp = to_sec(msg.header.stamp)
            if topic not in DATA:
                DATA[topic] = []
            
            if topic in ["/hdas/feedback_arm_left", "/hdas/feedback_arm_right"]:
                if stamp > unix_timestamp:
                    DATA[topic].append({
                        "timestamp": stamp, "position": list(msg.position)[:SINGLE_ARM_DOF],
                        # "velocity": list(msg.velocity)[:SINGLE_ARM_DOF], "effort": list(msg.effort)[:SINGLE_ARM_DOF],
                    })
                else:
                    DATA[topic].append({
                        "timestamp": timestamp * 1e-9, "position": list(msg.position)[:SINGLE_ARM_DOF],
                        # "velocity": list(msg.velocity)[:SINGLE_ARM_DOF], "effort": list(msg.effort)[:SINGLE_ARM_DOF],
                    })
            else:
                if stamp > unix_timestamp:
                    DATA[topic].append({
                        "timestamp": stamp, "position": list(msg.position),
                        # "velocity": list(msg.velocity), "effort": list(msg.effort),
                    })
                else:
                    DATA[topic].append({
                        "timestamp": timestamp * 1e-9, "position": list(msg.position),
                        # "velocity": list(msg.velocity), "effort": list(msg.effort),
                    })

    for k, v in list(DATA.items()):
        if len(v) == 0:
            DATA.pop(k)
    return DATA, TopicName2Type


if __name__ == "__main__":
    
    dataset_path = ".../r1lite/202512.use.the.right.gripper.to.pick.up.a.Sanhuang.Plan.and.hold.it/RB250715039_20251208172059030_RAW/" # folder, not mcap file
    
    bag_file_paths = [str(file) for ext in ("*.mcap", "*.db3") for file in Path(dataset_path).rglob(ext)]
    for bag_file_path in bag_file_paths:
        print("cfz 0 |", bag_file_path)

        unix_timestamp = datetime.datetime(2025, 5, 10, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        IMAGE_SHAPE = (224, 168)
        if "lite" in bag_file_path:
            SINGLE_ARM_DOF = 6
        elif "pro" in bag_file_path:
            SINGLE_ARM_DOF = 7
        else:
            raise ValueError("Need to assign a value to SINGLE_ARM_DOF")
        DATA, TopicName2Type = read_bag(bag_file_path, unix_timestamp, IMAGE_SHAPE, SINGLE_ARM_DOF)
        for key in sorted(DATA.keys()):
            print(key, len(DATA[key]))