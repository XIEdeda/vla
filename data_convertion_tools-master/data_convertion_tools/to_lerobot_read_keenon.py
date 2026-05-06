#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from keenon_msgs.msg import ArmData


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


conda create -y -n lerobot-2.1 python=3.10
conda activate lerobot-2.1
cd .../lerobot/
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

conda create -y -n lerobot-3.0 python=3.10
conda activate lerobot-3.0
pip install lerobot


pip install opencv-python
pip install matplotlib

# colcon 安装
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lark
pip install empy==3.3.4
pip3 install -U colcon-common-extensions

# 在非 conda 环境下运行编译
cd ./keenon_msgs
rm -rf build/ install/ log/
colcon build
source install/setup.bash
"""

def to_sec(t):
    return t.sec + t.nanosec * 1e-9


def read_images(folder, IMAGE_SHAPE):
    rows = []
    for path in sorted(folder.rglob("*.jpg")):
        stamp = float(path.stem.split('_')[1]) * 1e-9
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rows.append({"timestamp": stamp, "image": cv2.resize(img, IMAGE_SHAPE, interpolation=cv2.INTER_AREA),})
    return rows


def read_arms(path, SINGLE_ARM_DOF):
    if not path.exists():
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                ts_ns_str, values = line.split(" ", 1)
                stamp = float(ts_ns_str.strip()) * 1e-9
                arr = [float(value) for value in values.strip().split(",")]
                arr = arr[:SINGLE_ARM_DOF] + [arr[SINGLE_ARM_DOF + 1]]
                rows.append({"timestamp": stamp, "position": arr,})
        rows.sort(key=lambda x: x["timestamp"])
    except FileNotFoundError:
        print(f"{str(path)} not found")
    except PermissionError:
        print(f"no permission to read {str(path)}")
    return rows


def read_txt(left_arm_path, unix_timestamp, IMAGE_SHAPE, SINGLE_ARM_DOF, **kwargs):
    
    img_root = Path(left_arm_path.replace("arm_data", "temp_images")).parent # root_path / "temp_images" / episode_name
    arm_root = Path(left_arm_path).parent # root_path / "arm_data" / episode_name

    DATA = {}
    DATA["/camera3/camera_head/color/image_raw/compressed"] = read_images(img_root / "head_image_images", IMAGE_SHAPE)
    DATA["/camera1/camera_left/color/image_rect_raw/compressed"] = read_images(img_root / "left_wrist_image_images", IMAGE_SHAPE)
    DATA["/camera2/camera_right/color/image_rect_raw/compressed"] = read_images(img_root / "right_wrist_image_images", IMAGE_SHAPE)
    DATA["/left_data"] = read_arms(arm_root / "left_data.txt", SINGLE_ARM_DOF)
    DATA["/right_data"] = read_arms(arm_root / "right_data.txt", SINGLE_ARM_DOF)

    for k, v in list(DATA.items()):
        if len(v) == 0:
            DATA.pop(k)
    return DATA, {}


def read_bag(bag_file_path, unix_timestamp, IMAGE_SHAPE, SINGLE_ARM_DOF, **kwargs):
    if bag_file_path.endswith(".txt"):
        return read_txt(bag_file_path, unix_timestamp, IMAGE_SHAPE, SINGLE_ARM_DOF)
    
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
            /camera1/camera_left/color/image_rect_raw/compressed
            /camera2/camera_right/color/image_rect_raw/compressed
            /camera3/camera_head/color/image_raw/compressed
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


        elif TopicName2Type[topic] == "sc_ros2/msg/ArmData":
            """
            /left_data
            /right_data
            """
            msg = deserialize_message(data, ArmData)
            stamp = to_sec(msg.header.stamp)
            if topic not in DATA:
                DATA[topic] = []

            arr = list(msg.data)
            arr = arr[:SINGLE_ARM_DOF] + [arr[SINGLE_ARM_DOF + 1]]
        
            if stamp > unix_timestamp:
                DATA[topic].append({
                    "timestamp": stamp, "position": arr,
                    # "velocity": list(msg.velocity)[:SINGLE_ARM_DOF], "effort": list(msg.effort)[:SINGLE_ARM_DOF],
                })
            else:
                DATA[topic].append({
                    "timestamp": timestamp * 1e-9, "position": arr,
                    # "velocity": list(msg.velocity)[:SINGLE_ARM_DOF], "effort": list(msg.effort)[:SINGLE_ARM_DOF],
                })
        

    for k, v in list(DATA.items()):
        if len(v) == 0:
            DATA.pop(k)
    return DATA, TopicName2Type


if __name__ == "__main__":
    
    dataset_path = "/mnt/d/github/keenon/0421_cyw_coffee_02/" # folder, not mcap file
    
    bag_file_paths = [str(file) for ext in ("*.db3", "*.mcap", "*left_data.txt") for file in Path(dataset_path).rglob(ext)]
    for bag_file_path in bag_file_paths:
        print("bag_file_path:", bag_file_path)

        unix_timestamp = datetime.datetime(2025, 5, 10, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        IMAGE_SHAPE = (224, 168)
        SINGLE_ARM_DOF = 7
        DATA, TopicName2Type = read_bag(bag_file_path, unix_timestamp, IMAGE_SHAPE, SINGLE_ARM_DOF)
        for key in sorted(DATA.keys()):
            # print(key, len(DATA[key]))
            print("Topic:", key + ", num_frames:", len(DATA[key]))    