#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import numpy as np
# import os#, cv2
# import torch
import datetime
import argparse
from pathlib import Path
from to_lerobot_utils import *

"""
# V21
conda activate lerobot-2.1 && source ./keenon_msgs/install/setup.bash
python to_lerobot_main.py --robot_type keenon --ros_version 2 --lerobot_dataset_version 2.1 --delay 0.1

# V30
conda activate lerobot-3.0 && source ./keenon_msgs/install/setup.bash
python to_lerobot_main.py --robot_type keenon --ros_version 2 --lerobot_dataset_version 3.0 --delay 0.1
"""

# -----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--delay', type=float, default=0.1)
parser.add_argument('--ros_version', type=int, choices=[1, 2], default=2)
parser.add_argument('--lerobot_dataset_version', type=float, choices=[2.1, 3.0], default=2.1)
parser.add_argument('--robot_type', choices=['keenon', 'galaxea_r1lite', 'galaxea_r1pro'], default='galaxea_r1lite',
                    help='Machine model identifier (choices: galaxea_r1lite, galaxea_r1lite, ti5)')
args = parser.parse_args()
DELAY = args.delay
ROS_VERSION = args.ros_version
ROBOT_TYPE = args.robot_type
LEROBOT_DATASET_VERSION = args.lerobot_dataset_version
# FILTER_OBJECTS = "feedback and motion_target" # or "feedback"
FILTER_OBJECTS = "feedback"
BUILDFUTURETRAJ = True
CHUNKSIZE = 1
PLOT = True

print("ROS version =", ROS_VERSION)
print("lerobot dataset version =", LEROBOT_DATASET_VERSION)

if ROS_VERSION == 2 and ROBOT_TYPE == "keenon":
    from to_lerobot_frame_conversion_keenon import frame_conversion_keenon
    from to_lerobot_read_keenon import read_bag
elif ROS_VERSION == 2 and ROBOT_TYPE in ["galaxea_r1lite", "galaxea_r1pro"]:
    from to_lerobot_frame_conversion_galaxea import frame_conversion_galaxea
    from to_lerobot_read_galaxea_ros2 import read_bag
else:
    raise ValueError(f"ROS version {ROS_VERSION} + robot type {ROBOT_TYPE} are not currently supported.")


# IMAGE_SHAPE = (640, 480)
# IMAGE_SHAPE = (256, 256) # width, height
IMAGE_SHAPE = (224, 168) # width, height
unix_timestamp = datetime.datetime(2025, 5, 10, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()


if ROS_VERSION == 2 and ROBOT_TYPE == "galaxea_r1lite":
    MOTORS = [ # refer to: any4lerobot/agibot2lerobot/agibot_utils/config.py
        "effector.position.left_gripper",
        "effector.position.right_gripper",
        "joint.position.left_arm_0",
        "joint.position.left_arm_1",
        "joint.position.left_arm_2",
        "joint.position.left_arm_3",
        "joint.position.left_arm_4",
        "joint.position.left_arm_5",
        "joint.position.right_arm_0",
        "joint.position.right_arm_1",
        "joint.position.right_arm_2",
        "joint.position.right_arm_3",
        "joint.position.right_arm_4",
        "joint.position.right_arm_5",
        # "joint.position.torse_0",
        # "joint.position.torse_1",
        # "joint.position.torse_2",
        # "joint.position.torse_3",
    ]
elif ROS_VERSION == 2 and ROBOT_TYPE in ["keenon", "galaxea_r1pro"]:
    MOTORS = [ # refer to: any4lerobot/agibot2lerobot/agibot_utils/config.py
        "effector.position.left_gripper",
        "effector.position.right_gripper",
        "joint.position.left_arm_0",
        "joint.position.left_arm_1",
        "joint.position.left_arm_2",
        "joint.position.left_arm_3",
        "joint.position.left_arm_4",
        "joint.position.left_arm_5",
        "joint.position.left_arm_6",
        "joint.position.right_arm_0",
        "joint.position.right_arm_1",
        "joint.position.right_arm_2",
        "joint.position.right_arm_3",
        "joint.position.right_arm_4",
        "joint.position.right_arm_5",
        "joint.position.right_arm_6",
        # "joint.position.torse_0",
        # "joint.position.torse_1",
        # "joint.position.torse_2",
        # "joint.position.torse_3",
    ]
else:
    raise ValueError(f"ROS version {ROS_VERSION} + robot type {ROBOT_TYPE} are not currently supported.")

if FILTER_OBJECTS == "feedback and motion_target":
    # 截断时同时考虑 feedback 和 motion_target
    THRESHOLDS = [0.02 for _ in range(len(MOTORS) * 2)]
elif FILTER_OBJECTS == "feedback":
    # 截断时只考虑 feedback
    THRESHOLDS = [0.02 for _ in range(len(MOTORS))]
else:
    raise ValueError("FILTER_OBJECTS must be chosen from either 'feedback and motion_target', or from 'feedback' only")


paths = [
    # '20260408.coffee.by.cfz',
    "0421_cyw_coffee_02",
    # "", # all
]
for idx in range(len(paths)):
    paths[idx] = "/mnt/d/github/keenon/" + paths[idx]


isMerged = False
if isMerged:
    # 创建空数据集
    repo_id = input("pleas input dataset repo_id: ")
    if LEROBOT_DATASET_VERSION == 2.1:
        root = Path("./lerobot_data_v2.1/" + repo_id + ".img." + str(IMAGE_SHAPE[0]) + 'x' + str(IMAGE_SHAPE[1]) + '.' + chr(ord('N') - int(DELAY * 10) + 1) + ".dof." + str(len(MOTORS)))
    elif LEROBOT_DATASET_VERSION == 3.0:
        root = Path("./lerobot_data_v3.0/" + repo_id + + ".img." + str(IMAGE_SHAPE[0]) + 'x' + str(IMAGE_SHAPE[1]) + '.' + chr(ord('N') - int(DELAY * 10) + 1) + ".dof." + str(len(MOTORS)))
    else:
        raise ValueError("Currently only lerobot dataset versions 2.1 and 3.0 are supported")

    dataset = create_empty_dataset(
        repo_id,
        root,
        robot_type=ROBOT_TYPE,
        MOTORS=MOTORS,
        IMAGE_SHAPE=IMAGE_SHAPE,
        mode="video",
    )

for dataset_path in paths:
    bag_file_paths = sorted([str(file) for ext in ("*.db3", "*.mcap", "*left_data.txt") for file in Path(dataset_path).rglob(ext)])

    if not isMerged:
        # 创建空数据集
        repo_id = dataset_path.split('/')[-1]
        print("repo_id:", repo_id)
        if LEROBOT_DATASET_VERSION == 2.1:
            root = Path("./lerobot_data_v2.1/" + repo_id + ".img." + str(IMAGE_SHAPE[0]) + 'x' + str(IMAGE_SHAPE[1]) + '.' + chr(ord('N') - int(DELAY * 10) + 1) + ".dof." + str(len(MOTORS)))
        elif LEROBOT_DATASET_VERSION == 3.0:
            root = Path("./lerobot_data_v3.0/" + repo_id + ".img." + str(IMAGE_SHAPE[0]) + 'x' + str(IMAGE_SHAPE[1]) + '.' + chr(ord('N') - int(DELAY * 10) + 1) + ".dof." + str(len(MOTORS)))
        else:
            raise ValueError("Currently only lerobot dataset versions 2.1 and 3.0 are supported")

        dataset = create_empty_dataset(
            repo_id,
            root,
            robot_type=ROBOT_TYPE,
            MOTORS=MOTORS,
            IMAGE_SHAPE=IMAGE_SHAPE,
            BUILDFUTURETRAJ=BUILDFUTURETRAJ,
            CHUNKSIZE=CHUNKSIZE,
            mode="video",
        )

    # -----------------------------------------------------------------------------------------------------------------------
    # 第一次遍历所有 bag 包, 进行两端静止截断等各种过滤方法
    temp_count = 0 # todo: delete
    record_states = dict()
    for bag_file_path in bag_file_paths:
        if not os.path.exists(bag_file_path) or not os.path.isfile(bag_file_path):
            continue
        print("file:", bag_file_path)
        
        if '/' + ROBOT_TYPE + '/' in bag_file_path:
            task = ' '.join(bag_file_path.split('/' + ROBOT_TYPE + '/')[1].split('/')[0].split('.')[1:-2]) # todo: 文件名命名方式: data.XXXX.by.who
        else:
            task = "todo"

        print("task:", task)

        DATA, TOPICS = read_bag(bag_file_path, unix_timestamp=unix_timestamp, IMAGE_SHAPE=IMAGE_SHAPE, SINGLE_ARM_DOF=sum('left_arm' in m for m in MOTORS))
        
        # 跳过空数据
        if not DATA:
            continue

        if ROS_VERSION == 2 and ROBOT_TYPE == "keenon":
            if "/camera3/camera_head/color/image_raw/compressed" not in DATA:
                print(bag_file_path, "is missing head camera !!")
                delete_bag_file(bag_file_path)
                continue
            if "/camera1/camera_left/color/image_rect_raw/compressed" not in DATA:
                print(bag_file_path, "is missing left wrist camera !!")
                delete_bag_file(bag_file_path)
                continue
            if "/camera2/camera_right/color/image_rect_raw/compressed" not in DATA:
                print(bag_file_path, "is missing right wrist camera !!")
                delete_bag_file(bag_file_path)
                continue
        elif ROS_VERSION == 2 and ROBOT_TYPE in ["galaxea_r1lite", "galaxea_r1pro"]:
            if "/hdas/camera_head/right_raw/image_raw_color/compressed" not in DATA:
                print(bag_file_path, "is missing head camera !!")
                delete_bag_file(bag_file_path)
                continue
            if "/hdas/camera_wrist_left/color/image_raw/compressed" not in DATA and "/hdas/camera_wrist_left/color/image_rect_raw/compressed" not in DATA:
                print(bag_file_path, "is missing left wrist camera !!")
                delete_bag_file(bag_file_path)
                continue
            if "/hdas/camera_wrist_right/color/image_raw/compressed" not in DATA and "/hdas/camera_wrist_right/color/image_rect_raw/compressed" not in DATA:
                print(bag_file_path, "is missing right wrist camera !!")
                delete_bag_file(bag_file_path)
                continue
        else:
            raise ValueError(f"ROS version {ROS_VERSION} + robot type {ROBOT_TYPE} are not currently supported.")

        # 当前bag包中所有 topic 时间戳两侧对齐 todo: move to read_bag()
        start, end = float("inf"), 0
        for topic, value in DATA.items():
            start = min(value[0]["timestamp"], start)
            end = max(value[-1]["timestamp"], end)
            
            # check: 单调递增检查
            for idx in range(1, len(value)):
                if value[idx]["timestamp"] < value[idx - 1]["timestamp"]:
                    raise ValueError(topic, "is not monotonically increasing")

        for topic, value in DATA.items():
            idx_left = 0
            while idx_left < len(value) and value[idx_left]["timestamp"] < start:
                idx_left += 1
            idx_right = len(value) - 1
            while idx_right > -1 and value[idx_right]["timestamp"] > end:
                idx_right -= 1
            
            if idx_left != 0 or idx_right != len(value) - 1:
                print(topic, ", start idx =", idx_left, ", end idx =", idx_right, ", all =", len(value))
                value = value[idx_left : idx_right + 1]
        
        # ----------------------------------------------------------------------- #
        
        # 重构统一的时间戳序列 (针对单个 bag 包操作)
        timestamps = []
        base_topic = ""
        count = float("inf")
        for topic in DATA.keys():
            if ("camera" in topic or "image" in topic) and "depth" not in topic and len(DATA[topic]) < count:
                count = len(DATA[topic])
                base_topic = topic
        # print("base_topic =", base_topic)

        assert count > 1
        period = (DATA[base_topic][-1]["timestamp"] - DATA[base_topic][0]["timestamp"]) / (count - 1)
        # print("period =", period)

        for idx in range(len(DATA[base_topic])):
            if idx == 0:
                timestamps.append(DATA[base_topic][idx]["timestamp"])
            else:
                timestamp_new = DATA[base_topic][idx]["timestamp"]
                if timestamp_new - timestamps[-1] >= period * 0.5:
                    timestamps.append(timestamp_new)

        # check
        for idx in range(len(timestamps) - 1):
            assert timestamps[idx + 1] - timestamps[idx] >= period * 0.5

        if FILTER_OBJECTS == "feedback and motion_target":
            # 截断时同时考虑 feedback 和 motion_target
            states = [[] for _ in range(len(MOTORS) * 2)]
        elif FILTER_OBJECTS == "feedback":
            # 截断时只考虑 feedback
            states = [[] for _ in range(len(MOTORS))]
        else:
            raise ValueError("FILTER_OBJECTS must be chosen from either 'feedback and motion_target', or from 'feedback' only")
        # states = [[] for _ in range(len(MOTORS) * 2)]
        for timestamp in timestamps:

            if ROS_VERSION == 2 and ROBOT_TYPE == "keenon":
                SINGLE_ARM_DOF=sum('left_arm' in m for m in MOTORS)

                elem_left = extract_data(DATA, "/left_data", timestamp)
                idx_start = MOTORS.index("effector.position.left_gripper")
                states[idx_start].append(elem_left["position"][-1])

                elem_right = extract_data(DATA, "/right_data", timestamp)
                idx_start = MOTORS.index("effector.position.right_gripper")
                states[idx_start].append(elem_right["position"][-1])

                idx_start = MOTORS.index("joint.position.left_arm_0")
                assert len(elem_left["position"]) - 1 == SINGLE_ARM_DOF
                for idx in range(SINGLE_ARM_DOF):
                    states[idx_start + idx].append(elem_left["position"][idx])

                idx_start = MOTORS.index("joint.position.right_arm_0")
                assert len(elem_right["position"]) - 1 == SINGLE_ARM_DOF
                for idx in range(SINGLE_ARM_DOF):
                    states[idx_start + idx].append(elem_right["position"][idx])

                if FILTER_OBJECTS == "feedback and motion_target": # todo
                    idx_start = MOTORS.index("effector.position.left_gripper") + len(MOTORS)
                    states[idx_start].append(elem_left["position"][-1])

                    idx_start = MOTORS.index("effector.position.right_gripper") + len(MOTORS)
                    states[idx_start].append(elem_right["position"][-1])

                    idx_start = MOTORS.index("joint.position.left_arm_0") + len(MOTORS)
                    assert len(elem_left["position"]) - 1 == SINGLE_ARM_DOF
                    for idx in range(SINGLE_ARM_DOF):
                        states[idx_start + idx].append(elem_left["position"][idx])

                    idx_start = MOTORS.index("joint.position.right_arm_0") + len(MOTORS)
                    assert len(elem_right["position"]) - 1 == SINGLE_ARM_DOF
                    for idx in range(SINGLE_ARM_DOF):
                        states[idx_start + idx].append(elem_right["position"][idx])
                
            elif ROS_VERSION == 2 and ROBOT_TYPE in ["galaxea_r1lite", "galaxea_r1pro"]:
                SINGLE_ARM_DOF=sum('left_arm' in m for m in MOTORS)

                elem = extract_data(DATA, "/hdas/feedback_gripper_left", timestamp)
                idx_start = MOTORS.index("effector.position.left_gripper")
                assert len(elem["position"]) == 1
                for idx in range(len(elem["position"])):
                    states[idx_start + idx].append(elem["position"][idx])

                elem = extract_data(DATA, "/hdas/feedback_gripper_right", timestamp)
                idx_start = MOTORS.index("effector.position.right_gripper")
                assert len(elem["position"]) == 1
                for idx in range(len(elem["position"])):
                    states[idx_start + idx].append(elem["position"][idx])

                elem = extract_data(DATA, "/hdas/feedback_arm_left", timestamp)
                idx_start = MOTORS.index("joint.position.left_arm_0")
                assert len(elem["position"]) == SINGLE_ARM_DOF
                for idx in range(SINGLE_ARM_DOF):
                    states[idx_start + idx].append(elem["position"][idx])

                elem = extract_data(DATA, "/hdas/feedback_arm_right", timestamp)
                idx_start = MOTORS.index("joint.position.right_arm_0")
                assert len(elem["position"]) == SINGLE_ARM_DOF
                for idx in range(SINGLE_ARM_DOF):
                    states[idx_start + idx].append(elem["position"][idx])

                if FILTER_OBJECTS == "feedback and motion_target":
                    elem = extract_data(DATA, "/motion_target/target_position_gripper_left", timestamp)
                    idx_start = MOTORS.index("effector.position.left_gripper") + len(MOTORS)
                    assert len(elem["position"]) == 1
                    for idx in range(len(elem["position"])):
                        states[idx_start + idx].append(elem["position"][idx])

                    elem = extract_data(DATA, "/motion_target/target_position_gripper_right", timestamp)
                    idx_start = MOTORS.index("effector.position.right_gripper") + len(MOTORS)
                    assert len(elem["position"]) == 1
                    for idx in range(len(elem["position"])):
                        states[idx_start + idx].append(elem["position"][idx])

                    elem = extract_data(DATA, "/motion_target/target_joint_state_arm_left", timestamp)
                    idx_start = MOTORS.index("joint.position.left_arm_0") + len(MOTORS)
                    assert len(elem["position"]) == SINGLE_ARM_DOF
                    for idx in range(SINGLE_ARM_DOF):
                        states[idx_start + idx].append(elem["position"][idx])

                    elem = extract_data(DATA, "/motion_target/target_joint_state_arm_right", timestamp)
                    idx_start = MOTORS.index("joint.position.right_arm_0") + len(MOTORS)
                    assert len(elem["position"]) == SINGLE_ARM_DOF
                    for idx in range(SINGLE_ARM_DOF):
                        states[idx_start + idx].append(elem["position"][idx])
                
            else:
                raise ValueError(f"ROS version {ROS_VERSION} + robot type {ROBOT_TYPE} are not currently supported.")

        # 两端截断 (针对单个 bag 包操作)
        print("-" * 50)

        if "left.gripper" in bag_file_path and "right.gripper" not in bag_file_path:
            states_temp = [states[idx] for idx in range(len(states)) if "left" in MOTORS[idx % len(MOTORS)]]
            THRESHOLDS_temp = [THRESHOLDS[idx] for idx in range(len(THRESHOLDS)) if "left" in MOTORS[idx % len(MOTORS)]]
            print("parallel_two_ends_alternate input left shape:", len(states_temp), ",", len(states_temp[0]))
            subarrays = parallel_two_ends_alternate(states_temp, THRESHOLDS_temp)
        elif "right.gripper" in bag_file_path and "left.gripper" not in bag_file_path:
            states_temp = [states[idx] for idx in range(len(states)) if "right" in MOTORS[idx % len(MOTORS)]]
            THRESHOLDS_temp = [THRESHOLDS[idx] for idx in range(len(THRESHOLDS)) if "right" in MOTORS[idx % len(MOTORS)]]
            print("parallel_two_ends_alternate input right shape:", len(states_temp), ",", len(states_temp[0]))
            subarrays = parallel_two_ends_alternate(states_temp, THRESHOLDS_temp)
        else:
            print("parallel_two_ends_alternate input all shape:", len(states), ",", len(states[0]))
            subarrays = parallel_two_ends_alternate(states, THRESHOLDS)

        print("cfz | subarrays:", subarrays)
        if len(subarrays) > 0:
            if subarrays[-1][1] == len(states[0]) - 1:
                idx_end = subarrays[-1][0]
                for idx in range(len(states)):
                    states[idx] = states[idx][:idx_end]
                timestamps = timestamps[:idx_end]
                print("states new shape 1:", len(states), ",", len(states[0]))
            if subarrays[0][0] == 0:
                idx_start = subarrays[0][1] + 1
                for idx in range(len(states)):
                    states[idx] = states[idx][idx_start:]
                timestamps = timestamps[idx_start:]
                print("states new shape 2:", len(states), ",", len(states[0]))

        assert len(timestamps) == len(states[-1])
        states.append(timestamps) # 将时间戳追加到 states 最后一行
        if len(states) == 0:
            # print(f"len(states) == 0: {bag_file_path}")
            with open("./temp_record.txt", 'a', encoding="utf-8") as f:
                f.write(f"len(states) == 0: {bag_file_path}" + '\n')
            continue
        if len(states[-1]) == 0:
            with open("./temp_record.txt", 'a', encoding="utf-8") as f:
                f.write(f"len(states[-1]) == 0: {bag_file_path}" + '\n')
            continue
        record_states[bag_file_path] = states

        # temp_count += 1 # todo: for test
        # if temp_count > 20:
        #     break

    out_path="./plot_results/" + dataset_path.split('/')[-1] + '_' + chr(ord('N') - int(DELAY * 10) + 1) + '_' + datetime.datetime.now().strftime("%Y%m%d")
    print('-'*20, "bilateral truncation", '-'*20)
    if len(record_states) == 0:
        raise ValueError("1. record_states is empty!!")
    if PLOT:
        plot_and_save_hd(record_states, MOTORS, out_path=out_path + "_bilateral_truncation.png", suptitle=dataset_path.split('/')[-1])

    # for path, value in record_states.items():
    #     print("cfz test2:", path, len(value), len(value[-1]), value[-1][0], value[-1][-1])
    # raise ValueError("stop")

    """
    # todo: 将 record_states 中的 states 删掉(即前 len(MOTORS) 行), 只保留 timestamps(即最后一行), 减少内存压力
    for bag_file_path, value in record_states.items():
        value = value[-1]
    """
    # -----------------------------------------------------------------------------------------------------------------------
    # 第二次遍历所有 bag 包, 生成 lerobot 数据集
    for bag_file_path, states in record_states.items():
        if FILTER_OBJECTS == "feedback and motion_target":
            assert len(MOTORS) * 2 + 1 == len(states)
        elif FILTER_OBJECTS == "feedback":
            assert len(MOTORS) + 1 == len(states)
        else:
            raise ValueError("FILTER_OBJECTS must be chosen from either 'feedback and motion_target', or from 'feedback' only")
        
        timestamps = states[-1] # 两端截断后的时间序列
        # print("cfz test | timestamps 5:", len(timestamps), timestamps[0], timestamps[-1])

        if '/' + ROBOT_TYPE + '/' in bag_file_path:
            task = ' '.join(bag_file_path.split('/' + ROBOT_TYPE + '/')[1].split('/')[0].split('.')[1:-2]) # todo: data.XXXX.by.who
        else:
            task = "todo"

        print("task:", task)

        DATA, TOPICS = read_bag(bag_file_path, unix_timestamp=unix_timestamp, IMAGE_SHAPE=IMAGE_SHAPE, SINGLE_ARM_DOF=sum('left_arm' in m for m in MOTORS))
        
        # 跳过空数据
        if not DATA:
            continue
        
        # 当前bag包中所有 topic 时间戳两侧对齐 todo: move to read_bag()
        start, end = float("inf"), 0
        for topic, value in DATA.items():
            start = min(value[0]["timestamp"], start)
            end = max(value[-1]["timestamp"], end)
            
            # check: 单调递增检查
            for idx in range(1, len(value)):
                if value[idx]["timestamp"] < value[idx - 1]["timestamp"]:
                    raise ValueError(topic, "is not monotonically increasing")

        for topic, value in DATA.items():
            idx_left = 0
            while idx_left < len(value) and value[idx_left]["timestamp"] < start:
                idx_left += 1
            idx_right = len(value) - 1
            while idx_right > -1 and value[idx_right]["timestamp"] > end:
                idx_right -= 1
            
            if idx_left != 0 or idx_right != len(value) - 1:
                print(topic, ", start idx =", idx_left, ", end idx =", idx_right, ", all =", len(value))
                value = value[idx_left : idx_right + 1]
        
        # ----------------------------------------------------------------------- #

        for timestamp in timestamps:
            
            if ROS_VERSION == 2 and ROBOT_TYPE == "keenon":
                frame = frame_conversion_keenon(DATA, MOTORS, timestamp, DELAY, BUILDFUTURETRAJ, CHUNKSIZE)
            elif ROS_VERSION == 2 and ROBOT_TYPE in ["galaxea_r1lite", "galaxea_r1pro"]:
                frame = frame_conversion_galaxea(DATA, MOTORS, timestamp, DELAY)
            else:
                raise ValueError(f"ROS version {ROS_VERSION} + robot type {ROBOT_TYPE} are not currently supported.")

            if LEROBOT_DATASET_VERSION == 2.1:
                # frame["timestamp"] = timestamp # bug: https://github.com/huggingface/lerobot/issues/916
                dataset.add_frame(frame, task) # V21
            elif LEROBOT_DATASET_VERSION == 3.0:
                frame["task"] = task
                dataset.add_frame(frame) # V30
            else:
                raise ValueError("Currently only lerobot dataset versions 2.1 and 3.0 are supported")

        dataset.save_episode()
        print(bag_file_path + " end")
    
    if LEROBOT_DATASET_VERSION == 3.0:
        dataset.finalize()