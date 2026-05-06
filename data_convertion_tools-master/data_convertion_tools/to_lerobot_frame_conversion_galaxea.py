#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from to_lerobot_utils import extract_data

def frame_conversion_galaxea(DATA, MOTORS, timestamp, DELAY):
    concatenated_state = [0.] * len(MOTORS)
    concatenated_action = [0.] * len(MOTORS)
    SINGLE_ARM_DOF = sum('left_arm' in m for m in MOTORS)

    # ---- state ----
    elem = extract_data(DATA, "/hdas/feedback_gripper_left", timestamp)
    idx_start = MOTORS.index("effector.position.left_gripper")
    assert len(elem["position"]) == 1
    concatenated_state[idx_start : idx_start + len(elem["position"])] = elem["position"]

    elem = extract_data(DATA, "/hdas/feedback_gripper_right", timestamp)
    idx_start = MOTORS.index("effector.position.right_gripper")
    assert len(elem["position"]) == 1
    concatenated_state[idx_start : idx_start + len(elem["position"])] = elem["position"]
    
    elem = extract_data(DATA, "/hdas/feedback_arm_left", timestamp)
    idx_start = MOTORS.index("joint.position.left_arm_0")
    assert len(elem["position"]) == SINGLE_ARM_DOF
    concatenated_state[idx_start : idx_start + len(elem["position"])] = elem["position"]

    elem = extract_data(DATA, "/hdas/feedback_arm_right", timestamp)
    idx_start = MOTORS.index("joint.position.right_arm_0")
    assert len(elem["position"]) == SINGLE_ARM_DOF
    concatenated_state[idx_start : idx_start + len(elem["position"])] = elem["position"]

    # ---- action ----
    # elem = extract_data(DATA, "/hdas/feedback_gripper_left", timestamp + DELAY)
    elem = extract_data(DATA, "/motion_target/target_position_gripper_left", timestamp + DELAY)
    idx_start = MOTORS.index("effector.position.left_gripper")
    assert len(elem["position"]) == 1
    concatenated_action[idx_start : idx_start + len(elem["position"])] = elem["position"]

    # elem = extract_data(DATA, "/hdas/feedback_gripper_right", timestamp + DELAY)
    elem = extract_data(DATA, "/motion_target/target_position_gripper_right", timestamp + DELAY)
    idx_start = MOTORS.index("effector.position.right_gripper")
    assert len(elem["position"]) == 1
    concatenated_action[idx_start : idx_start + len(elem["position"])] = elem["position"]
    
    # elem = extract_data(DATA, "/hdas/feedback_arm_left", timestamp + DELAY)
    elem = extract_data(DATA, "/motion_target/target_joint_state_arm_left", timestamp + DELAY)
    idx_start = MOTORS.index("joint.position.left_arm_0")
    assert len(elem["position"]) == SINGLE_ARM_DOF
    concatenated_action[idx_start : idx_start + len(elem["position"])] = elem["position"]

    # elem = extract_data(DATA, "/hdas/feedback_arm_right", timestamp + DELAY)
    elem = extract_data(DATA, "/motion_target/target_joint_state_arm_right", timestamp + DELAY)
    idx_start = MOTORS.index("joint.position.right_arm_0")
    assert len(elem["position"]) == SINGLE_ARM_DOF
    concatenated_action[idx_start : idx_start + len(elem["position"])] = elem["position"]

    frame = {
        "observation.state": torch.tensor(concatenated_state),
        "action": torch.tensor(concatenated_action),
    }

    # image
    image = extract_data(DATA, "/hdas/camera_head/right_raw/image_raw_color/compressed", timestamp)["image"]
    # print("head: ", type(image), image.shape)
    frame["observation.images.head"] = image

    if "/hdas/camera_wrist_left/color/image_raw/compressed" in DATA:
        image = extract_data(DATA, "/hdas/camera_wrist_left/color/image_raw/compressed", timestamp)["image"]
        # print("hand_left: ", type(image), image.shape)
        frame["observation.images.hand_left"] = image
    elif "/hdas/camera_wrist_left/color/image_rect_raw/compressed" in DATA:
        image = extract_data(DATA, "/hdas/camera_wrist_left/color/image_rect_raw/compressed", timestamp)["image"]
        frame["observation.images.hand_left"] = image
    else:
        raise ValueError("!! Missing left hand camera!!")
    
    if "/hdas/camera_wrist_right/color/image_raw/compressed" in DATA:
        image = extract_data(DATA, "/hdas/camera_wrist_right/color/image_raw/compressed", timestamp)["image"]
        # print("hand_right: ", type(image), image.shape)
        frame["observation.images.hand_right"] = image
    elif "/hdas/camera_wrist_right/color/image_rect_raw/compressed" in DATA:
        image = extract_data(DATA, "/hdas/camera_wrist_right/color/image_rect_raw/compressed", timestamp)["image"]
        frame["observation.images.hand_right"] = image
    else:
        raise ValueError("!! Missing right hand camera!!")
    
    return frame