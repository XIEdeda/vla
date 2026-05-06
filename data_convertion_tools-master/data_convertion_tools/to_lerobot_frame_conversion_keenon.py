#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from to_lerobot_utils import extract_data

def frame_conversion_keenon(DATA, MOTORS, timestamp, DELAY, BUILDFUTURETRAJ=False, CHUNKSIZE=1):
    concatenated_state = [0.] * len(MOTORS)
    concatenated_action = [[] for _ in range(CHUNKSIZE)]  if BUILDFUTURETRAJ and CHUNKSIZE > 1 else []
    SINGLE_ARM_DOF = sum('left_arm' in m for m in MOTORS)

    # ---- state ----
    elem_left = extract_data(DATA, "/left_data", timestamp)
    idx_start = MOTORS.index("effector.position.left_gripper")
    concatenated_state[idx_start] = elem_left["position"][-1]

    elem_right = extract_data(DATA, "/right_data", timestamp)
    idx_start = MOTORS.index("effector.position.right_gripper")
    concatenated_state[idx_start] = elem_right["position"][-1]
    
    idx_start = MOTORS.index("joint.position.left_arm_0")
    assert len(elem_left["position"]) - 1 == SINGLE_ARM_DOF
    concatenated_state[idx_start : idx_start + SINGLE_ARM_DOF] = elem_left["position"][:-1]

    idx_start = MOTORS.index("joint.position.right_arm_0")
    assert len(elem_right["position"]) - 1 == SINGLE_ARM_DOF
    concatenated_state[idx_start : idx_start + SINGLE_ARM_DOF] = elem_right["position"][:-1]

    # ---- action ----
    if BUILDFUTURETRAJ and CHUNKSIZE > 1:
        for idx in range(CHUNKSIZE):
            elem_left = extract_data(DATA, "/left_data", timestamp + DELAY / CHUNKSIZE * (idx  + 1))
            concatenated_action[idx].append(elem_left["position"][-1])
          
            elem_right = extract_data(DATA, "/right_data", timestamp + DELAY / CHUNKSIZE * (idx  + 1))
            concatenated_action[idx].append(elem_right["position"][-1])
            
            assert len(elem_left["position"]) -1 == SINGLE_ARM_DOF
            for jj in range(SINGLE_ARM_DOF):
                concatenated_action[idx].append(elem_left["position"][jj])

            assert len(elem_right["position"]) -1 == SINGLE_ARM_DOF
            for jj in range(SINGLE_ARM_DOF):
                concatenated_action[idx].append(elem_right["position"][jj])

    else:
        elem_left = extract_data(DATA, "/left_data", timestamp + DELAY)
        concatenated_action.append(elem_left["position"][-1])

        elem_right = extract_data(DATA, "/right_data", timestamp + DELAY)
        concatenated_action.append(elem_right["position"][-1])
        
        assert len(elem_left["position"]) -1 == SINGLE_ARM_DOF
        for jj in range(SINGLE_ARM_DOF):
            concatenated_action.append(elem_left["position"][jj])

        assert len(elem_right["position"]) -1 == SINGLE_ARM_DOF
        for jj in range(SINGLE_ARM_DOF):
            concatenated_action.append(elem_right["position"][jj])

    frame = {
        "observation.state": torch.tensor(concatenated_state),
        "action": torch.tensor(concatenated_action),
    }

    # image
    image = extract_data(DATA, "/camera3/camera_head/color/image_raw/compressed", timestamp)["image"]
    # print("head: ", type(image), image.shape)
    frame["observation.images.head"] = image

    if "/camera1/camera_left/color/image_rect_raw/compressed" in DATA:
        image = extract_data(DATA, "/camera1/camera_left/color/image_rect_raw/compressed", timestamp)["image"]
        # print("hand_left: ", type(image), image.shape)
        frame["observation.images.hand_left"] = image
    else:
        raise ValueError("!! Missing left hand camera!!")
    
    if "/camera2/camera_right/color/image_rect_raw/compressed" in DATA:
        image = extract_data(DATA, "/camera2/camera_right/color/image_rect_raw/compressed", timestamp)["image"]
        # print("hand_right: ", type(image), image.shape)
        frame["observation.images.hand_right"] = image
    else:
        raise ValueError("!! Missing right hand camera!!")
    
    return frame