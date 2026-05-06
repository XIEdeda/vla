#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import threading
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from sc_ros2.msg import ArmData, JointSequence
from sc_ros2.srv import MulSetJointAngles

from openpi_client import websocket_client_policy


# ============================================================
# ROS Operator
# ============================================================
class RosOperator(Node):
    def __init__(self):
        super().__init__("pi0_ros1_semantic_operator")

        self.bridge = CvBridge()

        # -------------------------------
        # 队列（只存 msg，不做处理）
        # -------------------------------
        self.img_head_queue = deque(maxlen=50)
        self.img_left_queue = deque(maxlen=50)
        self.img_right_queue = deque(maxlen=50)
        self.dual_arm_queue = deque(maxlen=2000)

        # -------------------------------
        # Topics
        # -------------------------------
        self.head_topic = "/camera3/camera_head/color/image_raw"
        self.left_topic = "/camera1/camera_left/color/image_rect_raw"
        self.right_topic = "/camera2/camera_right/color/image_rect_raw"
        self.arm_topic = "/dual_data"

        self._init_ros_entities()

    def _init_ros_entities(self):
        # subscriptions
        self.create_subscription(Image, self.head_topic, self._head_cb, 10)
        self.create_subscription(Image, self.left_topic, self._left_cb, 10)
        self.create_subscription(Image, self.right_topic, self._right_cb, 10)
        self.create_subscription(ArmData, self.arm_topic, self._arm_cb, 200)
        print(f"subscriptions successfully !!!!")
        # arm service
        self.arm_client = self.create_client(MulSetJointAngles, "/MulSetJointAngles")
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arm service...")

    # --------------------------------------------------
    # callbacks
    # --------------------------------------------------
    def _head_cb(self, msg: Image):
        # print(f"head num:{len(self.img_head_queue)}")
        self.img_head_queue.append(msg)

    def _left_cb(self, msg: Image):
        # print(f"left num:{len(self.img_left_queue)}")
        self.img_left_queue.append(msg)

    def _right_cb(self, msg: Image):
        # print(f"right num:{len(self.img_right_queue)}")
        self.img_right_queue.append(msg)

    def _arm_cb(self, msg: ArmData):
        # print(f"arm num:{len(self.dual_arm_queue)}")
        self.dual_arm_queue.append(msg)

    # --------------------------------------------------
    # ROS semantic pop
    def _pop_at(self, queue, ts):
        """
        取 >= ts 的第一条
        """
        while queue:
            msg = queue[0]
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t < ts:
                queue.popleft()
            else:
                break
        if not queue:
            return None
        return queue.popleft()

    # --------------------------------------------------
    # get aligned frame
    def get_frame(self):

        if not (
            self.img_head_queue
            and self.img_left_queue
            and self.img_right_queue
            and self.dual_arm_queue
        ):
            return None

        # === 使用最新帧，不追历史 ===
        t_head = (
            self.img_head_queue[-1].header.stamp.sec
            + self.img_head_queue[-1].header.stamp.nanosec * 1e-9
        )
        t_left = (
            self.img_left_queue[-1].header.stamp.sec
            + self.img_left_queue[-1].header.stamp.nanosec * 1e-9
        )
        t_right = (
            self.img_right_queue[-1].header.stamp.sec
            + self.img_right_queue[-1].header.stamp.nanosec * 1e-9
        )

        frame_time = min(t_head, t_left, t_right)

        head = self._pop_at(self.img_head_queue, frame_time)
        left = self._pop_at(self.img_left_queue, frame_time)
        right = self._pop_at(self.img_right_queue, frame_time)
        arm = self._pop_at(self.dual_arm_queue, frame_time)

        if not all([head, left, right, arm]):
            return None

        img_head = self.bridge.imgmsg_to_cv2(head, "bgr8")
        img_left = self.bridge.imgmsg_to_cv2(left, "bgr8")
        img_right = self.bridge.imgmsg_to_cv2(right, "bgr8")
        print(f"frame_time:{frame_time}")

        return img_head, img_left, img_right, arm.data

    # --------------------------------------------------
    # arm control
    def control_arm_chunk(self, actions: np.ndarray):
        req = MulSetJointAngles.Request()
        req.sequences = []

        for step in actions:
            seq = JointSequence()
            seq.joints = step.tolist()
            req.sequences.append(seq)

        future = self.arm_client.call_async(req)

    def control_arm_chunk_freq(
        self, actions: np.ndarray, mode: int = 0, is_async: bool = False
    ):
        final_actions = []
        if mode == 0:  # 正常模式，以约13ms的间隔发送所有action_chunk中的动作
            final_actions = actions
        elif mode == 1:  # 正常模式，延迟夹爪
            M = len(actions)
            for i in range(0, M):
                joint_action = actions[i, :].copy()
                joint_action[[7, 15]] = actions[0, [7, 15]]
                final_actions.append(joint_action)
            for i in range(1, M):
                gripper_action = actions[M - 1, :].copy()
                gripper_action[[7, 15]] = actions[i, [[7, 15]]]
                final_actions.append(gripper_action)
        elif mode == 2:  # 高效模式，只发送action_chunk的5倍降采样关键帧动作
            final_actions = actions[
                np.unique(np.append(np.arange(0, len(actions), 5), len(actions) - 1)), :
            ]
        elif mode == 3:
            # 只发送action_chunk的5倍降采样关键帧动作，并且延迟gripper一帧，并且以sequence形式一起发
            downsample_actions = actions[
                np.unique(np.append(np.arange(0, len(actions), 5), len(actions) - 1)), :
            ]
            M = len(downsample_actions)
            req = MulSetJointAngles.Request()
            req.sequences = []
            for i in range(0, M):
                joint_action = downsample_actions[i, :].copy()
                if i > 0:
                    joint_action[[7, 15]] = downsample_actions[i - 1, [7, 15]]
                gripper_action = downsample_actions[i, :].copy()
                final_actions.append(joint_action)
                final_actions.append(gripper_action)
                for step in [joint_action, gripper_action]:
                    seq = JointSequence()
                    seq.joints = step.tolist()
                    req.sequences.append(seq)
            tic = time.time()
            if is_async:
                future = self.arm_client.call_async(req)
                time.sleep(0.013)
            else:
                future = self.arm_client.call(req)
            toc = time.time()
            print("One action time: ", toc - tic)
            with open("one_action_time_mode3.txt", "a") as f:
                f.write(str(toc - tic) + "\n")
            return  #!!! 这种方式直接返回即可
        elif mode == 4:
            # 只发送action_chunk的5倍降采样关键帧动作，并且延迟gripper多帧，并且以sequence形式一起发
            downsample_actions = actions[
                np.unique(np.append(np.arange(0, len(actions), 5), len(actions) - 1)), :
            ]
            M = len(downsample_actions)
            for i in range(0, M):
                joint_action = downsample_actions[i, :].copy()
                joint_action[[7, 15]] = downsample_actions[0, [7, 15]]
                final_actions.append(joint_action)
            for i in range(1, M):
                gripper_action = downsample_actions[M - 1, :].copy()
                gripper_action[[7, 15]] = downsample_actions[i, [[7, 15]]]
                final_actions.append(gripper_action)
        if is_async:
            for step in final_actions:
                req = MulSetJointAngles.Request()
                req.sequences = []

                seq = JointSequence()
                seq.joints = step.tolist()
                req.sequences.append(seq)
                future = self.arm_client.call_async(req)
                time.sleep(0.0134327)
        else:
            req = MulSetJointAngles.Request()
            req.sequences = []
            for step in final_actions:
                seq = JointSequence()
                seq.joints = step.tolist()
                req.sequences.append(seq)
            tic = time.time()
            future = self.arm_client.call(req)
            toc = time.time()
            print("Total action time: ", toc - tic)
            print(actions[:, 7])


# =======================
# ROS spin thread
# =======================
def start_ros_spin(node):
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    def _spin():
        executor.spin()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    return executor


# =======================
# observation builder
# =======================
def build_observation(ros_op: RosOperator):
    while rclpy.ok():
        res = ros_op.get_frame()
        if res is None:
            time.sleep(0.005)
            continue

        img_head, img_left, img_right, state = res

        # img_head = cv2.resize(img_head, (224, 224))[:, :, ::-1]
        # img_left = cv2.resize(img_left, (224, 224))[:, :, ::-1]
        # img_right = cv2.resize(img_right, (224, 224))[:, :, ::-1]
        img_head = img_head[:, :, ::-1]
        img_left = img_left[:, :, ::-1]
        img_right = img_right[:, :, ::-1]

        obs = {
            "observation/head_image": img_head,
            "observation/left_wrist_image": img_left,
            "observation/right_wrist_image": img_right,
            "observation/state": np.array(state),
            "prompt": "Prepare popcorn.",
        }
        return obs


# =========================
# gripper alignment
# =========================
def align_gripper_action_chunk(actions, g_t):
    actions = actions.copy()
    offset = g_t - actions[0, -1]
    actions[:, -1] += offset

    # 限制夹爪值在0-1之间
    actions[:, -1] = np.clip(actions[:, -1], 0.25, 1.0)

    return actions


# ============================
# actions postprocess
# ============================
last_actions = None
sliding_history = None
ema_state = None


def process_actions(actions, mode="mean", value=0):
    """
    对action_chunk的夹爪动作进行后处理
    参数:
        actions: 2D numpy array, shape (N, D), D >= 16
        mode: "mean", "rate_limit"
        value: some modes need value
    返回:
        processed_actions
    """
    global last_actions
    global sliding_history
    global ema_state
    actions = np.array(actions)
    if actions.ndim != 2:
        raise ValueError("actions 必须是二维数组")

    processed_actions = actions.copy()
    N, D = actions.shape
    if D < 16:
        raise ValueError("actions维度错误！")

    # target_cols = [7, 15]
    target_cols = list(range(16))

    if mode == "mean":
        for col in target_cols:
            col_data = actions[:, col]
            mean_val = np.mean(col_data)
            processed_actions[:, col] = mean_val
    elif mode == "sliding_mean":
        window = int(value)

        if window <= 0:
            raise ValueError("value 必须是正整数")

        # 初始化全局 history
        if sliding_history is None:
            sliding_history = {col: [] for col in target_cols}

        for col in target_cols:

            for i in range(N):
                sliding_history[col].append(actions[i, col])
                if len(sliding_history[col]) > window:
                    sliding_history[col].pop(0)
                processed_actions[i, col] = np.mean(sliding_history[col])

        last_actions = processed_actions[-1]

    elif mode == "rate_limit":
        if last_actions is None:
            last_actions = processed_actions[0, :]
        else:
            for col in target_cols:
                processed_actions[0, col] = min(
                    processed_actions[0, col], last_actions[col] + value
                )
                processed_actions[0, col] = max(
                    processed_actions[0, col], last_actions[col] - value
                )
        for i in range(1, N):
            for col in target_cols:
                processed_actions[i, col] = min(
                    processed_actions[i - 1, col], last_actions[col] + value
                )
                processed_actions[i, col] = max(
                    processed_actions[i - 1, col], last_actions[col] - value
                )

        last_actions = processed_actions[-1]
    elif mode == "ema":
        alpha = float(value)

        if not (0.0 < alpha <= 1.0):
            raise ValueError("EMA 的 value (alpha) 必须在 (0, 1] 之间")

        # 初始化 EMA 状态
        if ema_state is None:
            ema_state = {}
            for col in target_cols:
                if last_actions is not None:
                    ema_state[col] = last_actions[col]
                else:
                    ema_state[col] = actions[0, col]

        for i in range(N):
            for col in target_cols:
                prev = ema_state[col]
                curr = actions[i, col]
                ema = alpha * curr + (1.0 - alpha) * prev
                processed_actions[i, col] = ema
                ema_state[col] = ema

        last_actions = processed_actions[-1]

    else:
        raise NotImplementedError

    return processed_actions


BASE_THRESHOLDS = 0.03 * np.array(
    [
        0.24407958,
        0.24835201,
        0.36019514,
        0.1768939,
        0.2324477,
        0.10851968,
        0.15576612,
        27.84260593,
        0.28920391,
        0.10926549,
        0.24246325,
        0.05891056,
        0.04933415,
        0.07169942,
        0.13316296,
        30.1083496,
    ]
)


def remove_small_change_actions(
    action_chunk: np.ndarray, thresholds: np.ndarray = BASE_THRESHOLDS
):
    """
    去掉变化幅度过小的动作
    参数:
        action_chunk: shape (T, 16)
        thresholds:   shape (16,)
    返回:
        kept_actions
    """
    if action_chunk.shape[0] <= 1:
        return action_chunk

    kept_actions = [action_chunk[0]]
    last_kept = action_chunk[0]

    for i in range(1, action_chunk.shape[0]):
        curr = action_chunk[i]
        diff = np.abs(curr - last_kept)

        if np.all(diff < thresholds):
            continue

        kept_actions.append(curr)
        last_kept = curr

    return np.stack(kept_actions, axis=0)


# ============================
# main
# ============================
def main():
    rclpy.init()

    ros_op = RosOperator()
    executor = start_ros_spin(ros_op)
    print(f"start connect RDT policy server")
    policy = websocket_client_policy.WebsocketClientPolicy(
        host="172.16.10.14", port=5555
    )
    print(f"connect RDT policy server successfully !!!!!")

    cur_time = time.time()
    saved_actions_file = f"actions_{cur_time}.jsonl"
    saved_sub_actions_file = f"sub_actions_{cur_time}.jsonl"
    try:
        while rclpy.ok():
            obs = build_observation(ros_op)
            state = obs["observation/state"]
            print(f"state:{state}")
            current_gripper = state[-1]
            tic = time.time()
            actions = policy.infer(obs)["actions"]
            toc = time.time()
            print("推理耗时：", toc - tic)
            actions = np.array(actions)
            sub_actions = process_actions(
                actions, mode="sliding_mean", value=max(1, int(1.5 * len(actions)))
            )
            sub_actions[:, 7] = sub_actions[:, 7] * 70.0
            sub_actions[:, 15] = sub_actions[:, 15] * 70.0
            # sub_actions = align_gripper_action_chunk(sub_actions, current_gripper)
            tic = time.time()
            ros_op.control_arm_chunk_freq(sub_actions, mode=0)
            toc = time.time()
            print("动作执行耗时：", toc - tic)

            raw_list = actions.tolist()
            proc_list = sub_actions.tolist()
            with open(saved_actions_file, "a") as f:
                for action in raw_list:
                    f.write(json.dumps(action, ensure_ascii=False) + "\n")
            with open(saved_sub_actions_file, "a") as f:
                for action in proc_list:
                    f.write(json.dumps(action, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        ros_op.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
