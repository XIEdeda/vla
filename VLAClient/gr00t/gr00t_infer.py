#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T Policy Client for ROS2
- 机械臂控制和相机对齐: 参考 openpi_move.py
- ZMQ 通信: 参考 server_client.py (msgpack 序列化)
"""

import time
import threading
from collections import deque
import io
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from sc_ros2.msg import ArmData, JointSequence
from sc_ros2.srv import MulSetJointAngles

import zmq
import msgpack


# ============================================================
# Message Serializer (与 gr00t.server_client.MsgSerializer 兼容)
# ============================================================
class MsgSerializer:
    """
    消息序列化器，与 GR00T Policy Server 兼容
    使用 msgpack 序列化，支持 numpy 数组
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        """解码自定义类型"""
        if not isinstance(obj, dict):
            return obj
        # 解码 numpy 数组
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        """编码自定义类型"""
        # 编码 numpy 数组
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


# ============================================================
# ZMQ Policy Client (不依赖 gr00t)
# ============================================================
class ZMQPolicyClient:
    """
    ZMQ 客户端，用于与 GR00T Policy Server 通信
    使用 msgpack 序列化 (与服务端一致)
    """

    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[ZMQPolicyClient] Connected to tcp://{host}:{port}")

    def get_action(self, observation: dict) -> dict:
        """
        发送观测数据，获取动作预测

        Args:
            observation: 观测数据字典，包含:
                - video.xxx: 图像数据
                - state.xxx: 状态数据
                - annotation.human.action.task_description: 任务描述

        Returns:
            actions: 动作字典，包含:
                - action.xxx: 动作数据
        """
        request = {
            "endpoint": "get_action",
            "data": {"observation": observation, "options": None},
        }

        # 发送请求 (msgpack 序列化)
        self.socket.send(MsgSerializer.to_bytes(request))

        # 接收响应 (msgpack 反序列化)
        data = self.socket.recv()
        response = MsgSerializer.from_bytes(data)

        # response 是 (action, info) 元组
        if isinstance(response, (list, tuple)):
            action, info = response
            return action
        return response

    def reset(self, options: dict = None) -> dict:
        """重置策略状态"""
        request = {"endpoint": "reset", "data": {"options": options}}
        self.socket.send(MsgSerializer.to_bytes(request))
        data = self.socket.recv()
        return MsgSerializer.from_bytes(data)

    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()


# ============================================================
# ROS Operator (机械臂控制和相机对齐)
# ============================================================
class RosOperator(Node):
    def __init__(self):
        super().__init__("gr00t_client_node")

        self.bridge = CvBridge()

        # -------------------------------
        # 队列（只存 msg，不做处理）
        # -------------------------------
        self.img_head_queue = deque(maxlen=50)
        self.img_left_queue = deque(maxlen=50)
        self.img_right_queue = deque(maxlen=50)
        self.dual_arm_queue = deque(maxlen=2000)

        # -------------------------------
        # Topics (根据你的机器人配置修改)
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
        self.get_logger().info("Subscriptions initialized successfully!")

        # arm service
        self.arm_client = self.create_client(MulSetJointAngles, "/MulSetJointAngles")
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for arm service...")

    # --------------------------------------------------
    # callbacks
    # --------------------------------------------------
    def _head_cb(self, msg: Image):
        self.img_head_queue.append(msg)

    def _left_cb(self, msg: Image):
        self.img_left_queue.append(msg)

    def _right_cb(self, msg: Image):
        self.img_right_queue.append(msg)

    def _arm_cb(self, msg: ArmData):
        self.dual_arm_queue.append(msg)

    # --------------------------------------------------
    # ROS semantic pop: 取 >= ts 的第一条
    # --------------------------------------------------
    def _pop_at(self, queue, ts):
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
    # get aligned frame (相机时间戳对齐)
    # --------------------------------------------------
    def get_frame(self):
        if not (
            self.img_head_queue
            and self.img_left_queue
            and self.img_right_queue
            and self.dual_arm_queue
        ):
            return None

        # 使用最新帧，不追历史
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

        return img_head, img_left, img_right, arm.data

    # --------------------------------------------------
    # arm control
    # --------------------------------------------------
    def control_arm_chunk(self, actions: np.ndarray):
        """
        发送关节角度序列到机械臂控制器

        Args:
            actions: (N, 16) 数组，N 为动作步数，16 为关节维度
                     [left_arm_joints(7), left_gripper(1), right_arm_joints(7), right_gripper(1)]
        """
        req = MulSetJointAngles.Request()
        req.sequences = []
        for step in actions:
            seq = JointSequence()
            seq.joints = step.tolist()
            req.sequences.append(seq)

        self.arm_client.call_async(req)


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
def build_observation(ros_op: RosOperator, task_description: str = "pour tea"):
    """
    构建模型输入的观测数据

    Args:
        ros_op: ROS 操作器
        task_description: 任务描述文本

    Returns:
        obs: 观测字典，格式符合 GR00T 模型输入要求
             - video: dict[str, np.ndarray] - shape (B, T, H, W, C), uint8
             - state: dict[str, np.ndarray] - shape (B, T, D), float32
             - language: dict[str, list[list[str]]]
    """
    res = ros_op.get_frame()
    if res is None:
        return None

    img_head, img_left, img_right, state = res

    # 图像预处理: resize 到 640x480, BGR -> RGB
    img_head = cv2.resize(img_head, (640, 480))[:, :, ::-1]
    img_left = cv2.resize(img_left, (640, 480))[:, :, ::-1]
    img_right = cv2.resize(img_right, (640, 480))[:, :, ::-1]

    # 状态预处理: 夹爪值归一化 (假设原始值为 0-100)
    np_state = np.array(state, dtype=np.float32)
    np_state[7] = np_state[7] / 100.0  # left gripper
    np_state[15] = np_state[15] / 100.0  # right gripper

    # 构建观测字典 (符合 GR00T 输入格式 - 嵌套结构)
    # Video: (B, T, H, W, C), T=1 表示单帧, dtype=uint8
    # State: (B, T, D), T=1 表示单帧, dtype=float32
    # Language: list[list[str]], 外层是 batch，内层是时间维度
    obs = {
        "video": {
            "head_image": img_head.reshape(1, 1, 480, 640, 3).astype(np.uint8),
            "left_wrist_image": img_left.reshape(1, 1, 480, 640, 3).astype(np.uint8),
            "right_wrist_image": img_right.reshape(1, 1, 480, 640, 3).astype(np.uint8),
        },
        "state": {
            "left_arm_joints": np_state[0:7].reshape(1, 1, 7),
            "left_gripper_pose": np_state[7:8].reshape(1, 1, 1),
            "right_arm_joints": np_state[8:15].reshape(1, 1, 7),
            "right_gripper_pose": np_state[15:16].reshape(1, 1, 1),
        },
        "language": {
            "annotation.human.action.task_description": [[task_description]],
        },
    }
    return obs


# =========================
# gripper alignment
# =========================
def align_gripper_action_chunk(actions: np.ndarray, g_t: float):
    """
    对齐夹爪动作，避免突变

    Args:
        actions: 动作序列 (N, D)
        g_t: 当前夹爪状态

    Returns:
        aligned_actions: 对齐后的动作序列
    """
    actions = actions.copy()
    # 对齐左夹爪 (索引 7)
    if actions.shape[1] > 7:
        offset = g_t - actions[0, 7]
        actions[:, 7] += offset
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)
    return actions


# ============================
# main
# ============================
def main():
    rclpy.init()

    # 配置参数
    SERVER_HOST = "27.159.92.108"  # Policy Server IP
    SERVER_PORT = 5555
    TASK_DESCRIPTION = "Prepare popcorn"  # 任务描述

    CONTROL_HZ = 13  # 控制频率
    ACTION_HZ = 30  # 动作执行频率
    CONTROL_PERIOD = 1.0 / CONTROL_HZ
    ACTION_PERIOD = 1.0 / ACTION_HZ

    # 初始化 ROS
    ros_op = RosOperator()
    executor = start_ros_spin(ros_op)

    # 初始化 ZMQ 客户端
    print(f"[Main] Connecting to Policy Server at {SERVER_HOST}:{SERVER_PORT}...")
    policy_client = ZMQPolicyClient(host=SERVER_HOST, port=SERVER_PORT)
    print(f"[Main] Connected to Policy Server successfully!")

    actions_buffer = []
    action_idx = 0
    action_time = 0.0
    last_time = time.time()

    try:
        while rclpy.ok():
            now = time.time()
            dt = now - last_time
            last_time = now

            # 构建观测
            obs = build_observation(ros_op, TASK_DESCRIPTION)
            if obs is None:
                time.sleep(0.005)
                continue

            # 如果 action 用完，重新推理
            if action_idx >= len(actions_buffer):
                print(f"[Main] Getting action from server...")
                actions = policy_client.get_action(obs)

                # actions 是字典格式:
                # {
                #   'left_arm_joints': (B, T, 7),
                #   'left_gripper_pose': (B, T, 1),
                #   'right_arm_joints': (B, T, 7),
                #   'right_gripper_pose': (B, T, 1),
                # }
                # 按照 XMAN modality.json 的顺序拼接
                action_keys = [
                    "left_arm_joints",  # 7 维
                    "left_gripper_pose",  # 1 维
                    "right_arm_joints",  # 7 维
                    "right_gripper_pose",  # 1 维
                ]

                action_list = []
                for key in action_keys:
                    if key in actions:
                        arr = np.array(actions[key])
                        # arr shape: (B, T, D), 去掉 batch 维度
                        if arr.ndim == 3:
                            arr = arr[0]  # (T, D)
                        action_list.append(arr)
                        print(f"[Main] Action '{key}': shape {arr.shape}")

                if action_list:
                    # 拼接所有动作维度: (T, 16)
                    actions_buffer = np.concatenate(action_list, axis=-1)
                    print(f"[Main] Combined actions shape: {actions_buffer.shape}")
                    action_idx = 0
                    action_time = 0.0
                else:
                    print(
                        f"[Main] Warning: No action received, keys: {list(actions.keys())}"
                    )
                    continue

            # 获取当前动作步
            sub_actions = actions_buffer[action_idx : action_idx + 1].copy()

            # 夹爪值还原 (0-1 -> 0-100)
            # 索引: left_arm(0-6), left_gripper(7), right_arm(8-14), right_gripper(15)
            sub_actions[:, 7] *= 100.0  # left gripper
            sub_actions[:, 15] *= 100.0  # right gripper

            print(
                f"[Main] Executing action {action_idx}: "
                f"left_gripper={sub_actions[0, 7]:.2f}, "
                f"right_gripper={sub_actions[0, 15]:.2f}"
            )

            # 发送控制指令
            ros_op.control_arm_chunk(sub_actions)

            # 累积 action 时间
            action_time += dt
            if action_time >= ACTION_PERIOD:
                action_idx += 1
                action_time -= ACTION_PERIOD

            # 控制节拍
            sleep_time = CONTROL_PERIOD - (time.time() - now)
            time.sleep(max(0.0, sleep_time))

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    finally:
        policy_client.close()
        executor.shutdown()
        ros_op.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
