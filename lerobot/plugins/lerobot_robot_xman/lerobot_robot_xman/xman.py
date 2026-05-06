from __future__ import annotations

import threading
from functools import cached_property
from typing import Any
import time

import numpy as np
import sys
default_ros_path = ['/home/peanut/sc_ros/install/sc_ros2/local/lib/python3.10/dist-packages', '/mnt/data1/mlt/realsense/install/realsense2_camera_msgs/local/lib/python3.10/dist-packages', '/opt/ros/humble/lib/python3.10/site-packages', '/opt/ros/humble/local/lib/python3.10/dist-packages']
for v in default_ros_path:
    sys.path.remove(v)
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
for v in default_ros_path:
    sys.path.append(v)
from sc_ros2.msg import ArmData  # type: ignore[reportMissingImports]
from sc_ros2.srv import MulSetJointAngles  # type: ignore[reportMissingImports]

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_xman import XmanConfig


class Xman(Robot):
    """
    XmanR1双臂机器人插件
    修改`_connect_transport`, `_read_state`, and `_write_action`可以集成机器人特定的接口
    """

    config_class = XmanConfig
    name = "xman"

    def __init__(self, config: XmanConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True
        self._latest_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self._latest_state = np.zeros(self.config.state_dim, dtype=np.float32)
        self._smoothed_gripper = np.zeros(2, dtype=np.float32)
        self._bridge = None
        self._ros_node = None
        self._ros_executor = None
        self._ros_spin_thread: threading.Thread | None = None
        self._arm_client = None
        self._state_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._latest_head_image = np.zeros(
            (self.config.head_image_height, self.config.head_image_width, 3), dtype=np.uint8
        )
        self._latest_left_image = np.zeros(
            (self.config.wrist_image_height, self.config.wrist_image_width, 3), dtype=np.uint8
        )
        self._latest_right_image = np.zeros(
            (self.config.wrist_image_height, self.config.wrist_image_width, 3), dtype=np.uint8
        )
        self._interval = 0.1 # 1/FPS

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"joint_{idx}.pos": float for idx in range(self.config.action_dim)}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        state_features = {f"joint_{idx}.pos": float for idx in range(self.config.state_dim)}
        camera_features = {
            "head_image": (self.config.head_image_height, self.config.head_image_width, 3),
            "left_wrist_image": (self.config.wrist_image_height, self.config.wrist_image_width, 3),
            "right_wrist_image": (self.config.wrist_image_height, self.config.wrist_image_width, 3),
        }
        return {**state_features, **camera_features}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._connect_transport()
        self._is_connected = True
        if calibrate:
            self.calibrate()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._disconnect_transport()
        self._is_connected = False

    @check_if_not_connected
    def calibrate(self) -> None:
        # Replace with your homing / zero-offset routine.
        self._is_calibrated = True

    @check_if_not_connected
    def configure(self) -> None:
        # Optional: set control mode, gains, watchdog, etc.
        return

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._read_state()
        if state.shape != (self.config.state_dim,):
            raise ValueError(f"Expected state shape {(self.config.state_dim,)}, got {state.shape}")

        observation: RobotObservation = {
            f"joint_{idx}.pos": float(state[idx]) for idx in range(self.config.state_dim)
        }
        with self._image_lock:
            observation["head_image"] = self._latest_head_image.copy()
            observation["left_wrist_image"] = self._latest_left_image.copy()
            observation["right_wrist_image"] = self._latest_right_image.copy()

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        expected_keys = list(self.action_features.keys())
        missing = [k for k in expected_keys if k not in action]
        if missing:
            raise KeyError(f"Missing action keys: {missing}")

        action_vec = np.array([float(action[k]) for k in expected_keys], dtype=np.float32)
        self._write_action(action_vec)
        self._latest_action = action_vec

        return {k: float(v) for k, v in zip(expected_keys, action_vec, strict=True)}

    def _connect_transport(self) -> None:
        """
        Initialize ROS2 subscriptions/service for local onboard control.
        """


        self._bridge = CvBridge()

        if not rclpy.ok():
            rclpy.init()

        self._ros_node = Node("xman_lerobot_bridge")
        self._ros_node.create_subscription(
            ArmData,
            self.config.arm_topic,
            self._arm_state_cb,
            200,
        )
        self._ros_node.create_subscription(Image, self.config.head_topic, self._head_image_cb, 10)
        self._ros_node.create_subscription(Image, self.config.left_topic, self._left_image_cb, 10)
        self._ros_node.create_subscription(Image, self.config.right_topic, self._right_image_cb, 10)
        self._arm_client = self._ros_node.create_client(MulSetJointAngles, self.config.arm_service)
        while not self._arm_client.wait_for_service(timeout_sec=1.0):
            self._ros_node.get_logger().info(f"Waiting for {self.config.arm_service} service...")

        self._ros_executor = rclpy.executors.MultiThreadedExecutor()
        self._ros_executor.add_node(self._ros_node)
        self._ros_spin_thread = threading.Thread(target=self._ros_executor.spin, daemon=True)
        self._ros_spin_thread.start()

    def _disconnect_transport(self) -> None:
        """Cleanup ROS2 resources."""
        if self._ros_executor is not None:
            self._ros_executor.shutdown()
            self._ros_executor = None
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        self._arm_client = None
        self._ros_spin_thread = None

    def _arm_state_cb(self, msg: Any) -> None:
        state = np.asarray(msg.data, dtype=np.float32)
        if state.size < self.config.state_dim:
            padded = np.zeros(self.config.state_dim, dtype=np.float32)
            padded[: state.size] = state
            state = padded
        elif state.size > self.config.state_dim:
            state = state[: self.config.state_dim]

        with self._state_lock:
            self._latest_state = state

    def _head_image_cb(self, msg: Any) -> None:
        image_bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        image_rgb = image_bgr[:, :, ::-1]
        with self._image_lock:
            self._latest_head_image = image_rgb

    def _left_image_cb(self, msg: Any) -> None:
        image_bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        image_rgb = image_bgr[:, :, ::-1]
        with self._image_lock:
            self._latest_left_image = image_rgb

    def _right_image_cb(self, msg: Any) -> None:
        image_bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        image_rgb = image_bgr[:, :, ::-1]
        with self._image_lock:
            self._latest_right_image = image_rgb

    def _read_state(self) -> np.ndarray:
        """
        Return latest ArmData state from ROS callback.
        If no state has arrived yet, return last cached state (initialized as zeros).
        """
        with self._state_lock:
            return self._latest_state.copy()

    def _write_action(self, action: np.ndarray) -> None:
        """
        Send one action step through /MulSetJointAngles with basic gripper smoothing.

        Smoothing is EMA on gripper indices [7, 15]:
            g_smooth = alpha * g_new + (1 - alpha) * g_prev
        """
        if self._arm_client is None:
            raise RuntimeError("ROS arm client is not initialized. Did you call connect()?")
        from sc_ros2.msg import JointSequence  # type: ignore[reportMissingImports]
        from sc_ros2.srv import MulSetJointAngles  # type: ignore[reportMissingImports]

        cmd = action.copy()
        if cmd.size < 16:
            raise ValueError(f"Expected action dim >= 16 for grippers, got {cmd.size}")

        alpha = 0.2
        gripper_new = cmd[[7, 15]]
        self._smoothed_gripper = alpha * gripper_new + (1.0 - alpha) * self._smoothed_gripper
        cmd[[7, 15]] = self._smoothed_gripper

        req = MulSetJointAngles.Request()
        req.sequences = []
        seq = JointSequence()
        seq.joints = cmd.tolist()
        req.sequences.append(seq)
        time.sleep(self._interval)
        # self._arm_client.call(req) #!!!先不发送动作，测试流程
        print(cmd)
