#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLA inference node with integrated "move to init" functionality.
Merged move_to_init_ros2.py logic into VLANode so you can trigger
the initialization trajectory (via keyboard 'r') within the same node.

Requires: rclpy, numpy, scipy, torch, OpenCV, lerobot package etc.
"""

import os
import sys
import time
import threading
import termios
import fcntl

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import String, Header

import cv2
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# ---------- config ----------
IMAGE_SHAPE = (224, 168)
UPSAMPLING_FACTOR = 25
ORIGIN_FPS = 8
ACTION_DIM = 14
PRETRAINED_PATH = ".../checkpoints/202512.use.the.right.gripper.to.pick.up.a.Sanhuang.Plan.and.hold.it.CCN7.dof.14.VisionFreezeLLMFreeze.float32.zero1.gpu32.20260211131001/checkpoint-28735"  # todo

USERNAME = os.environ.get("USER", "")
if USERNAME == "nvidia":
    TARGET = {
        "arm_right": [0.0, 0.0, 0.0, -1.61808511, 0.0, 0.0, 0.0],
        "arm_left": [0.0, 0.0, 0.0, -1.61808511, 0.0, 0.0, 0.0],
        "torso": [-0.06230000, -0.01010000, -0.66160000, -0.00010000],
        "gripper_left": [100.0],
        "gripper_right": [100.0],
    }
else:  # r1lite
    TARGET = {
        "arm_right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "arm_left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "torso": [-0.08, 1.65, 0.8, 0.0],
        "gripper_left": [100.0],
        "gripper_right": [100.0],
    }

"""
# 创建虚拟环境
conda create -n lerobot-3.0 python=3.10
conda activate lerobot-3.0

# 必要包安装
pip install transformers==4.53.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install accelerate==1.4.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install deepspeed==0.17.4 -i https://mirrors.aliyun.com/pypi/simple/
pip install pytest==8.4.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install scipy -i https://mirrors.aliyun.com/pypi/simple/

conda search ffmpeg -c conda-forge
conda install ffmpeg==6.1.2 -c conda-forge



# 本地回放测试
in Terminal 1, run:batch["observation.images.head"]
    ros2 bag play --loop /media/vla/T7/galaxea_r1lite/202512.use.the.left.gripper.to.pick.up.a.Calamine.lotion.and.hold.it.by.WangHengze/RB250714001_20251210183413189_RAW/RB250714001_20251210183413189_RAW.mcap
in ros2 conda env of Terminal 2, run:
    conda activate lerobot-3.0
    cd .../lerobot/ && python src/lerobot/scripts/inference_ros2_v10.py
in Terminal 3, run:
    # 以 5 Hz 的频率重复发布
    ros2 topic pub -r 0.5 /language_instruction std_msgs/String "data: 'use the right gripper to pick up a Sanhuang Plan and hold it'"
    ros2 topic pub -r 0.5 /language_instruction std_msgs/String "data: 'use the left gripper to pick up a Calamine lotion and hold it'"
    ros2 topic pub -r 0.5 /language_instruction std_msgs/String "data: 'use the left gripper to pick up the Vitamin C Tablets and hold it'"
    ros2 topic pub -r 0.5 /language_instruction std_msgs/String "data: 'place the small coca-cola into the pink plate with the left hand'"
"""


class VLANode(Node):
    def __init__(self, pretrained_path: str):
        super().__init__("inference")
        # 日志
        self.get_logger().info(f"Loading model from: {pretrained_path}")
        # 模型加载
        self.policy = PI0Policy.from_pretrained(pretrained_name_or_path=pretrained_path)
        self.policy.to(device="cuda")  # todo: device
        # 从模型配置中读取 chunk_size 并保存
        self.chunk_size = self.policy.config.chunk_size
        # 设置发布频率
        self.publish_rate_hz = ORIGIN_FPS * UPSAMPLING_FACTOR

        # 数据缓存与锁
        self.lock = threading.Lock()
        self.camera_head = None
        self.camera_wrist_left = None
        self.camera_wrist_right = None
        self.language = None
        self.feedback_arm_left = None
        self.feedback_arm_right = None
        self.feedback_gripper_left = None
        self.feedback_gripper_right = None
        self.feedback_torso = None

        # 订阅 topics
        self.create_subscription(
            CompressedImage,
            "/hdas/camera_head/right_raw/image_raw_color/compressed",
            self.camera_head_callback,
            10,
        )
        self.create_subscription(
            CompressedImage,
            "/hdas/camera_wrist_left/color/image_raw/compressed",
            self.camera_wrist_left_callback,
            10,
        )
        self.create_subscription(
            CompressedImage,
            "/hdas/camera_wrist_right/color/image_raw/compressed",
            self.camera_wrist_right_callback,
            10,
        )
        self.create_subscription(
            String, "/language_instruction", self.language_callback, 10
        )
        self.create_subscription(
            JointState, "/hdas/feedback_arm_left", self.feedback_arm_left_callback, 10
        )
        self.create_subscription(
            JointState, "/hdas/feedback_arm_right", self.feedback_arm_right_callback, 10
        )
        self.create_subscription(
            JointState,
            "/hdas/feedback_gripper_left",
            self.feedback_gripper_left_callback,
            10,
        )
        self.create_subscription(
            JointState,
            "/hdas/feedback_gripper_right",
            self.feedback_gripper_right_callback,
            10,
        )
        try:
            self.create_subscription(
                JointState, "/hdas/feedback_torso", self.feedback_torso_callback, 10
            )
        except Exception:
            # if no torso topic available, leave as None
            pass

        # 发布 topics
        self.pub_gripper_left = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_left", 5
        )
        self.pub_gripper_right = self.create_publisher(
            JointState, "/motion_target/target_position_gripper_right", 5
        )
        self.pub_arm_left = self.create_publisher(
            JointState, "/motion_target/target_joint_state_arm_left", 5
        )
        self.pub_arm_right = self.create_publisher(
            JointState, "/motion_target/target_joint_state_arm_right", 5
        )
        self.pub_torso = self.create_publisher(
            JointState, "/motion_target/target_joint_state_torso", 5
        )

        # wait for a language instruction
        WAIT_TIMEOUT = 30.0
        deadline = self.get_clock().now() + Duration(seconds=WAIT_TIMEOUT)
        self.get_logger().info(
            f"Waiting for /language_instruction for up to {WAIT_TIMEOUT}s..."
        )
        while rclpy.ok() and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.lock:
                if self.language is not None:
                    self.get_logger().info(
                        f"✅ Received language instruction: {self.language.data}"
                    )
                    break

        # 启动键盘读取线程(用于非阻塞按键控制)
        self.key_state = "p"  # p: pause, s: start, r: reset(move_to_init)
        self._kb_thread = threading.Thread(target=self._keyboard_thread, daemon=True)
        self._kb_thread.start()

        # trick
        self.record_frame = np.zeros(ACTION_DIM, dtype=float)

    # ---------------- callbacks ----------------
    def camera_head_callback(self, msg: CompressedImage):
        with self.lock:
            self.camera_head = msg

    def camera_wrist_left_callback(self, msg: CompressedImage):
        with self.lock:
            self.camera_wrist_left = msg

    def camera_wrist_right_callback(self, msg: CompressedImage):
        with self.lock:
            self.camera_wrist_right = msg

    def language_callback(self, msg: String):
        with self.lock:
            self.language = msg

    def feedback_arm_left_callback(self, msg: JointState):
        with self.lock:
            self.feedback_arm_left = msg

    def feedback_arm_right_callback(self, msg: JointState):
        with self.lock:
            self.feedback_arm_right = msg

    def feedback_gripper_left_callback(self, msg: JointState):
        with self.lock:
            self.feedback_gripper_left = msg

    def feedback_gripper_right_callback(self, msg: JointState):
        with self.lock:
            self.feedback_gripper_right = msg

    def feedback_torso_callback(self, msg: JointState):
        with self.lock:
            self.feedback_torso = msg

    # ---------------- helper utils ----------------
    # Non-blocking key read
    def get_key_nonblock(self):
        fd = sys.stdin.fileno()
        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] &= ~termios.ICANON & ~termios.ECHO  # 关闭缓冲和回显
        oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
        try:
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            try:
                return sys.stdin.read(1)
            except IOError:
                return None
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

    def _all_inputs_ready(self) -> bool:
        with self.lock:
            ok = all(
                [
                    self.camera_head,
                    self.camera_wrist_left,
                    self.camera_wrist_right,
                    self.language,
                    self.feedback_arm_left,
                    self.feedback_arm_right,
                    self.feedback_gripper_left,
                    self.feedback_gripper_right,
                ]
            )
            # self.get_logger().info(f"camera_head: {self.camera_head is not None}")                                # todo: debug
            # self.get_logger().info(f"camera_wrist_left: {self.camera_wrist_left is not None}")                    # todo: debug
            # self.get_logger().info(f"camera_wrist_right: {self.camera_wrist_right is not None}")                  # todo: debug
            # self.get_logger().info(f"language: {self.language is not None}")                                      # todo: debug
            # self.get_logger().info(f"feedback_arm_left length: {len(self.feedback_arm_left.position)}")           # todo: debug
            # self.get_logger().info(f"feedback_arm_right length: {len(self.feedback_arm_right.position)}")         # todo: debug
            # self.get_logger().info(f"feedback_gripper_left length: {len(self.feedback_gripper_left.position)}")   # todo: debug
            # self.get_logger().info(f"feedback_gripper_right length: {len(self.feedback_gripper_right.position)}") # todo: debug
            self.get_logger().info(f"ok: {ok}")
        return ok

    def decode_compressed_image(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, IMAGE_SHAPE, interpolation=cv2.INTER_AREA)

    def linear_interpolate(self, start, end, n_points):
        s = np.asarray(start, dtype=float)
        e = np.asarray(end, dtype=float)
        if s.shape != e.shape:
            raise ValueError(f"start/end shape mismatch: {s.shape} vs {e.shape}")
        t = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)  # (n_points, 1)
        return (1.0 - t) * s.reshape(1, -1) + t * e.reshape(1, -1)

    # ---------------- inference once ----------------
    def process_once(self):
        # 在锁内将需要的数据复制到本地变量, 防止在处理过程中回调修改共享变量导致数据不一致或异常
        with self.lock:
            camera_head = self.camera_head
            camera_wrist_left = self.camera_wrist_left
            camera_wrist_right = self.camera_wrist_right
            language = self.language
            feedback_arm_left = self.feedback_arm_left
            feedback_arm_right = self.feedback_arm_right
            feedback_gripper_left = self.feedback_gripper_left
            feedback_gripper_right = self.feedback_gripper_right

        cv_image_head = self.decode_compressed_image(camera_head)
        cv_image_wrist_left = self.decode_compressed_image(camera_wrist_left)
        cv_image_wrist_right = self.decode_compressed_image(camera_wrist_right)
        prompt = language.data

        # -------- 主要处理流程 ---------
        start_time = time.time()

        # 将 gripper 与 arm 的位置拼接成一个一维状态向量
        state_np = np.concatenate(
            [
                feedback_gripper_left.position,
                feedback_gripper_right.position,
                feedback_arm_left.position,
                feedback_arm_right.position,
            ],
            axis=0,
        )

        dt = 1.0 / self.publish_rate_hz

        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0)

        batch = {
            "observation.images.head": torch.from_numpy(cv_image_head)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0,
            "observation.images.hand_left": torch.from_numpy(cv_image_wrist_left)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0,
            "observation.images.hand_right": torch.from_numpy(cv_image_wrist_right)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0,
            "observation.state": state_tensor,
            "task": prompt if isinstance(prompt, list) else [prompt],
        }
        device = "cuda"  # todo: device
        try:
            first_param = next(self.policy.parameters())
            model_dtype = first_param.dtype
        except StopIteration:
            model_dtype = torch.float32

        batch = {
            k: (
                v.to(device=device, dtype=model_dtype, non_blocking=device == "cuda")
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        with torch.inference_mode():
            actions = self.policy.select_action(batch)
        self.get_logger().info(
            f"cfz | action shape: {actions.shape}, dtype:{actions.dtype}"
        )

        vla_output = actions.cpu().numpy()[
            : self.chunk_size
        ]  # vla_output.shape: [50, action_dim]
        end_time = time.time()
        self.get_logger().info(
            f"VLA inference time: {end_time - start_time:.4f} seconds"
        )

        # Step 2: 原始时间点 t_original 和目标时间点 t_resampled
        t_original = np.linspace(0, 1, self.chunk_size)  # 原始采样时间点
        t_resampled = np.linspace(
            0, 1, self.chunk_size * UPSAMPLING_FACTOR
        )  # 目标采样时间点

        # Step 3: 对每列进行三次样条插值
        result_resampled = np.zeros(
            (len(t_resampled), ACTION_DIM)
        )  # 初始化新的结果矩阵
        for col in range(ACTION_DIM):  # 遍历每列
            cs = CubicSpline(t_original, vla_output[:, col])  # 使用三次样条插值
            result_resampled[:, col] = cs(t_resampled)

        for idx in range(len(t_resampled)):
            if not rclpy.ok() or self.key_state == "p":
                self.get_logger().info("process_once: interrupted by pause or shutdown")
                break

            row1 = result_resampled[idx, :]

            if idx % 10 == 0:  # todo: change 10
                self.publish_control_gripper_signals(row1, idx)
            self.publish_control_arm_signals(row1, idx)

            self.record_frame = row1.copy()

            if not rclpy.ok() or self.key_state == "p":
                break
            time.sleep(dt)

    def publish_control_gripper_signals(self, data_row, idx: int):
        stamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp

        # gripper left (col 0)
        js = JointState()
        js.header = header
        js.position = [float(data_row[0])]

        self.pub_gripper_left.publish(js)
        self.get_logger().info(f"{idx}-th left gripper: {js.position[0]}")  # todo

        # gripper right (col 1)
        js = JointState()
        js.header = header
        js.position = [float(data_row[1])]

        self.pub_gripper_right.publish(js)
        # self.get_logger().info(f"{idx}-th right gripper: {js.position[0]}") # todo

    def publish_control_arm_signals(self, data_row, idx: int):
        stamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = stamp

        # arm left (cols 2..7)
        js = JointState()
        js.header = header
        js.position = [float(x) for x in data_row[2 : 2 + int((ACTION_DIM - 2) / 2)]]
        self.pub_arm_left.publish(js)

        # arm right (cols 8..end)
        js = JointState()
        js.header = header
        js.position = [float(x) for x in data_row[-int((ACTION_DIM - 2) / 2) :]]
        self.pub_arm_right.publish(js)

    # ---------------- move_to_init ----------------
    def wait_for_move_feedback(self, timeout=60.0, spin_timeout=0.01):
        """Wait until feedback (arm/gripper/torso) are populated for move_to_init"""
        deadline = self.get_clock().now() + Duration(seconds=timeout)
        while rclpy.ok() and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=spin_timeout)
            with self.lock:
                fb = {
                    "arm_left": (
                        self.feedback_arm_left.position
                        if self.feedback_arm_left
                        else None
                    ),
                    "arm_right": (
                        self.feedback_arm_right.position
                        if self.feedback_arm_right
                        else None
                    ),
                    "gripper_left": (
                        self.feedback_gripper_left.position
                        if self.feedback_gripper_left
                        else None
                    ),
                    "gripper_right": (
                        self.feedback_gripper_right.position
                        if self.feedback_gripper_right
                        else None
                    ),
                    "torso": (
                        self.feedback_torso.position
                        if self.feedback_torso is not None
                        else None
                    ),
                }
            if all(v is not None for v in fb.values()):
                return fb
        missing = [k for k, v in fb.items() if v is None]
        self.get_logger().error(f"Timeout waiting move feedback, missing: {missing}")
        return None

    def _publish_move_js(self, publisher, positions, stamp):
        js = JointState()
        js.header = Header()
        js.header.stamp = stamp
        js.position = list(positions)
        publisher.publish(js)

    def move_to_init_and_publish(self):
        """
        Equivalent of original move_to_init(), but uses VLANode's feedback and publishers.
        Blocks until complete. Returns True on success.
        """
        fb = self.wait_for_move_feedback()
        if fb is None:
            return False

        # dimension checks
        if not (
            len(fb["arm_left"]) == len(TARGET["arm_left"])
            and len(fb["arm_right"]) == len(TARGET["arm_right"])
            and len(fb["gripper_left"]) == len(TARGET["gripper_left"])
            and len(fb["gripper_right"]) == len(TARGET["gripper_right"])
            and len(fb["torso"]) == len(TARGET["torso"])
        ):
            self.get_logger().error(
                "Feedback and target dimensions mismatch for move_to_init"
            )
            return False

        # prepare trajectories
        if USERNAME == "nvidia":
            traj = {
                k: self.linear_interpolate(
                    fb[k], TARGET[k], self.chunk_size * UPSAMPLING_FACTOR
                )
                for k in fb.keys()
            }
        else:
            # for r1lite, arm trajectories are first 6 dims
            traj = {
                "arm_left": self.linear_interpolate(
                    fb["arm_left"],
                    TARGET["arm_left"],
                    self.chunk_size * UPSAMPLING_FACTOR,
                )[:, :6],
                "arm_right": self.linear_interpolate(
                    fb["arm_right"],
                    TARGET["arm_right"],
                    self.chunk_size * UPSAMPLING_FACTOR,
                )[:, :6],
                "gripper_left": self.linear_interpolate(
                    fb["gripper_left"],
                    TARGET["gripper_left"],
                    self.chunk_size * UPSAMPLING_FACTOR,
                ),
                "gripper_right": self.linear_interpolate(
                    fb["gripper_right"],
                    TARGET["gripper_right"],
                    self.chunk_size * UPSAMPLING_FACTOR,
                ),
                "torso": self.linear_interpolate(
                    fb["torso"], TARGET["torso"], self.chunk_size * UPSAMPLING_FACTOR
                ),
            }

        self.get_logger().info("Starting move_to_init trajectory publishing...")
        dt = 100.0 / self.publish_rate_hz  # todo
        for i in range(self.chunk_size * UPSAMPLING_FACTOR):
            if not rclpy.ok() or self.key_state == "p":
                self.get_logger().info("move_to_init: interrupted by pause or shutdown")
                break
            ts = self.get_clock().now().to_msg()
            # grippers
            self._publish_move_js(self.pub_gripper_left, traj["gripper_left"][i], ts)
            self._publish_move_js(self.pub_gripper_right, traj["gripper_right"][i], ts)
            # arms
            self._publish_move_js(self.pub_arm_left, traj["arm_left"][i], ts)
            self._publish_move_js(self.pub_arm_right, traj["arm_right"][i], ts)
            # torso
            self._publish_move_js(self.pub_torso, traj["torso"][i], ts)
            self.get_logger().info(f"move_to_init step {i}")
            time.sleep(dt)
        self.get_logger().info("move_to_init finished.")
        time.sleep(0.05)
        return True

    # keyboard thread to change key_state
    def _keyboard_thread(self):
        while rclpy.ok():
            k = self.get_key_nonblock()
            if k is None:
                time.sleep(0.02)
                continue
            if k == "p":
                self.key_state = "p"
                self.get_logger().info("===> VLA 推理&信号已暂停")
            elif k == "s":
                self.key_state = "s"
                self.get_logger().info("===> VLA 推理&信号已开始")
            elif k == "r":
                self.key_state = "r"
                self.get_logger().info("===> VLA 复位 (move_to_init) 被触发")

    def run(self):
        while rclpy.ok():
            if self.key_state == "p":
                time.sleep(0.02)
                continue
            elif self.key_state == "s":
                if self._all_inputs_ready():
                    self.process_once()
            elif self.key_state == "r":
                self.get_logger().info("Executing move_to_init...")
                try:
                    ok = self.move_to_init_and_publish()
                    self.get_logger().info(
                        "move_to_init succeeded" if ok else "move_to_init failed"
                    )
                except Exception as e:
                    self.get_logger().error(f"move_to_init exception: {e}")
                time.sleep(1.0)
                self.key_state = "p"


def main(args=None):
    rclpy.init(args=args)
    node = VLANode(PRETRAINED_PATH)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
