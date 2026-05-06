#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PI0.5 推理节点

订阅（观测）:
/camera3/camera_head/color/image_raw           sensor_msgs/Image
/camera1/camera_left/color/image_rect_raw      sensor_msgs/Image
/camera2/camera_right/color/image_rect_raw     sensor_msgs/Image
/get_dual_arm                                  sc_ros2/ArmData  (16D)
/language_instruction                          std_msgs/String

发布（控制）:
/set_dual_arm       sc_ros2/ArmData  16D : [L0..L6, LG, R0..R6, RG]

ROS2 参数:
    policy_host              str    推理服务 IP       (default "172.16.10.14")
    policy_port              int    推理服务端口       (default 5555)
    control_frequency        float  动作下发频率 Hz    (default 10.0)
    prompt                   str    任务提示语         (default "Make coffee")

注：action_mode 已与推理路径强绑定 (async -> 5, sync -> 6)，不再可配置

键盘控制（运行时）:
    p — 暂停推理和动作下发
    s — 开始 / 恢复推理
    r — 回初始位
"""

import os
import sys
import time
import json
import math
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from sc_ros2.msg import ArmData

from openpi_client import websocket_client_policy

import termios
import fcntl
from scipy.interpolate import PchipInterpolator

# ================================================================
# 常量
# ================================================================
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

# 初始位姿（与 robot_control.cpp 的 zero_angle_l / zero_angle_r 一致）
INIT_LEFT_JOINTS = [
    23.628 * math.pi / 180.0,
    -55.034 * math.pi / 180.0,
    -13.216 * math.pi / 180.0,
    -112.641 * math.pi / 180.0,
    54.613 * math.pi / 180.0,
    -12.182 * math.pi / 180.0,
    -3.7888 * math.pi / 180.0,
]
INIT_RIGHT_JOINTS = [
    -24.459 * math.pi / 180.0,
    54.611 * math.pi / 180.0,
    13.531 * math.pi / 180.0,
    111.269 * math.pi / 180.0,
    -55.829 * math.pi / 180.0,
    12.532 * math.pi / 180.0,
    1.082 * math.pi / 180.0,
]
INIT_GRIPPER = 69.0

# 16D: [L0..L6, LG, R0..R6, RG]
INIT_POSE_16D = np.array(
    INIT_LEFT_JOINTS + [INIT_GRIPPER] + INIT_RIGHT_JOINTS + [INIT_GRIPPER]
)

# 16D 关节/夹爪列索引
JOINT_COLS_16D = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
GRIPPER_COLS_16D = [7, 15]

# 与底层 validate_* 保持一致，留少量余量避免浮点边界 reject
JOINT_LIMIT_RAD = math.pi - 1e-3
GRIPPER_LIMIT_MIN = 0.0
GRIPPER_LIMIT_MAX = 70.0


# ================================================================
# Pi05InferNode
# ================================================================
class Pi05InferNode(Node):

    ARM_JOINT_DIM = 7

    def __init__(self):
        super().__init__("pi05_infer_node")

        # ---- ROS2 参数 ----
        self.declare_parameter("policy_host", "172.16.10.14")
        self.declare_parameter("policy_port", 5555)
        self.declare_parameter("control_frequency", 80.0)
        self.declare_parameter("prompt", "Make coffee")
        self.declare_parameter("async_inference", True)
        self.declare_parameter("infer_interval", 0.64)
        self.declare_parameter("interpolation_scale", 10)
        self.declare_parameter("blend_steps", 10)
        # ===== 异步推理参数 =====
        # chunk_size_threshold:
        #   队列长度 / 已观察到的最大 chunk 长度 <= 该比例时触发新一次推理。
        #   越大越频繁更新，越小越接近同步
        # async_aggregate_alpha_new:
        #   重叠区"new chunk 在尾端的权重上限"。重叠区内 w 用 smoothstep 平滑
        #   alpha=1.0  => 头尾都完全平滑
        #   alpha<1.0  => 保留旧计划在尾部的影响，对新观测响应更钝
        self.declare_parameter("chunk_size_threshold", 0.5)
        self.declare_parameter("async_aggregate_alpha_new", 1.0)

        self._policy_host = self.get_parameter("policy_host").value
        self._policy_port = self.get_parameter("policy_port").value
        self._control_freq = self.get_parameter("control_frequency").value
        self._prompt = self.get_parameter("prompt").value
        self._async_inference = self.get_parameter("async_inference").value
        self._infer_interval = self.get_parameter("infer_interval").value
        self._interpolation_scale = self.get_parameter("interpolation_scale").value
        self._blend_steps = int(self.get_parameter("blend_steps").value)
        self._chunk_size_threshold = float(
            self.get_parameter("chunk_size_threshold").value
        )
        self._chunk_size_threshold = min(max(0.0, self._chunk_size_threshold), 1.0)
        self._aggregate_alpha_new = float(
            self.get_parameter("async_aggregate_alpha_new").value
        )
        self._aggregate_alpha_new = min(max(0.0, self._aggregate_alpha_new), 1.0)
        if self._async_inference:
            self.get_logger().info(
                f"async config: chunk_size_threshold={self._chunk_size_threshold}, "
                f"aggregate_alpha_new={self._aggregate_alpha_new}, "
                f"control_freq={self._control_freq:.1f}Hz"
            )

        self.bridge = CvBridge()

        # ---- 观测队列 ----
        self.img_head_queue = deque(maxlen=16)
        self.img_left_queue = deque(maxlen=16)
        self.img_right_queue = deque(maxlen=16)
        self.dual_arm_queue = deque(maxlen=128)

        # ---- 线程锁 ----
        self._obs_lock = threading.Lock()
        self._queue_lock = threading.Lock()

        # ---- 动作下发队列：deque[ (timestep:int, action_16d:np.ndarray) ] ----
        # 队列按 timestep 升序存储；timestep 是控制频率下的全局递增计数。
        # _global_step 表示"下一次 dispatch tick 应执行的 timestep"，
        # 每次 dispatch tick（无论是否实际拿到动作）都会自增 1
        self._action_queue: deque = deque()
        self._global_step: int = 0
        # 已观察到的最大 chunk 长度
        self._action_chunk_size_max: int = 1

        # ---- 推理耗时 EMA ----
        self._infer_time_ema = None
        self._infer_time_ema_alpha = 0.3

        # ---- 后处理状态 ----
        self._last_actions = None
        self._sliding_history = None
        self._ema_state = None

        # ---- 最近一次实际下发给底层的 16D 指令 ----
        self._last_commanded_16d = None

        # ---- 键盘控制状态 ----
        self._key_state = "p"  # 'p' 暂停 / 's' 推理 / 'r' 复位

        # ======================
        # 订阅（观测话题）
        # ======================
        self.create_subscription(
            Image, "/camera3/camera_head/color/image_raw", self._head_cb, 10
        )
        self.create_subscription(
            Image, "/camera1/camera_left/color/image_rect_raw", self._left_cb, 10
        )
        self.create_subscription(
            Image, "/camera2/camera_right/color/image_rect_raw", self._right_cb, 10
        )
        self.create_subscription(ArmData, "/get_dual_arm", self._arm_cb, 200)
        self.create_subscription(String, "/language_instruction", self._language_cb, 10)
        self.get_logger().info("Subscriptions created")

        # ======================
        # 发布（控制话题，匹配 robot_control.cpp 订阅端）
        # 使用 16D /set_dual_arm 通道下发
        # ======================
        self.pub_set_dual_arm = self.create_publisher(ArmData, "/set_dual_arm", 10)
        self.get_logger().info("Publishers created (dual mode)")

        # ======================
        # 动作下发定时器（仅异步推理模式使用队列 + 定时器）
        # 同步模式下在 _infer_loop 中直接按控制周期下发动作，不走队列
        # ======================
        self._dispatch_timer = None
        if self._async_inference:
            self._dispatch_timer = self.create_timer(
                1.0 / self._control_freq, self._dispatch_action_cb
            )

        # ======================
        # 连接推理服务
        # ======================
        self.get_logger().info(
            f"Connecting to policy server "
            f"{self._policy_host}:{self._policy_port} ..."
        )
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=self._policy_host, port=self._policy_port
        )
        self.get_logger().info("Policy server connected")

        # ---- 日志文件 ----
        cur_time = time.time()
        self._saved_actions_file = f"actions_{cur_time}.jsonl"
        self._saved_sub_actions_file = f"sub_actions_{cur_time}.jsonl"

        # ======================
        # 键盘监听线程
        # ======================
        self._infer_running = True
        self._kb_thread = threading.Thread(target=self._keyboard_thread, daemon=True)
        self._kb_thread.start()

        # ======================
        # 推理线程
        # ======================
        self._infer_count = 0 # 推理计数
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info(
            "Node ready  |  press 's' to start, 'p' to pause, 'r' to reset"
        )

    # ----------------------------------------------------------
    # 订阅回调（加锁保护）
    # ----------------------------------------------------------
    def _head_cb(self, msg: Image):
        with self._obs_lock:
            self.img_head_queue.append(msg)

    def _left_cb(self, msg: Image):
        with self._obs_lock:
            self.img_left_queue.append(msg)

    def _right_cb(self, msg: Image):
        with self._obs_lock:
            self.img_right_queue.append(msg)

    def _arm_cb(self, msg: ArmData):
        with self._obs_lock:
            self.dual_arm_queue.append(msg)

    def _language_cb(self, msg: String):
        old = self._prompt
        self._prompt = msg.data
        if msg.data != old:
            self.get_logger().info(f"Prompt updated: {self._prompt}")

    # ----------------------------------------------------------
    # 就绪检查 + 诊断日志
    # ----------------------------------------------------------
    def _all_inputs_ready(self) -> bool:
        missing = []
        with self._obs_lock:
            if not self.img_head_queue:
                missing.append("head_image")
            if not self.img_left_queue:
                missing.append("left_image")
            if not self.img_right_queue:
                missing.append("right_image")
            if not self.dual_arm_queue:
                missing.append("dual_arm")
        if missing:
            self.get_logger().warn(f"Waiting for: {missing}", throttle_duration_sec=2.0)
            return False
        return True

    # ----------------------------------------------------------
    # 时间对齐取帧（整体加锁）
    # ----------------------------------------------------------
    def _pop_at(self, queue, ts):
        """取 queue 中 timestamp >= ts 的第一条消息（需在 _obs_lock 内调用）"""
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

    def _get_current_state(self):
        """读取最新的 16D 关节状态，无数据时返回 None"""
        with self._obs_lock:
            if not self.dual_arm_queue:
                return None
            return np.array(self.dual_arm_queue[-1].data, dtype=np.float64)

    def _get_frame(self):
        with self._obs_lock:
            if not (
                self.img_head_queue
                and self.img_left_queue
                and self.img_right_queue
                and self.dual_arm_queue
            ):
                return None

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

        img_head = self.bridge.imgmsg_to_cv2(head, "rgb8")
        img_left = self.bridge.imgmsg_to_cv2(left, "rgb8")
        img_right = self.bridge.imgmsg_to_cv2(right, "rgb8")
        self.get_logger().debug(f"frame_time: {frame_time}")

        return img_head, img_left, img_right, arm.data

    # ----------------------------------------------------------
    # 构建观测
    # ----------------------------------------------------------
    def _build_observation(self):
        res = self._get_frame()
        if res is None:
            return None

        img_head, img_left, img_right, state = res

        return {
            "observation/head_image": img_head,
            "observation/left_wrist_image": img_left,
            "observation/right_wrist_image": img_right,
            "observation/state": np.array(state),
            "prompt": self._prompt,
        }

    # ==========================================================
    # 键盘控制
    # ==========================================================
    @staticmethod
    def _get_key_nonblock():
        """非阻塞读取单个按键，无输入返回 None"""
        fd = sys.stdin.fileno()
        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] &= ~termios.ICANON & ~termios.ECHO
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

    def _keyboard_thread(self):
        while self._infer_running and rclpy.ok():
            k = self._get_key_nonblock()
            if k is None:
                time.sleep(0.02)
                continue
            if k == "p":
                self._key_state = "p"
                self.get_logger().info("===> Paused")
            elif k == "s":
                self._key_state = "s"
                self.get_logger().info("===> Started")
            elif k == "r":
                self._key_state = "r"
                self.get_logger().info("===> Reset (move_to_init) triggered")

    # ==========================================================
    # move_to_init 回初始位
    # ==========================================================
    def _move_to_init(self):
        self.get_logger().info("move_to_init: sending init target ...")
        target = INIT_POSE_16D

        self._dispatch_actions_sync([target.copy()])
        if self._key_state == "p":
            self.get_logger().info("move_to_init: interrupted by pause")
            return

        self.get_logger().info("move_to_init: done")

    # ==========================================================
    # 推理线程主循环
    # ==========================================================
    def _infer_loop(self):
        while self._infer_running and rclpy.ok():
            # ---- 键盘状态检查 ----
            if self._key_state == "p":
                time.sleep(0.05)
                continue

            if self._key_state == "r":
                self._move_to_init()
                self._key_state = "p"
                continue

            # ---- key_state == 's': 正常推理 ----
            if not self._all_inputs_ready():
                time.sleep(0.1)
                continue

            if self._async_inference:
                # 队列利用率 <= chunk_size_threshold
                if not self._should_send_observation():
                    time.sleep(0.01)
                    continue
            else:
                if False: #self._infer_count >= 2 and self._infer_count <= 4:
                    time.sleep(1.0) # !!TRICK: 强制延时, 模拟长时间推理, 避开杯子未落下的区段
                else:
                    time.sleep(0.045) # !!TRICK: 增加延时, 减缓杯子抖动

            # ---- 采样观测，并锁定本次 chunk 的起始 timestep ----
            # obs_timestep = "本次观测对应的下一次 dispatch tick 应执行的 timestep"
            obs = self._build_observation()
            if obs is None:
                time.sleep(0.2)
                continue
            with self._queue_lock:
                obs_timestep = int(self._global_step)

            # ---- 推理 ----
            tic = time.time()
            output = self._policy.infer(obs)
            actions = output["actions"].copy()
            self._infer_count += 1
            toc = time.time()
            infer_dt = toc - tic
            self._update_infer_time_ema(infer_dt)
            self.get_logger().info(
                f"推理耗时: {infer_dt:.3f}s, 模型耗时: {output['server_timing']['infer_ms']/1000:.3f}s, "
                f"ema={self._infer_time_ema:.3f}s, obs_ts={obs_timestep}"
            )

            # ---- 夹爪值映射 ----
            actions[:, 7] = np.clip((actions[:, 7] * 100 - 45) * 70 / 25, 0, 70)
            actions[:, 15] = np.clip((actions[:, 15] * 100 - 38) * 70 / 32, 0, 70)

            # ---- 平滑后处理 ----
            sub_actions = self._process_actions(
                actions,
                mode="sliding_mean",
                value={
                    7: max(1, int(0.6 * len(actions))),   # left gripper
                    15: max(1, int(0.5 * len(actions))),  # right gripper
                },
            )

            # ---- 根据推理路径强制绑定 action_mode ----
            # 异步：mode 5（纯 PCHIP）
            # 同步：mode 6（PCHIP + 头部 smoothstep）
            mode = 5 if self._async_inference else 6
            final_actions = self._apply_action_mode(sub_actions, mode)

            # ---- 动作下发 ----
            if self._async_inference:
                # 异步：以 obs_timestep 为基准把新 chunk 聚合进队列
                self._aggregate_action_queue(final_actions, obs_timestep)
            else:
                # 同步：不走队列，按控制周期直接下发完整 chunk
                self._dispatch_actions_sync(final_actions)

            # ---- 保存日志 ----
            # self._save_actions(actions, sub_actions)

    # ==========================================================
    # 同步模式下发：按控制周期逐帧发布，直到下发完或被暂停
    # ==========================================================
    def _dispatch_actions_sync(self, actions):
        # period = 1.0 / self._control_freq
        # for step in actions:
        #     if not (self._infer_running and rclpy.ok()):
        #         return
        #     if self._key_state == "p":
        #         return
        #     self._publish_step(step)
        #     time.sleep(period)

        # 节拍控制，考虑发送和处理的时延
        period = 1.0 / self._control_freq
        next_t = time.time()
        for step in actions:
            if not (self._infer_running and rclpy.ok()):
                return
            if self._key_state == "p":
                return
            self._publish_step(step)
            next_t += period
            sleep_dt = next_t - time.time()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                next_t = time.time()

        time.sleep(3*period)

    # ==========================================================
    # 异步推理：基于 timestep 把新 chunk 聚合进队列
    #
    # 时间步约定：
    #   - obs_timestep    : 采样观测时锁定的"下一次 dispatch tick 应执行的 timestep"
    #   - 新 chunk[i]     : 绑定 timestep = obs_timestep + i
    #   - _global_step    : 当前 dispatch 线程"下一次 tick 即将执行"的 timestep
    #   - 任何 timestep < _global_step 的动作都视为已过期，直接丢弃
    #   - 与队列已有 timestep 重叠的动作按"smoothstep 渐变 + 加权"融合：
    #       重叠头 (ts=latest) w=0  -> 完全保留旧队列
    #       重叠尾 (ts=prev_tail) w=alpha -> 新 chunk 平滑衔接
    #   - 未来的 timestep 直接插入
    # ==========================================================
    def _aggregate_action_queue(self, new_chunk, obs_timestep: int):
        new_arr = np.asarray(list(new_chunk), dtype=np.float64)
        if new_arr.ndim != 2 or new_arr.shape[1] < 16 or new_arr.shape[0] == 0:
            self.get_logger().warn(
                f"_aggregate_action_queue: invalid chunk shape {new_arr.shape}, drop"
            )
            return

        N = new_arr.shape[0]
        a = float(self._aggregate_alpha_new)

        # ---- Phase 1: 锁内快照 ----
        with self._queue_lock:
            self._action_chunk_size_max = max(self._action_chunk_size_max, N)
            latest = int(self._global_step)
            q_len_before = len(self._action_queue)
            existing = {ts: act for ts, act in self._action_queue}
            # 旧队列的尾端 timestep；用于在锁外计算重叠区长度
            prev_tail = (
                self._action_queue[-1][0] if self._action_queue else latest - 1
            )

        # ---- Phase 2: 锁外做加权聚合 + 排序 ----
        overlap_lo = max(latest, obs_timestep)
        overlap_hi = min(prev_tail, obs_timestep + N - 1)
        K = max(0, overlap_hi - overlap_lo + 1)

        merged = dict(existing)
        dropped_stale = 0
        merged_overlap = 0
        appended_new = 0
        for i in range(N):
            ts = obs_timestep + i
            if ts < latest:
                dropped_stale += 1
                continue
            new_act = new_arr[i]
            if ts in existing:
                if K > 1:
                    u = (ts - overlap_lo) / (K - 1)
                    w = (3.0 * u * u - 2.0 * u * u * u) * a
                else:
                    w = a
                merged[ts] = (1.0 - w) * existing[ts] + w * new_act
                merged_overlap += 1
            else:
                merged[ts] = new_act
                appended_new += 1
        sorted_items = sorted(merged.items(), key=lambda x: x[0])

        # ---- Phase 3: 锁内写回 ----
        with self._queue_lock:
            latest2 = int(self._global_step)
            if latest2 > latest:
                sorted_items = [(ts, act) for ts, act in sorted_items if ts >= latest2]
                dropped_stale += latest2 - latest
            self._action_queue = deque(sorted_items)
            q_len_after = len(self._action_queue)
            head_ts = self._action_queue[0][0] if self._action_queue else None
            tail_ts = self._action_queue[-1][0] if self._action_queue else None

        self.get_logger().info(
            f"async-aggregate: obs_ts={obs_timestep}, latest={latest}->{latest2}, "
            f"chunk={N}, drop_stale={dropped_stale}, merge={merged_overlap}, "
            f"new={appended_new}, q={q_len_before}->{q_len_after} "
            f"[{head_ts}..{tail_ts}], chunk_max={self._action_chunk_size_max}"
        )

    # ----------------------------------------------------------
    # queue_size / chunk_size_max <= chunk_size_threshold => 触发新一次推理
    # ----------------------------------------------------------
    def _should_send_observation(self) -> bool:
        with self._queue_lock:
            q_len = len(self._action_queue)
            chunk_max = max(1, int(self._action_chunk_size_max))
        return (q_len / chunk_max) <= self._chunk_size_threshold

    # ----------------------------------------------------------
    # 推理耗时 EMA
    # ----------------------------------------------------------
    def _update_infer_time_ema(self, dt: float):
        dt = float(max(0.0, dt))
        if self._infer_time_ema is None:
            self._infer_time_ema = dt
        else:
            a = float(self._infer_time_ema_alpha)
            self._infer_time_ema = a * dt + (1.0 - a) * self._infer_time_ema

    # ----------------------------------------------------------
    # 通用 smoothstep 头部融合：
    #   把 seq[:n] 与给定 start 之间用 smoothstep(3u^2-2u^3) 做平滑过渡
    # 返回: (融合后 ndarray, 实际融合的帧数 n)
    # ----------------------------------------------------------
    def _smoothstep_blend_head(self, seq, start, blend_len):
        seq = np.asarray(seq, dtype=np.float64)
        if seq.size == 0 or blend_len <= 0 or start is None:
            return seq, 0

        start = np.asarray(start, dtype=np.float64).reshape(-1)
        if start.shape[0] != seq.shape[1]:
            return seq, 0

        n = min(int(blend_len), len(seq))
        if n <= 0:
            return seq, 0

        u = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])
        alpha = (3.0 * u * u - 2.0 * u * u * u)[:, None]

        out = seq.copy()
        head = (1.0 - alpha) * start.reshape(1, -1) + alpha * out[:n]

        thr = 0.5
        use_new_mask = (alpha[:, 0] >= thr)
        for col in GRIPPER_COLS_16D:
            head[:, col] = np.where(use_new_mask, out[:n, col], start[col])

        out[:n] = head
        return out, n

    # ==========================================================
    # 动作模式变换
    # ==========================================================
    def _apply_action_mode(self, actions, mode):
        """将 action_chunk 按 mode 变换为最终下发序列"""
        actions = np.asarray(actions)

        if mode == 0:
            return list(actions)

        elif mode == 1:
            final = []
            M = len(actions)
            for i in range(M):
                a = actions[i].copy()
                a[[7, 15]] = actions[0, [7, 15]]
                final.append(a)
            for i in range(1, M):
                a = actions[M - 1].copy()
                a[[7, 15]] = actions[i, [7, 15]]
                final.append(a)
            return final

        elif mode == 2:
            idx = np.unique(np.append(np.arange(0, len(actions), 5), len(actions) - 1))
            return list(actions[idx])

        elif mode == 3:
            idx = np.unique(np.append(np.arange(0, len(actions), 5), len(actions) - 1))
            ds = actions[idx]
            final = []
            for i in range(len(ds)):
                ja = ds[i].copy()
                if i > 0:
                    ja[[7, 15]] = ds[i - 1, [7, 15]]
                final.append(ja)
                final.append(ds[i].copy())
            return final

        elif mode == 4:
            idx = np.unique(np.append(np.arange(0, len(actions), 5), len(actions) - 1))
            ds = actions[idx]
            M = len(ds)
            final = []
            for i in range(M):
                a = ds[i].copy()
                a[[7, 15]] = ds[0, [7, 15]]
                final.append(a)
            for i in range(1, M):
                a = ds[M - 1].copy()
                a[[7, 15]] = ds[i, [7, 15]]
                final.append(a)
            return final

        elif mode == 5:
            # 纯 PCHIP 上采样
            src_steps = actions.shape[0]
            target_steps = int(self._interpolation_scale * src_steps)
            if src_steps <= 1:
                return list(np.repeat(actions, target_steps, axis=0))

            t_src = np.linspace(0.0, 1.0, src_steps)
            t_dst = np.linspace(0.0, 1.0, target_steps)
            result = np.empty((target_steps, actions.shape[1]), dtype=np.float64)
            for d in range(actions.shape[1]):
                pchip = PchipInterpolator(t_src, actions[:, d])
                result[:, d] = pchip(t_dst)

            return list(result)

        elif mode == 6:
            # PCHIP 上采样 + 与"上一次指令/当前位姿"的头部 smoothstep 融合
            # （同步路径专用：直接发布前一次性铺好头部过渡，避免 chunk 拼接处跳变）
            src_steps = actions.shape[0]
            target_steps = int(self._interpolation_scale * src_steps)

            if src_steps <= 1:
                return list(np.repeat(actions, target_steps, axis=0))

            # 形状保持插值，避免三次样条过冲
            t_src = np.linspace(0.0, 1.0, src_steps)
            t_dst = np.linspace(0.0, 1.0, target_steps)
            result = np.empty((target_steps, actions.shape[1]), dtype=np.float64)
            for d in range(actions.shape[1]):
                pchip = PchipInterpolator(t_src, actions[:, d])
                result[:, d] = pchip(t_dst)

            # 起点优先用"上一次实际下发的指令"，没有时回退到反馈态
            start = self._last_commanded_16d
            if start is None:
                start = self._get_current_state()

            # smoothstep 数值混合
            result, _ = self._smoothstep_blend_head(
                result, start, int(self._blend_steps)
            )

            return list(result)

        else:
            self.get_logger().warn(
                f"Unknown action_mode {mode}, fallback to mode 0 (raw chunk)"
            )
            return list(actions)

    # ==========================================================
    # 16D 指令限幅：关节 [-π, π]，夹爪 [0, 70]
    # ==========================================================
    @staticmethod
    def _clip_16d(arr):
        arr = np.asarray(arr, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr[JOINT_COLS_16D] = np.clip(
                arr[JOINT_COLS_16D], -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD
            )
            arr[GRIPPER_COLS_16D] = np.clip(
                arr[GRIPPER_COLS_16D], GRIPPER_LIMIT_MIN, GRIPPER_LIMIT_MAX
            )
        else:
            arr[:, JOINT_COLS_16D] = np.clip(
                arr[:, JOINT_COLS_16D], -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD
            )
            arr[:, GRIPPER_COLS_16D] = np.clip(
                arr[:, GRIPPER_COLS_16D], GRIPPER_LIMIT_MIN, GRIPPER_LIMIT_MAX
            )
        return arr

    # ==========================================================
    # 单步动作发布
    # ==========================================================
    def _publish_step(self, step):
        step = np.asarray(step, dtype=np.float64)
        step = self._clip_16d(step)

        # 16D 布局: [L0..L6, LG, R0..R6, RG]
        dual_msg = ArmData()
        dual_msg.header.stamp = self.get_clock().now().to_msg()
        dual_msg.data = [float(v) for v in step.tolist()]
        self.pub_set_dual_arm.publish(dual_msg)

        self._last_commanded_16d = step.copy()

    # ==========================================================
    # 定时器回调：异步模式下的动作下发（从队列取）
    # ==========================================================
    def _dispatch_action_cb(self):
        if self._key_state == "p":
            with self._queue_lock:
                self._action_queue.clear()
            if self._last_commanded_16d is not None:
                self._publish_step(self._last_commanded_16d)
            return

        step = None
        with self._queue_lock:
            # 队列头 timestep 与当前 _global_step 对齐时弹出执行
            if (
                self._action_queue
                and self._action_queue[0][0] == self._global_step
            ):
                _, step = self._action_queue.popleft()
            # 不论是否拿到动作，本次 tick 都推进时间步（与 wall-clock 对齐），
            # 这样推理过程中流逝的 wall-clock 会自动反映到 _global_step 的推进。
            self._global_step += 1

        # 队列空：保持上一条指令，避免"空窗帧"
        if step is None:
            step = self._last_commanded_16d
            if step is None:
                # 冷启动时尚未下发过动作，回退到当前反馈位姿
                step = self._get_current_state()
                if step is None:
                    return

        self._publish_step(step)

    # ==========================================================
    # 动作后处理（只平滑夹爪）
    # ==========================================================
    def _process_actions(self, actions, mode="mean", value=0):
        actions = np.array(actions)
        if actions.ndim != 2:
            raise ValueError("actions must be 2D")
        processed = actions.copy()
        N, D = actions.shape
        if D < 16:
            raise ValueError("actions dim < 16")

        target_cols = [7, 15]

        if mode == "mean":
            for col in target_cols:
                processed[:, col] = np.mean(actions[:, col])

        elif mode == "sliding_mean":
            if isinstance(value, dict):
                windows = {}
                for col in target_cols:
                    windows[col] = max(0, int(value.get(col, 0)))
            else:
                windows = {col: max(0, int(value)) for col in target_cols}

            if self._sliding_history is None:
                self._sliding_history = {col: [] for col in target_cols}
            for col in target_cols:
                window = windows[col]
                for i in range(N):
                    self._sliding_history[col].append(actions[i, col])
                    if len(self._sliding_history[col]) > window:
                        self._sliding_history[col].pop(0)
                    processed[i, col] = np.mean(self._sliding_history[col])
            self._last_actions = processed[-1]

        elif mode == "rate_limit":
            if self._last_actions is None:
                self._last_actions = processed[0, :]
            else:
                for col in target_cols:
                    processed[0, col] = min(
                        processed[0, col], self._last_actions[col] + value
                    )
                    processed[0, col] = max(
                        processed[0, col], self._last_actions[col] - value
                    )
            for i in range(1, N):
                for col in target_cols:
                    processed[i, col] = min(
                        processed[i - 1, col], self._last_actions[col] + value
                    )
                    processed[i, col] = max(
                        processed[i - 1, col], self._last_actions[col] - value
                    )
            self._last_actions = processed[-1]

        elif mode == "ema":
            alpha = float(value)
            if not (0.0 < alpha <= 1.0):
                raise ValueError("EMA alpha must be in (0, 1]")
            if self._ema_state is None:
                self._ema_state = {}
                for col in target_cols:
                    if self._last_actions is not None:
                        self._ema_state[col] = self._last_actions[col]
                    else:
                        self._ema_state[col] = actions[0, col]
            for i in range(N):
                for col in target_cols:
                    prev = self._ema_state[col]
                    curr = actions[i, col]
                    ema = alpha * curr + (1.0 - alpha) * prev
                    processed[i, col] = ema
                    self._ema_state[col] = ema
            self._last_actions = processed[-1]

        else:
            raise NotImplementedError(f"Unknown mode: {mode}")

        return processed

    @staticmethod
    def remove_small_change_actions(action_chunk, thresholds=BASE_THRESHOLDS):
        if action_chunk.shape[0] <= 1:
            return action_chunk
        kept = [action_chunk[0]]
        last_kept = action_chunk[0]
        for i in range(1, action_chunk.shape[0]):
            curr = action_chunk[i]
            if np.all(np.abs(curr - last_kept) < thresholds):
                continue
            kept.append(curr)
            last_kept = curr
        return np.stack(kept, axis=0)

    # ----------------------------------------------------------
    # 日志保存
    # ----------------------------------------------------------
    def _save_actions(self, actions, sub_actions):
        a_list = actions.tolist() if not isinstance(actions, list) else actions
        s_list = (
            sub_actions.tolist() if not isinstance(sub_actions, list) else sub_actions
        )
        with open(self._saved_actions_file, "a") as f:
            for a in a_list:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        with open(self._saved_sub_actions_file, "a") as f:
            for s in s_list:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ----------------------------------------------------------
    def destroy_node(self):
        self._infer_running = False
        if self._dispatch_timer is not None:
            self._dispatch_timer.cancel()
        if self._infer_thread.is_alive():
            self._infer_thread.join(timeout=3.0)
        super().destroy_node()


# ================================================================
# main
# ================================================================
def main():
    rclpy.init()
    node = Pi05InferNode()
    try:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
