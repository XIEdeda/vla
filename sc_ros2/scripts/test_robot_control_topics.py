#!/usr/bin/env python3
"""
集成测试：订阅所有 get_* 状态话题，并向所有 set_* 控制话题发布与当前状态一致的数据
（避免把机械臂拉到任意姿态）。

依赖：已 source ROS2 与工作空间（含 sc_ros2 消息），且 robot_control 节点在运行。

用法:
  ros2 run sc_ros2 test_robot_control_topics.py
  # 或
  python3 scripts/test_robot_control_topics.py

可选参数见 --help。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

try:
    from sc_ros2.msg import ArmData
except ImportError as e:
    print(
        "无法导入 sc_ros2.msg.ArmData。请先编译工作空间并执行:\n"
        "  source <ws>/install/setup.bash",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


# 与 src/robot_control.cpp 中发布/订阅约定一致
GET_TOPICS: Dict[str, int] = {
    "/get_dual_arm": 16,
    "/get_left_arm": 8,
    "/get_right_arm": 8,
    "/get_left_joint": 7,
    "/get_right_joint": 7,
    "/get_left_gripper": 1,
    "/get_right_gripper": 1,
    "/get_left_wrist": 7,
    "/get_right_wrist": 7,
}

SET_TOPICS_META = [
    ("set_dual_arm", 16),
    ("set_left_arm", 8),
    ("set_right_arm", 8),
    ("set_left_joint", 7),
    ("set_right_joint", 7),
    ("set_left_gripper", 1),
    ("set_right_gripper", 1),
]


def _qos() -> QoSProfile:
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.RELIABLE
    return q


class RobotControlTopicTester(Node):
    """
    订阅回调里会更新（加锁）：
    - _get_lengths[topic]：该话题最近一次消息的 data 长度
    - _get_last[topic]：该话题最近一次消息的 data 拷贝（浮点列表）
    用于 set 回显的首帧来自 /get_dual_arm，见 wait_for_dual_arm() 返回的 snap。
    """

    def __init__(self) -> None:
        super().__init__("robot_control_topic_tester")
        self._lock = threading.Lock()
        self._get_lengths: Dict[str, int] = {}
        self._get_last: Dict[str, List[float]] = {}

        for topic, _ in GET_TOPICS.items():

            def make_cb(name: str) -> Callable[[ArmData], None]:
                def _cb(msg: ArmData) -> None:
                    with self._lock:
                        self._get_lengths[name] = len(msg.data)
                        self._get_last[name] = list(msg.data)

                return _cb

            self.create_subscription(ArmData, topic, make_cb(topic), _qos())


def wait_for_dual_arm(node: RobotControlTopicTester, timeout_sec: float) -> Optional[List[float]]:
    """等到至少收到一帧 /get_dual_arm，返回该帧 data 的拷贝。"""
    end = time.monotonic() + timeout_sec
    while time.monotonic() < end:
        with node._lock:
            snap = node._get_last.get("/get_dual_arm")
            if snap is not None:
                return list(snap)
        time.sleep(0.02)
    return None


def wait_for_all_gets(node: RobotControlTopicTester, timeout_sec: float) -> bool:
    end = time.monotonic() + timeout_sec
    while time.monotonic() < end:
        with node._lock:
            if all(t in node._get_lengths for t in GET_TOPICS):
                return True
        time.sleep(0.02)
    return False


def build_set_payloads(dual: List[float]) -> Dict[str, List[float]]:
    """由 16 维 get_dual_arm 拼出各 set_ 所需向量。"""
    if len(dual) != 16:
        raise ValueError(f"get_dual_arm 期望 16 维，得到 {len(dual)}")

    lj = dual[0:7]
    lg = dual[7]
    rj = dual[8:15]
    rg = dual[15]

    return {
        "set_dual_arm": list(dual),
        "set_left_arm": lj + [lg],
        "set_right_arm": rj + [rg],
        "set_left_joint": lj,
        "set_right_joint": rj,
        "set_left_gripper": [lg],
        "set_right_gripper": [rg],
    }


def run_test(args: argparse.Namespace) -> int:
    rclpy.init()
    node = RobotControlTopicTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # 1) 等待至少一帧 get_dual_arm（用于安全回显）
        snap = wait_for_dual_arm(node, args.wait_dual_sec)
        if snap is None:
            node.get_logger().error(
                f"{args.wait_dual_sec}s 内未收到 /get_dual_arm。"
                "请确认 robot_control 已启动且发布正常。"
            )
            return 2

        payloads = build_set_payloads(snap)
        node.get_logger().info(
            f"已捕获 /get_dual_arm 一帧，将用相同数据测试各 set_ 话题。"
        )
        # snap 是「首帧」双机数据；后面等待期间话题仍在更新，故「各话题最近一次」可能与 snap 不完全同一时刻

        # 2) 校验所有 get_* 维度（再等一会凑齐各话题）
        if not wait_for_all_gets(node, args.wait_all_get_sec):
            node.get_logger().warn(
                f"{args.wait_all_get_sec}s 内未收齐全部 get_* 消息，"
                "将仅报告已收到的部分。"
            )

        if not args.no_print_gets:
            print("\n--- 用于 set 回显的 /get_dual_arm 首帧 (snap) ---")
            print(" ", snap)

        print("\n--- GET 话题维度检查 ---")
        with node._lock:
            last_copy = dict(node._get_last)
            for topic, expected in GET_TOPICS.items():
                got = node._get_lengths.get(topic)
                if got is None:
                    print(f"  {topic}: 未收到")
                elif got == expected:
                    print(f"  {topic}: OK (len={got})")
                else:
                    print(f"  {topic}: 期望 len={expected}, 实际 len={got}")

        if not args.no_print_gets:
            print("\n--- GET 话题 data 内容（各话题最近一次消息） ---")
            for topic in GET_TOPICS:
                row = last_copy.get(topic)
                if row is None:
                    print(f"  {topic}: (无数据)")
                else:
                    preview = ", ".join(f"{x:.6g}" for x in row)
                    print(f"  {topic}: [{preview}]")

        if args.gets_only:
            node.get_logger().info("已指定 --gets-only，跳过 set_ 发布。")
            return 0

        # 3) 发布所有 set_（名称与 C++ 一致：带前导 /）
        pubs = {}
        for name, _ in SET_TOPICS_META:
            t = "/" + name if not name.startswith("/") else name
            pubs[name] = node.create_publisher(ArmData, t, _qos())
        # 给 DDS 发现时间
        time.sleep(args.publisher_warmup_sec)

        print("\n--- SET 话题发布（回显当前姿态） ---")
        # payloads["set_left_joint"][-1] = payloads["set_left_joint"][-1]-0.1 # 修改一点姿态
        for name, dim in SET_TOPICS_META:
            msg = ArmData()
            msg.data = payloads[name]
            assert len(msg.data) == dim, (name, len(msg.data), dim)
            pubs[name].publish(msg)
            print(f"  已发布 /{name} len={dim}")
            time.sleep(args.set_interval_sec)

        node.get_logger().info("全部 set_ 测试消息已发送。")
        return 0
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gets-only",
        action="store_true",
        help="只订阅并检查 get_* 维度，不向 set_* 发布（最安全）。",
    )
    p.add_argument(
        "--no-print-gets",
        action="store_true",
        help="不打印各 get_* 的 data（默认会打印最近一次消息）。",
    )
    p.add_argument(
        "--wait-dual-sec",
        type=float,
        default=5.0,
        help="等待首帧 /get_dual_arm 的超时（秒）。",
    )
    p.add_argument(
        "--wait-all-get-sec",
        type=float,
        default=3.0,
        help="收齐全部 get_* 的额外等待时间（秒）。",
    )
    p.add_argument(
        "--set-interval-sec",
        type=float,
        default=0.08,
        help="相邻 set_* 发布间隔（秒），减轻总线负载。",
    )
    p.add_argument(
        "--publisher-warmup-sec",
        type=float,
        default=0.5,
        help="创建 Publisher 后等待时间，便于 DDS 匹配。",
    )
    args = p.parse_args()
    raise SystemExit(run_test(args))


if __name__ == "__main__":
    main()
