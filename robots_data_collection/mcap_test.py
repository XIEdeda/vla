#!/usr/bin/env python3
"""
常驻订阅相机 + 机械臂 topic；收到 /startend_data 后用 rosbag2_py 开写、停写（无 ros2 bag 子进程）。
"""
from __future__ import annotations

import argparse
import logging
import select
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import rosbag2_py
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64MultiArray

try:
    from sc_ros2.msg import ArmData
except ImportError:
    from sc_ros.msg import ArmData  # type: ignore


ROS2_CAMERA_TOPIC_MAP = {
    "head_image": "/camera3/camera_head/color/image_raw/compressed",
    "right_wrist_image": "/camera2/camera_right/color/image_rect_raw/compressed",
    "left_wrist_image": "/camera1/camera_left/color/image_rect_raw/compressed",
}

# 与 ROS2_CAMERA_TOPIC_MAP 一致即可：True=上面填 .../compressed 等 CompressedImage 话题；
# False=改为 Raw 路径并订阅 Image，例如 image_raw / image_rect_raw（无 /compressed）
CAMERA_MSG_IS_COMPRESSED: bool = True

ROS2_START_TOPIC = "/startend_data"

ROS2_ARM_TOPICS = ("/left_data", "/right_data")

logger = logging.getLogger(__name__)


def all_record_topics() -> List[str]:
    return list(ROS2_CAMERA_TOPIC_MAP.values()) + list(ROS2_ARM_TOPICS)


def next_episode_index(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    best = -1
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("episode_"):
            try:
                n = int(p.name.split("_", 1)[1])
                best = max(best, n)
            except ValueError:
                continue
    return best + 1


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec * 1_000_000_000 + stamp.nanosec)


def _qos_from_publishers(node: Node, topic: str) -> Optional[QoSProfile]:
    """若已有发布者，返回探测到的 QoSProfile（原始对象，仅用于再克隆）。"""
    try:
        infos = node.get_publishers_info_by_topic(topic)
    except Exception as e:
        logger.debug("get_publishers_info_by_topic(%s): %s", topic, e)
        return None
    if not infos:
        return None
    try:
        return infos[0].qos_profile
    except Exception:
        return None


def _clone_qos_for_subscription(raw: QoSProfile, *, min_depth: int) -> QoSProfile:
    """
    发布端探测到的 QoS 在 Python/RCL 里常出现 depth=0、history 枚举异常，
    直接 copy 或原样传入 create_subscription 会触发 Unknown QoS history / invalid allocator。
    这里只取可靠性与历史语义，用「新的」QoSProfile 建订阅。
    """
    hist = raw.history
    if hist not in (QoSHistoryPolicy.KEEP_LAST, QoSHistoryPolicy.KEEP_ALL):
        hist = QoSHistoryPolicy.KEEP_LAST

    depth_val = int(getattr(raw, "depth", 0) or 0)
    if hist == QoSHistoryPolicy.KEEP_LAST:
        depth_use = max(1, min_depth, depth_val)
    else:
        # KEEP_ALL：仍给 depth 下界，避免部分 RMW 对 0 敏感
        depth_use = max(1, min_depth, depth_val)

    rel = raw.reliability
    if rel not in (
        QoSReliabilityPolicy.RELIABLE,
        QoSReliabilityPolicy.BEST_EFFORT,
    ):
        rel = QoSReliabilityPolicy.RELIABLE

    dur = raw.durability
    if dur not in (
        QoSDurabilityPolicy.VOLATILE,
        QoSDurabilityPolicy.TRANSIENT_LOCAL,
    ):
        dur = QoSDurabilityPolicy.VOLATILE

    return QoSProfile(
        history=hist,
        depth=depth_use,
        reliability=rel,
        durability=dur,
    )


class PersistentRosbagNode(Node):
    """常驻订阅数据 topic；startend=1 打开 writer 落盘，=0 关闭 writer。"""

    def __init__(
        self,
        *,
        storage_id: str,
        arm_msg_type: str,
        image_qos_depth: int,
        arm_qos_depth: int,
        cam_qos_best_effort_fallback: bool,
        match_publisher_qos: bool,
        qos_setup_timeout_sec: float,
        qos_setup_poll_sec: float,
    ) -> None:
        super().__init__("mcap_episode_recorder_persistent")
        self._storage_id = storage_id
        self._arm_msg_type = arm_msg_type
        self._match_publisher_qos = match_publisher_qos
        self._qos_setup_deadline = time.monotonic() + qos_setup_timeout_sec
        self._qos_setup_poll_sec = qos_setup_poll_sec

        self._state_lock = threading.Lock()
        self.current_start_state = 2

        self._bag_lock = threading.Lock()
        self._recording = False
        self._writer: Optional[rosbag2_py.SequentialWriter] = None

        cam_rel = (
            QoSReliabilityPolicy.BEST_EFFORT
            if cam_qos_best_effort_fallback
            else QoSReliabilityPolicy.RELIABLE
        )
        self._cam_fallback = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=max(1, image_qos_depth),
            reliability=cam_rel,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._arm_fallback = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=max(10, arm_qos_depth),
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._image_qos_depth_cfg = max(1, image_qos_depth)
        self._arm_qos_depth_cfg = max(10, arm_qos_depth)

        self._subs_install_failed = False

        qos_cmd = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            ArmData, #Float64MultiArray
            ROS2_START_TOPIC,
            self._start_cb_float64,
            qos_cmd,
        )

        self._topic_metas: List[rosbag2_py.TopicMetadata] = []
        self._data_subs_ready = False
        self._setup_timer = self.create_timer(
            self._qos_setup_poll_sec, self._try_install_data_subscriptions
        )

        logger.info(
            "已订阅 %s；数据话题将%s安装（match_publisher_qos=%s，相机=%s）",
            ROS2_START_TOPIC,
            "按发布端 QoS 尝试匹配后" if match_publisher_qos else "用兜底 QoS",
            match_publisher_qos,
            "CompressedImage" if CAMERA_MSG_IS_COMPRESSED else "Image",
        )

    def _try_install_data_subscriptions(self) -> None:
        if self._data_subs_ready or self._subs_install_failed:
            return

        topics = list(ROS2_CAMERA_TOPIC_MAP.values()) + list(ROS2_ARM_TOPICS)
        qos_by_topic: Dict[str, QoSProfile] = {}
        missing: List[str] = []

        for t in topics:
            qp: Optional[QoSProfile] = None
            if self._match_publisher_qos:
                qp = _qos_from_publishers(self, t)
            if qp is None:
                missing.append(t)
            else:
                qos_by_topic[t] = qp

        timed_out = time.monotonic() >= self._qos_setup_deadline
        if self._match_publisher_qos and missing and not timed_out:
            return

        cam_topics = set(ROS2_CAMERA_TOPIC_MAP.values())
        for t in missing:
            if t in cam_topics:
                qos_by_topic[t] = self._cam_fallback
            else:
                qos_by_topic[t] = self._arm_fallback
            logger.warning(
                "话题 %s 使用兜底 QoS（%s）",
                t,
                "超时或未探测到发布者" if self._match_publisher_qos else "未启用匹配",
            )

        try:
            for t in topics:
                base = qos_by_topic[t]
                min_d = (
                    self._image_qos_depth_cfg
                    if t in cam_topics
                    else self._arm_qos_depth_cfg
                )
                qos = _clone_qos_for_subscription(base, min_depth=min_d)
                g = MutuallyExclusiveCallbackGroup()
                logger.info(
                    "订阅 %s  QoS history=%s reliability=%s depth=%s（独占回调组）",
                    t,
                    qos.history,
                    qos.reliability,
                    qos.depth,
                )

                if t in cam_topics:
                    if CAMERA_MSG_IS_COMPRESSED:
                        self.create_subscription(
                            CompressedImage,
                            t,
                            lambda msg, tn=t: self._on_compressed(tn, msg),
                            qos,
                            callback_group=g,
                        )
                    else:
                        self.create_subscription(
                            Image,
                            t,
                            lambda msg, tn=t: self._on_image(tn, msg),
                            qos,
                            callback_group=g,
                        )
                else:
                    self.create_subscription(
                        ArmData,
                        t,
                        lambda msg, tn=t: self._on_arm(tn, msg),
                        qos,
                        callback_group=g,
                    )
        except Exception as e:
            logger.error(
                "创建数据话题订阅失败（请试 --no-match-publisher-qos 或检查 DDS）: %s",
                e,
                exc_info=True,
            )
            self._subs_install_failed = True
            try:
                self._setup_timer.cancel()
            except Exception:
                pass
            return

        self._topic_metas.clear()
        cam_type = (
            "sensor_msgs/msg/CompressedImage"
            if CAMERA_MSG_IS_COMPRESSED
            else "sensor_msgs/msg/Image"
        )
        for topic in ROS2_CAMERA_TOPIC_MAP.values():
            self._topic_metas.append(
                rosbag2_py.TopicMetadata(
                    name=topic,
                    type=cam_type,
                    serialization_format="cdr",
                    offered_qos_profiles="",
                )
            )
        for topic in ROS2_ARM_TOPICS:
            self._topic_metas.append(
                rosbag2_py.TopicMetadata(
                    name=topic,
                    type=self._arm_msg_type,
                    serialization_format="cdr",
                    offered_qos_profiles="",
                )
            )

        self._data_subs_ready = True
        try:
            self._setup_timer.cancel()
        except Exception:
            pass
        logger.info("数据话题订阅已就绪（每路 topic 独立 MutuallyExclusiveCallbackGroup）")

    def subscriptions_install_failed(self) -> bool:
        return self._subs_install_failed

    def get_start_state(self) -> int:
        with self._state_lock:
            return self.current_start_state

    def is_recording(self) -> bool:
        with self._bag_lock:
            return self._recording and self._writer is not None

    def data_subscriptions_ready(self) -> bool:
        return self._data_subs_ready

    def _start_cb_float64(self, msg: Float64MultiArray) -> None:
        if not msg.data:
            logger.warning("收到空的 start 消息，忽略")
            return
        new_state = msg.data[0]
        if new_state not in [0, 1, 2]:
            logger.warning("无效 start 值 %s，仅支持 0/1/2", new_state)
            return
        new_state_i = int(new_state)
        with self._state_lock:
            if new_state_i != self.current_start_state:
                logger.info("start 状态: %s → %s", self.current_start_state, new_state_i)
                self.current_start_state = new_state_i

    def _on_image(self, topic_name: str, msg: Image) -> None:
        self._write(topic_name, msg, _stamp_to_ns(msg.header.stamp))

    def _on_compressed(self, topic_name: str, msg: CompressedImage) -> None:
        self._write(topic_name, msg, _stamp_to_ns(msg.header.stamp))

    def _on_arm(self, topic_name: str, msg: ArmData) -> None:
        self._write(topic_name, msg, _stamp_to_ns(msg.header.stamp))

    def _write(self, topic_name: str, msg, timestamp_ns: int) -> None:
        if not self._recording:
            return
        try:
            blob = serialize_message(msg)
        except Exception as e:
            logger.error("序列化失败 topic=%s: %s", topic_name, e, exc_info=True)
            return
        with self._bag_lock:
            if not self._recording or self._writer is None:
                return
            try:
                self._writer.write(topic_name, blob, timestamp_ns)
            except Exception as e:
                logger.error("写入 bag 失败 topic=%s: %s", topic_name, e, exc_info=True)

    def begin_episode(self, episode_dir: Path) -> bool:
        if not self._data_subs_ready:
            logger.error("数据订阅尚未就绪，无法开始录制")
            return False
        with self._bag_lock:
            if self._recording:
                logger.warning("已有打开的 writer，忽略重复 begin_episode")
                return False
            episode_dir = episode_dir.resolve()
            episode_dir.parent.mkdir(parents=True, exist_ok=True)
            if episode_dir.exists():
                shutil.rmtree(episode_dir)

            storage_options = rosbag2_py.StorageOptions(
                uri=str(episode_dir),
                storage_id=self._storage_id,
            )
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr",
                output_serialization_format="cdr",
            )
            try:
                writer = rosbag2_py.SequentialWriter()
                writer.open(storage_options, converter_options)
                for meta in self._topic_metas:
                    writer.create_topic(meta)
            except Exception as e:
                logger.error("打开 rosbag2 writer 失败: %s", e, exc_info=True)
                return False

            self._writer = writer
            self._recording = True
            logger.info("已开始写入 episode 目录: %s", episode_dir)
            return True

    def end_episode(self) -> None:
        writer_to_close: Optional[rosbag2_py.SequentialWriter] = None
        with self._bag_lock:
            self._recording = False
            writer_to_close = self._writer
            self._writer = None

        if writer_to_close is None:
            return
        try:
            if hasattr(writer_to_close, "close"):
                writer_to_close.close()
            del writer_to_close
        except Exception as e:
            logger.warning("关闭 writer 时: %s", e)
        logger.info("已结束本轮 bag 写入")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="常驻订阅 + rosbag2_py 分段录制（startend 控制）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data1/mcap_robot_datasets/20260415_subtask"),
        help="episode 子目录将创建在该路径下",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="mcap",
        choices=("sqlite3", "mcap"),
        help="rosbag2_py StorageOptions.storage_id（需本机已装对应插件）",
    )
    parser.add_argument(
        "--arm-msg-type",
        type=str,
        default="sc_ros2/msg/ArmData",
        help="与 ros2 topic info /left_data 的 Type 一致",
    )
    parser.add_argument(
        "--image-qos-depth",
        type=int,
        default=20,
        help="相机订阅最小队列深度；匹配发布端时会与发布 depth 取较大者",
    )
    parser.add_argument(
        "--cam-qos-best-effort",
        action="store_true",
        help="相机兜底 QoS 用 BEST_EFFORT（默认兜底 RELIABLE）；发布端可匹配时仍用发布端",
    )
    parser.add_argument(
        "--no-match-publisher-qos",
        action="store_true",
        help="不调用 get_publishers_info_by_topic，仅用兜底 QoS 建订阅",
    )
    parser.add_argument(
        "--qos-setup-timeout",
        type=float,
        default=15.0,
        help="等待各话题出现发布者以自动匹配 QoS 的最长时间（秒）",
    )
    parser.add_argument(
        "--qos-setup-poll",
        type=float,
        default=0.2,
        help="轮询发布者是否存在的周期（秒）",
    )
    parser.add_argument(
        "--arm-qos-depth",
        type=int,
        default=100,
        help="机械臂话题兜底 QoS 队列深度",
    )
    parser.add_argument(
        "--executor-threads",
        type=int,
        default=8,
        help="MultiThreadedExecutor 线程数",
    )
    args = parser.parse_args()

    if not rclpy.ok():
        rclpy.init(args=sys.argv)

    node = PersistentRosbagNode(
        storage_id=args.storage,
        arm_msg_type=args.arm_msg_type,
        image_qos_depth=args.image_qos_depth,
        arm_qos_depth=args.arm_qos_depth,
        cam_qos_best_effort_fallback=args.cam_qos_best_effort,
        match_publisher_qos=not args.no_match_publisher_qos,
        qos_setup_timeout_sec=args.qos_setup_timeout,
        qos_setup_poll_sec=args.qos_setup_poll,
    )
    n_exec = max(2, args.executor_threads)
    executor = MultiThreadedExecutor(num_threads=n_exec)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    t_wait = time.monotonic() + args.qos_setup_timeout + 2.0
    while (
        not node.data_subscriptions_ready()
        and not node.subscriptions_install_failed()
        and time.monotonic() < t_wait
    ):
        time.sleep(0.05)
    if node.subscriptions_install_failed():
        logger.error("数据话题订阅创建失败，请查看上方报错；可尝试 --no-match-publisher-qos")
    elif not node.data_subscriptions_ready():
        logger.error(
            "数据订阅仍未就绪（可调大 --qos-setup-timeout 或检查网络/发布节点）"
        )

    episode_id = next_episode_index(args.output_dir)
    last_start = node.get_start_state()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("===== rosbag2_py 常驻订阅录制 =====")
    print(f"输出根目录: {args.output_dir.resolve()}")
    print(f"  存储: {args.storage}（episode_XXXXXX/）")
    print(f"  发布端 QoS 匹配: {not args.no_match_publisher_qos}")
    print(f"  相机兜底 RELIABLE: {not args.cam_qos_best_effort} depth={args.image_qos_depth}")
    print(
        f"  相机消息类型: {'sensor_msgs/CompressedImage' if CAMERA_MSG_IS_COMPRESSED else 'sensor_msgs/Image'}（改文件顶 CAMERA_MSG_IS_COMPRESSED / ROS2_CAMERA_TOPIC_MAP）"
    )
    for t in all_record_topics():
        print(f"  录制: {t}")
    print(f"  控制: {ROS2_START_TOPIC} 发布 1=开始 / 0=停止")
    print(f"  ArmData 类型字符串: {args.arm_msg_type}")
    print(
        f"  性能: executor_threads={n_exec}；每 topic 独立 CallbackGroup 便于并行调度"
    )
    print("  键盘输入 q + Enter 退出")
    print("====================================")

    try:
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    break
                if line:
                    print("仅支持 q 退出")

            cur = node.get_start_state()
            if cur != last_start:
                if cur == 1:
                    if node.is_recording():
                        logger.warning("已在录制中，忽略新的「开始」")
                    else:
                        ep_dir = args.output_dir / f"episode_{episode_id:06d}"
                        logger.info("开始 episode %06d → %s", episode_id, ep_dir)
                        node.begin_episode(ep_dir)
                elif cur == 0:
                    if last_start == 1:
                        if node.is_recording():
                            logger.info("停止当前 episode 录制")
                            node.end_episode()
                        else:
                            logger.warning(
                                "收到停止(0) 时 writer 未开，仍推进 episode 编号"
                            )
                            node.end_episode()
                        episode_id += 1
                    else:
                        logger.warning("当前无进行中的录制（上次状态非 1），忽略「停止」")
                last_start = cur

            if not node.is_recording():
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        if node.is_recording():
            logger.info("退出前结束 bag 写入")
            node.end_episode()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("已退出")


if __name__ == "__main__":
    main()
