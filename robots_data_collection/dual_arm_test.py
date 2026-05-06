import os
import cv2
import json
import datetime
import numpy as np
import time
import pandas as pd
import sys
import select
import logging  
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import collections
import concurrent.futures

from threading import Thread, Lock
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Literal, Dict, Optional, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sc_ros2.msg import ArmData
# from topic_pub.msg import ArmData
# ROS2相关依赖
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor  # 多线程执行器
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

frequency: int = 30 #1秒30帧
# ------------------------------
# 1. 配置参数（相机ros2话题映射）
# ------------------------------
ROS2_CAMERA_TOPIC_MAP = {
    "head_image": "/camera3/camera_head/color/image_raw",  # 头部相机话题
    # "head_image": "/camera3/camera_head/color/image_rect_raw",  # 头部相机话题
    "right_wrist_image": "/camera2/camera_right/color/image_rect_raw",  # 右臂腕部相机话题
    "left_wrist_image": "/camera1/camera_left/color/image_rect_raw"  # 左臂腕部相机话题
}

ROS2_START_TOPIC = "/startend_data"

ROBOT_CONFIGS = {
    "Keenon_F1_DoubleArm": {  # 机器人类型名改为双臂
        "joint_names": {
            "left_arm": ['l_1', 'l_2', 'l_3', 'l_4', 'l_5', 'l_6', 'l_7'],  # 左臂7关节名
            "right_arm": ['r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7'],  # 右臂7关节名
        },
        "gripper": {
            "left_gripper": {"0": "closed", "1": "open"},  # 左臂夹爪状态
            "right_gripper": {"0": "closed", "1": "open"},  # 右臂夹爪状态
        },
        "cameras": ["head_image", "right_wrist_image","left_wrist_image"],  # 相机配置
        "state_fields": ["left_arm_joints", "right_arm_joints", "left_gripper", "right_gripper"],  # 双臂state字段
        "action_fields": ["left_arm_joints", "right_arm_joints", "left_gripper", "right_gripper"]   # 双臂action字段
    }
}

# 相机参数
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": frequency,
    "color_format": "bgr8",  # OpenCV默认格式
    "frame_timeout_ms": 1000  # 相机取帧超时时间
}

# 时间配置 
SYNC_CONFIG = {
    "arm_data_buffer_size": 5000,
    "allow_older_data": True,
    "interpolation_enabled": True,
    "max_time_diff_ms": 100
}

# 【双臂配置】
DOUBLE_ARM_STATE_LEN = 32  # 7左关节 + 7右关节 + 8左夹爪 + 8右夹爪
DOUBLE_ARM_TOTAL_LEN = 32  # 机械臂发布的state总长度
DOUBLE_ARM_ACTION_LEN = 32  # action长度与state一致
# DOUBLE_ARM_STATE_SPLIT = {
#     "left_arm_joints": (0, 7),    # 左臂7关节（索引0-6,共7个）
#     "left_gripper": (7, 15),      # 左夹爪（索引7-14,共8个）→ 前15位：0-14
#     "right_arm_joints": (15, 22), # 右臂7关节（索引15-21,共7个）
#     "right_gripper": (22, 30)     # 右夹爪（索引22-29,共8个）→ 后15位：15-29
# }
DOUBLE_ARM_STATE_SPLIT = {
    "left_arm_joints": (0, 7),    # 左臂7关节（索引0-6，共7个）
    "right_arm_joints": (7, 14),  # 右臂7关节（索引7-13，共7个）
    "left_gripper": (14, 23),     # 左臂夹爪
    "right_gripper": (23, 32)     # 右臂夹爪
}
DOUBLE_ARM_ACTION_SPLIT = DOUBLE_ARM_STATE_SPLIT  # action分割与state一致

# 分块配置
CHUNK_SIZE = 10000

# 初始化日志
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# 根据episode_id计算对应的chunk索引和名称（核心工具函数）
def get_chunk_info(episode_id: int) -> Tuple[int, str]:
    """
    根据episode_id计算chunk索引和chunk名称
    episode_id=99 → chunk_index=0 → chunk_name=chunk-000
    episode_id=100 → chunk_index=1 → chunk_name=chunk-001
    """
    chunk_index = episode_id // CHUNK_SIZE  # 整数除法,得到chunk索引
    chunk_name = f"chunk-{chunk_index:03d}"  # 格式化为3位数字（如chunk-001）
    return chunk_index, chunk_name

# ------------------------------
# 2. 机械臂数据缓存管理器（优化：拆分双臂锁）
# ------------------------------
class DoubleArmDataCache:
    def __init__(self, max_size=5000, data_timeout_sec=None):
        self.cache = {
            "left_arm": collections.deque(maxlen=max_size),
            "right_arm": collections.deque(maxlen=max_size)
        }
        self.max_size = max_size
        self.data_timeout_sec = data_timeout_sec
        # 新增：超时时间转纳秒
        self.data_timeout_ns = data_timeout_sec * 10**9 if data_timeout_sec else None
        self.last_used_idx = {
            "left_arm": -1,
            "right_arm": -1
        }
        self.locks = {
            "left_arm": Lock(),
            "right_arm": Lock()
        }

    def _is_data_expired(self, timestamp):
        """检查数据是否已过期适配19位纳秒级整数"""
        if self.data_timeout_sec is None:
            return False
        import time
        # 当前时间转19位纳秒级整数
        current_ns = int(time.time() * 10**9)
        # 纳秒级整数直接比较
        return (current_ns - timestamp) > self.data_timeout_ns

    def add_data(self, arm_type: Literal["left_arm", "right_arm"], timestamp: int, state: Dict):
        """添加单臂数据(线程安全),并清理过期数据。"""
        with self.locks[arm_type]:
            # 清理过期数据
            if self.data_timeout_sec is not None:
                while self.cache[arm_type] and self._is_data_expired(self.cache[arm_type][0][0]):
                    self.cache[arm_type].popleft()
            
            # deque会自动处理超出maxlen的情况
            self.cache[arm_type].append((timestamp, state))

    def get_matched_data(self, arm_type: Literal["left_arm", "right_arm"], target_timestamp: int) -> Tuple[Dict, int, float]:
        """
        获取与目标时间戳最匹配的单臂数据(线程安全)。
        返回: (状态, 机械臂时间戳, 时间差(ms))
        """
        with self.locks[arm_type]:
            arm_cache_deque = self.cache[arm_type]
            if not arm_cache_deque:
                return None, None, None # 返回None表示缓存为空

            # 转换为列表进行处理,并过滤过期数据
            arm_cache = list(arm_cache_deque)
            if self.data_timeout_sec is not None:
                arm_cache = [item for item in arm_cache if not self._is_data_expired(item[0])]
                if not arm_cache:
                    return None, None, None # 返回None表示所有数据都已过期

            time_diffs_ns = [abs(ts - target_timestamp) for ts, _ in arm_cache]  # 纳秒级时间差（整数）
            min_diff_idx = int(np.argmin(time_diffs_ns))
            
            if not SYNC_CONFIG["allow_older_data"] and min_diff_idx < self.last_used_idx[arm_type]:
                candidates = arm_cache[self.last_used_idx[arm_type]:]
                if candidates:
                    candidate_diffs = [abs(ts - target_timestamp) for ts, _ in candidates]
                    min_diff_idx = self.last_used_idx[arm_type] + int(np.argmin(candidate_diffs))
                else:
                    min_diff_idx = len(arm_cache) - 1
            
            self.last_used_idx[arm_type] = min_diff_idx
            ts, state = arm_cache[min_diff_idx]
            # time_diff = abs(ts - target_timestamp) * 1000 # 转为毫秒
            # 纳秒转毫秒（整数÷1e6,保留2位小数）
            time_diff_ms = round(time_diffs_ns[min_diff_idx] / 1e6, 2)
            
            return state, ts, time_diff_ms

    def reset(self):
        """重置双臂缓存（分别锁）"""
        for arm_type in self.cache.keys():
            with self.locks[arm_type]:
                self.cache[arm_type].clear()
                self.last_used_idx[arm_type] = -1

    def get_all_states(self, arm_type: Literal["left_arm", "right_arm"]) -> List[Dict]:
        """获取单臂所有历史状态"""
        with self.locks[arm_type]:
            return [item[1] for item in self.cache[arm_type]]

# ------------------------------
# 3. 机械臂ROS2接口
# ------------------------------
class ROS2RobotInterface(Node):
   
    def __init__(self, robot_type: str = "Keenon_F1_DoubleArm"):
        super().__init__("double_arm_data_recorder")
        self.robot_cfg = ROBOT_CONFIGS[robot_type] 
        
        # 使用优化后的缓存管理器,并设置超时
        self.arm_cache = DoubleArmDataCache(
            max_size=SYNC_CONFIG["arm_data_buffer_size"],
            data_timeout_sec=0.4 
        )

        # 用于心跳检测的变量（双臂独立）
        self.last_arm_data_timestamp = {
            "left_arm": None,
            "right_arm": None
        }
        self.arm_data_timeout_sec = 2 
        
        self.arm_data_paths = {}
        # 新增：控制是否允许写入txt文件（核心开关）
        self.enable_arm_data_write = False
        # 新增：文件句柄字典（保持文件打开，避免频繁开关）
        self.arm_data_file_handles = {
            "left_arm": None,
            "right_arm": None
        }
        # 新增：文件写入锁（每个臂独立锁，保护文件写入操作）
        self.arm_file_locks = {
            "left_arm": Lock(),
            "right_arm": Lock()
        }  
        
        # 订阅双臂数据话题（保持订阅，不销毁）
        self.left_sub = self.create_subscription(
            ArmData, "left_data", 
            lambda msg: self._arm_callback(msg, "left_arm"), 
            10
        )
        self.right_sub = self.create_subscription(
            ArmData, "right_data", 
            lambda msg: self._arm_callback(msg, "right_arm"), 
            10
        )
        
        self.cb_count = {
            "left_arm": 0,
            "right_arm": 0
        }
        self.cb_first_time = {
            "left_arm": None,
            "right_arm": None
        }
        self.cb_last_time = {
            "left_arm": None,
            "right_arm": None
        }
        
        logger.info("已订阅双臂话题:left_data(左臂)、right_data(右臂)")

    def _arm_callback(self, msg: ArmData, arm_type: Literal["left_arm", "right_arm"]):
        """单臂数据回调(验证长度,解析关节+夹爪)"""
        
        now = time.time()

        self.cb_count[arm_type] += 1
        self.cb_first_time[arm_type] = self.cb_first_time[arm_type] or now
        self.cb_last_time[arm_type] = now
        
        if len(msg.data) != 16:
            logger.warning(f"{arm_type}数据长度错误:实际{len(msg.data)},预期{16}(7关节+9夹爪)")
            return
        
        # 转换ROS时间戳为19位纳秒级整数
        arm_stamp = msg.header.stamp
        timestamp = arm_stamp.sec * 10**9 + arm_stamp.nanosec
        
        # 更新最后一次收到该臂数据的时间戳
        self.last_arm_data_timestamp[arm_type] = timestamp
        
        # 解析数据(关节+夹爪)
        if arm_type == "left_arm":
            state = {
                "left_arm_joints": msg.data[0:7],
                "left_gripper": msg.data[7:16]
            }
        else:
            state = {
                "right_arm_joints": msg.data[0:7],
                "right_gripper": msg.data[7:16]
            }
        
        # 【关键修改】只有开启写入开关时，才写入txt文件（使用保持打开的文件句柄）
        if self.enable_arm_data_write:
            file_handle = self.arm_data_file_handles.get(arm_type)
            if file_handle is not None:
                data_str = f"{timestamp} {','.join(map(str, msg.data))}\n"
                # 使用锁保护文件写入（线程安全）
                with self.arm_file_locks[arm_type]:
                    try:
                        file_handle.write(data_str)
                        # 使用行缓冲时，每写一行会自动刷新，无需手动flush（减少系统调用）
                    except Exception as e:
                        logger.error(f"{arm_type} 文件写入失败: {str(e)}", exc_info=True)
        
        # 加入缓存(线程安全)——即使停止写入，缓存仍会更新（不影响后续同步逻辑）
        self.arm_cache.add_data(arm_type, timestamp, state)

    def dump_cb_stats(self):
        for arm in ["left_arm", "right_arm"]:
            cnt = self.cb_count[arm]
            if cnt == 0:
                logger.warning(f"[CB] {arm}: 0 callbacks")
                continue

            dur = self.cb_last_time[arm] - self.cb_first_time[arm]
            hz = cnt / dur if dur > 0 else 0

            logger.warning(
                f"[CB] {arm}: callbacks={cnt}, duration={dur:.2f}s, avg_hz={hz:.1f}"
            )

    # 【修改后】暂停txt写入并清空缓存（不停止订阅）
    def pause_arm_data_write(self):
        """清空缓存 + 停止写入txt文件（订阅器继续接收数据，但不写入）"""
        # 1. 关闭写入开关
        self.enable_arm_data_write = False
        
        # 2. 关闭所有文件句柄（确保数据完全写入）
        for arm_type in ["left_arm", "right_arm"]:
            file_handle = self.arm_data_file_handles.get(arm_type)
            if file_handle is not None:
                with self.arm_file_locks[arm_type]:
                    try:
                        file_handle.flush()  # 最后刷新一次
                        file_handle.close()
                    except Exception as e:
                        logger.error(f"{arm_type} 关闭文件句柄失败: {str(e)}", exc_info=True)
                    finally:
                        self.arm_data_file_handles[arm_type] = None
        
        # 3. 清空txt文件路径（双重保险）
        self.arm_data_paths.clear()
        # 4. 清空缓存（避免上一轮数据残留）
        self.reset_cache()
        logger.info("已暂停机械臂数据写入txt文件，并清空缓存和文件句柄")

    # 【新增】恢复txt写入（供后续重新开始episode时调用）
    def resume_arm_data_write(self, arm_data_paths: Dict):
        """恢复txt写入（需要传入新的文件路径），并打开文件句柄"""
        # 1. 先关闭可能存在的旧文件句柄（安全措施）
        for arm_type in ["left_arm", "right_arm"]:
            old_handle = self.arm_data_file_handles.get(arm_type)
            if old_handle is not None:
                with self.arm_file_locks[arm_type]:
                    try:
                        old_handle.close()
                    except Exception as e:
                        logger.warning(f"{arm_type} 关闭旧文件句柄失败: {str(e)}")
                    finally:
                        self.arm_data_file_handles[arm_type] = None
        
        # 2. 保存文件路径
        self.arm_data_paths = arm_data_paths  # 传入新的left_arm/right_arm文件路径
        
        # 3. 打开新的文件句柄（追加模式，保持打开状态）
        for arm_type, file_path in arm_data_paths.items():
            try:
                file_handle = open(file_path, "a", encoding="utf-8", buffering=1)  # 行缓冲，减少系统调用
                self.arm_data_file_handles[arm_type] = file_handle
                logger.debug(f"{arm_type} 文件句柄已打开: {file_path}")
            except Exception as e:
                logger.error(f"{arm_type} 打开文件句柄失败: {file_path}, 错误: {str(e)}", exc_info=True)
                self.arm_data_file_handles[arm_type] = None
        
        # 4. 开启写入开关
        self.enable_arm_data_write = True
        logger.info("已恢复机械臂数据写入txt文件，文件句柄已打开")

    # 新增：检查指定机械臂数据是否超时
    def check_arm_data_timeout(self, arm_type: Literal["left_arm", "right_arm"]) -> bool:
        """检查自上次收到指定机械臂数据以来是否已超时。"""
        last_timestamp = self.last_arm_data_timestamp[arm_type]
        if last_timestamp is None:
            return False # 从未收到过数据,不认为是超时
        
        # 修改：时间戳比较改为19位整数
        current_ns = int(time.time() * 10**9)  # 当前时间转19位纳秒级整数
        time_since_last_data_ns = current_ns - last_timestamp  # 纳秒级时间差
        time_since_last_data_sec = time_since_last_data_ns / 10**9  # 转秒数比较
        
        return time_since_last_data_sec > self.arm_data_timeout_sec

    # 新增：检查任意一个机械臂是否超时
    def check_any_arm_timeout(self) -> bool:
        """
        检查左或右机械臂是否超时。
        :return: 如果任一臂超时,返回 True;否则返回 False。
        """
        return self.check_arm_data_timeout("left_arm") or self.check_arm_data_timeout("right_arm")

    def get_synced_double_arm_data(self, target_timestamp: int) -> Tuple[Dict, float]:
        """
        获取双臂同步数据。
        返回: (合并后的状态, 最大时间差(ms))
        """
        # 分别获取双臂匹配数据
        left_state, left_ts, left_diff = self.arm_cache.get_matched_data("left_arm", target_timestamp)
        right_state, right_ts, right_diff = self.arm_cache.get_matched_data("right_arm", target_timestamp)
        
        if not left_state or not right_state:
            missing_arms = []
            if not left_state: missing_arms.append("左臂")
            if not right_state: missing_arms.append("右臂")
            raise RuntimeError(f"{', '.join(missing_arms)}数据缓存为空或已过期,无法同步")
        
        # 合并双臂状态
        merged_state = {**left_state, **right_state}
        max_time_diff = max(left_diff, right_diff)
        
        arm_timestamp = (left_ts + right_ts) / 2
        
        # 时间差超过阈值警告,先不打印
        # if max_time_diff > SYNC_CONFIG["max_time_diff_ms"]:
        #     logger.warning(f"双臂数据时间差超标:{max_time_diff:.2f}ms(阈值{SYNC_CONFIG['max_time_diff_ms']}ms)")
        
        return merged_state, arm_timestamp,max_time_diff

    def get_all_double_arm_states(self) -> Tuple[List[Dict], List[Dict]]:
        """获取双臂所有历史状态(用于生成动作)"""
        left_states = self.arm_cache.get_all_states("left_arm")
        right_states = self.arm_cache.get_all_states("right_arm")
        return left_states, right_states

    def stop_subscribers(self):
        """停止ROS2订阅器，不再接收机械臂数据"""
        if self.subscribers_active:
            # 销毁订阅器（ROS2中销毁后将不再触发回调）
            self.destroy_subscription(self.left_sub)
            self.destroy_subscription(self.right_sub)
            self.subscribers_active = False
            logger.info("已停止双臂数据订阅")
        
        # 可选：清空缓存，避免残留数据
        self.reset_cache()

    def reset_cache(self):
        """重置双臂缓存"""
        self.arm_cache.reset()
        # 同时重置心跳检测的时间戳
        self.last_arm_data_timestamp = {"left_arm": None, "right_arm": None}

    def get_clock(self):
        """暴露ROS2时钟"""
        return super().get_clock()
       
# ------------------------------
# 4. ROS2相机订阅管理器
# ------------------------------
class ROS2CameraSubscriber(Node):
    def __init__(self, camera_names: List[str]):
        super().__init__("camera_subscriber_node")
        self.camera_names = camera_names
        self.bridge = CvBridge()
        
        # 分相机缓存和锁
        self.frame_cache = {
            cam_name: {
                # "frame_queue": deque(maxlen=50),  # 50帧缓存
                "frame_queue": deque(),  
                "lock": threading.Lock(),
                "subscriber": None,
                "encoding_warned": False,
                "latest_valid_frame": None   
            } for cam_name in camera_names
        }
        self.last_saved_ts = {cam_name: 0 for cam_name in self.camera_names}
        
        self.global_lock = threading.Lock()
        self.frame_counters = {cam_name: 0 for cam_name in camera_names}
        self.image_dirs = {}
        self.running = False
        self.episode_index = 0
        self.io_executor = None
        self.io_task_max_workers = 6  # 线程池大小
        self.cpu_executor = None
        self.cpu_task_max_workers = 9  # 线程池大小
        
        # 新增：存储异步任务结果（可选，用于错误处理或状态查询）
        self.async_tasks = {}  # key: (cam_name, frame_idx), value: future

        self._validate_camera_topics()

    def _validate_camera_topics(self):
        invalid_cams = [cam for cam in self.camera_names if cam not in ROS2_CAMERA_TOPIC_MAP]
        if invalid_cams:
            raise RuntimeError(f"以下相机无对应ROS2话题配置:{invalid_cams}\n已配置话题的相机：{list(ROS2_CAMERA_TOPIC_MAP.keys())}")
        logger.info("所有相机ROS2话题配置验证通过")

    def _frame_saver_loop(self):
        while self.running:
            any_consumed = False
            for cam_name in self.camera_names:
                cam_cache = self.frame_cache[cam_name]
                # 队列非空则持续消费，空则退出
                while True:
                    with cam_cache["lock"]:
                        if not cam_cache["frame_queue"]:
                            break
                        # 严格从队头出队，一帧只消费一次
                        frame = cam_cache["frame_queue"].popleft()
                    # 处理出队帧
                    self._submit_frame(cam_name, frame)
                    any_consumed = True
            # 无新帧时短睡眠，减少CPU空转
            if not any_consumed:
                time.sleep(0.002)

    def start(self, output_dir: Path):
        with self.global_lock:
            if self.running:
                logger.warning("相机订阅器已在运行，无需重复启动")
                return
            
            # 创建图片保存目录
            for cam_name in self.camera_names:
                self.image_dirs[cam_name] = output_dir / f"{cam_name}_images"
                self.image_dirs[cam_name].mkdir(parents=True, exist_ok=True)
                self.frame_counters[cam_name] = 0
            
            # CPU线程池（用于编码）
            self.cpu_executor = ThreadPoolExecutor(
                max_workers=self.cpu_task_max_workers,  
                thread_name_prefix="CameraImageEncoder"
            )
            
            # 初始化IO线程池
            self.io_executor = ThreadPoolExecutor(
                max_workers=self.io_task_max_workers,
                thread_name_prefix="CameraImageSaver"
            )
            
            # 创建ROS2订阅者
            for cam_name in self.camera_names:
                topic = ROS2_CAMERA_TOPIC_MAP[cam_name]
                qos_profile = QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=20,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.VOLATILE
                )

                self.frame_cache[cam_name]["subscriber"] = self.create_subscription(
                    msg_type=Image,
                    topic=topic,
                    callback=lambda msg, cam=cam_name: self._frame_callback(msg, cam),
                    qos_profile=qos_profile
                )
                logger.info(f"已启动 {cam_name} 订阅 → ROS2话题:{topic}（30帧缓存+异步编码保存）")

            self.running = True
            logger.info("所有相机订阅器启动完成，IO线程池已初始化")
            
            self._saver_thread = threading.Thread(target=self._frame_saver_loop, daemon=True)
            self._saver_thread.start()
            
    def _frame_callback(self, msg: Image, cam_name: str):
        cam_cache = self.frame_cache[cam_name]
        
        try:
            
            # 转换时间戳
            # ros2_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            # 修改后（19位纳秒级整数，和机械臂统一）
            # 1. 强制使用 ROS 系统时间戳（避免 RealSense 时间漂移）
            # now_stamp = self.get_clock().now().to_msg()
            # msg.header.stamp = now_stamp
            ros2_timestamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec  # 修改：19位整数
            
            # 编码转换
            if msg.encoding != "bgr8" and not cam_cache["encoding_warned"]:
                logger.warning(f"{cam_name} 图像编码为{msg.encoding}(预期bgr8)，自动转换为bgr8（仅提示一次）")
                cam_cache["encoding_warned"] = True
                cv_image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                cv_image = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
            else:
                if msg.encoding == "bgr8":
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                else:
                    cv_image_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    cv_image = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
            
            # 分辨率验证
            # expected_shape = (CAMERA_CONFIG["height"], CAMERA_CONFIG["width"], 3)
            # if cv_image.shape != expected_shape:
            #     logger.warning(f"{cam_name} 分辨率不匹配：实际{cv_image.shape[:2]},预期{expected_shape[:2]}，跳过此帧")
            #     return

            # 入队缓存
            with cam_cache["lock"]:
                frame = {
                    "cv_image": cv_image.copy(),
                    "timestamp": ros2_timestamp,
                    "is_valid": True
                }

                cam_cache["frame_queue"].append(frame)
                cam_cache["latest_valid_frame"] = frame
                
        except Exception as e:
            logger.error(f"{cam_name} 帧处理失败：{str(e)}", exc_info=True)
            # 无效帧入队
            current_ns = int(time.time() * 10**9)
            with cam_cache["lock"]:
                cam_cache["frame_queue"].append({
                    "cv_image": None,
                    "timestamp": ros2_timestamp if 'ros2_timestamp' in locals() else current_ns,
                    "is_valid": False
                })
        
    def _encode_frame(self, cam_name: str, frame_idx: int, cv_image: np.ndarray) -> Tuple[str, int, bool, Optional[bytes]]:
        """
        仅做 CPU 编码，返回字节流
        """
        try:
            ret, img_encoded = cv2.imencode('.jpg', cv_image)
            if not ret:
                logger.error(f"{cam_name} 帧{frame_idx}编码失败")
                return (cam_name, frame_idx, False, None)
            return (cam_name, frame_idx, True, img_encoded.tobytes())
        except Exception as e:
            logger.error(f"{cam_name} 帧{frame_idx}编码失败: {str(e)}", exc_info=True)
            return (cam_name, frame_idx, False, None)

    def _save_encoded_frame(self, cam_name: str, frame_idx: int, img_bytes: bytes, timestamp: int):
        """
        将已编码字节流写入硬盘
        """
        try:
            img_save_path = self.image_dirs[cam_name] / f"{frame_idx:04d}_{timestamp}.jpg"
            with open(img_save_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            logger.error(f"{cam_name} 帧{frame_idx}写盘失败: {str(e)}", exc_info=True)

    def get_frames(self) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        colors = {}
        timestamps = {}

        with self.global_lock:
            if not self.running:
                logger.warning("相机订阅器未启动，无法获取帧")
                return {}, {}

        for cam_name in self.camera_names:
            cam_cache = self.frame_cache[cam_name]
            with cam_cache["lock"]:
                frame = cam_cache["latest_valid_frame"]

            if frame is None or not frame["is_valid"]:
                logger.warning(f"{cam_name} 尚无有效帧，整组丢弃")
                return {}, {}

            colors[cam_name] = frame["cv_image"].copy()
            timestamps[cam_name] = frame["timestamp"]

        return colors, timestamps
    
    def _submit_frame(self, cam_name: str, frame: dict):
        if not frame["is_valid"]:
            return

        frame_idx = self.frame_counters[cam_name]
        timestamp = frame["timestamp"]
        cv_image = frame["cv_image"]

        # 提交编码任务
        future = self.cpu_executor.submit(
            self._encode_frame,
            cam_name,
            frame_idx,
            cv_image
        )

        # 编码完成后提交IO
        def callback(fut):
            _, _, success, img_bytes = fut.result()
            if success:
                self.io_executor.submit(
                    self._save_encoded_frame,
                    cam_name,
                    frame_idx,
                    img_bytes,
                    timestamp
                )
            if (cam_name, frame_idx) in self.async_tasks:
                del self.async_tasks[(cam_name, frame_idx)]

        future.add_done_callback(callback)
        self.async_tasks[(cam_name, frame_idx)] = future
        self.frame_counters[cam_name] += 1
        self.last_saved_ts[cam_name] = timestamp

    def stop(self, output_dirs: Dict[str, Path]):
        with self.global_lock:
            if not self.running:
                logger.warning("相机订阅器未运行，无需停止")
                return

            # 停止订阅器
            for cam_name in self.camera_names:
                subscriber = self.frame_cache[cam_name]["subscriber"]
                if subscriber is not None:
                    self.destroy_subscription(subscriber)
                    self.frame_cache[cam_name]["subscriber"] = None
            logger.info("所有相机ROS2订阅器已停止")
            
            # 等待所有异步图片编码/保存任务完成
            if hasattr(self, 'cpu_executor') and self.cpu_executor:
                logger.info("等待所有图片编码任务完成...")
                self.cpu_executor.shutdown(wait=True)
                self.cpu_executor = None
                logger.info("所有图片编码任务已完成")
            
            # 等待所有异步图片编码/保存任务完成
            if self.io_executor:
                logger.info("等待所有图片编码/保存任务完成...")
                self.io_executor.shutdown(wait=True)
                self.io_executor = None
                logger.info("所有图片编码/保存任务已完成")

            # 打印采集统计
            for cam_name in self.camera_names:
                logger.info(f"{cam_name} 本次采集共处理 {self.frame_counters[cam_name]} 张图片")

            #不合成视频了
            # 并行合成视频
            # if self.episode_index >= 0: # 确保有有效的episode索引
            #     logger.info("开始并行合成所有相机视频...")

            #     num_threads = len(self.camera_names)
                
            #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="VideoComposer") as executor:
            #         # 提交所有视频合成任务
            #         futures = []
            #         for cam_name in self.camera_names:
            #             video_filename = f"episode_{self.episode_index:06d}.mp4"
            #             video_save_dir = output_dirs[cam_name]
            #             video_save_dir.mkdir(parents=True, exist_ok=True)
            #             output_path = video_save_dir / video_filename
                        
            #             # 提交任务到线程池
            #             future = executor.submit(
            #                 self._images_to_video,
            #                 cam_name=cam_name,
            #                 img_dir=self.image_dirs[cam_name],
            #                 output_path=output_path,
            #                 fps=CAMERA_CONFIG["fps"]
            #             )
            #             futures.append((cam_name, future))

            #         # 等待所有任务完成并处理结果
            #         for cam_name, future in futures:
            #             try:
            #                 # 获取任务结果，这会阻塞直到任务完成
            #                 future.result()
            #                 logger.info(f"{cam_name} 视频合成成功: {output_path}")
            #             except Exception as e:
            #                 logger.error(f"{cam_name} 视频合成失败: {str(e)}", exc_info=True)
                
            #     logger.info("所有相机视频合成任务已全部完成。")

            self.running = False

    def _images_to_video(self, cam_name: str, img_dir: Path, output_path: Path, fps: int):
        img_paths = sorted(img_dir.glob("*.jpg"), key=lambda x: int(x.stem))
        if not img_paths:
            logger.warning(f"{cam_name} 对应目录 {img_dir} 下无jpg图片,跳过视频合成")
            return

        first_img = cv2.imread(str(img_paths[0]))
        if first_img is None:
            logger.error(f"{cam_name} 无法读取首帧图片 {img_paths[0]}，视频合成失败")
            return
        height, width = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for img_path in tqdm(img_paths, desc=f"合成{cam_name}视频"):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"{cam_name} 跳过损坏图片：{img_path}")
                continue
            video_writer.write(img)

        video_writer.release()
        logger.info(f"{cam_name} 视频合成完成：{output_path}（共{len(img_paths)}帧，帧率{fps}fps)")

    def reset_encoding_warning(self):
        for cam_name in self.camera_names:
            with self.frame_cache[cam_name]["lock"]:
                self.frame_cache[cam_name]["encoding_warned"] = False
        logger.info("所有相机编码警告标记已重置")

    def clear_frame_queues(self):
        for cam_name in self.camera_names:
            cam_cache = self.frame_cache[cam_name]
            with cam_cache["lock"]:
                cam_cache["frame_queue"].clear()
        logger.info("已清空所有相机的历史帧队列")


# ------------------------------
# 5. 数据集配置
# ------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True  
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: Optional[str] = None
    # 图像存储配置
    image_dtype: str = "uint8"
    image_format: str = "jpg"

# ------------------------------
# 6. 核心数据处理类
# ------------------------------
class DoubleArmDataProcessor(Node):
    def __init__(
        self,
        task_dir: str = "/home/user/mlt/PickandPlace_test",  
        robot_type: str = "Keenon_F1_DoubleArm",
        task_description: str = "Grab the cup and pour the popcorn onto the plate",
    ):
        super().__init__("double_arm_data_processor")
        self.task_dir = Path(task_dir)
        self.task_description = task_description
        # ===================== 新增：日志文件配置（当天即带日期后缀） =====================
        # 1. 定义日志保存目录（任务目录下的logs文件夹）
        log_dir = self.task_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)  # 自动创建logs目录

        # 2. 日志文件名：直接带当天日期（核心改动），格式如 data_collection_2025-12-10.log
        log_filename = f"data_collection_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
        log_file = log_dir / log_filename

        # 3. 清空原有logger处理器，避免重复输出
        logger.handlers.clear()

        # 4. 配置终端输出（保留原有终端打印，格式简洁）
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'  # 终端只显示时分秒，更简洁
        ))
        logger.addHandler(stream_handler)

        # 5. 配置文件输出（核心：当天日志直接写入带日期的文件，追加模式）
        file_handler = logging.FileHandler(
            filename=str(log_file),  # 带当天日期的日志文件
            mode='a',                # 追加模式（当天多次启动不会覆盖旧日志）
            encoding='utf-8'         # 避免中文乱码
        )
        # 文件日志格式（含完整时间、模块、行号，便于排查问题）
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

        # 6. 设置日志级别（保持原有INFO级别）
        logger.setLevel(logging.INFO)
        logger.info(f"日志已保存到：{log_file}")
        # ===================== 日志配置结束 =====================

        self.robot_type = robot_type
        self.frequency = frequency
        # 图像相关属性
        self.image_height = CAMERA_CONFIG["height"]
        self.image_width = CAMERA_CONFIG["width"]
        
        self.robot_cfg = ROBOT_CONFIGS[robot_type]
        self.camera_names = self.robot_cfg["cameras"]
        self.state_fields = self.robot_cfg["state_fields"]
        self.action_fields = self.robot_cfg["action_fields"]
        self.state_history = []  # 保存所有采集到的state（用于生成action）
        
        # --------------------------
        # 动态初始化chunk相关目录
        # --------------------------
        self.base_data_dir = self.task_dir / "data"  # 所有chunk的父目录
        self.base_videos_dir = self.task_dir / "videos"  # 所有chunk的父目录
        self.base_arm_data_dir = self.task_dir / "arm_data"  # arm_data根目录
        self.meta_dir = self.task_dir / "meta"
        self.camera_video_dirs = {}  # 动态生成（每个episode对应自己的chunk）
        self.counter_file = self.task_dir / "global_index_counter.json"  # 全局唯一计数器文件
        
        
        
        # start话题相关状态（线程安全）
        self.start_lock = Lock()  # 保护start状态的读写
        self.current_start_state = 2  # 当前start话题值（0/1/2）
        self.last_start_state = 2     # 上一次start话题值（用于检测变化）
        
        # 初始化组件
        self.camera_manager = ROS2CameraSubscriber(self.camera_names)
        time.sleep(1)  # 等待相机节点初始化稳定
        self.ros2_interface = ROS2RobotInterface(robot_type)
        self._init_start_subscriber()
        
        # 错误计数初始化
        self.camera_error_count = 0
        self.arm_error_count = 0
        self.ERROR_MAX = 5
        
        # --------------------------
        # 使用统一的多线程执行器管理所有节点
        # --------------------------
        try:
            from rclpy import executors as _rclpy_executors
            # 核心优化：指定线程数,避免默认单线程导致回调阻塞
            executor_obj = _rclpy_executors.MultiThreadedExecutor(num_threads=6)
        except Exception as e:
            logger.exception("创建 MultiThreadedExecutor 失败：%s", e)
            raise RuntimeError("无法创建 MultiThreadedExecutor") from e

        # 检查返回值
        if executor_obj is None:
            logger.error("MultiThreadedExecutor() 返回 None — 这通常意味着 rclpy 未正确初始化或环境异常")
            raise RuntimeError("MultiThreadedExecutor() returned None — ensure rclpy.init() was called")

        self.executor = executor_obj
        # logger.info("executor 创建成功：%r", self.executor)

        # 将节点加入 executor（加入之前再确认节点存在）
        for node_obj, name in [(self.camera_manager, "camera_manager"),
                            (self.ros2_interface, "ros2_interface"),
                            (self, "self_node")]:
            if node_obj is None:
                logger.error("尝试加入 executor 的节点 %s 为 None!", name)
                raise RuntimeError(f"节点 {name} 为 None,无法加入 executor")
            try:
                self.executor.add_node(node_obj)
                # logger.info("已把节点 %s 加入 executor", name)
            except Exception as e:
                logger.exception("向 executor 添加节点 %s 失败：%s", name, e)
                raise

        # 启动执行器线程（若执行器成功创建）
        self.executor_thread = Thread(target=self._spin_executor, daemon=True)
        self.executor_thread.start()
        # logger.info("executor 线程已启动")
        
        self.global_start_index = 0  # 默认起始值
        if self.counter_file.exists():
            # 1. 若计数器文件存在,直接读取
            try:
                with open(self.counter_file, "r", encoding="utf-8") as f:
                    counter_data = json.load(f)
                    last_max_index = counter_data.get("last_max_index", -1)
                    self.global_start_index = last_max_index + 1  # 下一个索引 = 最后一个索引 + 1
                logger.info(f"从计数器文件加载全局起始index: {self.global_start_index}")
            except Exception as e:
                logger.warning(f"计数器文件损坏（格式错误或内容异常）,将重新初始化。错误：{str(e)}")
                # 损坏时,先尝试从旧文件计算（兼容旧数据）
                max_index = self._compute_max_index_from_old_files()
                self.global_start_index = max_index + 1 if max_index != -1 else 0
                # 修复计数器文件
                self._update_counter_file(max_index)
        else:
            # 2. 若计数器文件不存在,首次运行：先检查是否有旧数据（兼容无计数器的历史数据）
            max_index = self._compute_max_index_from_old_files()  # 复用原遍历逻辑
            self.global_start_index = max_index + 1 if max_index != -1 else 0
            # 初始化计数器文件
            self._update_counter_file(max_index)
            logger.info(f"未找到计数器文件,已初始化(基于旧数据最大index):{self.global_start_index}")
        
        # --------------------------
        # 跨chunk获取最后一个episode_id
        # --------------------------
        last_episode_id, unfinished_episodes = self._get_last_episode_id()
        # 判断最大ID是否未完成,决定下一个ID
        if last_episode_id == -1:
            # 无历史数据,从0开始
            self.episode_id = 0
            logger.info(f"未检测到历史数据,从 Episode {self.episode_id:06d} 开始采集")
        else:
            if last_episode_id in unfinished_episodes:
                # 最大ID是未完成的,重新处理它
                self.episode_id = last_episode_id
                logger.info(f"检测到未完成的Episode ID:{last_episode_id:06d},将重新处理该Episode")
            else:
                # 最大ID已完成,从下一个ID开始
                self.episode_id = last_episode_id + 1
                logger.info(f"已检测到最大Episode ID:{last_episode_id:06d},从 Episode {self.episode_id:06d} 续采")

        self.item_id = self.global_start_index - 1
        self.episode_data = []  # 存储帧数据,无队列
        self.sync_stats = {"total_frames": 0, "total_diff_ms": 0, "max_diff_ms": 0}
        self.is_available = True
        self._init_task_dirs()

        # 新增：校验计数器与Parquet文件的一致性（防中断残留）
        if self.counter_file.exists() and self.base_data_dir.exists():
            try:
                # 读取计数器中的最后一个index
                with open(self.counter_file, "r", encoding="utf-8") as f:
                    counter_last = json.load(f)["last_max_index"]
                # 读取所有Parquet文件中的最大index
                parquet_last = -1
                for chunk_dir in self.base_data_dir.glob("chunk-???"):
                    for ep_file in chunk_dir.glob("episode_*.parquet"):
                        if os.path.getsize(ep_file) < 1024:
                            continue  # 跳过空文件
                        table = pq.read_table(ep_file, columns=["index"])
                        index_col = table["index"]
                        # 关键：合并分块数组（ChunkedArray → 连续数组）
                        if isinstance(index_col, pa.ChunkedArray):
                            index_col = index_col.combine_chunks()
                        file_max = pc.max(index_col).as_py()  # 现在可以安全调用max()

                        if file_max > parquet_last:
                            max_index = file_max
                # 若计数器中的index小于Parquet中的实际最大index,修正计数器
                if parquet_last > counter_last and parquet_last != -1:
                    self.global_start_index = parquet_last + 1
                    self._update_counter_file(parquet_last)
                    logger.warning(f"计数器与Parquet文件不一致:计数器last={counter_last},Parquet实际last={parquet_last},已修正计数器")
            except Exception as e:
                logger.warning(f"校验计数器一致性失败：{str(e)}")
        
        # 元数据初始化
        self.official_meta = self._init_official_meta()

        self._init_meta_files()

        # 全局数据缓存
        self.global_stats_cache = {
            "state": [], 
            "actions": [], 
            "timestamp": [], 
            "frame_index": [],
            "episode_index": [], 
            "index": [],
            "task_index": []
        }
        self._clean_invalid_episodes_in_meta(unfinished_episodes)
        
        logger.info("双臂数据处理器初始化完成")

    def _clean_invalid_episodes_in_meta(self, unfinished_episode_ids: set):

        episodes_path = self.meta_dir / "episodes.jsonl"
        if not episodes_path.exists():
            return  # 无文件则无需清理

        # 二次验证——检查未完成ID是否真的没有完整Parquet文件
        actual_unfinished = set()
        for ep_id in unfinished_episode_ids:
            # 计算该episode对应的chunk目录
            chunk_index, chunk_name = get_chunk_info(ep_id)
            parquet_file = self.base_data_dir / chunk_name / f"episode_{ep_id:06d}.parquet"
            # 判断：如果没有Parquet文件,或文件为空 → 真·未完成
            if not parquet_file.exists() or os.path.getsize(parquet_file) < 1024:
                actual_unfinished.add(ep_id)
            else:
                # 有完整Parquet文件 → 假·未完成（标记残留）,不加入清理列表
                logger.info(f"episode {ep_id:06d} 存在完整Parquet文件,忽略残留标记,不清理元数据")
        
        # 过滤episodes.jsonl,保留非未完成的条目（直接使用传递的集合）
        valid_episodes = []
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ep = json.loads(line.strip())
                    ep_id = ep["episode_index"]
                    if ep_id not in actual_unfinished:
                        valid_episodes.append(ep)
                    else:
                        logger.info(f"清理episodes.jsonl中未完成的条目: episode {ep_id}")
                except Exception as e:
                    logger.warning(f"跳过无效的episodes.jsonl条目: {line.strip()}, 错误: {e}")

        # 写回过滤后的有效条目
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in valid_episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        logger.info("episodes.jsonl未完成条目清理完成")

        
    def _compute_max_index_from_old_files(self) -> int:
        """仅在计数器文件不存在/损坏时调用（兼容旧数据）"""
        max_index = -1
        if self.base_data_dir.exists():
            # 按chunk索引排序,确保从旧到新遍历
            chunk_dirs = sorted(
                self.base_data_dir.glob("chunk-???"),
                key=lambda x: int(x.name.split("-")[-1])
            )
            for chunk_dir in chunk_dirs:
                # 遍历当前chunk下的所有episode文件
                for episode_file in chunk_dir.glob("episode_*.parquet"):
                    try:
                        # 跳过空文件
                        if os.path.getsize(episode_file) == 0:
                            continue
                        
                        # 读取"index"列（只加载需要的列,提高效率）
                        table = pq.read_table(episode_file, columns=["index"])
                        # 跳过空表（无数据行）
                        if table.num_rows == 0:
                            continue
                        
                        index_col = table["index"]
                        # 跳过全空列（无有效索引值）
                        if index_col.null_count == index_col.length:
                            logger.debug(f"文件 {episode_file.name} 的index列为全空,跳过")
                            continue
                        
                        # 关键修复：用pa.compute.max处理（支持ChunkedArray和普通Array）
                        # 无需手动合并分块,PyArrow会自动处理
                        file_max_idx = pc.max(index_col).as_py()
                        
                        # 更新全局最大index
                        if file_max_idx > max_index:
                            max_index = file_max_idx
                            logger.debug(f"更新最大index: {max_index}（来自文件 {episode_file.name})")
                        
                    except Exception as e:
                        logger.warning(f"读取旧文件 {episode_file.name} 失败,跳过：{str(e)}")
        return max_index


    def _update_counter_file(self, last_max_index: int):
        """更新计数器文件,记录当前最大index"""
        try:
            with open(self.counter_file, "w", encoding="utf-8") as f:
                json.dump({"last_max_index": last_max_index}, f, indent=2)
            logger.debug(f"计数器文件已更新,最新max_index: {last_max_index}")
        except Exception as e:
            logger.error(f"更新计数器文件失败（可能导致索引重复）：{str(e)}")

    # start话题订阅初始化
    def _init_start_subscriber(self):
        """初始化start话题订阅"""
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.start_subscriber = self.create_subscription(
            msg_type=ArmData,
            # msg_type=Float64MultiArray,
            topic=ROS2_START_TOPIC,
            callback=self._start_topic_callback,
            qos_profile=qos_profile
        )
        logger.info(f"已启动 start 话题订阅 → ROS2话题:{ROS2_START_TOPIC}(0=停止,1=开始)")

    def _start_topic_callback(self, msg: Float64MultiArray):
        """start话题回调:更新当前状态(0/1),线程安全"""
        with self.start_lock:
            # 简化逻辑：取数组第一个元素（默认控制信号仅一个值）
            if not msg.data:  # 防止空数组
                logger.warning("收到空的start话题数据,跳过")
                return
            new_state = msg.data[0]  # 提取第一个元素
            
            # 仅处理0/1/2
            if new_state not in [0, 1, 2]:
                logger.warning(f"收到无效start状态:{new_state},仅支持0.0(停止)、1.0(开始)、2.0(初始)")
                return
            
            # 更新状态
            if new_state != self.current_start_state:
                logger.info(f"start话题状态变化:{self.current_start_state} → {new_state}")
                self.current_start_state = new_state
    
    def _get_last_episode_id(self) -> Tuple[int, set]:

        max_episode_id = -1
        unfinished_episode_ids = set()  # 新增：收集所有未完成的episode_id

        # 步骤1：遍历所有chunk目录,收集最大ID和未完成ID
        if self.base_data_dir.exists():
            chunk_dirs = self.base_data_dir.glob("chunk-???")
            for chunk_dir in chunk_dirs:
                # 子步骤1.1：先处理未完成标记文件（_in_progress.txt）
                in_progress_files = list(chunk_dir.glob("episode_*_in_progress.txt"))
                for file in in_progress_files:
                    try:
                        # 解析文件名：episode_000005_in_progress.txt → 提取5
                        file_stem = file.stem  # 去掉后缀："episode_000005_in_progress"
                        ep_id_str = file_stem.split("_")[1]  # 按"_"分割,取第2个元素："000005"
                        ep_id = int(ep_id_str)  # 转为整数：5
                        unfinished_episode_ids.add(ep_id)  # 加入未完成集合
                        # 未完成的ID也可能是最大ID,需要更新max_episode_id
                        if ep_id > max_episode_id:
                            max_episode_id = ep_id
                        logger.debug(f"检测到未完成episode:{ep_id}（来自{file.name})")
                    except (IndexError, ValueError) as e:
                        logger.warning(f"忽略格式错误的进度文件：{file.name},错误：{str(e)}")

                # 子步骤1.2：再处理已完成的parquet文件
                episode_files = list(chunk_dir.glob("episode_*.parquet"))
                for file in episode_files:
                    try:
                        # 解析文件名：episode_000005.parquet → 提取5
                        file_stem = file.stem  # "episode_000005"
                        ep_id = int(file_stem.split("_")[1])
                        if ep_id > max_episode_id:
                            max_episode_id = ep_id  # 更新最大ID
                    except (IndexError, ValueError) as e:
                        logger.warning(f"忽略格式错误的Parquet文件:{file.name},错误：{str(e)}")

            # 步骤2：清理「已完成episode」的残留进度标记（避免误删未完成的标记）
            if max_episode_id >= 0:
                # 只有当最大ID不在未完成集合中时,才清理它的残留标记
                if max_episode_id not in unfinished_episode_ids:
                    chunk_index, chunk_name = get_chunk_info(max_episode_id)
                    chunk_dir = self.base_data_dir / chunk_name
                    residual_mark = chunk_dir / f"episode_{max_episode_id:06d}_in_progress.txt"
                    if residual_mark.exists():
                        residual_mark.unlink()
                        logger.warning(f"删除chunk {chunk_name} 中残留的进度标记：{residual_mark.name}")
                else:
                    # 最大ID是未完成的,保留其进度标记
                    logger.debug(f"最大episode_id {max_episode_id} 未完成,保留其进度标记")

        return max_episode_id, unfinished_episode_ids  # 返回「最大ID」和「未完成集合」

    # --------------------------
    # 统一执行器自旋函数
    # --------------------------
    def _spin_executor(self):
        """统一执行器自旋线程,管理所有ROS2节点"""
        try:
            self.executor.spin()
        except KeyboardInterrupt:
            logger.info("执行器自旋被键盘中断")
        except Exception as e:
            logger.error(f"执行器自旋出错: {e}", exc_info=True)

    def _init_task_dirs(self):
        """创建目录"""
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _init_meta_files(self):
        """初始化元数据文件"""
        # info.json
        official_meta_path = self.meta_dir / "info.json"
        if not official_meta_path.exists():
            with open(official_meta_path, "w", encoding="utf-8") as f:
                json.dump(self.official_meta, f, indent=4, ensure_ascii=False)
        
        # tasks.jsonl
        tasks_path = self.meta_dir / "tasks.jsonl"
        if not tasks_path.exists():
            with open(tasks_path, "w", encoding="utf-8") as f:
                task = {"task_index": 0, "task": self.task_description}
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        
        # episodes_stats.jsonl
        
        # episodes_stats_path = self.meta_dir / "episodes_stats.jsonl"
        # if not episodes_stats_path.exists():
        #     with open(episodes_stats_path, "w", encoding="utf-8") as f:
        #         pass

    def _init_official_meta(self) -> Dict:
        """初始化元数据结构(lerobotv2.1版本)"""
        state_names = (
            self.robot_cfg["joint_names"]["left_arm"] +  # 左关节：l_1~l_7
            self.robot_cfg["joint_names"]["right_arm"] + # 右关节：r_1~r_7
            ["left_gripper", "right_gripper"]            # 夹爪：左+右
        )
        action_names = state_names
        initial_total_episodes = 0
        initial_total_videos = initial_total_episodes * len(self.camera_names)
        
        return {
            "codebase_version": "v2.1",  
            "robot_type": self.robot_type,
            "total_episodes": initial_total_episodes,
            "total_frames": 0,
            "total_tasks": 1,
            "total_videos": initial_total_videos,
            "total_chunks": 1,
            "chunks_size": CHUNK_SIZE,
            "fps": self.frequency,
            "splits": {"train": f"0:{initial_total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                **{
                    cam:{
                        "dtype": "image",
                        "shape": [CAMERA_CONFIG["height"], CAMERA_CONFIG["width"], 3],
                        "names": ["height", "width", "channel"]
                    } for cam in self.camera_names
                },
                "state": {
                    "dtype": "float32", 
                    "shape": [DOUBLE_ARM_STATE_LEN], 
                    "names": "state"
                },
                "actions": {
                    "dtype": "float32", 
                    "shape": [DOUBLE_ARM_ACTION_LEN], 
                    "names": "actions"
                },
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]}
            }
        }


    def start_episode(self) -> bool:
        """开始Episode"""
        if not self.is_available:
            logger.warning("无法创建新episode当前任务未完成")
            return False

        current_episode_id = self.episode_id
        self.current_chunk_index, self.current_chunk_name = get_chunk_info(current_episode_id)
        # 动态生成当前chunk的data目录（如data/chunk-001）
        self.current_data_dir = self.base_data_dir / self.current_chunk_name
        # 动态生成当前chunk的video目录（如videos/chunk-001/observation.images.image）
        self.current_camera_video_dirs = {
            cam: self.base_videos_dir / self.current_chunk_name / f"observation.images.{cam}" 
            for cam in self.camera_names}

        # 创建当前episode的arm_data目录
        self.current_arm_data_dir = self.base_arm_data_dir / f"episode_{current_episode_id:06d}"
        self.current_arm_data_dir.mkdir(parents=True, exist_ok=True)
        # 初始化左右臂数据文件路径
        self.left_arm_data_path = self.current_arm_data_dir / "left_data.txt"
        self.right_arm_data_path = self.current_arm_data_dir / "right_data.txt"
        
        # 清空文件（避免追加旧数据）- 在打开文件句柄之前清空
        open(self.left_arm_data_path, "w").close()
        open(self.right_arm_data_path, "w").close()
        
        arm_data_paths = {
            "left_arm": str(self.left_arm_data_path),  # 转换为字符串路径（兼容文件操作）
            "right_arm": str(self.right_arm_data_path)
        }
        # 【新增】恢复机械臂数据写入txt（传入新的文件路径，内部会打开文件句柄）
        self.ros2_interface.resume_arm_data_write(arm_data_paths)
    
        # 初始化当前episode的参数
        self.item_id = self.global_start_index - 1
        self.episode_data = []
        self.sync_stats = {"total_frames": 0, "total_diff_ms": 0, "max_diff_ms": 0}
        self.global_stats_cache = {k: [] for k in self.global_stats_cache.keys()}
        self.current_episode_frame_idx = 0
        self.camera_manager.episode_index = current_episode_id
        
        # 初始化当前episode的frame_index计数器（从0开始）
        self.current_data_dir.mkdir(parents=True, exist_ok=True)  # 创建chunk的data目录
        for cam_dir in self.current_camera_video_dirs.values():
            cam_dir.mkdir(parents=True, exist_ok=True)  # 创建chunk的video子目录

        # 创建进度标记文件
        self.in_progress_mark = self.current_data_dir / f"episode_{current_episode_id:06d}_in_progress.txt"
        if self.in_progress_mark.exists():
            self.in_progress_mark.unlink()
            logger.info(f"删除chunk {self.current_chunk_name} 中残留的进度标记：{self.in_progress_mark.name}(为新Episode {current_episode_id:06d} 做准备)")
        self.in_progress_mark.touch()
        logger.info(f"在chunk {self.current_chunk_name} 中创建进度标记：{self.in_progress_mark.name}(Episode {current_episode_id:06d})")
        
        # 清理临时目录
        self.temp_image_dir = self.task_dir / "temp_images" / f"episode_{current_episode_id:06d}"
        if self.temp_image_dir.exists():
            shutil.rmtree(self.temp_image_dir, ignore_errors=True)
            logger.warning(f"清理残留临时目录：{self.temp_image_dir}")
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)
        
        # 重置缓存和历史
        self.state_history = []
        self.ros2_interface.reset_cache()
        
        # 清空相机历史帧队列
        self.camera_manager.clear_frame_queues()
        # 启动相机
        self.camera_manager.start(self.temp_image_dir)
        # 等待订阅连接建立
        time.sleep(0.3)
        self.is_available = False
        
        print("\n" + "*" * 30)
        logger.info(f"开始新episode: {current_episode_id:06d}")
        print(f"可以开始采集")
        print("*" * 30 + "\n")
        
        self.camera_error_count = 0
        self.arm_error_count = 0
        
        return True

    def record_frame(self):
        # 1. 获取相机帧（时间戳已是19位整数）
        # colors, cam_timestamps, encoded_bytes, _ = self.camera_manager.get_frames()
        colors, cam_timestamps = self.camera_manager.get_frames()
        
        if len(colors) != len(self.camera_names):
            self.camera_error_count += 1
            logger.warning(f"相机帧不完整（{len(colors)}/{len(self.camera_names)}），跳过当前帧")
            
            if self.camera_error_count > self.ERROR_MAX:
                logger.info("超过相机无效帧次数，关闭程序，请检查相机状态")
                self.close(is_normal_exit=False)
                sys.exit(1)
            return
        
        # 2. 计算相机帧的平均时间戳
        valid_timestamps = [ts for ts in cam_timestamps.values() if ts is not None]
        if not valid_timestamps:
            logger.warning("无有效相机时间戳，跳过当前帧")
            return
        
        # 平均时间戳用整数计算
        avg_timestamp = int(np.mean(valid_timestamps)) 
        logger.debug(f"相机平均时间戳：{avg_timestamp}")

        # 3. 获取机械臂同步数据
        try:
            current_state, arm_timestamp, time_diff = self.ros2_interface.get_synced_double_arm_data(avg_timestamp)
            required_fields = ["left_arm_joints", "left_gripper", "right_arm_joints", "right_gripper"]
            if not all(f in current_state for f in required_fields):
                logger.error(f"双臂state缺少必要字段: {[f for f in required_fields if f not in current_state]}，跳过当前帧")
                return
        except RuntimeError as e:
            self.arm_error_count += 1
            logger.error(f"获取机械臂数据失败: {str(e)}，跳过当前帧")
            if self.arm_error_count > self.ERROR_MAX:
                logger.info("超过机械臂数据失败次数，关闭程序，请检查机械臂状态")
                self.close(is_normal_exit=False)
                sys.exit(1)
            return

        # 3. 更新索引和统计
        self.item_id += 1
        # self.sync_stats["total_frames"] += 1
        # self.sync_stats["total_diff_ms"] += time_diff
        # if time_diff > self.sync_stats["max_diff_ms"]:
        #     self.sync_stats["max_diff_ms"] = time_diff
        # 跳过前3帧的统计（前N帧忽略时间差）
        if self.current_episode_frame_idx >= 0:
            self.sync_stats["total_frames"] += 1
            self.sync_stats["total_diff_ms"] += time_diff
            if time_diff > self.sync_stats["max_diff_ms"]:
                self.sync_stats["max_diff_ms"] = time_diff
        # --------------------------
        # 新增：时间差超过阈值时，打印包含详细时间戳的警告
        # --------------------------
        if self.current_episode_frame_idx >= 0 and time_diff > SYNC_CONFIG["max_time_diff_ms"]:
            # 格式化相机时间戳字符串（每个相机名称+时间戳）
            cam_ts_detail = ", ".join([
                f"{cam}: {ts:.6f}s"  # 保留6位小数，精确到微秒
                for cam, ts in cam_timestamps.items()
            ])
            logger.warning(
                f"数据警告: 时间差 {time_diff:.2f}ms 超过阈值 {SYNC_CONFIG['max_time_diff_ms']}ms\n"
                f"  双臂平均时间戳: {arm_timestamp:.6f}s\n"
                f"  相机平均时间戳: {cam_ts_detail}\n"
                f"  当前帧索引: {self.current_episode_frame_idx}"  # 补充帧索引，方便定位问题帧
            )

        relative_timestamp = (self.current_episode_frame_idx + 1) * (1 / self.frequency)

        # 4. 更新state历史和统计缓存
        self.state_history.append(current_state)
        # state_array = np.concatenate([
        #     np.array(current_state["left_arm_joints"], dtype=np.float32),  # 左臂关节（7个）
        #     np.array(current_state["left_gripper"], dtype=np.float32),     # 左夹爪（8个）→ 前15位
        #     np.array(current_state["right_arm_joints"], dtype=np.float32), # 右臂关节（7个）
        #     np.array(current_state["right_gripper"], dtype=np.float32)     # 右夹爪（8个）→ 后15位
        # ])
        state_array = np.concatenate([
            np.array(current_state["left_arm_joints"], dtype=np.float32),
            np.array(current_state["right_arm_joints"], dtype=np.float32),
            np.array(current_state["left_gripper"], dtype=np.float32),
            np.array(current_state["right_gripper"], dtype=np.float32)
        ])
        
        self.global_stats_cache["state"].append(state_array)
        self.global_stats_cache["actions"].append(None)
        self.global_stats_cache["timestamp"].append([relative_timestamp])
        self.global_stats_cache["frame_index"].append(None)
        self.global_stats_cache["index"].append([self.item_id])
        self.global_stats_cache["episode_index"].append([self.episode_id])
        self.global_stats_cache["task_index"].append([0])

        # 5. 处理图像数据（核心优化：直接使用异步编码的字节流）
        # image_data = {}
        # for cam_name in self.camera_names:
        #     # 从 encoded_bytes 获取异步编码结果（无需主循环编码）
        #     img_bytes = encoded_bytes.get(cam_name, b"")
            
        #     # 构建图像数据（路径仍需记录，用于视频合成）
        #     # img_filename = f"{cam_name}_frame_{self.current_episode_frame_idx:06d}.jpg"

        #     cam_timestamp = cam_timestamps[cam_name]
        #     img_filename = f"{cam_name}_frame_{self.current_episode_frame_idx:06d}_{cam_timestamp}.png"
            
        #     img_path = self.temp_image_dir / img_filename

        #     image_data[cam_name] = {
        #         "bytes": img_bytes,  # 直接使用异步编码的字节流
        #         "path": str(img_path),
        #         "height": CAMERA_CONFIG["height"],
        #         "width": CAMERA_CONFIG["width"],
        #         "format": "jpg"
        #     }

        # # 6. 存储帧数据
        # self.episode_data.append({
        #     "idx": self.item_id,
        #     "state": current_state,
        #     "frame_index": self.current_episode_frame_idx,
        #     "images": image_data,
        #     "timestamps": {
        #         "camera": relative_timestamp,
        #         "robot": relative_timestamp,
        #         "sync_diff_ms": time_diff
        #     }
        # })
        self.current_episode_frame_idx += 1

        
    def end_episode(self):
        """结束Episode并保存数据"""
        # self.ros2_interface.pause_arm_data_write()
        
        # 1. 停止相机（含视频合成）
        self.camera_manager.stop(self.current_camera_video_dirs)  
                  
        self.ros2_interface.pause_arm_data_write()

        # 2. 生成action列表
        n_states = len(self.state_history)
        action_list = []
        if n_states >= 2:
            # 步骤1：生成前n_states-1个action（action[i] = state[i+1]）
            for i in range(n_states - 1):
                action_list.append(self.state_history[i + 1])
            
            # 步骤2：最后一帧action复用倒数第二帧的action（满足需求）
            # 此时action_list已有n_states-1个元素,添加最后一个使其长度为n_states
            action_list.append(action_list[-1])  # 最后一帧 = 倒数第二帧

        elif n_states == 1:
            # 特殊情况：只有1个state时
            # 生成与episode_data等长的action列表,且所有元素相同（最后一帧自然与前一帧一致）
            action_list = [self.state_history[0]] * len(self.episode_data)

        else:
            logger.warning("无state数据无法生成action")
            # 清理进度标记和临时目录
            if hasattr(self, "in_progress_mark") and self.in_progress_mark.exists():
                self.in_progress_mark.unlink()
            if hasattr(self, "temp_image_dir") and self.temp_image_dir.exists():
                shutil.rmtree(self.temp_image_dir, ignore_errors=True)
            self.is_available = True
            return

        # 确保action_list长度与episode_data完全一致（避免截断）
        if len(action_list) != len(self.episode_data):
            # 极端情况：长度不匹配时,用最后一个action填充至相同长度
            diff = len(self.episode_data) - len(action_list)
            if diff > 0:
                action_list.extend([action_list[-1]] * diff)
            else:
                action_list = action_list[:len(self.episode_data)]
        
         # 3. 补充action到episode_data和统计缓存
        min_len = min(len(self.episode_data), len(action_list))
        
        # 填充frame_index真实值（从episode_data中提取预存的frame_index）
        for i in range(min_len):
            # 从episode_data中获取当前帧的frame_index（0开始递增）
            self.global_stats_cache["frame_index"][i] = [self.episode_data[i]["frame_index"]]
        
        # 截取有效长度（避免越界）
        self.episode_data = self.episode_data[:min_len]
        for key in self.global_stats_cache:
            self.global_stats_cache[key] = self.global_stats_cache[key][:min_len]
        # 填充action
        for i in range(min_len):
            self.episode_data[i]["action"] = action_list[i]
            # 更新统计缓存中的action
            # action_array = np.concatenate([
            # np.array(action_list[i]["left_arm_joints"], dtype=np.float32),   # 左臂关节（7维）
            # np.array(action_list[i]["left_gripper"], dtype=np.float32),      # 左夹爪（8维）→ 前15位
            # np.array(action_list[i]["right_arm_joints"], dtype=np.float32),  # 右臂关节（7维）
            # np.array(action_list[i]["right_gripper"], dtype=np.float32)      # 右夹爪（8维）→ 后15位
            # ])
            action_array = np.concatenate([
            np.array(action_list[i]["left_arm_joints"], dtype=np.float32),   # 左臂关节（7维）
            np.array(action_list[i]["right_arm_joints"], dtype=np.float32),  # 右臂关节（7维）
            np.array(action_list[i]["left_gripper"], dtype=np.float32),      # 左臂夹爪（8维）
            np.array(action_list[i]["right_gripper"], dtype=np.float32)      # 右臂夹爪（8维）
            ])
            self.global_stats_cache["actions"][i] = action_array

            
        # 4. 输出统计
        avg_diff = self.sync_stats["total_diff_ms"] / self.sync_stats["total_frames"] if self.sync_stats["total_frames"] > 0 else 0
        logger.info(
            f"****************************************************\n"
            f"统计帧数={self.sync_stats['total_frames']}, "
            f"平均时间差={avg_diff:.2f}ms, 最大时间差={self.sync_stats['max_diff_ms']:.2f}ms\n"
            f"****************************************************"
        )
        
        self.ros2_interface.dump_cb_stats()

        # 5. 保存Parquet数据
        self._save_episode()
        
        # 6. 清理当前chunk的进度标记
        # if self.temp_image_dir.exists():
        #     shutil.rmtree(self.temp_image_dir, ignore_errors=True)
        #     logger.info(f"清理临时目录：{self.temp_image_dir}")
        

    def _save_episode(self):
        """保存Parquet文件"""
        current_episode_id = self.episode_id 
        logger.info(f"开始保存episode {current_episode_id:06d} 到 chunk {self.current_chunk_name}(目录：{self.current_data_dir})")

        # 准备数据
        indices = []
        states = []
        actions = []
        timestamps = []
        episode_indices = []
        frame_indices = []
        task_indices = []
        # 图像数据
        # image_structs = {cam: [] for cam in self.camera_names}

        for frame in self.episode_data:
            if "action" not in frame:
                continue
            
            indices.append(frame["idx"])
            episode_indices.append(self.episode_id)
            task_indices.append(0)
            timestamps.append(frame["timestamps"]["camera"])
            frame_indices.append(frame["frame_index"])
            # 状态和动作
            # state = np.concatenate([
            #     np.array(frame["state"]["left_arm_joints"], dtype=np.float32),   # 左臂关节（7维）
            #     np.array(frame["state"]["left_gripper"], dtype=np.float32),      # 左臂夹爪（8维）
            #     np.array(frame["state"]["right_arm_joints"], dtype=np.float32),  # 右臂关节（7维）
            #     np.array(frame["state"]["right_gripper"], dtype=np.float32)      # 右臂夹爪（8维）
            # ])
            # action = np.concatenate([
            #     np.array(frame["action"]["left_arm_joints"], dtype=np.float32),   # 左臂关节（7维）
            #     np.array(frame["action"]["left_gripper"], dtype=np.float32),      # 左臂夹爪（8维）
            #     np.array(frame["action"]["right_arm_joints"], dtype=np.float32),  # 右臂关节（7维）
            #     np.array(frame["action"]["right_gripper"], dtype=np.float32)      # 右臂夹爪（8维）
            # ])
            state = np.concatenate([
                np.array(frame["state"]["left_arm_joints"], dtype=np.float32),   # 左臂关节（7维）
                np.array(frame["state"]["right_arm_joints"], dtype=np.float32),  # 右臂关节（7维）
                np.array(frame["state"]["left_gripper"], dtype=np.float32),      # 左臂夹爪（8维）
                np.array(frame["state"]["right_gripper"], dtype=np.float32)      # 右臂夹爪（8维）
            ])
            action = np.concatenate([
                np.array(frame["action"]["left_arm_joints"], dtype=np.float32),   # 左臂关节（7维）
                np.array(frame["action"]["right_arm_joints"], dtype=np.float32),  # 右臂关节（7维）
                np.array(frame["action"]["left_gripper"], dtype=np.float32),      # 左臂夹爪（8维）
                np.array(frame["action"]["right_gripper"], dtype=np.float32)      # 右臂夹爪（8维）
            ])
            states.append(state)
            actions.append(action)
            
        
        # 创建Parquet表
        data = {
            "state": pa.array(states, type=pa.list_(pa.float32())),
            "actions": pa.array(actions, type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()), 
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64())
        }

        
        # 构建huggingface元数据
        huggingface_metadata = {
            "info": {
                "features": {
                    # "image": {"_type": "Image"},
                    # "wrist_image": {"_type": "Image"},
                    "state": {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "float32"}},
                    "actions": {"_type": "Sequence", "feature": {"_type": "Value", "dtype": "float32"}},
                    "timestamp": {"_type": "Value", "dtype": "float32"},
                    "frame_index": {"_type": "Value", "dtype": "int64"},
                    "episode_index": {"_type": "Value", "dtype": "int64"},
                    "index": {"_type": "Value", "dtype": "int64"},
                    "task_index": {"_type": "Value", "dtype": "int64"}
                }
            }
        }
        schema_metadata = {
            b"huggingface": json.dumps(huggingface_metadata).encode("utf-8")
        }
        
        # 定义完整Schema（包含huggingface元数据所需的字段类型）
        schema = pa.schema([
            # pa.field("image", image_field),  
            # pa.field("wrist_image", image_field),  
            pa.field("state", pa.list_(pa.float32())),
            pa.field("actions", pa.list_(pa.float32())),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64())
        ], metadata=schema_metadata)
        # 生成表并写入（指定schema和元数据）
        
        table = pa.Table.from_pydict(data, schema=schema)
        output_path = self.current_data_dir / f"episode_{self.episode_id:06d}.parquet"
        pq.write_table(
            table, 
            output_path, 
            compression="snappy"
            # metadata=schema_metadata  # 写入元数据
        )
        
        logger.info(f"Parquet保存完成: {output_path}")
        
        # 删除_in_progress标记（必须在计数器更新前）
        mark_deleted = False  # 标记是否删除成功
        if hasattr(self, "in_progress_mark") and self.in_progress_mark.exists():
            try:
                self.in_progress_mark.unlink()
                logger.info(f"episode {current_episode_id:06d} 进度标记已删除：{self.in_progress_mark.name}")
                mark_deleted = True  # 标记删除成功
            except Exception as e:
                logger.error(f"删除episode {current_episode_id:06d} 进度标记失败：{str(e)}")
                mark_deleted = False  # 标记删除失败
        
        # 只有标记删除成功,才更新计数器
        if indices and mark_deleted:
            last_index = indices[-1]  # 当前episode的最后一个index
            self._update_counter_file(last_index)  # 写入计数器文件
            self.global_start_index = last_index + 1  # 更新下一次起始索引
            logger.info(f"episode {current_episode_id:06d} 完全完成,更新计数器:最新index={last_index},下一次起始index={self.global_start_index}")
        elif indices and not mark_deleted:
            # 标记删除失败,不更新计数器（避免后续重复）
            logger.warning(f"episode {current_episode_id:06d} Parquet已保存,但标记删除失败,暂不更新计数器!下次启动会重新处理并修正索引")
        else:
            logger.warning("当前episode无有效数据,不更新计数器文件")
        
        # self._generate_episode_stats(indices, states, actions, current_episode_id)
    
        self.episode_id += 1  # 递增,避免下一次重复
        
        # 更新元数据
        self._update_meta_data(current_episode_id)
        
        self.is_available = True
        logger.info(f"episode {current_episode_id:06d} 保存到chunk {self.current_chunk_name} 完成")
 

    def _update_meta_data(self, current_episode_id: int):
        """更新元数据文件（优化：补充统计日志）"""
        # 1. 更新episodes.jsonl
        episodes_path = self.meta_dir / "episodes.jsonl"
        episode_meta = {
            "episode_index": current_episode_id,
            "tasks": [self.task_description],
            "length": self.sync_stats["total_frames"],  # 最终有效帧数
        }

        # 读取已有条目,过滤掉相同episode_index的旧数据
        existing_episodes = []
        if episodes_path.exists():
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ep = json.loads(line.strip())
                        # 只保留非当前episode_index的条目
                        if ep["episode_index"] != current_episode_id:
                            existing_episodes.append(ep)
                    except Exception as e:
                        logger.warning(f"跳过无效的episodes.jsonl条目:{line.strip()},错误：{e}")

        # 写入：先写回过滤后的旧条目,再追加新条目
        with open(episodes_path, "w", encoding="utf-8") as f:
            # 写回保留的旧条目
            for ep in existing_episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            # 追加当前新条目
            f.write(json.dumps(episode_meta, ensure_ascii=False) + "\n")
        logger.info(f"已更新episodes.jsonl,episode {current_episode_id} 去重后保留最新数据")

        # 2. 更新info.json
        official_meta_path = self.meta_dir / "info.json"
        total_episodes = self.episode_id
        
        # 计算总chunk数（最大chunk索引 + 1）
        if total_episodes == 0:
            total_chunks = 1  # 无数据时默认1个chunk
        else:
            max_chunk_index = (total_episodes - 1) // CHUNK_SIZE  # 最后一个episode的chunk索引
            total_chunks = max_chunk_index + 1  # 总chunk数=最大索引+1
        
        # 计算总视频数（每个episode对应len(camera_names)个视频）
        total_videos = total_episodes * len(self.camera_names)
        
        # 计算总帧数（跨所有chunk累加）
        total_frames = 0
        if self.base_data_dir.exists():
            chunk_dirs = self.base_data_dir.glob("chunk-???")
            for chunk_dir in chunk_dirs:
                for episode_file in chunk_dir.glob("episode_*.parquet"):
                    try:
                        frame_count = pq.read_table(episode_file).num_rows
                        total_frames += frame_count
                        logger.debug(f"chunk {chunk_dir.name} / {episode_file.name}:{frame_count} 帧")
                    except Exception as e:
                        logger.warning(f"读取文件 {episode_file.name} 帧数失败,跳过：{e}")
        
        # 更新info.json
        self.official_meta.update({
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_videos": total_videos,
            "total_chunks": total_chunks,  # 动态更新总chunk数
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"0:{total_episodes}"}  # 训练集范围覆盖所有episode
        })
        with open(official_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.official_meta, f, indent=4, ensure_ascii=False)

    def close(self, is_normal_exit: bool = False):
        """释放资源。
        - 正常退出(is_normal_exit=True): 完成当前episode保存,保留所有temp_images/arm_data目录,仅清理残留_in_progress.txt标记。
        - 异常退出(is_normal_exit=False): 不保存当前episode,清理临时图片+对应arm_data的txt文件,保留_in_progress.txt标记。
        """
        logger.info("开始关闭处理器...")

        # 1. 先停止执行器（这会停止执行器线程处理回调）
        try:
            self.executor.shutdown()
        except Exception as e:
            logger.warning(f"执行器shutdown时出现异常: {str(e)}")
        
        # 2. 等待执行器线程完全退出（确保执行器不再使用节点）
        if hasattr(self, 'executor_thread') and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=2.0)
            if self.executor_thread.is_alive():
                logger.warning("执行器线程在超时后仍未退出")
        
        # 3. 从执行器移除节点（在shutdown之后是安全的，但可能不是必需的）
        try:
            self.executor.remove_node(self.ros2_interface)
        except Exception as e:
            logger.debug(f"移除ros2_interface节点时出现异常（可能已经被移除）: {str(e)}")
        
        try:
            self.executor.remove_node(self.camera_manager)
        except Exception as e:
            logger.debug(f"移除camera_manager节点时出现异常（可能已经被移除）: {str(e)}")
        
        try:
            self.executor.remove_node(self)
        except Exception as e:
            logger.debug(f"移除self节点时出现异常（可能已经被移除）: {str(e)}")
        
        # 4. 销毁节点（必须在执行器完全停止之后）
        try:
            self.ros2_interface.destroy_node()
        except Exception as e:
            logger.warning(f"销毁ros2_interface节点时出现异常: {str(e)}")
        
        try:
            self.camera_manager.destroy_node()
        except Exception as e:
            logger.warning(f"销毁camera_manager节点时出现异常: {str(e)}")
        
        try:
            self.destroy_subscription(self.start_subscriber)
            self.destroy_node()
        except Exception as e:
            logger.warning(f"销毁self节点时出现异常: {str(e)}")

        # 区分正常/异常退出处理
        if not self.is_available:  # 如果当前正在采集
            if is_normal_exit:
                # 1. 正常退出：完成保存,保留所有temp_images/arm_data目录（包含txt和图片）
                logger.info("检测到正常退出,正在处理未完成的episode...")
                self.end_episode()
                while not self.is_available:
                    time.sleep(0.01)
                
                # 正常退出时,执行计数器校验（原有逻辑保留）
                if hasattr(self, "current_data_dir") and self.current_data_dir.exists():
                    current_ep_id = self.episode_id - 1  # 当前处理的episode_id(已递增,需减1)
                    ep_file = self.current_data_dir / f"episode_{current_ep_id:06d}.parquet"
                    if ep_file.exists() and os.path.getsize(ep_file) > 1024:
                        try:
                            table = pq.read_table(ep_file, columns=["index"])
                            index_col = table["index"]
                            if isinstance(index_col, pa.ChunkedArray):
                                index_col = index_col.combine_chunks()
                            ep_last_index = pc.max(index_col).as_py() 

                            if self.counter_file.exists():
                                with open(self.counter_file, "r", encoding="utf-8") as f:
                                    counter_last = json.load(f)["last_max_index"]
                                if counter_last > ep_last_index:
                                    self._update_counter_file(ep_last_index)
                                    logger.info(f"程序退出,修正计数器:从 {counter_last} 回滚到 {ep_last_index}(匹配Parquet实际index)")
                        except Exception as e:
                            logger.warning(f"校验计数器一致性失败:{str(e)}")
            else:
                # 2. 异常退出：清理临时图片 + 对应arm_data的txt文件,保留_in_progress.txt标记
                logger.error("检测到异常退出,保留进度标记,清理临时图片和对应arm_data的txt文件...")
                current_ep_id = self.episode_id - 1 if hasattr(self, "episode_id") else None
                
                # 【修改1：清理temp_images下的当前episode图片目录】
                temp_root = self.task_dir / "temp_images"
                if temp_root.exists():
                    current_ep_temp_dir = temp_root / f"episode_{current_ep_id:06d}" if current_ep_id is not None else None
                    if current_ep_temp_dir and current_ep_temp_dir.exists():
                        try:
                            shutil.rmtree(current_ep_temp_dir)
                            logger.info(f"已删除异常episode的临时图片目录: {current_ep_temp_dir}")
                        except Exception as e:
                            logger.warning(f"删除临时图片目录 {current_ep_temp_dir} 失败: {str(e)}")

                # 【修改2：清理相机订阅器的临时图片目录（每个相机的独立目录）】
                if hasattr(self.camera_manager, "image_dirs"):
                    for cam_name, cam_temp_dir in self.camera_manager.image_dirs.items():
                        if cam_temp_dir.exists():
                            try:
                                shutil.rmtree(cam_temp_dir)
                                logger.info(f"异常退出,删除 {cam_name} 相机临时图片目录: {cam_temp_dir}")
                            except Exception as e:
                                logger.warning(f"删除 {cam_name} 相机临时目录失败: {str(e)}")

                # 【核心恢复：异常退出时清理当前episode的arm_data目录（包含left/right_data.txt）】
                if hasattr(self, "current_arm_data_dir") and self.current_arm_data_dir.exists():
                    try:
                        shutil.rmtree(self.current_arm_data_dir)
                        logger.info(f"异常退出,删除当前episode的arm_data目录（含txt文件）: {self.current_arm_data_dir}")
                    except Exception as e:
                        logger.warning(f"删除arm_data目录（含txt文件）失败: {str(e)}")

                # 重置状态,避免死循环
                self.is_available = True

        # 【修改3：正常退出时仅清理残留_in_progress.txt标记,不碰任何数据目录】
        if is_normal_exit:
            logger.info("开始清理所有残留的in_progress.txt标记（保留所有temp_images/arm_data目录）...")
            if self.base_data_dir.exists():
                chunk_dirs = self.base_data_dir.glob("chunk-???")
                for chunk_dir in chunk_dirs:
                    residual_marks = chunk_dir.glob("episode_*_in_progress.txt")
                    for mark in residual_marks:
                        try:
                            mark.unlink()
                            logger.info(f"删除残留标记:{mark}")
                        except Exception as e:
                            logger.warning(f"删除标记 {mark} 失败:{str(e)}")

        # 【修改4：temp_images根目录处理逻辑】
        temp_root = self.task_dir / "temp_images"
        if temp_root.exists():
            try:
                if is_normal_exit:
                    # 正常退出：无论是否为空,都保留temp_images根目录（含所有图片）
                    logger.info(f"正常退出,永久保留temp_images根目录: {temp_root}（包含所有采集的图片）")
                else:
                    # 异常退出：仅当根目录为空时删除,非空则保留（避免误删其他episode数据）
                    if not any(temp_root.iterdir()):
                        shutil.rmtree(temp_root)
                        logger.info(f"异常退出,删除空的temp_images根目录:{temp_root}")
                    else:
                        logger.info(f"异常退出,temp_images根目录非空,保留剩余数据: {temp_root}")
            except Exception as e:
                logger.warning(f"检查temp_images根目录失败:{str(e)}")

        # 【日志补充】明确告知清理/保留状态
        if is_normal_exit:
            logger.info("正常退出完成：所有图片、arm_data的txt文件均已保留")
        else:
            logger.info("异常退出完成：已清理当前episode的图片和arm_data的txt文件,保留进度标记")

        logger.info("处理器已关闭")


def main():
    # 加强ROS2初始化检查
    if not rclpy.ok():
        try:
            rclpy.init(args=sys.argv)
        except Exception as e:
            print(f"ROS2初始化失败: {str(e)}")
            return

    if not rclpy.ok():
        print("ROS2初始化后仍处于异常状态,无法启动程序")
        return

    processor = None
    try:
        processor = DoubleArmDataProcessor(
            task_dir="/mnt/data1/robot_datasets/0430_cyw_coffee_02",
            # task_dir="/home/peanut/mlt/robot_datasets",
            robot_type="Keenon_F1_DoubleArm",
            task_description="Organize your clothes on the table"
        )
        print("===== 双臂抓取任务数据采集系统 =====")
        print("操作说明：")
        print(f"  1. 向ROS2话题 '{ROS2_START_TOPIC}' 发布1 → 开始新episode")
        print(f"  2. 向ROS2话题 '{ROS2_START_TOPIC}' 发布0 → 停止当前episode并保存")
        print("  3. 输入'q'并按Enter → 退出程序")
        print("====================================")

        # 非阻塞主循环
        while True:
            # 新增：机械臂数据心跳检测
            if processor.ros2_interface.check_any_arm_timeout():
                logger.error("机械臂数据超时2s,未在规定时间内收到数据,程序将自动退出。")
                processor.close(is_normal_exit=False)
                # 退出主循环
                break
            # --------------------------
            # 1. 处理键盘输入
            # --------------------------
            if select.select([sys.stdin], [], [], 0.001)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == 'q':
                    print("\n准备退出程序...")
                    processor.close(is_normal_exit=True)
                    break
                else:
                    print(f"无效输入：{user_input},仅支持'q'退出")

            # --------------------------
            # 2. 处理 Start 话题状态变化
            # --------------------------
            with processor.start_lock:
                current_start = processor.current_start_state
                last_start = processor.last_start_state

            if current_start != last_start:
                if current_start == 1:
                    if processor.is_available:
                        success = processor.start_episode()
                        if not success:
                            logger.error("启动episode失败(当前有未完成任务)")
                    else:
                        logger.warning(f"当前正在保存 episode {processor.episode_id - 1:06d},暂不允许启动新episode,请等待保存完成")
                elif current_start == 0:
                    if not processor.is_available:
                        print(f"\n开始停止episode {processor.episode_id - 1:06d}...")
                        processor.end_episode()
                        print("\n" + "*" * 30)
                        print(f"episode {processor.episode_id - 1:06d} 停止并保存完成")
                        print("*" * 30 + "\n")
                    else:
                        logger.warning("当前无运行中的episode,无需停止")

                with processor.start_lock:
                    processor.last_start_state = current_start

            # --------------------------
            # 3. 记录帧数据 (带动态 sleep 调整)
            # --------------------------
            if not processor.is_available:
                # 记录当前时间戳,作为这一帧开始处理的时间
                frame_start_time = time.time()

                # 执行帧采集和处理
                processor.record_frame()

                # 计算这一帧总共花费了多长时间
                frame_processing_time = time.time() - frame_start_time

                # 计算为了达到目标帧率,还需要休眠多久
                # 目标间隔 = 1 / 帧率
                target_interval = 1.0 / processor.frequency
                sleep_time = target_interval - frame_processing_time

                # 如果计算出的休眠时间为正,则进行休眠
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # （可选）打印实际帧率,用于调试
                actual_fps = 1.0 / (time.time() - frame_start_time)
                print(f"实际帧率: {actual_fps:.2f} FPS", end='\r')

            # 如果不在录制状态,短暂休眠以降低CPU占用
            else:
                time.sleep(0.01)

    except Exception as e:
        logging.error(f"程序出错：{str(e)}", exc_info=True)
        if processor:
            processor.close(is_normal_exit=False)

    finally:
        if rclpy.ok():
            rclpy.shutdown()
        print("程序已退出")

if __name__ == "__main__":
    main()
