import os
import glob
import dataclasses
import copy
import json
import shutil
from pathlib import Path
from typing import Literal
import bisect

import numpy as np
import pandas as pd
import torch
import tqdm
import tyro
import cv2

import datasets.features.features as _datasets_features
import pyarrow as pa

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _ensure_sequence_in_features_dict(d: dict) -> dict:
    """Recursively replace _type='List' with _type='Sequence' so parquet metadata is compatible with datasets."""
    if not isinstance(d, dict):
        return d
    d = copy.deepcopy(d)
    if d.get("_type") == "List":
        d["_type"] = "Sequence"
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _ensure_sequence_in_features_dict(v)
    return d


def _patched_arrow_schema(self):
    """Build arrow_schema with List -> Sequence in metadata so written parquet is loadable by datasets."""
    hf_metadata = {"info": {"features": _ensure_sequence_in_features_dict(self.to_dict())}}
    return pa.schema(self.type).with_metadata({"huggingface": json.dumps(hf_metadata)})


# Patch Features so parquet files written by LeRobotDataset use _type="Sequence" not "List"
_datasets_features.Features.arrow_schema = property(_patched_arrow_schema)


# ============================================================
# 配置
# ============================================================

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# ============================================================
# 创建空数据集
# ============================================================

def create_empty_dataset(
    repo_id: str,
    root: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    fps: int = 30,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """
    输入：
        repo_id: str - 数据集唯一标识名称（如 "keenon_dataset1114"）
        root: str - 转换后数据集的存储根路径（如 "/path/to/result/data/keenon_dataset1114"）
        robot_type: str - 机器人类型（如 "keenon_arm"）
        mode: Literal["video", "image"] - 图像存储模式（默认 "image"，单帧图片；"video" 合并为MP4）
        fps: int - 数据集帧率（默认 15 FPS，需与原始数据帧率一致）
        dataset_config: DatasetConfig - 数据集处理配置（默认使用 DEFAULT_DATASET_CONFIG）
    输出：
        LeRobotDataset - 初始化后的空 LeRobot 数据集实例，已定义特征结构（state/actions/相机图像）
    约束：
        - mode 仅支持 "video" 或 "image"
        - fps 需为正整数
        - root 路径需可写
    """
    motors = [
        "left_joint0",
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
        "left_gripper_pose",
        "right_joint0",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
        "right_gripper_pose",
    ]
    cameras = [
        "observation.images.head_image",
        "observation.images.left_wrist_image",
        "observation.images.right_wrist_image",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for cam in cameras:
        features[cam] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# ============================================================
# 时间戳工具
# ============================================================

def ns_to_sec_nsec_str(ts_ns: int) -> str:
    sec = ts_ns // 1_000_000_000
    nsec = ts_ns % 1_000_000_000
    return f"{sec}.{nsec:09d}"


# ============================================================
# 图像加载 & 对齐工具
# ============================================================

def parse_image_filename(fname: str) -> int:
    base = os.path.basename(fname).replace(".jpg", "")
    _, ts_ns_str = base.split("_")
    return int(ts_ns_str)


def load_images(folder: str):
    mapping = {}
    for f in glob.glob(os.path.join(folder, "*.jpg")):
        frame_id = int(os.path.basename(f).split("_")[0])
        ts_ns = parse_image_filename(f)

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mapping[frame_id] = {
            "ts_ns": ts_ns,
            "ts_str": ns_to_sec_nsec_str(ts_ns),
            "bytes": img,
        }
    return mapping


def images_to_sorted_list(img_mapping: dict):
    lst = list(img_mapping.values())
    lst.sort(key=lambda x: x["ts_ns"])
    return lst


def find_nearest_image(img_list, target_ts_ns, max_diff_ns=None):
    ts_list = [x["ts_ns"] for x in img_list]
    idx = bisect.bisect_left(ts_list, target_ts_ns)

    candidates = []
    if idx < len(img_list):
        candidates.append(img_list[idx])
    if idx > 0:
        candidates.append(img_list[idx - 1])

    if not candidates:
        return None

    best = min(candidates, key=lambda x: abs(x["ts_ns"] - target_ts_ns))

    if max_diff_ns is not None:
        if abs(best["ts_ns"] - target_ts_ns) > max_diff_ns:
            return None

    return best


# ============================================================
# 机械臂数据
# ============================================================

def load_arm_txt(txt_path: str):
    data = []
    if not os.path.exists(txt_path):
        return data

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ts_ns, rest = line.split(" ", 1)
            arm_list = list(map(float, rest.split(",")))

            data.append({
                "ts_ns": int(ts_ns),
                "ts_str": ns_to_sec_nsec_str(int(ts_ns)),
                "data": arm_list,
            })
    return data


def find_next_arm_timestamp(arm_list, img_ts_ns: int):
    for a in arm_list:
        if a["ts_ns"] > img_ts_ns:
            return a
    return None


# ============================================================
# 计算最大差值和次数
# ============================================================
def max_and_count(values, atol=10):
    """
    values: list of float (ms)
    返回:
        max_val: float or np.nan
        count: int
    """
    if values is None or len(values) == 0:
        return np.nan, 0

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]   # 关键：过滤 NaN

    if arr.size == 0:
        return np.nan, 0

    max_val = arr.max()
    count = int(np.sum(np.abs(arr - max_val) <= atol))

    return float(max_val), count




# ============================================================
# Episode 处理
# ============================================================

def populate_dataset(
    dataset: LeRobotDataset,
    root_path: str,
    task: str,
    ep_files,
    debug_dir,
    action_gripper,
):

    MAX_IMG_DIFF_NS = 40_000_000  # 40 ms
    summary_stats = []  # 用于汇总每个 episode 的最大差值

    for ep_name in tqdm.tqdm(ep_files):
        print(f"\nProcessing {ep_name}")

        img_root = os.path.join(root_path, "temp_images", ep_name)
        arm_root = os.path.join(root_path, "arm_data", ep_name)
        # 任一目录不存在或为空，跳过
        if not os.path.isdir(img_root) or not os.listdir(img_root):
            print(f"[SKIP] img_root missing or empty: {img_root}")
            continue

        if not os.path.isdir(arm_root) or not os.listdir(arm_root):
            print(f"[SKIP] arm_root missing or empty: {arm_root}")
            continue

        head_imgs  = images_to_sorted_list(load_images(os.path.join(img_root, "head_image_images")))
        left_imgs  = images_to_sorted_list(load_images(os.path.join(img_root, "left_wrist_image_images")))
        right_imgs = images_to_sorted_list(load_images(os.path.join(img_root, "right_wrist_image_images")))

        # 图像数量检查,跳过少于100个数据的episode
        if min(len(head_imgs), len(left_imgs), len(right_imgs)) < 100:
            print(
                f"[SKIP] not enough images: "
                f"head={len(head_imgs)}, left={len(left_imgs)}, right={len(right_imgs)}")
            continue

        left_arm  = load_arm_txt(os.path.join(arm_root, "left_data.txt"))
        right_arm = load_arm_txt(os.path.join(arm_root, "right_data.txt"))

        rows = []
        # 先收集所有的observation.state数据
        observation_states = []

        for h in head_imgs[30:-5]:
            head_ts = h["ts_ns"]
            
            l = find_nearest_image(left_imgs, head_ts, MAX_IMG_DIFF_NS)
            r = find_nearest_image(right_imgs, head_ts, MAX_IMG_DIFF_NS)

            if l is None or r is None:
                continue

            ts_max_img = max(h["ts_ns"], l["ts_ns"], r["ts_ns"])

            left_match  = find_next_arm_timestamp(left_arm, ts_max_img)
            right_match = find_next_arm_timestamp(right_arm, ts_max_img)

            if left_match is None or right_match is None:
                continue

            left_arm_data  = left_match["data"]
            right_arm_data = right_match["data"]
            
            dual_arm_state = left_arm_data[:7] + [left_arm_data[8] / 100.] + right_arm_data[:7] + [right_arm_data[8] / 100.]
            dual_arm_action = left_arm_data[:7] + [left_arm_data[7]] + right_arm_data[:7] + [right_arm_data[7]]

            dual_arm_state_arr = np.array(dual_arm_state,dtype=np.float32)
            dual_arm_state_tensor = torch.from_numpy(dual_arm_state_arr)
            
            dual_arm_action_arr = np.array(dual_arm_action,dtype=np.float32)
            dual_arm_action_tensor = torch.from_numpy(dual_arm_action_arr)
            
            observation_states.append({
                "observation_state": dual_arm_state_tensor,
                "observation_action": dual_arm_action_tensor,
                "head_img": h["bytes"],
                "left_img": l["bytes"],
                "right_img": r["bytes"],
                "head_ts": h["ts_str"],
                "left_ts": l["ts_str"],
                "right_ts": r["ts_str"],
                
                "head_ts_ns": h["ts_ns"],
                "left_ts_ns": l["ts_ns"],
                "right_ts_ns": r["ts_ns"], 
                
                "left_arm_ts": left_match["ts_str"] if left_match else None,
                "right_arm_ts": right_match["ts_str"] if right_match else None,
            })
        
        # 收集帧数据并计算每帧差值
        cam_align_diffs = []
        head_diffs = []
        left_diffs = []
        right_diffs = []    
        
            
        # 现在为每个时间步设置action为下一个时间步的observation.state
        for i in range(len(observation_states)):
            current = observation_states[i]
            
            # 如果当前不是最后一个时间步，action是下一个时刻的observation.state
            if i < len(observation_states) - 1:
                if action_gripper != "bin":
                    # action使用开度
                    next_observation = observation_states[i + 1]["observation_state"]
                    action = next_observation
                else:
                    # action 使用 0/1
                    next_observation = observation_states[i + 1]["observation_action"]
                    action = next_observation

            else:
                if action_gripper != "bin":
                    # 最后时刻的action等于本时刻的observation.state
                    action = current["observation_state"]
                else:
                    action = current["observation_action"]

            frame = {
                "task": task,
                "observation.images.head_image": current["head_img"],
                "observation.images.left_wrist_image": current["left_img"],
                "observation.images.right_wrist_image": current["right_img"],
                "observation.state": current["observation_state"],
                "action": action,
            }
            dataset.add_frame(frame)
            
            # ===== 计算三相机对齐最大差 =====
            cam_ts = [
                current["head_ts_ns"],
                current["left_ts_ns"],
                current["right_ts_ns"],
            ]
            cam_align_max_diff_ms = (max(cam_ts) - min(cam_ts)) / 1e6
            cam_align_diffs.append(cam_align_max_diff_ms)
            
            # ===== 新增同一路相机帧间时间差 =====
            if i > 0:
                prev = observation_states[i - 1]
                head_dt_ms  = (current["head_ts_ns"]  - prev["head_ts_ns"])  / 1e6
                left_dt_ms  = (current["left_ts_ns"]  - prev["left_ts_ns"])  / 1e6
                right_dt_ms = (current["right_ts_ns"] - prev["right_ts_ns"]) / 1e6
            else:
                head_dt_ms = left_dt_ms = right_dt_ms = np.nan

            head_diffs.append(head_dt_ms)
            left_diffs.append(left_dt_ms)
            right_diffs.append(right_dt_ms)
            
            rows.append({
                # 时间戳
                "head_image_ts": current["head_ts"],
                "left_image_ts": current["left_ts"],
                "right_image_ts": current["right_ts"],

                # 机械臂（严格在图像之后）
                "left_arm_ts": current["left_arm_ts"],
                "right_arm_ts": current["right_arm_ts"],
                "is_last_frame": (i == len(observation_states) - 1),  # 标记是否为最后一帧
                
                # 时间戳差值
                "cam_align_max_diff_ms": cam_align_max_diff_ms,
                "head_dt_ms": head_dt_ms,
                "left_dt_ms": left_dt_ms,
                "right_dt_ms": right_dt_ms,
            })
            
        dataset.save_episode()
        print(f"data/xxx.parquet save")
        # ===== 保存 rows 到 CSV（每个 episode 一个文件） =====
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)

        csv_path = os.path.join(debug_dir, f"{ep_name}.csv")

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV → {csv_path}")
        
        # ===== 汇总每个 episode 的最大差值 =====
        # cam_max, cam_max_cnt = max_and_count(cam_align_diffs)
        head_max, head_max_cnt = max_and_count(head_diffs)
        left_max, left_max_cnt = max_and_count(left_diffs)
        right_max, right_max_cnt = max_and_count(right_diffs)
        
        summary_stats.append({
            "episode": ep_name,

            # 三相机对齐
            "cam_align_max_diff_ms": max(cam_align_diffs),

            # 同一路相机帧间差
            "head_max_dt_ms": head_max,
            "head_max_dt_cnt": head_max_cnt,

            "left_max_dt_ms": left_max,
            "left_max_dt_cnt": left_max_cnt,

            "right_max_dt_ms": right_max,
            "right_max_dt_cnt": right_max_cnt,
        })

   
    return dataset, summary_stats


# ============================================================
# 主入口
# ============================================================

def port_keenon(
    raw_dir: Path,
    repo_id: str,
    dst_dir: Path,
    task: str,
    robot_type: str = "Keenon_arm",
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    action_gripper_mode: Literal["open_value", "bin"] = "open_value",
):
    """
    转换 Keenon 数据集到 LeRobot 格式
    
    参数:
        raw_dir: 原始数据根目录路径，包含 temp_images 和 arm_data 文件夹
        repo_id: 数据集标识名称
        dst_dir: 转换后数据集的存储根路径
        robot_type: 机器人类型
        task: 任务名称
        mode: 图像存储模式
        dataset_config: 数据集配置
    """
    
    episodes = sorted(
        d for d in os.listdir(raw_dir / "temp_images")
        if d.startswith("episode_")
    )
    episodes = episodes[:-1]
    dataset = create_empty_dataset(
        repo_id=repo_id,
        root=str(dst_dir / repo_id),
        robot_type=robot_type,
        mode=mode,
        dataset_config=dataset_config,
    )

    dataset, summary_stats = populate_dataset(
        dataset,
        root_path=str(raw_dir),
        task=task,
        ep_files=episodes,
        debug_dir=str(dst_dir / repo_id / "debug"),
        action_gripper=action_gripper_mode,
    )


    # ===== 生成汇总 CSV =====
    summary_csv_path = os.path.join(dst_dir / repo_id, "summary_max_diffs.csv")
    pd.DataFrame(summary_stats).to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV → {summary_csv_path}")
    
    # dataset.finalize()


if __name__ == "__main__":
    tyro.cli(port_keenon)

