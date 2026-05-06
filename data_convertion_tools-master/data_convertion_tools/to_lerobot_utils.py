#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import dataclasses
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from typing import Dict, List, Tuple, Sequence, Literal
import math
import numpy as np
import matplotlib.pyplot as plt

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = os.cpu_count()
    image_writer_threads: int = os.cpu_count()
    video_backend: str | None = None # todo


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    root: str | Path,
    robot_type: str,
    MOTORS: list[str],
    IMAGE_SHAPE: tuple[int],
    BUILDFUTURETRAJ: bool = False,
    CHUNKSIZE: int = 1,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    if BUILDFUTURETRAJ and CHUNKSIZE > 1:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(MOTORS),),
                "names": [
                    MOTORS,
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (CHUNKSIZE, len(MOTORS),),
                "names": [
                    MOTORS,
                ],
            },
        }
    else:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(MOTORS),),
                "names": [
                    MOTORS,
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(MOTORS),),
                "names": [
                    MOTORS,
                ],
            },
        }

    # if has_velocity:
    #     features["observation.velocity"] = {
    #         "dtype": "float32",
    #         "shape": (len(MOTORS),),
    #         "names": [
    #             MOTORS,
    #         ],
    #     }

    # if has_effort:
    #     features["observation.effort"] = {
    #         "dtype": "float32",
    #         "shape": (len(MOTORS),),
    #         "names": [
    #             MOTORS,
    #         ],
    #     }

    cameras = [
        "head",
        "hand_left",
        "hand_right",
        # "cam_left_head",
        # "cam_right_head",
        # "cam_left_wrist",
        # "cam_right_wrist",
    ]

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (IMAGE_SHAPE[1], IMAGE_SHAPE[0], 3),
            "names": [
                "height",
                "width",
                "rgb",
            ],
            # "shape": (3, IMAGE_SHAPE[1], IMAGE_SHAPE[0]),
            # "names": [
            #     "channels",
            #     "height",
            #     "width",
            # ],
        }

    if Path(root).exists():
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos, # todo: check = True?
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads, # todo: print
        video_backend=dataset_config.video_backend,
    )

# -----------------------------------------------------------------------------------------------------------------------------------

def binary_search(arr, key, timestamp):
    left, right = 0, len(arr) -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid][key] >= timestamp:
            right = mid - 1
        else:
            left = mid + 1
    return left

def extract_data(data, topic, timestamp):
    idx = binary_search(data[topic], "timestamp", timestamp)
    if idx == 0:
        # print(topic + " timestamp:", data[topic][idx]["timestamp"])
        pass
    elif 0 < idx < len(data[topic]):
        if abs(data[topic][idx - 1]["timestamp"] - timestamp) < abs(data[topic][idx]["timestamp"] - timestamp):
            # print(topic + " timestamp:", data[topic][idx - 1]["timestamp"])
            idx = idx - 1
        else:
            # print(topic + " timestamp:", data[topic][idx]["timestamp"])
            pass
    else:
        # print(topic + " timestamp:", data[topic][idx - 1]["timestamp"])
        idx = idx - 1
    return data[topic][idx]

# -----------------------------------------------------------------------------------------------------------------------------------

def delete_bag_file(bag_file_path):
    parent_dir = os.path.dirname(bag_file_path)
    json_path = parent_dir + ".json"
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
        except FileNotFoundError:
            pass
    if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
        try:
            shutil.rmtree(parent_dir)
        except FileNotFoundError:
            pass
                
def plot_and_save_hd(record,
                     MOTORS,
                     out_path="test.png",
                     dpi=300,
                     figsize=(20, 24),
                     suptitle="Record trajectories",
                     fmt=None):
    """
    Plot subplots from record and save to a high-resolution image.

    Parameters:
    - record: dict, key -> value, value: [traj0,...,traj13, time]
    - out_path: output file path (if fmt provided, that format used; else inferred from ext)
    - dpi: dots per inch for saved image (e.g. 300, 600, 1200)
    - figsize: tuple (width_in_inches, height_in_inches)
    - suptitle: overall title
    - fmt: optional override format, e.g. "png", "tiff"
    """
    if not record:
        raise ValueError("Empty record")

    n_sub = None
    for key, value in record.items():
        if not n_sub:
            n_sub = len(value) - 1
        else:
            assert n_sub == len(value) - 1

    if not n_sub or n_sub < 0:
        raise ValueError("n_sub error")
    
    ncols = 2
    nrows = math.ceil(n_sub / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=False)
    axes = axes.flatten() # 将返回的 axes（形状为 (nrows, ncols) 的数组）展平成一维数组，便于按索引访问每个子图（axes[i]）

    # Prepare color map
    keys = list(record.keys())
    cmap = plt.get_cmap("tab20") # 选择 matplotlib 的 colormap "tab20"，一个有最多 20 个区分颜色的调色板
    key_colors = {k: cmap(i % cmap.N) for i, k in enumerate(keys)} # 为每个 key 分配一个颜色

    plotted_any = [False] * n_sub # 初始化一个布尔列表，用于记录每个子图（0..n_sub-1）是否绘制过至少一条曲线。后面用于决定是否显示图例或显示 “no data”

    for key in keys:
        value = record[key]
        
        t_raw = [elem for elem in value[-1]]
        # 绝对时间转相对时间(从零开始)
        for idx in range(len(t_raw) - 1, -1, -1):
            t_raw[idx] -= t_raw[0]
        
        try:
            t = np.asarray(t_raw, dtype=float)
        except Exception:
            t = np.arange(len(t_raw))

        for i in range(n_sub):
            traj = value[i]
            if traj is None:
                continue
            y = np.asarray(traj, dtype=float)
            # align lengths
            assert y.shape[0] == t.shape[0]

            ax = axes[i]
            # label = os.path.basename(key) if isinstance(key, str) else str(key)
            ax.plot(t, y, color=key_colors.get(key))
            plotted_any[i] = True

    # Finalize subplots
    LEFT_DOF=sum('left' in m for m in MOTORS)
    for idx in range(n_sub):
        ax = axes[idx]
        
        if idx == 0:
            ax.set_title("left gripper feedback", fontsize=10)
        elif idx == 1:
            ax.set_title("right gripper feedback", fontsize=10)
        elif 2 <= idx < 2 + LEFT_DOF:
            ax.set_title(f"{idx - 2}-th left arm feedback", fontsize=10)
        elif 2 + LEFT_DOF <= idx < len(MOTORS):
            ax.set_title(f"{idx - 2 - LEFT_DOF}-th right arm feedback", fontsize=10)
        elif idx == len(MOTORS):
            ax.set_title("left gripper target", fontsize=10)
        elif idx == len(MOTORS) + 1:
            ax.set_title("right gripper target", fontsize=10)
        elif len(MOTORS) + 2 <= idx < len(MOTORS) + 2 + LEFT_DOF:
            ax.set_title(f"{idx - len(MOTORS) - 2}-th left arm target", fontsize=10)
        elif len(MOTORS) + 2 + LEFT_DOF <= idx < len(MOTORS) * 2:
            ax.set_title(f"{idx - len(MOTORS) - 2 - LEFT_DOF}-th right arm target", fontsize=10)

        ax.grid(True, linewidth=0.5, alpha=0.6)
        if plotted_any[idx]:
            # smaller legend to avoid clutter
            ax.legend(fontsize="x-small", loc="best", framealpha=0.7)
        else: # 若没有数据
            ax.set_xticks([]) # 隐藏刻度
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray", transform=ax.transAxes)

    # 在整张图上方设置一个超级标题
    fig.suptitle(suptitle, fontsize=16)

    # Improve layout: use constrained_layout or tight_layout
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    except Exception:
        pass

    # Determine format
    if fmt is None:
        _, ext = os.path.splitext(out_path)
        fmt = ext.lstrip(".") if ext else "png"
    # Ensure parent dir exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    # Save with high DPI and tight bbox
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.1}
    # recommended formats: png, tiff (uncompressed supported by matplotlib if Pillow installed)
    print(f"Saving figure to {out_path} at {dpi} DPI (figsize={figsize} inches).")
    fig.savefig(out_path, format=fmt, **save_kwargs)

    plt.close(fig)
    print("Saved successfully.")

# -----------------------------------------------------------------------------------------------------------------------------------

def parallel_two_ends_alternate(
    arrs: List[List[float]],
    thresholds: List[float],
    min_len: int = 2,
    start_from_left: bool = True
) -> List[Tuple[int, int]]:
    """
    并行处理多个等长数组：在全局 [L,R] 上交替从两端扩展找到公共子区间，
    要求对所有数组 i: max(arrs[i][interval]) - min(arrs[i][interval]) <= thresholds[i].
    过滤掉长度 < min_len（默认 2）。
    返回被选子区间列表 [(start, end), ...]（0-based，闭区间），按发现顺序。
    """
    if not arrs:
        return []

    m = len(arrs)
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("All arrays must have the same length")

    if len(thresholds) != m:
        raise ValueError("thresholds length must match number of arrays")

    res: List[Tuple[int, int]] = []
    left, right = 0, n - 1
    take_from_left = start_from_left

    # 主循环：交替从左右两端尝试扩展，直到 left > right
    while left <= right:
        if take_from_left:
            # 从 left 向右扩展，寻找最大的 end (<= right)
            j = left

            while j <= right:
                ok = True
                # 更新每个数组的队列并检查约束
                for k in range(m):
                    temp = arrs[k][left : j + 1]
                    # print("-> j =", j, ", k =", k,", arr =", temp, ", diff:", max(temp) - min(temp), "vs", thresholds[k])
                    if max(temp) - min(temp) > thresholds[k]:
                        ok = False
                        break  # 任何一个数组不满足都不能继续扩展
                if ok:
                    j += 1
                else:
                    break

            # print("left =", left, ", j =", j, "|", (left, j))
            if (j - left) >= min_len:
                res.append((left, j - 1)) # 前闭后闭区间

            left = j
            take_from_left = False

        else:
            # 从 right 向左扩展, 寻找最小的 start (>= left)
            j = right

            while j >= left:
                ok = True
                for k in range(m):
                    temp = arrs[k][j: right + 1]
                    # print("<- arr =", temp, ", k =", k, ", diff:", max(temp) - min(temp), "vs", thresholds[k])
                    if max(temp) - min(temp) > thresholds[k]:
                        ok = False
                        break
                if ok:
                    j -= 1
                else:
                    break

            # print("right =", right, ", j =", j, "|", (j + 1, right + 1))
            if (right - j) >= min_len:
                res.append((j + 1, right)) # 前闭后闭区间

            right = j
            take_from_left = True

    res.sort()
    return res