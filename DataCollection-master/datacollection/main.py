from pathlib import Path
import os
import zipfile
from PIL import Image
import numpy as np
import pandas as pd
from pprint import pprint


def _check_empty_image(image_array):
    """
    空图像判定：
    - 全黑: 像素最大值 <= 20 的像素占比 > 99%
    - 全白: 像素最小值 >= 230 的像素占比 > 99%
    返回 (is_empty, reason)
    """
    if image_array.ndim == 2: # depth
        black_ratio = np.mean(image_array <= 100) # 小于10厘米
        white_ratio = 0 # 深度图像全白无法判断
    elif image_array.ndim == 3: # color
        black_ratio = np.mean(np.max(image_array, axis=2) <= 20)
        white_ratio = np.mean(np.min(image_array, axis=2) >= 230)
    else:
        return True, f"图像维度异常: ndim={image_array.ndim}"

    if black_ratio > 0.99:
        return True, f"全黑, {black_ratio:.4f}"

    if white_ratio > 0.99:
        return True, f"全白, {white_ratio:.4f}"

    return False, ""

def _sample_frames(frames, ratio=0.05, min_count=5):
    sample_count = min(len(frames), max(min_count, int(np.ceil(len(frames) * ratio))))
    if sample_count <= 0:
        return []
    if sample_count == len(frames):
        return frames
    sample_indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    return [frames[i] for i in sample_indices]

def _inspect_camera_stream(episode_dir_path, stream_name, suffix, expected_sizes):
    camera_frames = {}
    camera_timestamps = {}
    for camera_name in ["head", "hand_left", "hand_right"]:
        camera_dir = episode_dir_path / "camera" / camera_name / stream_name
        frames = list(camera_dir.glob(f"*.{suffix}"))
        frames.sort(key=lambda x: int(x.stem))
        camera_frames[camera_name] = frames
        camera_timestamps[camera_name] = [int(x.stem) for x in frames]

    # 检查项1-3：图像完整、像素值正常、分辨率正常（抽查5%）
    # for camera_name in ["head", "hand_left", "hand_right"]:
    #     frames = camera_frames[camera_name]
    #     if len(frames) == 0:
    #         return False, f"{stream_name} {camera_name} 相机无图像帧", {}
    #     sampled_frames = _sample_frames(frames, ratio=0.05, min_count=5)
    #     for frame_path in sampled_frames:
    #         try:
    #             image_array = np.array(Image.open(frame_path))
    #         except Exception:
    #             return False, f"{stream_name} 图像无法解码: {camera_name}, {frame_path.name}", {}
    #         is_empty, reason = _check_empty_image(image_array)
    #         if is_empty:
    #             return False, f"{stream_name} 空图像: {camera_name}, {frame_path.name}, {reason}", {}
    #         h, w = image_array.shape[:2]
    #         expected_w, expected_h = expected_sizes[camera_name]
    #         if (w, h) != (expected_w, expected_h):
    #             return False, (
    #                 f"{stream_name} 图像分辨率错误: {camera_name}, {frame_path.name}, "
    #                 f"actual={w}x{h}, expected={expected_w}x{expected_h}"
    #             ), {}

    N_head = len(camera_frames["head"])
    N_left = len(camera_frames["hand_left"])
    N_right = len(camera_frames["hand_right"])

    # 检查项4：整体帧数是否一致
    if N_head == 0:
        return False, f"{stream_name} 头部相机无图像帧", {}
    if abs(N_head-N_left)/N_head > 0.03 or abs(N_head-N_right)/N_head > 0.03:
        return False, f"{stream_name} 相机帧数不一致", {}

    # 检查项5：时间帧连续性
    # head_time_diff = np.diff(camera_timestamps["head"]) / 1_000_000
    # left_time_diff = np.diff(camera_timestamps["hand_left"]) / 1_000_000
    # right_time_diff = np.diff(camera_timestamps["hand_right"]) / 1_000_000

    # head_time_diff_valid_ratio = 1.0 if len(head_time_diff) == 0 else np.sum(head_time_diff <= 35) / len(head_time_diff)
    # left_time_diff_valid_ratio = 1.0 if len(left_time_diff) == 0 else np.sum(left_time_diff <= 35) / len(left_time_diff)
    # right_time_diff_valid_ratio = 1.0 if len(right_time_diff) == 0 else np.sum(right_time_diff <= 35) / len(right_time_diff)
    # if head_time_diff_valid_ratio < 0.98 or left_time_diff_valid_ratio < 0.98 or right_time_diff_valid_ratio < 0.98:
    #     return False, f"{stream_name} 相机时间帧不连续", {}
    return True, "", {
        "head": N_head,
        "hand_left": N_left,
        "hand_right": N_right,
    }

def inspect_camera(episode_dir_path):
    expected_sizes = {
        "head": (1280, 720),
        "hand_left": (640, 480),
        "hand_right": (640, 480),
    }

    # 先分别检查 color 和 depth
    is_valid, info, color_counts = _inspect_camera_stream(
        episode_dir_path=episode_dir_path,
        stream_name="color",
        suffix="jpg",
        expected_sizes=expected_sizes,
    )
    if not is_valid:
        return False, info

    is_valid, info, depth_counts = _inspect_camera_stream(
        episode_dir_path=episode_dir_path,
        stream_name="depth",
        suffix="png",
        expected_sizes=expected_sizes,
    )
    if depth_counts: # 有深度图才需要检查
        if not is_valid:
            return False, info

        # 最后检查 color/depth 帧数一致性
        for camera_name in ["head", "hand_left", "hand_right"]:
            color_n = color_counts[camera_name]
            depth_n = depth_counts[camera_name]
            if abs(color_n - depth_n) / color_n > 0.03:
                return False, (
                    f"color/depth 帧数不一致: {camera_name}, "
                    f"color={color_n}, depth={depth_n}"
                )
    return True, ""

def inspect_joint(episode_dir_path):
    # 检查项：关节状态文件存在、格式正确、时间帧连续
    def _read_joint_data_file(file_path):
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"{file_path.name} 第{line_no}行格式错误，缺少空格分隔")

                ts_str, values_str = parts
                try:
                    timestamp = int(ts_str)
                except ValueError as e:
                    raise ValueError(f"{file_path.name} 第{line_no}行时间戳非法: {ts_str}") from e

                values_text = values_str.split(",")
                if len(values_text) != 16:
                    raise ValueError(
                        f"{file_path.name} 第{line_no}行维度错误: 期望16维，实际{len(values_text)}维"
                    )
                try:
                    values = [float(x) for x in values_text]
                except ValueError as e:
                    raise ValueError(f"{file_path.name} 第{line_no}行存在非法数值") from e

                rows.append([timestamp] + values)

        columns = ["timestamp"] + [f"raw_{i}" for i in range(16)]
        joint_data = pd.DataFrame(rows, columns=columns)
        timestamps = joint_data["timestamp"].values
        time_diff = np.diff(timestamps) / 1_000_000
        time_diff_valid_ratio = 1.0 if len(time_diff) == 0 else np.sum(time_diff <= 15) / len(time_diff)
        if time_diff_valid_ratio < 0.99:
            raise ValueError("关节时间帧不连续")
        return joint_data

    def _check_gripper_values(data, side_name):
        # raw_7 为夹爪动作值，raw_8 为夹爪开度(mm)
        action = data["raw_7"].values
        width = data["raw_8"].values

        # 动作值只允许 0/1
        valid_action = np.isin(action, [0, 1])
        if not np.all(valid_action):
            bad_idx = int(np.where(~valid_action)[0][0])
            bad_val = action[bad_idx]
            return False, f"{side_name} 夹爪动作值非法: timestamp={data["timestamp"][bad_idx]}, value={bad_val}"

        # 开度范围 0~70 mm
        valid_width = (width >= 0) & (width <= 70)
        if not np.all(valid_width):
            bad_idx = int(np.where(~valid_width)[0][0])
            bad_val = width[bad_idx]
            return False, f"{side_name} 夹爪开度越界: timestamp={data["timestamp"][bad_idx]}, value={bad_val:.4f}, range=[0,70]"

        # 动作值与开度变化同步
        if len(width) >= 2:
            prev_w = width[:-1]
            cur_w = width[1:]
            cur_action = action[1:]
            eps_close = 5
            eps_open = 1

            close_mask = (cur_action == 0) & (prev_w + eps_close < cur_w)
            if np.any(close_mask):
                bad_idx = int(np.where(close_mask)[0][0]) + 1
                return False, (
                    f"{side_name} 夹爪动作-开度不同步: timestamp={data["timestamp"][bad_idx]}, action=0, "
                    f"width_prev={prev_w[bad_idx-1]:.4f}, width_cur={cur_w[bad_idx-1]:.4f}"
                )

            open_mask = (cur_action == 1) & (prev_w - eps_open > cur_w)
            if np.any(open_mask):
                bad_idx = int(np.where(open_mask)[0][0]) + 1
                return False, (
                    f"{side_name} 夹爪动作-开度不同步: timestamp={data["timestamp"][bad_idx]}, action=1, "
                    f"width_prev={prev_w[bad_idx-1]:.4f}, width_cur={cur_w[bad_idx-1]:.4f}"
                )

        return True, ""

    left_data_path = episode_dir_path / "record" / "left_data.txt"
    right_data_path = episode_dir_path / "record" / "right_data.txt"
    if not left_data_path.exists() or not right_data_path.exists():
        return False, "关节状态文件不存在"

    try:
        left_data = _read_joint_data_file(left_data_path)
        right_data = _read_joint_data_file(right_data_path)
    except ValueError as e:
        return False, str(e)

    if len(left_data) == 0 or len(right_data) == 0:
        return False, "关节状态文件为空"

    # 联调检查：以 head color 图像时间窗筛选机械臂数据
    # head_color_dir = episode_dir_path / "camera" / "head" / "color"
    # head_color_frames = list(head_color_dir.glob("*.jpg"))
    # if len(head_color_frames) == 0:
    #     return False, "head color 相机无图像帧"
    # head_timestamps = sorted(int(x.stem) for x in head_color_frames)
    # t_start = head_timestamps[0]
    # t_end = head_timestamps[-1]
    # n_head = len(head_timestamps)

    # left_win = left_data[(left_data["timestamp"] >= t_start) & (left_data["timestamp"] <= t_end)].sort_values("timestamp")
    # right_win = right_data[(right_data["timestamp"] >= t_start) & (right_data["timestamp"] <= t_end)].sort_values("timestamp")
    # n_left = len(left_win)
    # n_right = len(right_win)
    # if n_left == 0 or n_right == 0:
    #     return False, f"机械臂数据不在相机时间窗内: left={n_left}, right={n_right}"

    # 单路图像与机械臂数据比例：head/left ≈ head/right <= 5/14
    # max_ratio = 5 / 14
    # ratio_left = n_head / n_left
    # ratio_right = n_head / n_right
    # if ratio_left > max_ratio or ratio_right > max_ratio:
    #     return False, (
    #         "图像-机械臂数据量比例超阈值: "
    #         f"head/left={ratio_left:.4f}, head/right={ratio_right:.4f}, limit={max_ratio:.4f}"
    #     )
    # ratio_diff = abs(ratio_left - ratio_right)
    # if ratio_diff > 0.035:
    #     return False, (
    #         "左右机械臂比例不一致: "
    #         f"head/left={ratio_left:.4f}, head/right={ratio_right:.4f}, diff={ratio_diff:.4f}"
    #     )

    # 夹爪动作值、开度值、动作-开度同步
    # ok, msg = _check_gripper_values(left_win.reset_index(drop=True), "left")
    # if not ok:
    #     return False, msg
    # ok, msg = _check_gripper_values(right_win.reset_index(drop=True), "right")
    # if not ok:
    #     return False, msg

    return True, ""

# 测试部分，正式使用时请注释掉
# os.environ["INPUT_PATH"] = "test_data.zip"
# os.environ["OUTPUT_PATH"] = "result.txt"

INPUT_PATH = Path(os.getenv("INPUT_PATH"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH"))
DATA_PATH = Path("/workspace/data")

print(INPUT_PATH)
print(OUTPUT_PATH)

# 解压
with zipfile.ZipFile(INPUT_PATH, "r") as zip_ref:
    zip_ref.extractall(DATA_PATH)

results = []

# TODO: 一个压缩文件里是否只有一个task 目录
task_dirs = list(os.listdir(DATA_PATH))
for task_dir in task_dirs:
    task_dir_path = DATA_PATH / task_dir
    for episode_dir in os.listdir(task_dir_path):
        if episode_dir=="task_info":
            continue
        episode_dir_path = task_dir_path / episode_dir
        is_valid = True
        info = ""
        for inspect_func in [inspect_camera, inspect_joint]:
            is_valid_cur, info_cur = inspect_func(episode_dir_path)
            if not is_valid_cur:
                is_valid = False
                info = info_cur
                break
        results.append({
            "sub_task_id": int(episode_dir),
            "result": "pass" if is_valid else "failed",
            "fail_error": info,
        })

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, sep="\t")