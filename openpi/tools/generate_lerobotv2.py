"""
Script to convert keenon data to the LeRobot dataset v2.0 format.

Example usage: python generate_lerobotv2.py \
                      --raw-dir /path/to/raw/data \
                      --repo-id <dataset-name> \
                      --dst_dir /path/to/result/data/ \
                      --robot_type <robot-name> \
                      --task "DEBUG"
# python generate_lerobotv2.py --raw-dir /root/wyx/VLA-dataset/lerobot_data_test/dataset_src \
                      --repo-id keenon_dataset1114 \
                      --dst_dir  /root/wyx/VLA-dataset/lerobot_data_test/ \
                      --robot_type keenon_arm
                      --task "Grab the cup and pour the popcorn onto the plate"
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 20
    image_writer_threads: int = 10
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    root: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    fps: int = 30,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # if not os.path.exists(root):
    #     os.makedirs(root)
    motors = [
        "joint0",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper",
        "right_joint0",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
        "right_gripper",

    ]
    cameras = [
        "head_image",
        "left_wrist_image",
        "right_wrist_image",
    ]

    features = {
        "state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ]
        },
        "actions": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ]
        },
    }
    
    for cam in cameras:
        features[f"{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root= root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_lerobot_video(video_path, episode_name, frame_index, cameras: list[str]):
    imgs_per_cam = {}
    for camera in cameras:
        video_file = os.path.join(video_path, f"observation.images.{camera}/{episode_name}.mp4")
        print("video:", video_file)
        image_array = extract_frames_opencv(video_file, frame_index)
        imgs_per_cam[camera] = image_array
    return imgs_per_cam


def extract_frames_opencv(video_path, frame_indices):
    """
    读取视频帧
    """
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return []
    
    imgs_array = []
    valid_indices = sorted([idx for idx in frame_indices if idx >= 0])
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 过滤有效帧号
    valid_indices = [idx for idx in valid_indices if idx < total_frames]
    
    current_frame = -1
    
    for frame_idx in valid_indices:
        # 如果需要的帧在当前帧之后，可以顺序读取
        if frame_idx == current_frame + 1:
            ret, frame = cap.read()
            current_frame += 1
        else:
            # 否则跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            current_frame = frame_idx
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs_array.append(frame_rgb)
        else:
            print(f"警告：无法读取第 {frame_idx} 帧")
    
    cap.release()
    return np.array(imgs_array) if imgs_array else []


def extract_and_normalize(arr, cols_keep, col_norm):
    others = arr[:, cols_keep]
    norm = arr[:, col_norm] / 1000.0
    norm = np.clip(norm, 0.0, 1.0)
    print("others = ", type(others), others.shape)
    print("norm = ", type(norm), norm.shape)


    return np.concatenate([others, norm], axis=1)


def unwrap_parquet_array(pd_data_np):
    # """ pd.read data to 2D numpy"""
    # pd_data_np = pd_data.to_numpy()

    if isinstance(pd_data_np[0], (list, np.ndarray)) and isinstance(pd_data_np[0][0], np.ndarray):
        return np.stack([x[0] for x in pd_data_np])

    if pd_data_np.dtype == object:
        return np.stack(pd_data_np)

    return pd_data_np


def load_raw_lerobot_data(
    ep_path: Path,
    video_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    state = pd.read_parquet(ep_path, columns=["state"])
    action = pd.read_parquet(ep_path, columns=["actions"])
    frame_index = pd.read_parquet(ep_path, columns=["frame_index"])
    # print(f"state:{type(state)}, {state}")
    # print(f"action:{type(action)}, {action}")
    
    state_col = state.to_numpy()
    state_arr = unwrap_parquet_array(state_col)
    
    action_col = action.to_numpy()
    action_arr = unwrap_parquet_array(action_col)
    
    joint_cols = [0, 1, 2, 3, 4, 5, 6]     # joint1-7
    gripper_bin_cols = [7]
    left_gripper_bin_cols = [14]
    right_gripper_bin_cols = [22] 
    right_gripper_bin_cols_width = [23]
    right_cols = [7,8,9,10,11,12,13]
    gripper_width_cols = [8]
    pose_cols = [9, 10, 11, 12, 13, 14]  # ee pose
    
    # for sigle arm ont include gripper
    # new_state_arr = state_arr[:, joint_cols]
    # new_action_arr = action_arr[:, joint_cols]

    # 提取7Dof + gripper open-1\close-0 
    # new_state_arr = state_arr[:, joint_cols + left_gripper_bin_cols+right_cols+right_gripper_bin_cols]
    # new_action_arr = action_arr[:, joint_cols + left_gripper_bin_cols+right_cols+right_gripper_bin_cols]

    # 提取7Dof + gripper open-1\close-0 
    new_state_arr = state_arr[:, joint_cols + left_gripper_bin_cols+right_cols+right_gripper_bin_cols_width]
    if np.any(new_state_arr[:,15]>2):
        new_state_arr[:,15] = new_state_arr[:,15]/1000
    new_action_arr = action_arr[:, joint_cols + left_gripper_bin_cols+right_cols+right_gripper_bin_cols_width]
    if np.any(new_action_arr[:,15]>2):
        new_action_arr[:,15] = new_action_arr[:,15]/1000
    
    # 提取7Dof + gripper width
    # new_state_arr = extract_and_normalize(state_arr, joint_cols, gripper_width_cols)
    # new_action_arr = extract_and_normalize(action_arr, joint_cols, gripper_width_cols)
    
    # print("new_state_arr:", type(new_state_arr), new_state_arr.shape, new_state_arr)
    
    state_tensor = torch.from_numpy(new_state_arr)
    action_tensor =  torch.from_numpy(new_action_arr)
    # print("state:", type(state_tensor), state_tensor.shape, state_tensor)
    
    # print(type(state_arr), state_arr)
    frame_index_arr = frame_index.to_numpy()
    # print(type(frame_index_arr), frame_index_arr)
    frame_index_list = frame_index_arr.squeeze().tolist()
    # print(frame_index_list)
    
    episode_name = os.path.basename(ep_path).split(".")[0]
    print("episode_name:", episode_name)
    imgs_per_cam = load_lerobot_video(
            video_path,
            episode_name,
            frame_index_list,
            [
                "head_image",
                "left_wrist_image",
                "right_wrist_image",
            ],
        )


    return imgs_per_cam, state_tensor, action_tensor


def populate_dataset(
    dataset: LeRobotDataset,
    task: str,
    video_dir: Path,
    ep_files: list[Path],
) -> LeRobotDataset:

    episodes = range(len(ep_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = ep_files[ep_idx]
        video_path = video_dir
        imgs_per_cam, state, action = load_raw_lerobot_data(ep_path, video_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            # print(f"state{i}, {type(state[i])}, {state[i].shape}, {state[i]}")
            frame = {
                "task": task,
                "state": state[i],
                "actions": action[i],
            }
            
            for camera, img_array in imgs_per_cam.items():
                frame[f"{camera}"] = img_array[i]
            
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_keenon(
    raw_dir: Path,
    repo_id: str, # "dataset113_test"
    dst_dir: Path, # "/root/wyx/test_lerobot"
    robot_type: str = "Keenon_arm",
    task: str = "pick up the cup and pour popcorn into it",
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):  
    # raw_dir = "/media/xiededa/PortableSSD/vla_datas/test1124"
    raw_dir = raw_dir
    # repo_id = "test1124_left_arm"
    repo_id =repo_id
    # dst_dir = "/media/xiededa/PortableSSD/vla_datas"
    dst_dir = dst_dir
    ep_dir = Path(os.path.join(raw_dir, "data/chunk-000"))
    video_dir = os.path.join(raw_dir, "videos/chunk-000") 
    ep_files = sorted(ep_dir.glob("episode_*.parquet"))
    print(f"ep_files:{ep_files}")
    dataset = create_empty_dataset(
        repo_id,
        root=os.path.join(dst_dir, repo_id),
        robot_type=robot_type,
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        task,
        video_dir,
        ep_files,
    )
    dataset.finalize()


if __name__ == "__main__":
    # tyro.cli(port_keenon)
    port_keenon(raw_dir = "/root/xie/dual_arm_wyx_1204",
    repo_id = "dual_arm_train_1211",
    dst_dir = "/mnt/nvme1n1/vla_datas")


