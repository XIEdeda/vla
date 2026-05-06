import dataclasses
import logging
import socket
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow
import tyro
import random
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    dataset_repo_id: str = (
        "/mnt/datas/vla_datasets/fold_clothes/022_20260123_cyw_fold_clothes_5_1"  # LeRobot数据集路径或ID
    )
    port: int = 5555
    delay: float = 0.5  # 秒级延时
    chunk_size: int = 50
    mode: str = "lerobot"


# class ExpertDatasetPolicyOld:
#     def __init__(self, dataset_repo_id: str, chunk_size: int, delay: float):
#         self.dataset = LeRobotDataset(dataset_repo_id)
#         self.chunk_size = chunk_size
#         self.delay = delay
#         self.episode_indices = self.dataset.episode_data_index
#         self.num_episodes = self.dataset.num_episodes

#         self.current_episode_id = -1
#         self.end_frame = 0
#         self.ptr = 0
#         self.last_action = np.zeros(16)  # 用于缓存最后一帧

#     def reset_session(self):
#         """
#         只有新连接建立时才会选择新episode
#         """
#         self.current_episode_id = random.randint(0, self.num_episodes - 1)
#         # self.current_episode_id = 44
#         self.ptr = self.episode_indices["from"][self.current_episode_id].item()
#         self.end_frame = self.episode_indices["to"][self.current_episode_id].item()

#         self.last_action = self.dataset[self.end_frame - 1]["action"].numpy()

#         logging.info(
#             f"[Session Start] Episode: {self.current_episode_id} | Range: {self.ptr}-{self.end_frame}"
#         )

#     def infer(self, observation: dict) -> dict:
#         actual_read_start = self.ptr

#         num_to_read = min(self.chunk_size, self.end_frame - self.ptr)

#         actions = []
#         if num_to_read > 0:
#             batch_data = self.dataset.hf_dataset.select(
#                 range(self.ptr, self.ptr + num_to_read)
#             )
#             batch_actions = list(batch_data["action"])

#             actions = torch.stack(batch_actions).numpy()
#             actions[:, 7] = actions[:, 7] * 100
#             actions[:, 15] = actions[:, 15] * 100
#             actions[:, 7][actions[:, 7] < 2] = 0
#             actions[:, 15][actions[:, 15] < 2] = 0
#             actions = actions.tolist()

#             self.ptr += num_to_read
#             self.last_action = actions[-1]

#         while len(actions) < self.chunk_size:
#             actions.append(self.last_action)

#         actions = np.array(actions, dtype=np.float32)

#         if actual_read_start < self.end_frame:
#             logging.info(
#                 f"Episode {self.current_episode_id}: Progress {self.ptr}/{self.end_frame}"
#             )
#         time.sleep(self.delay)
#         return {"actions": actions}

#     @property
#     def metadata(self):
#         return {
#             "action_shape": (self.chunk_size, 16),
#             "observation_shape": {},
#         }


class LerobotExpertDatasetPolicy:
    def __init__(
        self, dataset_repo_id: str, chunk_size: int, delay: float, verbose=True
    ):
        self.verbose = verbose
        self.repo_dir = Path(dataset_repo_id)
        self.chunk_size = chunk_size
        self.delay = delay

        self.episode_files = sorted(
            list(self.repo_dir.rglob("data/chunk-*/episode_*.parquet"))
        )
        if not self.episode_files:
            self.episode_files = sorted(
                list(self.repo_dir.rglob("data/episode_*.parquet"))
            )

        self.num_episodes = len(self.episode_files)
        print("Total num episodes: ", self.num_episodes)

        self.current_episode_id = -1
        self.current_episode_actions = None
        self.ptr = 0
        self.end_frame = 0
        self.last_action = np.zeros(16)

    def reset_session(self):
        t0 = time.time()
        # self.current_episode_id = random.randint(0, self.num_episodes - 1)
        self.current_episode_id = 25
        target_file = self.episode_files[self.current_episode_id]

        try:
            df = pd.read_parquet(target_file, columns=["action", "timestamp"])
        except (pyarrow.lib.ArrowInvalid, KeyError, ValueError):
            df = pd.read_parquet(target_file, columns=["actions", "timestamp"])
            df = df.rename(columns={"actions": "action"})

        # timestamps = df["timestamp"].values
        # if len(timestamps) > 1:
        #     # np.diff 计算的是 t[n] - t[n-1]
        #     avg_diff = np.diff(timestamps).mean()
        #     print(f"[Timer] Average step duration: {avg_diff:.4f}s (FPS: {1/avg_diff:.1f})")
        # else:
        #     print("[Timer] Average step duration: N/A (Only 1 frame)")

        t1 = time.time()
        actions_all = np.stack(df["action"].values).astype(np.float32)

        # actions_all[:, 7] *= 100
        # actions_all[:, 15] *= 100
        actions_all[:, 7][actions_all[:, 7] < 2] = 0
        actions_all[:, 15][actions_all[:, 15] < 2] = 0

        self.current_episode_actions = actions_all
        self.ptr = 0
        self.end_frame = len(actions_all)
        self.last_action = actions_all[-1]
        t2 = time.time()

        if self.verbose:
            print(
                f"[Timer] reset_session: total={t2-t0:.4f}s (io={t1-t0:.4f}s, proc={t2-t1:.4f}s)"
            )

        logging.info(
            f"[Session Start] Episode File: {target_file.name} | Total Frames: {self.end_frame}"
        )

    def infer(self, observation: dict) -> dict:
        t0 = time.time()
        if self.current_episode_actions is None:
            raise RuntimeError("请先调用 reset_session() 初始化 Episode")

        actual_read_start = self.ptr
        num_to_read = min(self.chunk_size, self.end_frame - self.ptr)
        actions_out = np.tile(self.last_action, (self.chunk_size, 1))

        if num_to_read > 0:
            actions_out[:num_to_read] = self.current_episode_actions[
                self.ptr : self.ptr + num_to_read
            ]
            self.ptr += num_to_read
            self.last_action = actions_out[num_to_read - 1]

        if actual_read_start < self.end_frame:
            logging.info(
                f"Episode Index {self.current_episode_id}: Progress {self.ptr}/{self.end_frame}"
            )

        t1 = time.time()
        if self.verbose:
            print(f"[Timer] infer: {t1-t0:.4f}s")

        time.sleep(self.delay)
        return {"actions": actions_out}

    @property
    def metadata(self):
        return {
            "action_shape": (self.chunk_size, 16),
            "observation_shape": {},
        }


class RawExpertDatasetPolicy:
    def __init__(
        self, dataset_repo_id: str, chunk_size: int, delay: float, verbose=False
    ):
        self.verbose = verbose
        self.repo_dir = Path(dataset_repo_id)
        self.chunk_size = chunk_size
        self.delay = delay

        # 搜索所有的 episode 文件夹
        self.episode_dirs = sorted(list((self.repo_dir / "arm_data").glob("episode_*")))
        self.num_episodes = len(self.episode_dirs)
        print("Total num episodes: ", self.num_episodes)

        self.current_episode_id = -1
        self.current_episode_actions = None
        self.ptr = 0
        self.end_frame = 0
        self.last_action = np.zeros(16)

    def _parse_txt(self, file_path):
        """解析单臂数据：7个关节 + 第9个数字(夹爪值)"""
        data = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                # 取时间戳后面的逗号分隔数值
                values = [float(x) for x in parts[1].split(",")]
                # 提取：7个关节(values[0:7]) + 夹爪开合值(values[8])
                arm_action = values[0:7] + [values[8]]
                data.append(arm_action)
        return np.array(data, dtype=np.float32)

    def reset_session(self):
        t0 = time.time()
        # self.current_episode_id = random.randint(0, self.num_episodes - 1)
        self.current_episode_id = 25
        target_dir = self.episode_dirs[self.current_episode_id]

        # 读取左右臂数据
        left_file = target_dir / "left_data.txt"
        right_file = target_dir / "right_data.txt"

        t1 = time.time()
        left_actions = self._parse_txt(left_file)
        right_actions = self._parse_txt(right_file)

        # 确保左右臂行数对齐（取最小值防止越界）
        min_frames = min(len(left_actions), len(right_actions))
        # 拼接成 16 维: [left(8), right(8)]
        actions_all = np.concatenate(
            [left_actions[:min_frames], right_actions[:min_frames]], axis=1
        )

        # 和 LeRobot 一致归一化
        actions_all[:, 7] /= 100
        actions_all[:, 15] /= 100
        actions_all[:, 7][actions_all[:, 7] < 2] = 0
        actions_all[:, 15][actions_all[:, 15] < 2] = 0

        self.current_episode_actions = actions_all
        self.ptr = 0
        self.end_frame = len(actions_all)
        self.last_action = actions_all[-1]
        t2 = time.time()

        if self.verbose:
            print(
                f"[Timer] reset_session (Raw): total={t2-t0:.4f}s (io_parse={t1-t0:.4f}s, proc={t2-t1:.4f}s)"
            )

        logging.info(
            f"[Session Start] Episode Dir: {target_dir.name} | Total Frames: {self.end_frame}"
        )

    def infer(self, observation: dict) -> dict:
        t0 = time.time()
        if self.current_episode_actions is None:
            raise RuntimeError("请先调用 reset_session() 初始化 Episode")

        actual_read_start = self.ptr
        num_to_read = min(self.chunk_size, self.end_frame - self.ptr)
        actions_out = np.tile(self.last_action, (self.chunk_size, 1))

        if num_to_read > 0:
            actions_out[:num_to_read] = self.current_episode_actions[
                self.ptr : self.ptr + num_to_read
            ]
            self.ptr += num_to_read
            self.last_action = actions_out[num_to_read - 1]

        if actual_read_start < self.end_frame:
            logging.info(
                f"Episode Index {self.current_episode_id}: Progress {self.ptr}/{self.end_frame}"
            )

        t1 = time.time()
        if self.verbose:
            print(f"[Timer] infer: {t1-t0:.4f}s")

        time.sleep(self.delay)
        return {"actions": actions_out}

    @property
    def metadata(self):
        return {
            "action_shape": (self.chunk_size, 16),
            "observation_shape": {},
        }


class SessionAwareServer(websocket_policy_server.WebsocketPolicyServer):
    async def _handler(self, websocket):
        if hasattr(self._policy, "reset_session"):
            self._policy.reset_session()
        await super()._handler(websocket)


def main(args: Args) -> None:
    if args.mode == "lerobot":
        policy = LerobotExpertDatasetPolicy(
            dataset_repo_id=args.dataset_repo_id,
            chunk_size=args.chunk_size,
            delay=args.delay,
        )
    else:
        policy = RawExpertDatasetPolicy(
            dataset_repo_id=args.dataset_repo_id,
            chunk_size=args.chunk_size,
            delay=args.delay,
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = SessionAwareServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
