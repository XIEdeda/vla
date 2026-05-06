#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    """
    print("cfz test 2 | ds_meta:", ds_meta)
    cfz test 2 | ds_meta: LeRobotDatasetMetadata({
        Repository ID: '/home/user/vla/keenon/20260408.coffee.by.cfz.img.224x168.N.dof.16',
        Total episodes: '1',
        Total frames: '2574',
        Features: '['observation.state', 'action', 'observation.images.head', 'observation.images.hand_left', 'observation.images.hand_right', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']',
    })',
    print("cfz test 2 | ds_meta.fps:", ds_meta.fps)
    cfz test 2 | ds_meta.fps: 30
    """
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            """
            print("cfz test 2 | cfg.action_delta_indices:", cfg.action_delta_indices)
            cfg.action_delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
            """
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        """
        print("cfz test 0 | delta_timestamps:", delta_timestamps)
        cfz test 0 | delta_timestamps: {
            'action': [0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666, 0.2, 0.23333333333333334, 0.26666666666666666, 0.3, 0.3333333333333333, 0.36666666666666664, 0.4, 0.43333333333333335, 0.4666666666666667, 0.5, 0.5333333333333333, 0.5666666666666667, 0.6, 0.6333333333333333, 0.6666666666666666, 0.7, 0.7333333333333333, 0.7666666666666667, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9, 0.9333333333333333, 0.9666666666666667, 1.0, 1.0333333333333334, 1.0666666666666667, 1.1, 1.1333333333333333, 1.1666666666666667, 1.2, 1.2333333333333334, 1.2666666666666666, 1.3, 1.3333333333333333, 1.3666666666666667, 1.4, 1.4333333333333333, 1.4666666666666666, 1.5, 1.5333333333333334, 1.5666666666666667, 1.6, 1.6333333333333333]
        }
        print("cfz test 0 | cfg.dataset.repo_id:", cfg.dataset.repo_id)
        print("cfz test 0 | cfg.dataset.root:", cfg.dataset.root)
        print("cfz test 0 | cfg.dataset.episodes:", cfg.dataset.episodes)
        print("cfz test 0 | image_transforms:", image_transforms)
        print("cfz test 0 | cfg.dataset.revision:", cfg.dataset.revision)
        print("cfz test 0 | cfg.dataset.video_backend:", cfg.dataset.video_backend)
        cfz test 0 | cfg.dataset.repo_id: /home/user/vla/keenon/20260408.coffee.by.cfz.img.224x168.N.dof.16
        cfz test 0 | cfg.dataset.root: None
        cfz test 0 | cfg.dataset.episodes: None
        cfz test 0 | image_transforms: None
        cfz test 0 | cfg.dataset.revision: None
        cfz test 0 | cfg.dataset.video_backend: torchcodec
        """

        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        """
        for key, value in dataset[len(dataset)-10].items():
            if isinstance(value, torch.Tensor):
                print("cfz test 0 |", key, ":", value.shape)
            else:
                print("cfz test 0 |", key, ":", value)        
        cfz test 0 | observation.images.head : torch.Size([3, 168, 224])
        cfz test 0 | observation.images.hand_left : torch.Size([3, 168, 224])
        cfz test 0 | observation.images.hand_right : torch.Size([3, 168, 224])
        cfz test 0 | observation.state : torch.Size([16])
        cfz test 0 | action : torch.Size([50, 16])
        cfz test 0 | timestamp : torch.Size([])
        cfz test 0 | frame_index : torch.Size([])
        cfz test 0 | episode_index : torch.Size([])
        cfz test 0 | index : torch.Size([])
        cfz test 0 | task_index : torch.Size([])
        cfz test 0 | action_is_pad : torch.Size([50])
        cfz test 0 | task : coffee
        """

    elif isinstance(cfg.dataset.repo_id, list):   
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id[0], root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )
    else:
        raise NotImplementedError("dataset.repo_id only supports a string or a list of strings")

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
