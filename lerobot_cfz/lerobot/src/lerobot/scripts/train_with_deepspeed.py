#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""
# 激活虚拟环境
conda activate lerobot-2.1
cd ~/vla/lerobot
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.53.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate==1.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install deepspeed==0.17.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行 deepspeed 版

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
# # PyTorch CUDA 架构列表（只保留需要的架构，例如 8.9）
# export TORCH_CUDA_ARCH_LIST="8.9"
# # 强制启用 CUDA（按需）
# export FORCE_CUDA=1

单机
torchrun --standalone --nproc_per_node=1 src/lerobot/scripts/train_with_deepspeed.py \
--policy.path=/home/user/vla/hf_models/pi0 \
--dataset.repo_id=/home/user/vla/keenon/20260408.coffee.by.cfz.img.224x168.N.dof.16

在 master (172.16.10.70) 上运行：
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=172.16.10.70 --master_port=11212 src/lerobot/scripts/train_with_deepspeed.py \
--policy.path=/home/user/vla/hf_models/pi0 \
--dataset.repo_id=/home/user/vla/keenon/20260408.coffee.by.cfz.img.224x168.N.dof.16

在 worker (172.16.9.50)上运行：
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=172.16.10.70 --master_port=11212 src/lerobot/scripts/train_with_deepspeed.py \
--policy.path=/home/user/vla/hf_models/pi0 \
--dataset.repo_id=/home/user/vla/keenon/20260408.coffee.by.cfz.img.224x168.N.dof.16

Common configurations:
batch_size      (defined in src/lerobot/configs/train.py)
save_freq       (defined in src/lerobot/configs/train.py)
max_state_dim   (defined in src/lerobot/policies/pi0/configuration_pi0.py)
"""

import logging
import time
from typing import Any, Optional
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.train_utils import load_training_state, update_last_checkpoint
from lerobot.utils.utils import format_big_number, init_logging
from lerobot.utils.wandb_utils import WandBLogger

from transformers import Trainer, TrainingArguments, TrainerCallback
import json, os, math, datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class HFTrainer(Trainer):
    def __init__(
        self,
        cfg,
        train_tracker: "MetricsTracker",
        wandb_logger: Optional["WandBLogger"],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.train_tracker = train_tracker
        self.wandb_logger = wandb_logger
 
        self._last_output_dict = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", getattr(self.args, "local_rank", 0)))
        self.device = torch.device(f"cuda:{self.local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        self.global_rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else int(os.environ.get("RANK", "0")) 
        self.num_gpus = int(os.environ.get("WORLD_SIZE", "1"))


    def training_step(self, model, inputs, *args, **kwargs):
        start_time = time.perf_counter()
        model.train()
        
        # ========== Forward ==========

        param_dtype = torch.float32
        if self.args.deepspeed["fp16"]["enabled"] == True:
            param_dtype = torch.float16
        if self.args.deepspeed["bf16"]["enabled"] == True:
            param_dtype = torch.bfloat16
        
        # todo
        # 尽量使用 Trainer 的 _prepare_inputs 来处理 inputs
        # inputs = self._prepare_inputs(inputs)

        def _cast_inputs_to_dtype_and_device(obj, dtype, device):
            if isinstance(obj, torch.Tensor):
                if obj.dtype.is_floating_point:
                    return obj.to(device=device, dtype=dtype, non_blocking=True)
                else:
                    return obj.to(device=device, non_blocking=True)
            elif isinstance(obj, dict):
                return {k: _cast_inputs_to_dtype_and_device(v, dtype, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_cast_inputs_to_dtype_and_device(x, dtype, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(_cast_inputs_to_dtype_and_device(x, dtype, device) for x in obj)
            else:
                return obj

        inputs = _cast_inputs_to_dtype_and_device(inputs, param_dtype, self.device)

        # print("[rank" + str(self.local_rank) + "] --- cfz ---")
        # for key, value in inputs.items():
        #     if isinstance(value, torch.Tensor):
        #         print("[rank" + str(self.local_rank) + "] cfz test 0| key =", key, ", value =", value.shape, value.dtype, value.device)
        #     else:
        #         print("[rank" + str(self.local_rank) + "] cfz test 0| key =", key, ", value =", value)

        loss, output_dict = model(inputs)
        self._last_output_dict = output_dict
        
        # ========== Backward ==========
        model.backward(loss)
   
        # ========== Step ==========
        model.step()
        model.zero_grad()        
        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.step()
            
        # ========== Update metrics tracker ==========
        Time = time.perf_counter() - start_time
        self.train_tracker.update_s = Time
        
        step_info = " | ".join([
            f"loss.dtype: {loss.dtype}", f"zero{self.args.deepspeed['zero_optimization']['stage']}", f"gpus: {self.global_rank}/{self.num_gpus}",
            f"global step: {getattr(self, 'state', None).global_step}", f"loss: {loss.item()}",
            f"batch size: {self.cfg.batch_size}", f"time per batch: {Time}",
            # f"loss.dtype: {loss.dtype}", f"device: {self.device}", f"loss.device: {loss.device}", f"batch size: {self.args.deepspeed['train_micro_batch_size_per_gpu']}"
        ])
        logging.info(step_info)

        self.train_tracker.loss = loss.item()
        self.train_tracker.lr = self.optimizer.param_groups[0]["lr"]
        return loss.detach()


    def get_train_dataloader(self):
        dataset = self.train_dataset
        cfg = self.cfg

        if hasattr(cfg.policy, "drop_n_last_frames"):
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.episode_data_index,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=True,
            )
        else:
            shuffle = True
            sampler = None

        if torch.distributed.is_available() and torch.distributed.is_initialized() and sampler is None:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )


class DeepSpeedCheckpointCallback(TrainerCallback):

    def __init__(self, cfg, update_last_fn=None, trainer=None):
        self.cfg = cfg
        self.update_last_fn = update_last_fn
        self.trainer = trainer


    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            raise RuntimeError("DeepSpeedCheckpointCallback: trainer not available, skipping on_step_end")
        
        global_step = int(getattr(state, "global_step", 0))
        is_log_step = self.cfg.log_freq > 0 and global_step % self.cfg.log_freq == 0
        is_saving_step = (global_step > 0 and global_step % self.cfg.save_freq == 0) or global_step == self.cfg.steps
        
        if is_log_step:
            logging.info(self.trainer.train_tracker)
            if self.trainer.wandb_logger:
                output_dict = self.trainer._last_output_dict
                wandb_log_dict = self.trainer.train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                self.trainer.wandb_logger.log_dict(wandb_log_dict, global_step)
            self.trainer.train_tracker.reset_averages()

        self.trainer.train_tracker.step()

        if not is_saving_step:
            return

        checkpoint_dir = self.cfg.output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
       
        if self.trainer.global_rank == 0:
            try:
                self.cfg.policy.save_pretrained(checkpoint_dir)
                logging.exception(f"config saving to {checkpoint_dir} on rank 0")
            except Exception:
                logging.exception("config failed on rank 0")
                raise
        
        if self.trainer.local_rank == 0:
            assert self.update_last_fn is not None
            try:
                self.update_last_fn(checkpoint_dir)
            except Exception:
                logging.exception("Failed to update last checkpoint symlink on rank 0")

            if self.trainer.wandb_logger is not None:
                try:
                    self.trainer.wandb_logger.log_policy(checkpoint_dir)
                except Exception:
                    logging.exception("wandb_logger.log_policy failed")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def get_synced_time_str(fmt="%Y%m%d%H%M%S"):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ts = datetime.datetime.now().strftime(fmt) if torch.distributed.get_rank() == 0 else None
        obj_list = [ts]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        return obj_list[0]
    else:
        return datetime.datetime.now().strftime(fmt)


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    cfg.validate()
    # logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    # ------------------- 重要参数定义 ------------------------
    cfg.policy.freeze_vision_encoder = cfg.policy.train_expert_only = True
    torch_float = "float32"
    ds_stage = "zero1"
    
    cfg.policy.train_state_proj = True

    # deepspeed stage parameter
    if ds_stage == "zero1":
        ds_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero1.json")
    elif ds_stage == "zero2":
        ds_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero2.json")
    elif ds_stage == "zero2_offload":
        ds_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero2_offload.json")
    elif ds_stage == "zero3":
        ds_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero3.json")
    elif ds_stage == "zero3_offload":
        ds_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config_zero3_offload.json")
    else:
        raise ValueError(
            f"Unsupported DeepSpeed stage '{ds_stage}'. "
            f"Valid options are: zero1, zero2, zero2_offload, zero3, zero3_offload"
        )
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    # torch.float stage parameter
    if torch_float == "float32":
        ds_config["fp16"]["enabled"] = "auto"
        ds_config["bf16"]["enabled"] = "auto"
    elif torch_float == "bfloat16":
        ds_config["fp16"]["enabled"] = "auto"
        ds_config["bf16"]["enabled"] = True
    elif torch_float == "float16":
        ds_config["fp16"]["enabled"] = True
        ds_config["bf16"]["enabled"] = "auto"
    else:
        raise ValueError(
            f"Unsupported torch float '{torch_float}'. "
            f"Valid options are: float32, bfloat16, float16"
        )
    
    # number of GPUs
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))

    # freeze parameters
    label_freeze_or_train = ""
    if cfg.policy.freeze_vision_encoder == True and cfg.policy.train_expert_only == True:
        label_freeze_or_train = "VisionFreezeLLMFreeze"
        if torch_float == "float32":
            if ds_stage == "zero1":
                cfg.batch_size = 1
            elif ds_stage == "zero2":
                cfg.batch_size = 1
            elif ds_stage == "zero2_offload":
                cfg.batch_size = 1
            elif ds_stage == "zero3":
                cfg.batch_size = 1
            elif ds_stage == "zero3_offload":
                cfg.batch_size = 1
        elif torch_float == "bfloat16":
            if ds_stage == "zero1":
                cfg.batch_size = 1
            elif ds_stage == "zero2":
                cfg.batch_size = 1
            elif ds_stage == "zero2_offload":
                cfg.batch_size = 1
            elif ds_stage == "zero3":
                cfg.batch_size = 1
            elif ds_stage == "zero3_offload":
                cfg.batch_size = 1
    elif cfg.policy.freeze_vision_encoder == True and cfg.policy.train_expert_only == False:
        label_freeze_or_train = "VisionFreezeLLMTrain"
        if torch_float == "float32":
            if ds_stage == "zero1":
                cfg.batch_size = 1
            elif ds_stage == "zero2":
                cfg.batch_size = 1
            elif ds_stage == "zero2_offload":
                cfg.batch_size = 1
            elif ds_stage == "zero3":
                cfg.batch_size = 1
            elif ds_stage == "zero3_offload":
                cfg.batch_size = 1
        elif torch_float == "bfloat16":
            if ds_stage == "zero1":
                cfg.batch_size = 1
            elif ds_stage == "zero2":
                cfg.batch_size = 1
            elif ds_stage == "zero2_offload":
                cfg.batch_size = 1
            elif ds_stage == "zero3":
                cfg.batch_size = 1
            elif ds_stage == "zero3_offload":
                cfg.batch_size = 1
    elif cfg.policy.freeze_vision_encoder == False and cfg.policy.train_expert_only == False:
        label_freeze_or_train = "VisionTrainLLMTrain"
        if torch_float == "float32":
            if ds_stage == "zero1":
                cfg.batch_size = 1
            elif ds_stage == "zero2":
                cfg.batch_size = 1
            elif ds_stage == "zero2_offload":
                cfg.batch_size = 1
            elif ds_stage == "zero3":
                cfg.batch_size = 1
            elif ds_stage == "zero3_offload":
                cfg.batch_size = 1
        elif torch_float == "bfloat16":
            if ds_stage == "zero1":
                cfg.batch_size = 1
            elif ds_stage == "zero2":
                cfg.batch_size = 1
            elif ds_stage == "zero2_offload":
                cfg.batch_size = 1
            elif ds_stage == "zero3":
                cfg.batch_size = 1
            elif ds_stage == "zero3_offload":
                cfg.batch_size = 1
    else: # cfg.policy.freeze_vision_encoder == False and cfg.policy.train_expert_only == True
        raise ValueError(
            "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
        )

    ds_config["train_micro_batch_size_per_gpu"] = cfg.batch_size
    ds_config["gradient_clipping"] = cfg.optimizer.grad_clip_norm

    # save path
    dir_suffix = '.'.join([
        label_freeze_or_train,  # freeze parameters
        torch_float,            # torch.float stage parameter
        ds_stage,               # deepspeed stage parameter
        'gpu'+str(num_gpus),    # number of GPUs
        get_synced_time_str(),  # time
    ])

    save_path = ".../checkpoints/outputs/"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if isinstance(cfg.dataset.repo_id, list):
        folder = input("Please enter the folder name to create: ").strip()
        cfg.output_dir = Path(save_path + folder + '.' + dir_suffix)
    else:
        cfg.output_dir = Path(save_path + cfg.dataset.repo_id.split('/')[-1] + '.' + dir_suffix)

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    logging.info("Creating policy")
    cfg.policy.device = "cpu"
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    if torch_float == "float32" and "zero3" in ds_stage:
        policy.to(torch.float32)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    save_total = 5
    cfg.save_freq = math.ceil(dataset.num_frames / cfg.batch_size / num_gpus * 10) # 10 个 epoch 存一次
    cfg.steps = cfg.save_freq * save_total # 共运行 50 个 epoch
    
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        max_steps=cfg.steps,
        logging_steps=cfg.log_freq if cfg.log_freq > 0 else 1000,
        save_steps=cfg.save_freq if cfg.save_freq > 0 else cfg.steps,
        fp16=ds_config["fp16"]["enabled"] == True,
        bf16=ds_config["bf16"]["enabled"] == True,
        max_grad_norm=ds_config["gradient_clipping"],
        gradient_accumulation_steps=1,
        dataloader_drop_last=False,
        dataloader_num_workers=cfg.num_workers,
        save_total_limit=5,
        report_to=["wandb"] if cfg.wandb.enable else [],
        remove_unused_columns=False,
        deepspeed=ds_config,
    )

    step = 0
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    trainer = HFTrainer(
        cfg=cfg,
        train_tracker=train_tracker,
        wandb_logger=wandb_logger,
        model=policy,
        args=training_args,
        train_dataset=dataset,
        optimizers=(optimizer, lr_scheduler),
    )
    cb = DeepSpeedCheckpointCallback(cfg, update_last_fn=update_last_checkpoint, trainer=trainer)
    trainer.add_callback(cb)

    logging.info("Start offline training on a fixed dataset")
    start_time = time.perf_counter()
    trainer.train(resume_from_checkpoint=None)
    end_time = time.perf_counter()
    print(f"#### Time per epoch: {(end_time - start_time) / save_total:.2f} seconds")

    logging.info("End of training")

if __name__ == "__main__":
    init_logging()
    main()