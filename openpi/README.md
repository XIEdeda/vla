## 环境与代码位置
统一使用openpi作为conda环境名

conda环境位置：
+ 训练服务器：/root/miniconda3/envs/openpi
+ 推理服务器：/home/xman/miniforge3/envs/openpi

代码位置：
+ 训练服务器：/root/openpi
+ 推理服务器：/home/xman/openpi


重新安装流程：
```
git clone http://27.159.92.108:3000/KeenonVLA/openpi
cd openpi
conda create -n openpi python=3.12 pip uv -c conda-forge
GIT_LFS_SKIP_SMUDGE=1 uv sync -i https://pypi.tuna.tsinghua.edu.cn/simple
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install chex==0.1.89 pytest==8.3.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

(可选) git submodule update --init --recursive
```

###  openpi环境升级lerobot v3

```
conda activate openpi
conda install ffmpeg=7.1.1 -c conda-forge
pip install torchcodec==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple # 或者载入数据集使用 pyav后端
pip install transformers -U -i https://pypi.tuna.tsinghua.edu.cn/simple
git clone https://github.com/huggingface/lerobot.git # 或者git clone http://27.159.92.108:3000/KeenonVLA/lerobot.git
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
lerobot.common. 改为lerobot.
```

### 可能需要的措施
```
pip install datasets av
pip install torchcodec==0.10.0 # 对应torch 2.10
```

## 训练

### 撰写训练配置
在src/openpi/training/config.py文件中新增一个配置。请放置在Training config部分，例如：
```
    ################################################
    # Training config start
    ################################################
    ...,
    TrainConfig(
        name="pi05_xman_pour_wine",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            ),
        data=LeRobotKeenonXmanDualArmConfig(
            repo_id="/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/019_20260115_lyx_pour_wine_8_3_hzttest",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        val_data=LeRobotKeenonXmanDualArmConfig(
            repo_id="/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/013_20260114_lyx_pour_wine_8_1",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=10000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10000,
        save_interval=1000,
        val_interval=1000,
    ),
    ...
    ################################################
    # Training config end
    ################################################
```
### 开始训练
可参考或使用根目录的train.sh

```
#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7                     # 指定使用哪些GPU
export HF_LEROBOT_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"     # 指定HF数据集缓存目录
export HF_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"             # 指定HF数据集缓存目录
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c" # 指定wandb API KEY
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9                       # 指定预占用显存比例

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi

## 计算数据集的统计数据，每个数据集只需运行一次
# python scripts/compute_norm_stats.py --config-name=pi05_xman_pour_wine

## 开始训练
python scripts/train_val.py pi05_xman_pour_wine \               # 指定训练配置名字
    --exp-name=hzt_experiment_tmp \                             # 指定实验名字
    --checkpoint_base_dir=/mnt/nvme0n1/nvme1n1/hzt/VLA-models/keenon-model \ # 指定模型权重保存路径
    --overwrite \                                               # 是否覆盖保存模型权重
    --fsdp-devices=8                                            # 分布式训练使用的设备数量，需和CUDA_VISIBLE_DEVICES保持一致
```
训练结束后，可在wandb自己的账号查看训练损失曲线。如果设置了val_interval和val_data，那么还可以查看验证集上的损失情况。


## 推理
### 撰写推理配置
在src/openpi/training/config.py文件中新增一个配置。请放置在Inference config部分，例如：
```
    ################################################
    # Inference config start
    ################################################
    ...,
    TrainConfig(
        name="pi05_xman_pour_wine_inference",   # 加_inference后缀
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            ),
        data=LeRobotKeenonXmanDualArmConfig(
            repo_id="/mnt/datas/finetune_model/hzt_experiment_tmp/1000/assets", # 此文件夹放置了norm_stats.json
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=10000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10000,
        save_interval=1000,
    ),
    ...
    ################################################
    # Inference config end
    ################################################
```
推理配置可直接从训练配置复制过来，但有**两个注意事项**：
1. name加上`_inference`后缀以示区分
2. 由于推理主机和训练服务器可能不是同一台，需要将数据集的norm_stats.json复制到推理主机，并对应修改data.repo_id目录

### 传输模型权重
在推理主机上：
```
scp -r root@27.159.92.108:/root/openpi/VLA-models/keenon-model/pi05_xman_pour_wine/hzt_experiment_tmp /mnt/datas/finetune_model/
mkdir /mnt/datas/finetune_model/hzt_experiment_tmp/1000/assets
scp root@27.159.92.108:/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/019_20260115_lyx_pour_wine_8_3_hzttest/norm_stats.json /mnt/datas/finetune_model/hzt_experiment_tmp/1000/assets/
```
传输前，可以删除train_state文件夹

### 启动推理服务
可参考或使用根目录的inference.sh
```
source /home/xman/miniforge3/etc/profile.d/conda.sh
conda activate openpi
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_xman_pour_wine_inference \                 # 配置名字
    --policy.dir=/mnt/datas/finetune_model/hzt_experiment_tmp/1000  # 模型权重
```
启动后，会在推理主机的5555端口启动websocket服务

`source .venv/bin/activate`应该也可以激活环境

## 其他事项
0. 额外的工具脚本统一写入`tools/`文件夹
1. 关于  python scripts/compute_norm_stats.py --config-name pi05_libero 进行数据归一化的一些问题 
    1. 数据得放在 /home/xman/.cache/huggingface/lerobot/lerobot_dataset_0915 （以lerobot_dataset_0915这个数据格式为例、config.py里面的repo_id="lerobot_dataset_0915",这个得改成自己数据集的名称  config-name 是pi05_libero 就得改相应的 TrainConfig中的repo_id
    2. '/home/xman/.cache/huggingface/lerobot/lerobot_dataset_0915/meta/info.json'


2. 推理服务指令 python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/xman/xman/openpi-main/checkpoints/params30000

3. 训练启动命令 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_libero --exp-name=my_experiment  --fsdp-devices=8 --overwrite
4. 断点训练  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_libero --exp-name=my_experiment --resume

5. 转换成pytorch 命令   python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/xman/xman/checkpoints/pi0.5/pi05_base --config-name pi05_libero  --output_path /home/xman/xman/checkpoints
6. 关于wandb的应用 export https_proxy=http://127.0.0.1:7890;export http_proxy=http://127.0.0.1:7890;export all_proxy=socks5://127.0.0.1:7890    wandb sync /home/xiededa/project/xman/Isaac-GR00T-main/wandb/offline-run-20250916_110152-k86nty0d 同步到服务器端
7. yU7CQ13EOIlY2kwxa6S08nZXgPiWf594
8. python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/xman/xman/openpi-main/jscloud/29999 a98579a1-c823-4c95-9518-aafbfa766202

