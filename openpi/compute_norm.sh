#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_LEROBOT_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export HF_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi

python scripts/compute_norm_stats.py --config-name=pi05_xman_coffee_lora_downsample_filter_0427
