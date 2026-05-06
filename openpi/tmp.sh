
#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_LEROBOT_HOME="/DATA/disk0/cache"
export HF_HOME="/DATA/disk0/cache"
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi


# pi train
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python scripts/train.py pi05_xman_coffee_a800 --exp-name=tmp --checkpoint_base_dir=/tmp --overwrite --fsdp-devices=8 