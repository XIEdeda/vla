#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_LEROBOT_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export HF_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi


# pi train
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# python scripts/train_val.py pi05_xman_coffee_lora_downsample_chunk16 \
#     --exp-name=pi05_xman_coffee_lora_downsample_chunk16_0326 \
#     --checkpoint_base_dir=/mnt/nvme0n1/nvme1n1/hzt/VLA-models/keenon-model \
#     --overwrite \
#     --fsdp-devices=4

# python scripts/train_val.py pi05_xman_coffee_lora_downsample_chunk16 --exp-name=pi05_xman_coffee_lora_downsample_chunk16_0326 --checkpoint_base_dir=/mnt/nvme0n1/nvme1n1/hzt/VLA-models/keenon-model --overwrite --fsdp-devices=4

python scripts/train_val.py pi05_xman_coffee_CU_SD_a800_0413 --exp-name=pi05_xman_coffee_CU_SD_a800_0413 --checkpoint_base_dir=/mnt/nvme0n1/nvme1n1/hzt/VLA-models/keenon-model --resume --fsdp-devices=8