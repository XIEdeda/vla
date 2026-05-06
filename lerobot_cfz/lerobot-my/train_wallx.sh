export HF_ENDPOINT=https://hf-mirror.com
export HF_LEROBOT_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export HF_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c"
# export ACCELERATE_MIXED_PRECISION=bf16

lerobot-train \
    --dataset.repo_id=/mnt/datas/vla_datasets/coffee_src_data/coffee_motion_downsample_0401 \
    --policy.type=wall_x \
    --output_dir=/mnt/datas/finetune_model/wallx_coffee_motion_downsample_0401 \
    --job_name=wallx_training \
    --policy.pretrained_name_or_path=x-square-robot/wall-oss-flow \
    --policy.prediction_mode=diffusion \
    --policy.attn_implementation=eager \
    --steps=3000 \
    --wandb.enable=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=32