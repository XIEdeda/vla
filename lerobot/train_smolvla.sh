export HF_ENDPOINT=https://hf-mirror.com
export HF_LEROBOT_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export HF_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c"
export ACCELERATE_MIXED_PRECISION=bf16

OUTPUT_DIR=/mnt/datas/finetune_model/smolvla_coffee_motion_downsample_0401
rm -rf "$OUTPUT_DIR"

lerobot-train \
  --dataset.repo_id=/mnt/datas/vla_datasets/coffee_src_data/coffee_motion_downsample_0401 \
  --policy.path=lerobot/smolvla_base \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_coffee_motion_downsample_0401 \
  --policy.device=cuda \
  --wandb.enable=true \
  --step=30000 \
  --batch_size=64 \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.head_image": "observation.images.camera1", "observation.images.left_wrist_image": "observation.images.camera2", "observation.images.right_wrist_image": "observation.images.camera3"}'