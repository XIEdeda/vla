export HF_LEROBOT_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export HF_HOME="/mnt/nvme0n1/nvme1n1/cache_dataset"
export WANDB_API_KEY="43c090c35e1a2e20da2a888a9bfcf270df14232c"

python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8090 \
  --fps=10 \
  --inference_latency=0.03 \
  --obs_queue_timeout=1