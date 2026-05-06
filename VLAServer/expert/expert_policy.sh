export HF_LEROBOT_HOME="/mnt/datas/cache_dataset"
export HF_HOME="/mnt/datas/cache_dataset"

# conda activate pi
python expert_policy.py --dataset_repo_id "/mnt/datas/vla_datasets/fold_clothes/018_20260115_cyw_fold_clothes_5_3" --port 5556 --delay 0.302 --mode raw
# python expert_policy.py --dataset_repo_id "/mnt/datas/vla_datasets/fold_clothes/018_20260115_cyw_fold_clothes_5_3_" --port 5556 --delay 1.4 --mode lerobot
