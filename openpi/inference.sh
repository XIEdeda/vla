source /home/xman/miniforge3/etc/profile.d/conda.sh
conda activate openpi
# source .venv/bin/activate
# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_xman_coffee_lora_inference \
#     --policy.dir=/mnt/datas/finetune_model/hzt_experiment_coffee_lora_0319/49999

    
# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_xman_coffee_lora_downsample_inference \
#     --policy.dir=/mnt/datas/finetune_model/hzt_experiment_coffee_lora_downsample_0323/30000


python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_xman_coffee_lora_downsample_0423_inference \
    --policy.dir=/mnt/datas/finetune_model/pi05_xman_coffee_lora_downsample_0423/53669

# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_xman_coffee_lora_downsample_0421_inference \
#     --policy.dir=/mnt/datas/finetune_model/pi05_xman_coffee_lora_downsample_0421/40000

    
# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_xman_coffee_only_ae_0410_inference \
#     --policy.dir=/mnt/datas/finetune_model/pi05_xman_coffee_lora_downsample_0417/40000


# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_xman_coffee_CU_SD_a800_0413_inference \
#     --policy.dir=/mnt/datas/finetune_model/pi05_xman_coffee_CU_SD_a800_0413/19999

# python scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_xman_coffee_lora_downsample_a800_0414_inference \
#     --policy.dir=/mnt/datas/finetune_model/pi05_xman_coffee_lora_downsample_a800_0414/18000

    