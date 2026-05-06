#!/usr/bin/env bash

set -euo pipefail

DELAY_HOURS=0
DELAY_SECONDS=$((DELAY_HOURS * 3600))


REMOTE_HOST="root@27.159.92.108"
REMOTE_BASE="/mnt/nvme0n1/nvme1n1/hzt/VLA-models/keenon-model/pi05_xman_coffee_lora_downsample_0423/pi05_xman_coffee_lora_downsample_0423"
LOCAL_BASE="/mnt/datas/finetune_model/pi05_xman_coffee_lora_downsample_0423"
PASSWORD="KEENON-robotics-2010!"

log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_rsync() {
	local step="$1"
	local ckpt="$2"
	log "Start step ${step}: sync checkpoint ${ckpt}"
	sshpass -p "$PASSWORD" rsync -avP -e "ssh -o StrictHostKeyChecking=no" \
		"${REMOTE_HOST}:${REMOTE_BASE}/${ckpt}" \
		"${LOCAL_BASE}"
	log "Finish step ${step}: checkpoint ${ckpt}"
}

mkdir -p "${LOCAL_BASE}"

log "Script started. Will wait ${DELAY_HOURS} hours before transfer."
sleep "${DELAY_SECONDS}"

run_rsync 1 53669

log "All transfers completed successfully."
