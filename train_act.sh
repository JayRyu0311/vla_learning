#!/bin/bash

read -p "dataset.repo_id (예: jinw0o0/pick_and_place_v2_50): " DATASET_REPO_ID
read -p "batch_size (기본값 16): " BATCH_SIZE
read -p "steps (기본값 200000): " STEPS
read -p "num_workers (기본값 4): " NUM_WORKERS

DATASET_NAME="${DATASET_REPO_ID##*/}"
OUTPUT_DIR="outputs/train/${DATASET_NAME}"
JOB_NAME="${DATASET_NAME}"
POLICY_REPO_ID="jae0311/${DATASET_NAME}"

echo ""
echo "학습 설정:"
echo "  dataset.repo_id : ${DATASET_REPO_ID}"
echo "  output_dir      : ${OUTPUT_DIR}"
echo "  job_name        : ${JOB_NAME}"
echo "  policy.repo_id  : ${POLICY_REPO_ID}"
echo "  batch_size      : ${BATCH_SIZE}"
echo "  steps           : ${STEPS}"
echo "  num_workers     : ${NUM_WORKERS}"
echo ""

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --policy.type=act \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.use_amp=true \
  --wandb.enable=true \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=200
