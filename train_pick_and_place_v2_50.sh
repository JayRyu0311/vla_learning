#!/bin/bash

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=jinw0o0/pick_and_place_v2_50 \
  --policy.type=act \
  --output_dir=outputs/train/pick_and_place_v2_50 \
  --job_name=pick_and_place_v2_50 \
  --policy.device=cuda \
  --policy.repo_id=jae0311/pick_and_place_v2_50 \
  --policy.use_amp=true \
  --wandb.enable=true \
  --batch_size=16 \
  --steps=200000 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=200
