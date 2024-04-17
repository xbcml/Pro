#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 \
python train.py --exp-dir experiment/ \
--dist-url 'tcp://localhost:5678' \
--world-size 1 \
--rank 0 \
--batch-size 8 \
--gpu 0
# --multiprocessing-distributed \
