#!/usr/bin/env bash
python train.py --exp-dir experiment/ --dist-url 'tcp://localhost:5678' --multiprocessing-distributed --world-size 1 --rank 0
