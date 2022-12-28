#!/bin/bash

# python bm_train.py --epoch 10
NNODES=1
GPU_PER_NODE=1
torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} bm_train.py

