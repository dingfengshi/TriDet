#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/hacs_slowfast.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/hacs_slowfast.yaml ckpt/hacs_slowfast_pretrained/epoch_010.pth.tar

