#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/thumos_i3d.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pretrained/epoch_039.pth.tar
