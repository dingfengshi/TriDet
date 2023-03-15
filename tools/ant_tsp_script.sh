##!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/anet_tsp.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/anet_tsp.yaml ckpt/anet_tsp_pretrained/epoch_014.pth.tar
