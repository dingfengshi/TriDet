#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/epic_slowfast_noun.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/epic_slowfast_noun.yaml ckpt/epic_slowfast_noun_pretrained/epoch_018.pth.tar

