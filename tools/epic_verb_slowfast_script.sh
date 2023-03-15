#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/epic_slowfast_verb.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/epic_slowfast_verb.yaml ckpt/epic_slowfast_verb_pretrained/epoch_022.pth.tar

