#!/bin/sh

CUDA_VISIBLE_DEVICES=0 \
python multi_continual.py -ds market1501 -dt cuhksysu msmt17 -a resnet50 \
  --num-instances 16 --lr 0.0002 --iters 400 -b 128 --epochs 60 --stages 1 \
  --dropout 0 --eps 0.6 --eval-step 10 --data-dir ../data --momentum 0.1 \
  --logs-dir ../logs/color-prompting/fulluda-trans-prompter --trans --prompter \
  --resume ../logs/color-prompting/pretrain-market-trans-prompter/model_best.pth.tar \
  --prompter-path ../logs/color-prompting/pretrain-market-trans-prompter/prompter.pth.tar
