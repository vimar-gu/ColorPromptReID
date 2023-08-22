#!/bin/sh

CUDA_VISIBLE_DEVICES=0 \
python source_pretrain.py -ds market1501 -dt msmt17 -a resnet50 --seed 0 --margin 0.0 \
	--num-instances 4 -b 64 -j 16 --warmup-step 10 --lr 0.00035 \
	--milestones 40 70 --iters 400 --data-dir ../data \
	--logs-dir ../logs/color-prompting/pretrain-market-trans-prompter --trans --prompter
