#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 nohup python train_rpn.py --dataset_option=normal > train_rpn.out &

