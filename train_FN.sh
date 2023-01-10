#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 nohup python train_FN.py --dataset_option=normal --path_opt=options/models/VG-MSDN-TRAIN.yaml --rpn output/trained_models/RPN.h5 > train_FN.out &
