#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python train_fn.py --evaluate --dataset_option=normal --path_opt options/models/msdn.yaml --pretrained_model output/trained_models/Model-VG-MSDN.h5 > eval.out &
