#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python train_FN.py --evaluate --dataset_option=normal --path_opt options/models/VG-MSDN-EVAL.yaml --pretrained_model output/trained_models/Model-VG-MSDN.h5 > eval.out &
