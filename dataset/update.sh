#!/bin/bash
dataset=~/workspaces/dataset/nia/info

cp ${dataset}/objects.json .
cp ${dataset}/categories.json .
cp ${dataset}/dict.json .
cp ${dataset}/inverse_weight.json .
cp ${dataset}/train.json .
cp ${dataset}/test.json .
cp ${dataset}/train_fat.json .
cp ${dataset}/test_fat.json .
cp ${dataset}/train_small.json .
cp ${dataset}/test_small.json .

