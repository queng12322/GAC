#!/bin/bash

augmentations=$1
dataset_names=$2
model_type=$3
model_path=$4
eval_task_type=$5

for dataset_name in $dataset_names
do
    for ntrain in $ntrains
    do 
        for augmentation in $augmentations
        do
            export CUDA_VISIBLE_DEVICES=0
            export BETA=0.4
            export THRES=0.1
            echo "-----Testing dataset: $dataset_name; Testing few-shot number: $ntrain; If augmentation: $augmentation-----"
            
            python main.py \
                --ckpt_dir $model_path \
                --model_type $model_type \
                --calibrate 0 \
                --do_augmentation $augmentation \
                --task_type $eval_task_type \
                --dataset $dataset_name
        done
    done
done
