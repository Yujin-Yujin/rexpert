#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
# export DATASET=../../../../rexpert/dataset/benchmark/socialiqa
export DATASET=../../../../rexpert/dataset/benchmark/commonsense/origin
export PRETRAINED_ADAPTER_DIR_PATH=../../../../rexpert/output/best/adapters/full
export OUTPUT_DIR=../../../../rexpert/output/analysis
export BATCH=8
# export PRETRAINED_FUSION_LAYER=../../../../rexpert/output/fusions/full/atomic,cwwv/atomic,cwwv
# export PRETRAINED_FUSION_LAYER=/home/yujin/rexpert/output/fusions/blend/10k/atomic,cwwv/atomic,cwwv/atomic,cwwv
export PRETRAINED_FUSION_LAYER=/home/bwoo/workspace/rexpert/att_sup/0.1/atomic,cwwv/atomic,cwwv

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice_custom.py \
    --task_name csqa \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --fusion_path $PRETRAINED_FUSION_LAYER\
    --wandb_project "fusion-analysis" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES" \
    --test_fusion \
    --do_eval \
    --seed 42 \
    --data_dir $DATASET \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output 