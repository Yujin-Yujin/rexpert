#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export TASK_NAME=siqa
export DATASET=../../../dataset/benchmark/socialiqa
export PRETRAINED_ADAPTER_DIR_PATH=../../../output/best/adapters/full
export PRETRAINED_FUSION_PATH=/home/yujin/rexpert/output/fusions/stack-res/atomic,cwwv
export OUTPUT_DIR=../../../../rexpert/output/fusions/stack
export BATCH=8

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice_custom.py \
    --task_name $TASK_NAME \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-stack" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-full" \
    --fusion_path $PRETRAINED_FUSION_PATH/$ADAPTER_NAMES \
    --test_fusion \
    --do_eval \
    --do_predict \
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