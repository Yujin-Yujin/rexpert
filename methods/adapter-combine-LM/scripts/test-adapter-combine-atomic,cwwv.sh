#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# export TASK_NAME=siqa
# export DATASET=../../../dataset/benchmark/socialiqa
export TASK_NAME=csqa
export DATASET=../../../dataset/benchmark/commonsense/origin
export PRETRAINED_ADAPTER_DIR_PATH=../../../output/best/adapters/full
export OUTPUT_DIR=../../../output/fusions/adapter-combine/atomic,cwwv
export BATCH=4
export ATTENTION_LAYER_PATH=../../../output/fusions/adapter-combine/10k/attn-lm/atomic,cwwv/pytorch_model.bin
export ADAPTER_NAMES=atomic,cwwv
python ../inference_multiple_choice_attention.py \
    --task_name $TASK_NAME \
    --model_name_or_path roberta-large \
    --adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-attention" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-test" \
    --attention_layer_path $ATTENTION_LAYER_PATH \
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