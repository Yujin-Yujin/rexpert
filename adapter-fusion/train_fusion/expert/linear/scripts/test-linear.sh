#!/bin/bash
# cd ../../../../adapter-transformers-linear

# pip install .

# cd ../train_fusion/expert/linear/scripts
export CUDA_VISIBLE_DEVICES=3
# export DATASET=../../../../dataset/socialiqa/origin
export DATASET=../../../../dataset/commonsense/origin
# export DATASET=../../../../dataset/atomic/full
export PRETRAINED_ADAPTER_DIR_PATH=../../../../output/best/expert/adapters/full
export OUTPUT_DIR=../../../../output/expert/fusions/linear
export BATCH=8
export BEST_MODEL_PATH=../../../../output/best/expert/fusions/linear
export LINEAR_MODEL_PATH=../../../../output/expert/fusions/linear-full/10k/atomic,cwwv/pytorch_model.bin


export ADAPTER_NAMES=atomic,cwwv
python ../inference-with-linear.py \
    --task_name csqa \
    --model_name_or_path roberta-large \
    --linear_model_path $LINEAR_MODEL_PATH \
    --adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-linear" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES" \
    --do_eval \
    --seed 42 \
    --data_dir $DATASET \
    --best_model_path $BEST_MODEL_PATH \
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