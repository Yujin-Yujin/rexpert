export CUDA_VISIBLE_DEVICES=1
export DATASET=/home/yujin/rexpert/dataset/kg-dataset/small
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/rexpert/output/best/adapters/full
export OUTPUT_DIR=/home/yujin/rexpert/output/adapters
export BATCH=8

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice_custom.py \
    --task_name multikg \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-contrastive" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-test" \
    --train_fusion \
    --do_train \
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