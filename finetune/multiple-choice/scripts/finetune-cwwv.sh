export CUDA_VISIBLE_DEVICES=2
export OUTPUT_DIR=../../../output/finetune
export DATASET_DIR=../../../dataset/kg-dataset/cwwv/full
export EPOCH=5
export TASK_NAME=multikg

python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --seed 42 \
    --model_name_or_path roberta-large \
    --wandb_project "finetune-kg" \
    --wandb_entity "rexpert" \
    --wandb_name "finetune-cwwv-0716" \
    --do_train \
    --do_eval \
    --data_dir $DATASET_DIR \
    --learning_rate 5e-6 \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/cwwv/"5e-6" \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output  