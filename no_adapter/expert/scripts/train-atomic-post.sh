export CUDA_VISIBLE_DEVICES=0
export DATASET_DIR=/home/yujin/data2/yujin/dataset/atomic_2019/post_sample
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/finetune/post
export BATCH=8

python ../run_multiple_choice.py \
    --task_name atomic \
    --seed 42 \
    --model_name_or_path roberta-large \
    --wandb_project "finetune-atomic" \
    --wandb_entity "rexpert" \
    --wandb_name "roberta-post" \
    --do_train \
    --do_eval \
    --data_dir $DATASET_DIR \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output  