cd /home/yujin/r-expert/adapter-transformers
pip install .
cd /home/yujin/r-expert/finetune/scripts

export CUDA_VISIBLE_DEVICES=3
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/finetune
export DATASET_DIR=/home/yujin/r-expert/dataset/atomic/full
export EPOCH=1
export TASK_NAME=atomic

python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --seed 42 \
    --model_name_or_path roberta-large \
    --wandb_project "finetune-kg" \
    --wandb_entity "rexpert" \
    --wandb_name "finetune-$TASK_NAME" \
    --do_train \
    --do_eval \
    --data_dir $DATASET_DIR \
    --learning_rate 1e-5 \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$TASK_NAME \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output  