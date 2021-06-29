export SIQA_DIR=/home/airesearch04/workspace/task-augmentation/dataset/social-iqa/social-iqa-10k/default
export OUTPUT_DIR=/home/airesearch04/workspace/task-augmentation/output/fusions/10k-multi/attn-15
export PRETRAINED_ADAPTER_DIR_PATH=/home/airesearch04/workspace/task-augmentation/output/best/adapters/10k
export ADAPTER_NAME=default,xAttr,xEffect,xIntent,xNeed,xReact,xWant

python ../../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --train_fusion \
    --pretrained_adapter_names $ADAPTER_NAME \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --do_train \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 500 \
    --overwrite_output 


