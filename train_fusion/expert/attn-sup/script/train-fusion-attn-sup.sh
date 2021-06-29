export SIQA_DIR=/home/airesearch04/workspace/task-augmentation/dataset/social_iqa
export OUTPUT_DIR=/home/airesearch04/workspace/task-augmentation/output/fusions/multiple-fusion/10-epochs-8-batch
export PRETRAINED_ADAPTER_DIR_PATH=/home/airesearch04/workspace/task-augmentation/output/best/adapters/10-epochs-16-batch

python ../../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --train_fusion \
    --pretrained_adapter_names "default,xAttr,xEffect,xIntent,xNeed,xWant,xReact" \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --do_train \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/"attn-sup-hyp-20" \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 500 \
    --overwrite_output 

