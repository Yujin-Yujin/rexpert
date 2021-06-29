export SIQA_DIR=/home/airesearch04/workspace/task-augmentation/dataset/social-iqa/social_iqa
export PRETRAINED_ADAPTER_DIR_PATH=home/airesearch04/workspace/task-augmentation/output/best/adapters/10k
export PRETRAINED_FUSION_DIR_PATH=/home/airesearch04/workspace/task-augmentation/output/best/fusions/10k/attn-10

export ADAPTER_NAMES=default,xAttr,xEffect
python ../../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --test_fusion \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --fusion_path $PRETRAINED_FUSION_DIR_PATH/$ADAPTER_NAMES \
    --do_predict \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --max_seq_length 128 \
    --output_dir $PRETRAINED_FUSION_DIR_PATH/$ADAPTER_NAMES \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --overwrite_output 

export ADAPTER_NAMES=default,xAttr,xEffect,xIntent,xNeed,xReact,xWant
python ../../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --test_fusion \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --fusion_path $PRETRAINED_FUSION_DIR_PATH/$ADAPTER_NAMES \
    --do_predict \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --max_seq_length 128 \
    --output_dir $PRETRAINED_FUSION_DIR_PATH/$ADAPTER_NAMES \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --overwrite_output 