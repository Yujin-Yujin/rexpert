export CUDA_VISIBLE_DEVICES=2
export SIQA_DIR=/home/yujin/data2/yujin/dataset/socialiqa
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/aaai_2021/output/adapters/selected
export OUTPUT_DIR=/home/yujin/aaai_2021/output/fusions
export BATCH=8

export ADAPTER_NAMES=pre,post
python ../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --test_fusion \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --do_predict \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_eval_batch_size=$BATCH \
    --gradient_accumulation_steps 1 \
    --overwrite_output 
