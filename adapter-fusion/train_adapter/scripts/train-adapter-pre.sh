export CUDA_VISIBLE_DEVICES=4
export OUTPUT_DIR=/home/yujin/aaai_2021/output/adapters2
export DATASET_DIR=/home/yujin/data2/yujin/dataset/atomic_2019/pre_sample
export ADAPTER_NAME="pre"
export EPOCH=10

python ../run_multiple_choice.py \
    --task_name atomic \
    --seed 42 \
    --model_name_or_path roberta-large \
    --train_adapter \
    --adapter_name $ADAPTER_NAME\
    --do_train \
    --do_eval \
    --data_dir $DATASET_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output true 