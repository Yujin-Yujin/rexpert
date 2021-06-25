cd /home/yujin/r-expert/adapter-transformers-mr

pip install .

cd /home/yujin/r-expert/train_adapter/expert/mr/scripts

export CUDA_VISIBLE_DEVICES=3
# export DATASET_DIR=/home/yujin/r-expert/dataset/cwwv/full
export DATASET_DIR=/home/yujin/r-expert/dataset/cwwv/10k

export OUTPUT_DIR=/home/yujin/r-expert/output/expert/adapters/mr
export ADAPTER_NAME=cwwv_mr
export EPOCH=5

python ../run_mr_pp.py \
    --model_name_or_path roberta-large \
    --task_name cwwv \
    --adapter_name $ADAPTER_NAME \
    --train_file $DATASET_DIR/train_rand_split.jsonl \
    --validation_file $DATASET_DIR/dev_rand_split.jsonl \
    --train_adapter \
    --num_train_epochs $EPOCH \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --per_device_train_batch_size 8\
    --per_device_train_batch_size 8\
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir