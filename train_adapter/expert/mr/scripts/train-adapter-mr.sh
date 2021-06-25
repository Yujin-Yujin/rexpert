cd /home/yujin/r-expert/adapter-transformers-mr

pip install .

cd /home/yujin/r-expert/train_adapter/expert/mr/scripts

export CUDA_VISIBLE_DEVICES=4
export DATASET=/home/yujin/r-expert/dataset/atomic/10k
export OUTPUT_PATH=/home/yujin/r-expert/output/test
export ADAPTER_NAME=atomic


python ../run_mr.py \
    --model_name_or_path roberta-large \
    --task_name atomic \
    --adapter_name $ADAPTER_NAME \
    --train_file $DATASET/train_random.jsonl \
    --validation_file $DATASET/dev_random.jsonl \
    --epoch 1\
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_PATH \
    --overwrite_output_dir