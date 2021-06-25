export CUDA_VISIBLE_DEVICES=2
# export DATA_FILE="/home/yujin/r-expert/dataset/socialiqa/origin/socialIQa_v1.4_dev.jsonl"
# export DATA_FILE="/home/yujin/r-expert/dataset/socialiqa/origin/socialIQa_v1.4_tst.jsonl"
# export TASK_NAME=socialiqa

export DATA_FILE="/home/yujin/r-expert/dataset/commonsense/origin/dev_rand_split.jsonl"
export TASK_NAME=commonsenseqa

export PRETRAINED_ADAPTER_PATH=/home/yujin/r-expert/output/expert/adapters/mr/atomic
# export PRETRAINED_FUSION_PATH=/home/yujin/r-expert/output/best/expert/fusions/full
export ADAPTER_NAMES=atomic

# python evaluate_adapter_tune.py \
#     --lm roberta-large \
#     --dataset_file $DATA_FILE \
#     --out_dir /home/yujin/r-expert/output \
#     --reader $TASK_NAME \
#     --adapter_load $PRETRAINED_ADAPTER_PATH \
#     --adapter_names $ADAPTER_NAMES \
#     --fusion_path $PRETRAINED_FUSION_PATH/$ADAPTER_NAMES

python evaluate_adapter_tune.py \
    --lm roberta-large \
    --dataset_file $DATA_FILE \
    --out_dir /home/yujin/r-expert/output \
    --reader $TASK_NAME \
    --adapter_load $PRETRAINED_ADAPTER_PATH \
    --adapter_names $ADAPTER_NAMES

