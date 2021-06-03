export DATA_FILE="/home/yujin/r-expert/HyKAS-CSKG/tasks/commonsenseqa_dev.jsonl"
export TASK_NAME=commonsenseqa

python evaluate_RoBERTa.py \
    --lm roberta-large \
    --dataset_file $DATA_FILE \
    --out_dir ../../results \
    --device 1 \
    --reader $TASK_NAME