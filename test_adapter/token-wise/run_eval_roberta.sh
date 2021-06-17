export DATA_FILE="/home/yujin/r-expert/dataset/socialiqa/origin/socialIQa_v1.4_dev.jsonl"
export TASK_NAME=socialiqa

python evaluate_RoBERTa.py \
    --lm roberta-large \
    --dataset_file $DATA_FILE \
    --out_dir /home/yujin/r-expert/output \
    --device 1 \
    --reader $TASK_NAME