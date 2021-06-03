export CUDA_VISIBLE_DEVICES=1
export OUTPUT_DIR=/home/yujin/r-expert/output/best/expert/adapters/10k
export DATASET=/home/yujin/r-expert/dataset/commonsense/origin

# export OUTPUT_DIR=/home/yujin/r-expert/output/best/expert/adapters/
# export DATASET=/home/yujin/r-expert/dataset/socialiqa/origin

export ADAPTER_NAME="cwwv"
export TASK_NAME="csqa"

python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --seed 42 \
    --model_name_or_path roberta-large \
    --adapter_name $ADAPTER_NAME\
    --load_adapter $OUTPUT_DIR/$ADAPTER_NAME \
    --do_eval \
    --data_dir $DATASET \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --per_device_eval_batch_size=8 \
    --overwrite_output 