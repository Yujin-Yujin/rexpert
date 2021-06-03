export CUDA_VISIBLE_DEVICES=1
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/finetune
# export DATASET_DIR=/home/yujin/data2/yujin/dataset/commonsense/our-split
export DATASET_DIR=/home/yujin/r-expert/dataset/origin

python ../run_multiple_choice.py \
    --task_name csqa \
    --seed 42 \
    --model_name_or_path roberta-large \
    --do_eval \
    --data_dir $DATASET_DIR \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size=32 \
    --overwrite_output 