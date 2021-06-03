export CUDA_VISIBLE_DEVICES=1
export OUTPUT_DIR=/home/yujin/r-expert/output/0.1/finetune/checkpoint-2090
export DATASET_DIR=/home/yujin/data2/yujin/dataset/social-iqa-0.1/default

python ../run_multiple_choice.py \
    --task_name siqa \
    --seed 42 \
    --model_name_or_path "/home/yujin/r-expert/output/0.1/finetune/checkpoint-2090" \
    --do_predict \
    --data_dir $DATASET_DIR \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size=8 \
    --overwrite_output 