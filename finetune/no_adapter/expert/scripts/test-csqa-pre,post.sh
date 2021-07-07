export CUDA_VISIBLE_DEVICES=4
export DATASET_DIR=/home/yujin/data2/yujin/dataset/commonsense/our-split
export OUTPUT_DIR=/home/yujin/r-expert/output/best/expert/finetune/pre,post
export BATCH=8

python ../run_multiple_choice.py \
    --task_name csqa \
    --seed 42 \
    --model_name_or_path $OUTPUT_DIR \
    --wandb_project "finetune-atomic" \
    --wandb_entity "rexpert" \
    --wandb_name "csqa-pre,post-test" \
    --do_predict \
    --data_dir $DATASET_DIR \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size=$BATCH \
    --overwrite_output  