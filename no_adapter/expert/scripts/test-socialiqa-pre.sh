export CUDA_VISIBLE_DEVICES=4
export SIQA_DIR=/home/yujin/data2/yujin/dataset/socialiqa
export OUTPUT_DIR=/home/yujin/r-expert/output/best/expert/finetune/pre
export BATCH=8

python ../run_multiple_choice.py \
    --task_name siqa \
    --seed 42 \
    --model_name_or_path $OUTPUT_DIR \
    --wandb_project "finetune-atomic" \
    --wandb_entity "rexpert" \
    --wandb_name "roberta-pre-test" \
    --do_predict \
    --data_dir $SIQA_DIR \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size=$BATCH \
    --overwrite_output  