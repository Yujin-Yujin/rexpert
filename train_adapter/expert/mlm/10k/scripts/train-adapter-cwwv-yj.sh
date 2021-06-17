export CUDA_VISIBLE_DEVICES=4
export DATASET=/home/yujin/r-expert/dataset/cwwv/10k
export OUTPUT_PATH=/home/yujin/r-expert/output/expert/adapters/10k/mlm

# python ../run_mlm.py \
#     --model_type roberta-mlm \
#     --model_name_or_path roberta-large \
#     --task_name cwwv \
#     --output_dir $OUTPUT_PATH \
#     --max_sequence_per_time 200 \
#     --train_file $DATASET/train_random.jsonl \
#     --dev_file $DATASET/dev_random.jsonl \
#     --max_seq_length 128 \
#     --max_words_to_mask 6 \
#     --do_train \
#     --do_eval \
#     --per_gpu_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 1 \
#     --warmup_proportion 0.05 \
#     --evaluate_during_training \
#     --per_gpu_eval_batch_size 8  \
#     --save_steps 6500 \
#     --margin 1.0


python ../run_mlm_yj.py \
    --model_name_or_path roberta-large \
    --task_name cwwv \
    --train_file $DATASET/train_rand_split.jsonl \
    --validation_file $DATASET/dev_rand_split.jsonl \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --max_words_to_mask 6 \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --output_dir /home/yujin/r-expert/output/test \
    --overwrite_output_dir
