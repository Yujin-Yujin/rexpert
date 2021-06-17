export CUDA_VISIBLE_DEVICES=4
export DATASET=/home/yujin/r-expert/dataset/cwwv/10k
export OUTPUT_PATH=/home/yujin/r-expert/output/expert/adapters/10k/mlm

python ../run_mlm.py \
    --model_type roberta-mlm \
    --model_name_or_path roberta-large \
    --task_name cskg \
    --output_dir $OUTPUT_PATH \
    --train_file $DATASET/train_random.jsonl \
    --dev_file $DATASET/dev_random.jsonl \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --warmup_proportion 0.05 \
    --evaluate_during_training\
    --save_steps 6500 \
    --margin 1.0
