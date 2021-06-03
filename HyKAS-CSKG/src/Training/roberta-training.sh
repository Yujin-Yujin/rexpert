export CUDA_VISIBLE_DEVICES=0 

python run_pretrain.py \ 
    --model_type roberta-mlm \
    --model_name_or_path roberta-large --task_name cskg --output_dir ../../out_dir --max_sequence_per_time 200 \
--train_file ../../data/ATOMIC/train_random.jsonl --second_train_file ../../data/CWWV/train_random.jsonl --dev_file ../../data/ATOMIC/dev_random.jsonl --second_dev_file \
../../data/CWWV/dev_random.jsonl --max_seq_length 128 --max_words_to_mask 6 --do_train --do_eval --per_gpu_train_batch_size 2 --gradient_accumulation_steps 16 \
--learning_rate 1e-5 --num_train_epochs 1 --warmup_proportion 0.05 --evaluate_during_training --per_gpu_eval_batch_size 8  --save_steps 6500 --margin 1.0
