# cd /home/yujin/rexpert/adapter-transformers

# pip install .

# cd /home/yujin/rexpert/train_fusion/expert/origin/scripts


export CUDA_VISIBLE_DEVICES=2
export DATASET=/home/yujin/rexpert/dataset/benchmark/commonsense/origin
export OUTPUT_DIR=/home/yujin/rexpert/output/fusions/siqa,csqa
export BATCH=8
export EPOCH=1
export PRETRAINED_FUSION_PATH=/home/yujin/rexpert/output/fusions/siqa,csqa/siqa,csqa

python ../run_multiple_choice_pretrained.py \
    --task_name csqa \
    --model_name_or_path roberta-base \
    --wandb_project "fusion-default" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-siqa,csqa-csqa" \
    --test_fusion \
    --fusion_path $PRETRAINED_FUSION_PATH\
    --do_eval \
    --seed 42 \
    --data_dir $DATASET \
    --learning_rate 5e-5 \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output 