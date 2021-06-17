export CUDA_VISIBLE_DEVICES=3
export DATASET=/home/yujin/r-expert/dataset/multikg/10k
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/r-expert/output/best/expert/adapters/10k
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/fusions/10k
export BATCH=8
export BEST_MODEL_PATH=/home/yujin/r-expert/output/best/expert/fusions/10k

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice.py \
    --task_name multikg \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-multikg" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES" \
    --train_fusion \
    --do_train \
    --do_select \
    --seed 42 \
    --data_dir $DATASET \
    --best_model_path $BEST_MODEL_PATH \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output 