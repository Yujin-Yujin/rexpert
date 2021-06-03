export CUDA_VISIBLE_DEVICES=2
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/adapters/full-5
export DATASET_DIR=/home/yujin/r-expert/dataset/cwwv/full
export ADAPTER_NAME="cwwv"
export EPOCH=5
export BEST_MODEL_PATH=/home/yujin/r-expert/output/best/expert/adapters/full-5

python ../run_multiple_choice.py \
    --task_name csqa \
    --seed 42 \
    --model_name_or_path roberta-large \
    --adapter_name $ADAPTER_NAME\
    --wandb_project "adapters-cwwv" \
    --wandb_entity "rexpert" \
    --wandb_name "adapter-$ADAPTER_NAME-full" \
    --do_train \
    --train_adapter \
    --do_eval \
    --best_model_path $BEST_MODEL_PATH \
    --data_dir $DATASET_DIR \
    --learning_rate 5e-5 \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output  