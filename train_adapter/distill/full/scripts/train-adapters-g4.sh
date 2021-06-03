export CUDA_VISIBLE_DEVICES=4
export OUTPUT_DIR=/home/yujin/r-expert/output/full/adapters/kdonly
export DATASET_DIR=/home/yujin/r-expert/dataset/socialiqa/socialiqa-full
export EPOCH=10
export TEACHER_MODEL_PATH=/home/yujin/r-expert/output/best/full/adapters/kdonly/default
export BEST_MODEL_PATH=/home/yujin/r-expert/output/best/full/adapters/kdonly

export ADAPTER_NAME="xReact"

python ../run_multiple_choice.py \
    --task_name siqa \
    --seed 42 \
    --model_name_or_path roberta-large \
    --train_adapter \
    --adapter_name $ADAPTER_NAME\
    --wandb_project "adapters-full" \
    --wandb_entity "rexpert" \
    --wandb_name "adapter-$ADAPTER_NAME-kdonly" \
    --teacher_model_path $TEACHER_MODEL_PATH \
    --do_train \
    --best_model_path $BEST_MODEL_PATH \
    --data_dir $DATASET_DIR/$ADAPTER_NAME \
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
    --overwrite_output  ;

export ADAPTER_NAME="xWant"

python ../run_multiple_choice.py \
    --task_name siqa \
    --seed 42 \
    --model_name_or_path roberta-large \
    --train_adapter \
    --adapter_name $ADAPTER_NAME\
    --wandb_project "adapters-full" \
    --wandb_entity "rexpert" \
    --wandb_name "adapter-$ADAPTER_NAME-kdonly" \
    --teacher_model_path $TEACHER_MODEL_PATH \
    --do_train \
    --best_model_path $BEST_MODEL_PATH \
    --data_dir $DATASET_DIR/$ADAPTER_NAME \
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