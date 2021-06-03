export CUDA_VISIBLE_DEVICES=4
export SIQA_DIR=/home/yujin/r-expert/dataset/socialiqa/origin
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/r-expert/output/best/full/adapters/kdonly
export OUTPUT_DIR=/home/yujin/r-expert/output/full/fusions/kdonly
export BATCH=8
export BEST_MODEL_PATH=/home/yujin/r-expert/output/best/full/fusions/kdonly

export ADAPTER_NAMES=default,xReact
python ../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-full" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-kdonly" \
    --train_fusion \
    --do_train \
    --do_select \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --best_model_path $BEST_MODEL_PATH \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output ;

export ADAPTER_NAMES=default,xWant
python ../run_multiple_choice.py \
    --task_name siqa \
    --model_name_or_path roberta-large \
    --train_fusion \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-0.1" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-kdonly" \
    --do_train \
    --do_select \
    --seed 42 \
    --data_dir $SIQA_DIR \
    --best_model_path $BEST_MODEL_PATH \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output 