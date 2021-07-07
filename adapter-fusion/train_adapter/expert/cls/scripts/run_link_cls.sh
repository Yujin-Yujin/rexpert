export CUDA_VISIBLE_DEVICES=4
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/adapters/link_cls
export ADAPTER_NAME="atomic_link_cls"
export EPOCH=1
export TRAIN_FILE=/home/yujin/r-expert/dataset/atomic/link_cls/link_cls_trn.jsonl
export EVAL_FILE=/home/yujin/r-expert/dataset/atomic/link_cls/link_cls_dev.jsonl

python ../run_glue.py \
    --model_name_or_path roberta-large \
    --wandb_project "adapters-atomic" \
    --wandb_entity "rexpert" \
    --wandb_name "adapter-$ADAPTER_NAME" \
    --adapter_name $ADAPTER_NAME \
    --train_file $TRAIN_FILE\
    --validation_file $EVAL_FILE\
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
     --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --train_adapter \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output \
    --adapter_config pfeiffer