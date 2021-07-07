
cd /home/yujin/r-expert/adapter-transformers

pip install .

cd /home/yujin/r-expert/train_adapter/expert/mlm/scripts

export CUDA_VISIBLE_DEVICES=4
export TRAIN_FILE_PATH=/home/yujin/r-expert/dataset/atomic/mlm/mlm_trn.txt
export EVAL_FILE_PATH=/home/yujin/r-expert/dataset/atomic/mlm/mlm_dev.txt
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/adapters/mlm
export ADAPTER_NAME="atomic_mlm"
export EPOCH=1

python ../run_mlm.py \
    --model_name_or_path roberta-large \
    --wandb_project "adapters-atomic" \
    --wandb_entity "rexpert" \
    --wandb_name "adapter-$ADAPTER_NAME" \
    --adapter_name $ADAPTER_NAME\
    --train_file $TRAIN_FILE_PATH \
    --validation_file $EVAL_FILE_PATH \
    --learning_rate 5e-5 \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --line_by_line \
    --train_adapter \
    --do_train \
    --do_eval \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --overwrite_output \
    --output_dir $OUTPUT_DIR
