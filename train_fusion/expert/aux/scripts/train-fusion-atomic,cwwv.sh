cd /home/yujin/r-expert/adapter-transformers-aux
pip install .
cd /home/yujin/r-expert/train_fusion/expert/aux/scripts

export CUDA_VISIBLE_DEVICES=4
export DATASET=/home/yujin/r-expert/dataset/multikg/10k
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/r-expert/output/best/expert/adapters/full
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/fusions/classifier
export BATCH=8
export BEST_MODEL_PATH=/home/yujin/r-expert/output/best/expert/fusions/classifier

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice.py \
    --task_name multikg \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-classifier" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES" \
    --train_fusion \
    --do_train \
    --do_select \
    --seed 42 \
    --data_dir $DATASET \
    --best_model_path $BEST_MODEL_PATH \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAMES \
    --per_device_train_batch_size=$BATCH \
    --per_device_eval_batch_size=$BATCH \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --gradient_accumulation_steps 1 \
    --overwrite_output 