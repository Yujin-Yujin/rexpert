export CUDA_VISIBLE_DEVICES=1
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/r-expert/output/best/expert/adapters/full
# export OUTPUT_DIR=/home/yujin/r-expert/output/expert/fusions/full
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/fusions/blend/10k/atomic,cwwv
export BATCH=8
export BEST_MODEL_PATH=/home/yujin/r-expert/output/best/expert/fusions/10k
export PRETRAINED_FUSION_PATH=/home/yujin/r-expert/output/expert/fusions/blend/10k/atomic,cwwv
# export TASK_NAME=siqa
# export DATASET=/home/yujin/r-expert/dataset/socialiqa/origin
export TASK_NAME=csqa
export DATASET=/home/yujin/r-expert/dataset/commonsense/origin

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-multikg" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-test" \
    --fusion_path $PRETRAINED_FUSION_PATH/$ADAPTER_NAMES \
    --test_fusion \
    --do_eval \
    --seed 42 \
    --data_dir $DATASET \
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