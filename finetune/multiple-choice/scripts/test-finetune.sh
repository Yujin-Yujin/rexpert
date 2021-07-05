# cd /home/yujin/r-expert/adapter-transformers
# pip install .
# cd /home/yujin/r-expert/finetune/scripts

export CUDA_VISIBLE_DEVICES=3
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/finetune
export EPOCH=1
# export TASK_NAME=siqa
# export DATASET=/home/yujin/r-expert/dataset/socialiqa/origin
export TASK_NAME=csqa
export DATASET=/home/yujin/r-expert/dataset/commonsense/origin
export PRETRAINED_MODEL_PATH=/home/yujin/r-expert/output/expert/finetune/multikg

python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --seed 42 \
    --model_name_or_path $PRETRAINED_MODEL_PATH \
    --wandb_project "finetune-kg" \
    --wandb_entity "rexpert" \
    --wandb_name "finetune-$TASK_NAME" \
    --do_eval \
    --data_dir $DATASET \
    --num_train_epochs $EPOCH \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$TASK_NAME \
    --per_device_eval_batch_size=8 \
    --overwrite_output  