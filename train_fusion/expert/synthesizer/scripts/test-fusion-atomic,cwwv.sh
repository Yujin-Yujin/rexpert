cd /home/yujin/r-expert/adapter-transformers-synthesizer

pwd

pip install .

cd /home/yujin/r-expert/train_fusion/expert/synthesizer/scripts

export CUDA_VISIBLE_DEVICES=1
export DATASET=/home/yujin/r-expert/dataset/commonsense/origin
export PRETRAINED_ADAPTER_DIR_PATH=/home/yujin/r-expert/output/expert/fusions/synthe/atomic,cwwv
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/fusions/synthe/atomic,cwwv
export BATCH=8

export ADAPTER_NAMES=atomic,cwwv
python ../run_multiple_choice.py \
    --task_name csqa \
    --model_name_or_path roberta-large \
    --pretrained_adapter_names $ADAPTER_NAMES \
    --pretrained_adapter_dir_path $PRETRAINED_ADAPTER_DIR_PATH \
    --wandb_project "fusion-synthe" \
    --wandb_entity "rexpert" \
    --wandb_name "fusion-$ADAPTER_NAMES-full-test" \
    --test_fusion \
    --do_eval \
    --seed 42 \
    --data_dir $DATASET \
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