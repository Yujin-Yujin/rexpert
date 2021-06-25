# export CUDA_VISIBLE_DEVICES=3
# export OUTPUT_DIR=/home/yujin/r-expert/output/best/expert/adapters/pre
# export =/home/yujin/data2/yujin/dataset/socialiqa

# export ADAPTER_NAME="xAttr"

# python ../run_multiple_choice.py \
#     --task_name siqa \
#     --seed 42 \
#     --model_name_or_path roberta-large \
#     --adapter_name $ADAPTER_NAME\
#     --load_adapter $OUTPUT_DIR/$ADAPTER_NAME \
#     --do_predict \
#     --data_dir $SIQA_DIR \
#     --max_seq_length 128 \
#     --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
#     --per_device_eval_batch_size=8 \
#     --overwrite_output 

export CUDA_VISIBLE_DEVICES=1
# export OUTPUT_DIR=/home/yujin/aaai_2021/output/pretrained/mnli/
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/adapters/mr/atomic
export DATASET_DIR=/home/yujin/r-expert/dataset/socialiqa/origin
export TASK_NAME=siqa
export ADAPTER_NAME="atomic"

python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --seed 42 \
    --model_name_or_path roberta-large \
    --adapter_name $ADAPTER_NAME\
    --load_adapter $OUTPUT_DIR/$ADAPTER_NAME \
    --do_eval \
    --do_predict \
    --data_dir $DATASET_DIR \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --per_device_eval_batch_size=8 \
    --overwrite_output 