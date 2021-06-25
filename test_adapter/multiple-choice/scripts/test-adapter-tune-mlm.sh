cd /home/yujin/r-expert/adapter-transformers
pip install .
cd /home/yujin/r-expert/test_adapter/multiple-choice/scripts

export CUDA_VISIBLE_DEVICES=4
export OUTPUT_DIR=/home/yujin/r-expert/output/expert/adapters/mlm
# export DATASET=/home/yujin/r-expert/dataset/socialiqa/origin
# export DATASET=/home/yujin/r-expert/dataset/commonsense/origin

export ADAPTER_NAME="atomic_mlm"
# export TASK_NAME="csqa"

# python ../run_multiple_choice.py \
#     --task_name $TASK_NAME \
#     --seed 42 \
#     --model_name_or_path roberta-large \
#     --adapter_name $ADAPTER_NAME\
#     --load_adapter $OUTPUT_DIR/$ADAPTER_NAME \
#     --do_eval \
#     --data_dir $DATASET \
#     --max_seq_length 128 \
#     --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
#     --per_device_eval_batch_size=8 \
#     --overwrite_output ;

export DATASET=/home/yujin/r-expert/dataset/socialiqa/origin
export TASK_NAME="siqa"

python ../run_multiple_choice.py \
    --task_name $TASK_NAME \
    --seed 42 \
    --model_name_or_path roberta-large \
    --adapter_name $ADAPTER_NAME\
    --load_adapter $OUTPUT_DIR/$ADAPTER_NAME \
    --do_predict \
    --data_dir $DATASET \
    --max_seq_length 128 \
    --output_dir $OUTPUT_DIR/$ADAPTER_NAME \
    --per_device_eval_batch_size=8 \
    --overwrite_output 