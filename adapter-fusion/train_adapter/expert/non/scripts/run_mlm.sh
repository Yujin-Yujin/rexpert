# cd /home/yujin/r-expert/adapter-transformers

# pip install .

cd /home/yujin/r-expert/train_adapter/expert/mlm/10k/scripts

export CUDA_VISIBLE_DEVICES=4
export DATASET=/home/yujin/r-expert/dataset/atomic/10k
export OUTPUT_PATH=/home/yujin/r-expert/output/test
export ADAPTER_NAME=atomic

python ../run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm

# python ../run_mlm.py \
#     --model_name_or_path roberta-base \
#     --train_file path_to_train_file \
#     --validation_file path_to_validation_file \
#     --do_train \
#     --do_eval \
#     --output_dir /tmp/test-mlm
