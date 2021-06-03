# python generate_from_ATOMIC.py \
#     --train_KG /home/yujin/aaai_2021/atomic2020_data-feb2021/train.tsv \
#     --dev_KG /home/yujin/aaai_2021/atomic2020_data-feb2021/dev.tsv \
#     --strategy random \
#     --out_dir /home/yujin/aaai_2021/dataset/pretraining_dataset


# python generate_from_ATOMIC.py \
#     --train_KG /home/yujin/aaai_2021/atomic2020_data-feb2021/train.tsv \
#     --dev_KG /home/yujin/aaai_2021/atomic2020_data-feb2021/dev.tsv \
#     --strategy random \
#     --out_dir /home/yujin/aaai_2021/dataset/pretraining_dataset

python generate_from_ATOMIC.py \
    --train_KG /home/yujin/data2/yujin/kg/atomic_2019/v4_atomic_trn.csv \
    --dev_KG /home/yujin/data2/yujin/kg/atomic_2019/v4_atomic_dev.csv \
    --strategy random \
    --out_dir /home/yujin/r-expert/dataset/atomic
