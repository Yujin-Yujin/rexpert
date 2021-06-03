# For CWWV, download the cskg_connected.tsv from here and cache.pkl from here, then run:

# python generate_from_CWWV.py \
#     --cskg_file /home/yujin/data2/yujin/kg/cwwv/cskg_connected.tsv \
#     --lex_cache /home/yujin/data2/yujin/kg/cwwv/cache.pkl \
#     --out_dir /home/yujin/r-expert/dataset/cwwv \
#     --strategy random

python filter_CWWV.py \
    --input_file /home/yujin/r-expert/dataset/cwwv/random.jsonl 