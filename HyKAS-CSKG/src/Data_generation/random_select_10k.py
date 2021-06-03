import pandas as pd
from random import sample
import json

# trn_path = "/home/yujin/r-expert/dataset/atomic/full/train_random.jsonl"
# dev_path = "/home/yujin/r-expert/dataset/atomic/full/dev_random.jsonl"
# trn_output_path = "/home/yujin/r-expert/dataset/atomic/10k/train_random.jsonl"
# dev_output_path = "/home/yujin/r-expert/dataset/atomic/10k/dev_random.jsonl"

trn_path = "/home/yujin/r-expert/dataset/cwwv/full/train_rand_split.jsonl"
dev_path = "/home/yujin/r-expert/dataset/cwwv/full/dev_rand_split.jsonl"
trn_output_path = "/home/yujin/r-expert/dataset/cwwv/10k/train_rand_split.jsonl"
dev_output_path = "/home/yujin/r-expert/dataset/cwwv/10k/dev_rand_split.jsonl"

output_path_list = [trn_output_path, dev_output_path]
dataset_num_list = [10000,1000]
for index, d_path in enumerate([trn_path, dev_path]):
    df = pd.read_json(d_path, lines=True)

    range_list = list(range(1,df.shape[0]+1))
    random_ids = sample(range_list,dataset_num_list[index])

    sel_df = df[df.index.isin(random_ids)]

    df_json = sel_df.to_json(orient='records')

    with open(output_path_list[index], 'w') as f:
        for item in json.loads(df_json):
            json.dump(item, f)
            f.write("\n")

print("done!")