import pandas as pd
import os
from tqdm import tqdm

# train_dataset_path = "/home/yujin/aaai_2021/atomic2020_data-feb2021/train.tsv"
save_dir_path = "/home/yujin/aaai_2021/dataset/pretraining_dataset"
# df = pd.read_csv(train_dataset_path, delimiter="\t", names=["hyp","relation","event"])



# # precondition dataset
# precondition_dataset_path = os.path.join(save_dir_path, "precondition_all.jsonl")
# precondition_rel = ["isAfter", "xReason", "xNeed", "xIntent"]


# # postcondition dataset
# postcondition_rel = ["causes", "xEffect", "isBefore", "xReact", "xWant", "oEffect", "oReact", "oWant"]


# for index, row in df.iterrows():
#     if row["relation"] in precondition_rel:
        

# print(df)

org_qa_path = "/home/yujin/aaai_2021/dataset/atomic_2019/train_random.jsonl"

df = pd.read_json(org_qa_path, lines=True)

precondition_rel = ["isAfter", "xReason", "xNeed", "xIntent"]
# columns = ['id', 'dim', 'context', 'correct', 'candidates', 'keywords']

pre_df = pd.DataFrame()
post_df = pd.DataFrame()

with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        pbar.update(1)
        if row['dim'] in precondition_rel:
            pre_df = pre_df.append(row, ignore_index=True)
        else:
            post_df = post_df.append(row, ignore_index=True)

# save pre dataset
pre_trn_path = os.path.join(save_dir_path, "pre_trn.jsonl")
pre_dev_path = os.path.join(save_dir_path, "pre_dev.jsonl")
pre_test_path = os.path.join(save_dir_path, "pre_test.jsonl")

split_idx_list = [round(pre_df.shape[0] * 0.8), round(pre_df.shape[0] * 0.9)]
trn_json = pre_df.iloc[:split_idx_list[0]].to_json(orient='records')
dev_json = pre_df.iloc[split_idx_list[0]:split_idx_list[1]].to_json(orient='records')
test_json = pre_df.iloc[split_idx_list[1]:].to_json(orient='records')

with open(pre_trn_path, 'w') as f:
    for item in json.loads(trn_json):
        json.dump(item, f)
        f.write("\n")

with open(pre_dev_path, 'w') as f:
    for item in json.loads(dev_json):
        json.dump(item, f)
        f.write("\n")


with open(pre_test_path, 'w') as f:
    for item in json.loads(test_json):
        json.dump(item, f)
        f.write("\n")

print("pre_train_size :", len(trn_json))
print("pre_dev_size :", len(dev_json))
print("pre_test_size :", len(test_json))

# save post dataset
post_trn_path = os.path.join(save_dir_path, "post_trn.jsonl")
post_dev_path = os.path.join(save_dir_path, "post_dev.jsonl")
post_test_path = os.path.join(save_dir_path, "post_test.jsonl")

split_idx_list = [round(post_df.shape[0] * 0.8), round(post_df.shape[0] * 0.9)]
trn_json = post_df.iloc[:split_idx_list[0]].to_json(orient='records')
dev_json = post_df.iloc[split_idx_list[0]:split_idx_list[1]].to_json(orient='records')
test_json = post_df.iloc[split_idx_list[1]:].to_json(orient='records')

with open(post_trn_path, 'w') as f:
    for item in json.loads(trn_json):
        json.dump(item, f)
        f.write("\n")

with open(post_dev_path, 'w') as f:
    for item in json.loads(dev_json):
        json.dump(item, f)
        f.write("\n")


with open(post_test_path, 'w') as f:
    for item in json.loads(test_json):
        json.dump(item, f)
        f.write("\n")

print("post_train_size :", len(trn_json))
print("post_dev_size :", len(dev_json))
print("post_test_size :", len(test_json))


