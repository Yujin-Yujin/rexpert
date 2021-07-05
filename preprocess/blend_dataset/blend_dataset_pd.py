import pandas as pd
from tqdm import tqdm
import random
import json

atomic_dataset_path = ["/home/yujin/r-expert/dataset/atomic/full/train_random.jsonl","/home/yujin/r-expert/dataset/atomic/full/dev_random.jsonl"]
cwwv_dataset_path = ["/home/yujin/r-expert/dataset/cwwv/full/train_rand_split.jsonl","/home/yujin/r-expert/dataset/cwwv/full/dev_rand_split.jsonl"]
output_path = ["/home/yujin/r-expert/dataset/blended/full/train.jsonl","/home/yujin/r-expert/dataset/blended/full/dev.jsonl"]

def cwwv_convert(example):
    context = example["question"]["stem"]
    raw_candidates = example["question"]["choices"]
    answer_label = example["answerKey"]
    answer = [x['text'] for x in raw_candidates if x['label'] == answer_label ]
    distractors = [x['text'] for x in raw_candidates if x['label'] != answer_label ]
    answer.extend(distractors)

    return (context, answer)

def atomic_convert(example):
    context = example["context"]
    raw_candidates = example["candidates"]
    answer_label = example["correct"]
    answer = [raw_candidates[answer_label]]
    distractors = [x for x in raw_candidates if raw_candidates.index(x) != answer_label ]
    answer.extend(distractors)

    return (context, answer)

def answer_index_shuffle(candidates):
    index_list = list(range(len(candidates)))
    random.shuffle(index_list)

    shuffled_candidates = [candidates[i] for i in index_list ]
    answer_index = index_list.index(0)
    return (answer_index, shuffled_candidates)


for i in range(len(atomic_dataset_path)):
    # load original data
    atomic = pd.read_json(atomic_dataset_path[i], lines=True)
    cwwv = pd.read_json(cwwv_dataset_path[i], lines=True)

    blend_df = pd.DataFrame(columns=["context", "correct", "candidates"])

    for index, row in tqdm(atomic.iterrows(), total=atomic.shape[0]):
        a_context, a_candidates = atomic_convert(row)
        while index > cwwv.shape[0] - 1 :
            index = index - cwwv.shape[0]
        c_context, c_candidates = cwwv_convert(cwwv.iloc[index])

        n_candidates = [a_candidates[x] + " " + c_candidates[x] for x in range(len(a_candidates))]
        n_answer_index, n_candidates = answer_index_shuffle(n_candidates)

        blend_df = blend_df.append({"context" : a_context + " " + c_context,
        "correct" : n_answer_index,
        "candidates" : n_candidates}, ignore_index=True)
    
    
    blend_json = blend_df.to_json(orient='records')
    with open(output_path[i], 'w') as f:
        for item in json.loads(blend_json):
            json.dump(item, f)
            f.write("\n")
    
    print(output_path[i], "is saved.")