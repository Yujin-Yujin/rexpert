import pandas as pd
from tqdm import tqdm
import random
import json

atomic_dataset_path = ["/home/yujin/r-expert/dataset/atomic/full/train_random.jsonl","/home/yujin/r-expert/dataset/atomic/full/dev_random.jsonl"]
# atomic_dataset_path = ["/home/yujin/r-expert/dataset/atomic/small-test/train_random.jsonl","/home/yujin/r-expert/dataset/atomic/small-test/dev_random.jsonl"]
cwwv_dataset_path = ["/home/yujin/r-expert/dataset/cwwv/full/train_rand_split.jsonl","/home/yujin/r-expert/dataset/cwwv/full/dev_rand_split.jsonl"]
output_path = ["/home/yujin/r-expert/dataset/blended/full/train.jsonl","/home/yujin/r-expert/dataset/blended/full/dev.jsonl"]

def cwwv_convert(example):
    example = json.loads(example.strip("\n"))
    context = example["question"]["stem"]
    raw_candidates = example["question"]["choices"]
    answer_label = example["answerKey"]
    answer = [x['text'] for x in raw_candidates if x['label'] == answer_label ]
    distractors = [x['text'] for x in raw_candidates if x['label'] != answer_label ]
    answer.extend(distractors)

    return (context, answer)

def atomic_convert(example):
    example = json.loads(example.strip("\n"))
    context = example["context"]
    raw_candidates = example["candidates"]
    answer_label = example["correct"]
    answer = [raw_candidates[answer_label]]
    distractors = [x for idx, x in enumerate(raw_candidates) if idx != answer_label ]
    answer.extend(distractors)

    return (context, answer)

def answer_index_shuffle(candidates):
    index_list = list(range(len(candidates)))
    random.shuffle(index_list)

    shuffled_candidates = [candidates[i] for i in index_list ]
    answer_index = index_list.index(0)
    return (answer_index, shuffled_candidates)

def read_json(input_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        return lines

for i in range(len(atomic_dataset_path)):
    # load original data
    atomic = read_json(atomic_dataset_path[i])
    cwwv = read_json(cwwv_dataset_path[i])
    blend_df = pd.DataFrame(columns=["context", "correct", "candidates"])

    for index, row in tqdm(enumerate(atomic),total=len(atomic)):
        a_context, a_candidates = atomic_convert(row)
        while index > len(cwwv) - 1 :
            index = index - len(cwwv)
        c_context, c_candidates = cwwv_convert(cwwv[index])

        assert len(a_candidates) == len(a_candidates) and len(a_candidates) == 3, "candidates error"

        n_candidates = [a_candidates[x] + " " + c_candidates[x] for x in range(len(a_candidates))]
        assert len(n_candidates) == 3, "candidates error2"

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