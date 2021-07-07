import pandas as pd
import json
with open("/home/yujin/r-expert/dataset/atomic/cls/past/cls_trn.jsonl", 'r') as f:
    for line in f:
        print(line)
        json_line = json.loads(line)
# b = {"sentence1": "Ash scrapes together a ___.", "sentence2": "Ash thenAsh yells \" ow \" as they step on a sharp object in the pantry.", "label": "False"}
# a = json.loads(b)
# df = pd.read_json("/home/yujin/r-expert/dataset/atomic/cls/past/cls_trn.jsonl", lines=True)

# print(df)