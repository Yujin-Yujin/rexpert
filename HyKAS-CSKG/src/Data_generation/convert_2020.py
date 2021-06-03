import pandas as pd
import ast
from tqdm import tqdm
import numpy as np

xset = ['PersonX', 'Personx', 'personX', 'personx', 'Person X', 'Person x', 'person X', 'person x']
yset = ['PersonY', 'Persony', 'personY', 'persony', 'Person Y', 'Person y', 'person Y', 'person y']
zset = ['PersonZ', 'Personz', 'personZ', 'personz', 'Person Z', 'Person z', 'person Z', 'person z']

origin_file="/home/yujin/data2/yujin/kg/atomic_2019/v4_atomic_trn.csv"
# df = pd.read_csv(origin_file)
# print(df)
# raise RuntimeError()

atomic_2020_file="/home/yujin/data2/yujin/kg/atomic2020_data-feb2021/train.tsv"
df = pd.read_csv(atomic_2020_file, delimiter='\t', names=["h_event", "rel", "t_event"])
print(df.columns)
print(df)

split_type = "trn"

columns = ["event", "ObjectUse", "AtLocation", "MadeUpOf", "HasProperty","CapableOf", 
            "Desires", "NotDesires", "isAfter", "HasSubEvent", "isBefore",
            "HinderedBy", "Causes", "xReason", "isFilledBy", "xNeed",
            "xAttr", "xEffect", "xReact", "xWant", "xIntent",
            "oEffect", "oReact", "oWant", 'prefix', 'split']
final_df = pd.DataFrame(columns=columns)

def find_prefix(event_sent):
    event_sent = event_sent.split(" ")
    prefix = []
    for s in event_sent:
        flag = False
        for x in xset:
            if x in s:
                flag=True
                break
        if flag:
            break
        for y in yset:
            if y in s:
                flag=True
                break
        if flag:
            break
        for z in zset:
            if z in s:
                flag=True
                break
        if flag:
            break
        
        if '___' in s:
            flag = True
            break

        else:
            prefix.append(s)

    return prefix
        
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if final_df['event'].str.contains(row['h_event']).any():
        # _temp = str(final_df.loc[final_df['event'] == row['h_event'], row['rel']].item())
        try:
            _temp = str(final_df.loc[final_df['event'] == row['h_event'], row['rel']].item())
        except:
            print(row['h_event'])
            
        _temp = ast.literal_eval(_temp)
        if row['t_event'] == np.nan or pd.isna(row['t_event']):
            _temp.append("none")
        else:
            _temp.append(row["t_event"])
        
        final_df.loc[final_df['event'] == row['h_event'], row['rel']] = str(_temp)

    else:
        _prefix = find_prefix(row['h_event'])
        
        final_df = final_df.append({
            "event": row["h_event"], 
            "ObjectUse": [], 
            "AtLocation": [], 
            "MadeUpOf": [], 
            "HasProperty": [],
            "CapableOf": [], 
            "Desires": [], 
            "NotDesires": [], 
            "isAfter": [], 
            "HasSubEvent": [], 
            "isBefore": [],
            "HinderedBy": [], 
            "Causes": [], 
            "xReason": [], 
            "isFilledBy": [], 
            "xNeed": [],
            "xAttr": [], 
            "xEffect": [], 
            "xReact": [], 
            "xWant": [], 
            "xIntent": [],
            "oEffect": [], 
            "oReact": [], 
            "oWant": [], 
            'prefix': _prefix, 
            'split': split_type
        }, ignore_index=True)

        if row['t_event'] == np.nan or row['t_event'] is None:
            final_df.loc[final_df['event'] == row['h_event'], row['rel']] = str(["none"])
        else:
            final_df.loc[final_df['event'] == row['h_event'], row['rel']] = str([row['t_event']])

final_df.to_csv("/home/yujin/r-expert/HyKAS-CSKG/src/Data_generation/a.csv")

# new_df = pd.read_csv("/home/yujin/r-expert/HyKAS-CSKG/src/Data_generation/a.csv")
# print(new_df)