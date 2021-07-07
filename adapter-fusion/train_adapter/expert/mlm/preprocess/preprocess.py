  
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from fairseq.data.data_utils import collate_tokens
import random
import json
# from fairseq.data.data_utils import collate_tokens


ATOMIC2NL = {'xIntent': 'Because PersonX wanted ',
            'xNeed': 'Before, PersonX needed ', 
            'xAttr': 'PersonX is seen as ', 
            'xEffect': 'As a result, PersonX feels ', 
            'xWant': 'As a result, PersonX wants ',
            'xReact': 'PersonX then ', 
            'oEffect': 'As a result, others feel ',
            'oWant': 'As a result, others want ', 
            'oReact': 'Others then '}

class AtomicPreprocess():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def convert_file(self, raw_data_file_path):
        output_path = os.path.join(self.output_dir, "atomic_sentence.csv")
        if os.path.isfile(output_path):
            print("generate dataset using {} file.".format(output_path))
            df_new = pd.read_csv(output_path, index_col=None)
        else:
            # convert into sentence
            df = pd.read_csv(raw_data_file_path)
            df_new = pd.DataFrame(columns=["premise","continuation"])
            event_col = df.columns[0]
            prop_cols = df.columns[1:10]

            for index, row in df.iterrows():
                event = row[event_col]
                for prop in prop_cols:
                    prop_events = json.loads(row[prop])
                    for item in prop_events:
                        if item != "none":
                            following_event = ATOMIC2NL[prop] + item
                            df_new= df_new.append({df_new.columns[0]: event, df_new.columns[1]: following_event}, ignore_index=True)
                if index % 1000 == 0 :
                    print(index/1000, df.shape[0])    

            # premise - continuation format
            df_new.to_csv(output_path, index=False)
            print("converted file is generated in {}.".format(output_path))

        return df_new

    def genearte_dataset(self, origin_file_path, task_type, shuffle=True):
        df = pd.read_json(origin_file_path, lines=True)

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        # split into train / dev (9:1)
        split_idx_list = [round(df.shape[0] * 0.9)]

        if "mlm" in task_type:
            mlm_output_path = os.path.join(self.output_dir,"mlm")
            if not os.path.isdir(mlm_output_path):
                os.mkdir(mlm_output_path)
            mlm_trn_output_path = os.path.join(mlm_output_path,"mlm_trn.txt")
            mlm_dev_output_path = os.path.join(mlm_output_path,"mlm_dev.txt")

            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                df.loc[index,'mlm'] = row['context'] + row['candidates'][row['correct']]
            
            np.savetxt(mlm_trn_output_path, df['mlm'].iloc[:split_idx_list[0]].values, fmt="%s")
            np.savetxt(mlm_dev_output_path, df['mlm'].iloc[split_idx_list[0]:].values, fmt="%s")

            print("mlm dataset is saved in {}.".format(mlm_output_path))

        if "cls" in task_type:
            cls_output_path = os.path.join(self.output_dir,"cls")
            if not os.path.isdir(cls_output_path):
                os.mkdir(cls_output_path)
            cls_trn_output_path = os.path.join(cls_output_path,"cls_trn.jsonl")
            cls_dev_output_path = os.path.join(cls_output_path,"cls_dev.jsonl")

            cls_df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                cls_df.loc[index,'sentence1'] = "".join(row["context"].split(".")[:-1]) + "."
                for c_idx, candi in enumerate(row['candidates']):
                    cls_df.loc[index,'sentence2'] = row["context"].split(".")[-1].lstrip() + candi
                    if c_idx == row['correct']:
                        cls_df.loc[index,'label'] = "True"
                    else:
                        cls_df.loc[index,'label'] = "False"
            
            # split into train / dev (9: 1)
            trn_json = cls_df.iloc[:split_idx_list[0]].to_json(orient='records')
            dev_json = cls_df.iloc[split_idx_list[0]:].to_json(orient='records')
            
            with open(cls_trn_output_path, 'w') as f:
                for item in json.loads(trn_json):
                    json.dump(item, f)
                    f.write("\n")

            with open(cls_dev_output_path, 'w') as f:
                for item in json.loads(dev_json):
                    json.dump(item, f)
                    f.write("\n")

            # with open(cls_trn_output_path, 'w') as f:
            #     for item in json.loads(trn_json):
            #         f.write(str(item) + '\n')

            # with open(cls_dev_output_path, 'w') as f:
            #     for item in json.loads(dev_json):
            #         f.write(str(item) + '\n')

            print("cls dataset is saved in {}.".format(cls_output_path))

        if "link_cls" in task_type:
            link_cls_output_path = os.path.join(self.output_dir,"link_cls")
            if not os.path.isdir(link_cls_output_path):
                os.mkdir(link_cls_output_path)
            link_cls_trn_output_path = os.path.join(link_cls_output_path,"link_cls_trn.jsonl")
            link_cls_dev_output_path = os.path.join(link_cls_output_path,"link_cls_dev.jsonl")

            link_cls_df = pd.DataFrame(columns=['sentence1', 'sentence2', 'label'])
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                link_cls_df.loc[index,'sentence1'] = "".join(row["context"].split(".")[:-1]) + "."
                link_cls_df.loc[index,'sentence2'] = row["context"].split(".")[-1].lstrip() + row['candidates'][row['correct']]
                link_cls_df.loc[index,'label'] = row["dim"]

            # split into train / dev (9: 1)
            trn_json = link_cls_df.iloc[:split_idx_list[0]].to_json(orient='records')
            dev_json = link_cls_df.iloc[split_idx_list[0]:].to_json(orient='records')

            with open(link_cls_trn_output_path, 'w') as f:
                for item in json.loads(trn_json):
                    json.dump(item, f)
                    f.write("\n")

            with open(link_cls_dev_output_path, 'w') as f:
                for item in json.loads(dev_json):
                    json.dump(item, f)
                    f.write("\n")

            # with open(link_cls_trn_output_path, 'w') as f:
            #     for item in json.loads(trn_json):
            #         f.write(str(item) + '\n')

            # with open(link_cls_dev_output_path, 'w') as f:
            #     for item in json.loads(dev_json):
            #         f.write(str(item) + '\n')

            print("cls dataset is saved in {}.".format(link_cls_output_path))



        # elif "sc" in task_type:
        #     sc_output_path = os.path.join(self.output_dir,"sc")
        #     if not os.path.isdir(sc_output_path):
        #         os.mkdir(sc_output_path)
        #     sc_trn_output_path = os.path.join(sc_output_path,"sc_trn.jsonl")
        #     sc_dev_output_path = os.path.join(sc_output_path,"sc_dev.jsonl")
        #     sc_test_output_path = os.path.join(sc_output_path,"sc_test.jsonl")

        #     neg_sample_num = 0
        #     pos_sample_num = 0


        #     # use only 50%
        #     sample_df = df.sample(frac=0.2, replace=False, random_state=1)
        #     print("original df",df.shape)
        #     print("sampled df",sample_df.shape)

        #     # generate positive sample with kg
        #     sc_df = pd.DataFrame(columns=["premise", "hypothesis","label"])
        #     sc_df[["premise", "hypothesis"]] = sample_df[sample_df.columns]
        #     sc_df["label"] = 1
        #     pos_sample_num = sc_df.shape[0]

        #     # generate negative sample with pretrained mnli
        #     roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli', force_reload=True)
        #     roberta.cuda()
        #     roberta.eval()

        #     premise_list = df[df.columns[0]].unique().tolist()

        #     hypothesis_list = df[df.columns[1]].unique().tolist()

        #     for premise in tqdm.tqdm(random.sample(premise_list, 10000), desc="generate negative sample"):
        #         hypothesis_candidates = df.loc[df[df.columns[0]] != premise][df.columns[1]].unique().tolist()

        #         hypothesis_candidates_rand = random.sample(hypothesis_candidates, 100)
        #         for candi in hypothesis_candidates_rand:
        #             tokens = roberta.encode(premise, candi)
        #             prediction = roberta.predict('mnli', tokens)
        #             m = nn.Softmax(dim=1)
        #             probability = m(prediction)
        #             if probability[0][0] > 0.85:
        #                 sc_df = sc_df.append({"premise":premise,
        #                                 "hypothesis":candi,
        #                                 "label": 0}, ignore_index=True)
        #                 neg_sample_num += 1


                

        #     # shuffle and split and save to file
        #     sc_df = sc_df.sample(frac=1).reset_index(drop=True)

        #     # split into train / dev / test (8:1:1)
        #     sc_split_idx_list = [round(sc_df.shape[0] * 0.8), round(sc_df.shape[0] * 0.9)]
        #     trn_json = sc_df.iloc[:sc_split_idx_list[0]].to_json(orient='records')
        #     dev_json = sc_df.iloc[sc_split_idx_list[0]:sc_split_idx_list[1]].to_json(orient='records')
        #     test_json = sc_df.iloc[sc_split_idx_list[1]:].to_json(orient='records')

        #     with open(sc_trn_output_path, 'w') as f:
        #         for item in json.loads(trn_json):
        #             f.write(str(item) + '\n')

        #     with open(sc_dev_output_path, 'w') as f:
        #         for item in json.loads(dev_json):
        #             f.write(str(item) + '\n')

        #     with open(sc_test_output_path, 'w') as f:
        #         for item in json.loads(test_json):
        #             f.write(str(item) + '\n')
        #     print("number of pos_sample_num : {} \n".format(pos_sample_num))
        #     print("number of neg_sample_num : {} \n".format(neg_sample_num))
        #     print("sc dataset is saved in {}.".format(sc_output_path))

                
def main():

    # inputs
    origin_file_path = "/home/yujin/r-expert/dataset/atomic/full/train_random.jsonl"
    output_path = "/home/yujin/r-expert/dataset/atomic"

    # mlm - masked language model
    # nsp - next sentence prediction
    # sc - squence classification
    # qa - question answering
    # cls - classification
    # link_cls - link classification

    # task_type=["mlm", "nsp", "sc", "qa"]
    task_type=["link_cls"]
    # task_type=["cls"]


    preprocess = AtomicPreprocess(output_path)
    
    df = preprocess.genearte_dataset(origin_file_path, task_type)





if __name__ == '__main__':
    main()