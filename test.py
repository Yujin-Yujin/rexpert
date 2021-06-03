# import json
# import torch
# from fairseq.models.roberta import RobertaModel
# # from examples.roberta import commonsense_qa  # load the Commonsense QA task
# roberta = RobertaModel.from_pretrained('roberta-large')
# roberta.eval()  # disable dropout
# roberta.cuda()  # use the GPU (optional)
# nsamples, ncorrect = 0, 0
# with open('/home/yujin/r-expert/dataset/origin/dev_rand_split.jsonl') as h:
#     for line in h:
#         example = json.loads(line)
#         scores = []
#         for choice in example['question']['choices']:
#             input = roberta.encode(
#                 'Q: ' + example['question']['stem'],
#                 'A: ' + choice['text'],
#                 no_separator=True
#             )
#             score = roberta.predict('sentence_classification_head', input, return_logits=True)
#             scores.append(score)
#         pred = torch.cat(scores).argmax()
#         answer = ord(example['answerKey']) - ord('A')
#         nsamples += 1
#         if pred == answer:
#             ncorrect += 1

# print('Accuracy: ' + str(ncorrect / float(nsamples)))
# # Accuracy: 0.7846027846027847

label_map = {label: i for i, label in enumerate(["A","B","C","D","E"])}

print(label_map["A"])