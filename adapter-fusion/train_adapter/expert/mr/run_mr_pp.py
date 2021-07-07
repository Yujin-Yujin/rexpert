#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch 
import numpy as np
import random
# from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.adapters.configuration import AdapterConfig
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from data_utils import accuracy, myprocessors, convert_examples_to_features
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from filelock import FileLock
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(myprocessors.keys())})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    adapter_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_words_to_mask: Optional[int] = field(
        default=6,
        metadata={
            "help":"The maximum number of tokens to mask when computing scores"
        }
    )
    margin: Optional[float] = field(
        default=1.0,
        metadata={
            "help":"The margin for ranking loss"
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt","jsonl"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt","jsonl"], "`validation_file` should be a csv, a json or a txt file."

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, data, pad_token, mask_token, max_words_to_mask):
		self.data = data
		self.pad_token = pad_token
		self.mask_token = mask_token
		self.max_words_to_mask = max_words_to_mask

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample, self.pad_token, self.mask_token, self.max_words_to_mask
def load_and_cache_examples(args, output_dir, task, tokenizer, evaluate=False):
	# if args.local_rank not in [-1, 0] and not evaluate:
		# torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
	processor = myprocessors[task](args)
	if evaluate:
	    data_type = "dev"
	else:
	    data_type = "train"
	cached_features_file = os.path.join(output_dir, 'cached_{}_{}_{}'.format(
		data_type,
		str(args.max_seq_length),
		str(task)))

	lock_path = cached_features_file + ".lock"
	with FileLock(lock_path):
		if os.path.exists(cached_features_file):
			features = torch.load(cached_features_file)
		else:
			logger.info("Saving features into cached file %s", cached_features_file)
			examples = processor.get_dev_examples() if evaluate else processor.get_train_examples()
			features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length)
			torch.save(features, cached_features_file)
	# if args.local_rank == 0 and not evaluate:
		# torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
	print ('max_words_to_mask is %s for pretraining tasks %s' % (args.max_words_to_mask, task))
	return MyDataset(features, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask)

def mCollateFn(batch):
	batch_input_ids = []
	batch_input_mask = []
	batch_input_labels = []
	batch_label_ids = []
	features = [b[0] for b in batch]
	pad_token = batch[0][1]
	mask_token = batch[0][2]
	MAX_WORDS_TO_MASK = batch[0][3]
	max_len = max([len(cand) for f in features for cand in f[0]])
	for f in features:
		batch_input_ids.append([])
		batch_input_mask.append([])
		batch_input_labels.append([])
		batch_label_ids.append(f[2])
		for i in range(len(f[0])):
			masked_sequences = []
			masked_labels = []
			this_att_mask = []
			sequence = f[0][i] + [pad_token]*(max_len-len(f[0][i]))
			label_sequence = f[1][i]+[-100]*(max_len-len(f[1][i]))
			valid_indices = [l_i for l_i, l in enumerate(label_sequence) if l != -100]
			if len(valid_indices) > MAX_WORDS_TO_MASK:
				rm_indices = random.sample(valid_indices, (len(valid_indices)-MAX_WORDS_TO_MASK))
				label_sequence = [-100 if l_i in rm_indices else l for l_i, l in enumerate(label_sequence)]
			for j, t in enumerate(label_sequence):
				if t == -100:
					continue
					masked_sequences.append(sequence)
					masked_labels.append([-100]*max_len)
				else:
					masked_sequences.append(sequence[:j]+[mask_token]+sequence[j+1:])
					masked_labels.append([-100]*j+[sequence[j]]+[-100]*(max_len-j-1))
				this_att_mask.append([1]*len(f[0][i])+[0]*(max_len-len(f[0][i])))
			batch_input_ids[-1].append(torch.tensor(masked_sequences, dtype=torch.long))
			batch_input_mask[-1].append(torch.tensor(this_att_mask, dtype=torch.long))
			batch_input_labels[-1].append(torch.tensor(masked_labels, dtype=torch.long))
	
	# result_batch = {
	#     "input_ids" : batch_input_ids,
	#     "attention_mask" : batch_input_mask,
	#     "labels" : batch_input_labels
	# }
	return batch_input_ids, batch_input_mask, batch_input_labels, torch.tensor(batch_label_ids, dtype=torch.long)
    
	# return result_batch

def train(args, data_args,train_dataset, model, tokenizer, eval_dataset):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

	train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=mCollateFn)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	# warmup_steps = args.warmup_steps if args.warmup_steps != 0 else int(args.warmup_proportion * t_total)
	warmup_steps = 0
	logger.info("warm up steps = %d", warmup_steps)
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.98))
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	if args.fp16:
		try:
			from apex import amp
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
		model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank,
														  find_unused_parameters=True)
	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
	logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
				   train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	# set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
	set_seed(args.seed)
	curr_best = 0.0
	CE = torch.nn.CrossEntropyLoss(reduction='none')
	loss_fct = torch.nn.MultiMarginLoss(margin=data_args.margin)
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, batch in enumerate(epoch_iterator):
			model.train()
			# num_cand = len(batch[0][0])
			num_cand = len(batch[0][0])
			choice_loss = []
			choice_seq_lens = np.array([0]+[len(c) for sample in batch[0] for c in sample])
			choice_seq_lens = np.cumsum(choice_seq_lens)
			input_ids = torch.cat([c for sample in batch[0] for c in sample], dim=0).to(args.device)
			att_mask = torch.cat([c for sample in batch[1] for c in sample], dim=0).to(args.device)
			input_labels = torch.cat([c for sample in batch[2] for c in sample], dim=0).to(args.device)
			if len(input_ids) < data_args.max_seq_length:
				inputs = {'input_ids': input_ids,
						  'attention_mask': att_mask}
				outputs = model(**inputs)
				ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels.view(-1))
				ce_loss = ce_loss.view(outputs[0].size(0), -1).sum(1)
			else:
				ce_loss = []
				for chunk in range(0, len(input_ids), data_args.max_seq_length):
					inputs = {'input_ids': input_ids[chunk:chunk+data_args.max_seq_length],
						  'attention_mask': att_mask[chunk:chunk+data_args.max_seq_length]}
					outputs = model(**inputs)
					tmp_ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels[chunk:chunk+data_args.max_seq_length].view(-1))
					tmp_ce_loss = tmp_ce_loss.view(outputs[0].size(0), -1).sum(1)
					ce_loss.append(tmp_ce_loss)
				ce_loss = torch.cat(ce_loss, dim=0)
			# all tokens are valid
			for c_i in range(len(choice_seq_lens)-1):
				start = choice_seq_lens[c_i]
				end =  choice_seq_lens[c_i+1]
				choice_loss.append(-ce_loss[start:end].sum()/(end-start))

			choice_loss = torch.stack(choice_loss)
			choice_loss = choice_loss.view(-1, num_cand)
			loss = loss_fct(choice_loss, batch[3].to(args.device))

			if args.n_gpu > 1:
				loss = loss.mean() # mean() to average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps

			if args.fp16:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					# Log metrics
					tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
					tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
					tb_writer.add_scalar('Batch_loss', loss.item()*args.gradient_accumulation_steps, global_step)
					logger.info(" global_step = %s, average loss = %s", global_step, (tr_loss - logging_loss)/args.logging_steps)
					logging_loss = tr_loss

				pooh_save_val = len(train_dataset) / train_batch_size
				print(pooh_save_val)
				if args.local_rank == -1 and True and global_step % (pooh_save_val) == 0:
					results = evaluate(args, data_args, model, tokenizer, eval_dataset)
					for key, value in results.items():
						tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
					if results['acc'] > curr_best:
						curr_best = results['acc']
						# Save model checkpoint
						output_dir = args.output_dir
						if not os.path.exists(output_dir):
							os.makedirs(output_dir)
						model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
						model_to_save.save_pretrained(output_dir)
						tokenizer.save_pretrained(output_dir)
						torch.save(args, os.path.join(output_dir, 'training_args.bin'))
						logger.info("Saving model checkpoint to %s", output_dir)
					

			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break
	results = evaluate(args, data_args, model, tokenizer, eval_dataset)
	for key, value in results.items():
		tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
	if results['acc'] > curr_best:
		curr_best = results['acc']
		# Save model checkpoint
		output_dir = args.output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)
		torch.save(args, os.path.join(output_dir, 'training_args.bin'))
		logger.info("Saving model checkpoint to %s", output_dir)
		model.save_all_adapters(output_dir)

	if args.local_rank in [-1, 0]:
		tb_writer.close()
	return global_step, tr_loss / global_step

def evaluate(args, data_args, model, tokenizer, eval_dataset):
	results = {}
	if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(args.output_dir)
	
	eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=mCollateFn)

	# Eval!
	logger.info("***** Running evaluation *****")
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", eval_batch_size)
	CE = torch.nn.CrossEntropyLoss(reduction='none')
	preds = []
	out_label_ids = []
	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		model.eval()
		with torch.no_grad():
			num_cand = len(batch[0][0])
			choice_loss = []
			choice_seq_lens = np.array([0]+[len(c) for sample in batch[0] for c in sample])
			choice_seq_lens = np.cumsum(choice_seq_lens)
			input_ids = torch.cat([c for sample in batch[0] for c in sample], dim=0).to(args.device)
			att_mask = torch.cat([c for sample in batch[1] for c in sample], dim=0).to(args.device)
			input_labels = torch.cat([c for sample in batch[2] for c in sample], dim=0).to(args.device)
			if len(input_ids) < data_args.max_seq_length:
				inputs = {'input_ids': input_ids,
						  'attention_mask': att_mask}
				outputs = model(**inputs)
				ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels.view(-1))
				ce_loss = ce_loss.view(outputs[0].size(0), -1).sum(1)
			else:
				ce_loss = []
				for chunk in range(0, len(input_ids), data_args.max_seq_length):
					inputs = {'input_ids': input_ids[chunk:chunk+data_args.max_seq_length],
						  'attention_mask': att_mask[chunk:chunk+data_args.max_seq_length]}
					outputs = model(**inputs)
					tmp_ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels[chunk:chunk+data_args.max_seq_length].view(-1))
					tmp_ce_loss = tmp_ce_loss.view(outputs[0].size(0), -1).sum(1)
					ce_loss.append(tmp_ce_loss)
				ce_loss = torch.cat(ce_loss, dim=0)
			for c_i in range(len(choice_seq_lens)-1):
				start = choice_seq_lens[c_i]
				end =  choice_seq_lens[c_i+1]
				choice_loss.append(-ce_loss[start:end].sum()/(end-start))
			choice_loss = torch.stack(choice_loss)
			choice_loss = choice_loss.view(-1, num_cand)
		preds.append(choice_loss)
		out_label_ids.append(batch[3].numpy())
	preds = torch.cat(preds, dim=0).cpu().numpy()
	save_logits(preds.tolist(), os.path.join(args.output_dir, 'logits_test.txt'))
	preds = np.argmax(preds, axis=1)
	result = accuracy(preds, np.concatenate(out_label_ids, axis=0))
	results.update(result)
	output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
	with open(output_eval_file, "w") as writer:
		logger.info("***** Eval results *****")
		for key in sorted(result.keys()):
			logger.info("  %s = %s", key, str(result[key]))
			writer.write("%s = %s\n" % (key, str(result[key])))
	return results

def write_data(filename, data):
    with open(filename, 'w') as fout:
        for sample in data:
            fout.write(json.dumps(sample))
            fout.write('\n')

def save_logits(logits_all, filename):
	with open(filename, "w") as f:
		for i in range(len(logits_all)):
			for j in range(len(logits_all[i])):
				f.write(str(logits_all[i][j]))
				if j == len(logits_all[i])-1:
					f.write("\n")
				else:
					f.write(" ")

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Setup adapters for adapter tuning
    if adapter_args.train_adapter:
        task_name = data_args.task_name
        # check if adapter already exists otherwise add it
        if task_name not in model.config.adapters:
            # resolve adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load adapter from hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(adapter_args.load_adapter, config=adapter_config, load_as=task_name)
            else:
                model.add_adapter(data_args.adapter_name, config=adapter_config)
        # optionally load  a pretrained language adapter
        if adapter_args.load_lang_adapter:
            # resolve language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those in this adapter
        model.train_adapter(data_args.adapter_name)
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(Fuse(lang_adapter_name, task_name))
        else:
            model.set_active_adapters(data_args.adapter_name)
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter_training"
            )

    for (n,p) in model.named_parameters():
        print(n, p.requires_grad)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        train_dataset = load_and_cache_examples(data_args, training_args.output_dir, data_args.task_name, tokenizer, evaluate=False)
        eval_dataset = load_and_cache_examples(data_args, training_args.output_dir, data_args.task_name, tokenizer, evaluate=True)
    elif training_args.do_eval:
        eval_dataset = load_and_cache_examples(data_args,training_args.output_dir, data_args.task_name, tokenizer, evaluate=True)


    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=mCollateFn,
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
    )

    # Training
    if training_args.do_train:
        global_step, tr_loss = train(training_args, data_args, train_dataset, model, tokenizer, eval_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # if last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        # else:
        #     checkpoint = None
        # print("checkpoint",checkpoint)
        # train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        # metrics = train_result.metrics

        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
