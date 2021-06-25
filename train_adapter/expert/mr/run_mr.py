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
        default=None,
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
	if evaluate and os.path.exists(cached_features_file):
		features = torch.load(cached_features_file)
	else:
		examples = processor.get_dev_examples() if evaluate else processor.get_train_examples()
		features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length)
		if evaluate:
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
	
	print("batch_input_ids",type(batch_input_ids))
	result_batch = {
	    "input_ids" : batch_input_ids,
	    "attention_mask" : batch_input_mask,
	    "labels" : batch_input_labels
	}
    # return batch_input_ids, batch_input_mask, batch_input_labels, torch.tensor(batch_label_ids, dtype=torch.long)
    
	return result_batch

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

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        train_dataset = load_and_cache_examples(data_args, training_args.output_dir, data_args.task_name, tokenizer, evaluate=False)
        eval_dataset = load_and_cache_examples(data_args,training_args.output_dir, data_args.task_name, tokenizer, evaluate=True)
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
        data_collator=data_collator,
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        print("checkpoint",checkpoint)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

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
