# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import (
    AdapterConfig,
    AdapterFusionConfig,
    AdapterType,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils_multiple_choice import MultipleChoiceDataset, Split, processors
import pandas as pd
import wandb
import shutil
logger = logging.getLogger(__name__)



def simple_accuracy(preds, labels):

    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    adapter_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    data_dir: str = field(default=None, metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "wandb project name"}
    )
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "wandb team name"}
    )
    wandb_name: Optional[str] = field(
        default=None, metadata={"help": "wandb name"}
    )
    best_model_path: Optional[str] = field(
        default=None, metadata={"help": "best model path"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    wandb.init(project=data_args.wandb_project, entity=data_args.wandb_entity, name=data_args.wandb_name)


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Setup adapters
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
            model.set_active_adapters(Fuse(lang_adapter_name, data_args.adapter_name))
        else:
            model.set_active_adapters(data_args.adapter_name)
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter_training"
            )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    for (n,p) in model.named_parameters():
        print(n, p.requires_grad)

    # Get datasets
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval or training_args.do_predict
        else None
    )

    test_dataset = (
    MultipleChoiceDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        task=data_args.task_name,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.test,
    )
    if training_args.do_predict
    else None
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        # do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # # Evaluation
    # results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     result = trainer.evaluate()

    #     output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    #     if trainer.is_world_process_zero():
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key, value in result.items():
    #                 logger.info("  %s = %s", key, value)
    #                 writer.write("%s = %s\n" % (key, value))

    #             results.update(result)
    
    # select best adapter
    df = pd.DataFrame(columns=["checkpoint","dev_acc","test_acc"])
    folder_path = os.path.join(training_args.output_dir)
    sub_folders = [ f.path for f in os.scandir(folder_path) if f.is_dir() ]
    sub_folders = [ x for x in sub_folders if "checkpoint" in x]
    for sub_folder in sub_folders:
        checkpoint = sub_folder.split("/")[-1]
        checkpoint_adapter = os.path.join(sub_folder, data_args.adapter_name)
        print(checkpoint_adapter)

        config = AutoConfig.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    num_labels=num_labels,
                    finetuning_task=data_args.task_name,
                    cache_dir=model_args.cache_dir,
                )
        tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                )
        checkpoint_model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        adapter_path = os.path.join(checkpoint_adapter)
        adapter = checkpoint_model.load_adapter(adapter_path)
        checkpoint_model.set_active_adapters(adapter)

        temp_trainer = Trainer(
            model=checkpoint_model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # do_save_full_model=not adapter_args.train_adapter,
            do_save_adapters=adapter_args.train_adapter,
        )

        eval_result = temp_trainer.evaluate()
        test_result = temp_trainer.predict(test_dataset).metrics

        df = df.append({"checkpoint": checkpoint, "dev_acc": eval_result["eval_acc"], "test_acc":test_result["test_acc"]}, ignore_index=True)
    
    # select the best and move
    df.to_csv(os.path.join(training_args.output_dir, "accuracy.csv"))
    best_checkpoint = df.loc[df.dev_acc.idxmax(), 'checkpoint']
    best_dev_acc = df.loc[df.dev_acc.idxmax(), 'dev_acc']
    best_test_acc = df.loc[df.dev_acc.idxmax(), 'test_acc']


    best_path = os.path.join(training_args.output_dir, best_checkpoint, data_args.adapter_name)
    to_best_path = os.path.join(data_args.best_model_path, data_args.adapter_name)
    if os.path.isdir(to_best_path):
        shutil.rmtree(to_best_path)
    shutil.copytree(best_path, to_best_path)
    print("the best checkpoint is : {}".format(best_checkpoint) )
    print("accuracy at the best check point is dev: {} / test: {} ".format(best_dev_acc, best_test_acc))
    print("the best adapter is saved at : {}".format(to_best_path))
    print(df)

    # Predict
    if training_args.do_predict:
        results = {}
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        results = {}
        logger.info("*** Test ***")

        result = trainer.predict(test_dataset)
        result = result.metrics

        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

        return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()