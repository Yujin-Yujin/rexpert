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
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformers import (
    AdapterConfig,
    AdapterFusionConfig,
    AdapterType,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    # Trainer,
    TrainingArguments,
    set_seed,
)
from utils_multiple_choice import MultipleChoiceDataset, Split, processors
import pandas as pd
import wandb
import shutil

from custom.models.roberta.modeling_roberta_custom import RobertaForMultipleChoiceCustom
from custom.trainer_custom import Trainer
from custom.adapters.modeling_custom import BertFusion
# from custom.adapters.layer_custom import adapter_fusion
from transformers.adapters.composition import AdapterCompositionBlock, Fuse
from custom.modeling_outputs_custom import (
    MultipleChoiceModelOutput,
)


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

    # adapter_name: Optional[str] = field(metadata={"help": "The name of the adapter to train on: " + ", ".join(processors.keys())})

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
    do_select: bool = field(
        default=False, metadata={"help": "select best fusion model"}
    )
    adapter_names: Optional[str] = field(default=None, metadata={"help": "list of pretrained adapter names"})


@dataclass
class FusionArguments:
    """
    Arguments for fusion.
    """
    train_fusion: bool = field(
        default=False, metadata={"help": "train fusion. default is false."}
    )
    test_fusion: bool = field(
        default=False, metadata={"help": "test fusion. default is false."}
    )
    fusion_path: Optional[str] = field(default=None, metadata={"help": "Should contain the fusion files for the test."})
    temperature: bool = field(
        default=False, metadata={"help": "train fusion. default is false."}
    )
    pretrained_adapter_names: Optional[str] = field(
        default=None, metadata={"help": "list of pretrained adapter names"}
    )
    pretrained_adapter_dir_path: Optional[str] = field(
        default=None, metadata={"help": "dir_path of pretrained adapters"}
    )
    fusion_attention_supervision: bool = field(
        default=False, metadata={"help": "test fusion. default is false."}
    )

class AttentionFusion(nn.Module):
    def __init__(self, config, training_args, data_args, model_args, fusion_args, num_labels):
        super().__init__()      
        self.config = config
        self.dense_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.num_labels = num_labels
        self.device = training_args.device

        # pooh initialize is needed?
        # self.init_weights()
        adapter_names = [data_args.adapter_names.split(',')] # get adapter names

        adapter_fusion_config = AdapterFusionConfig.load(config="dynamic")
        self.config.adapter_fusion = adapter_fusion_config

        fusion = BertFusion(self.config)
        fusion.train(self.training)
        self.adapter_fusion_layer= fusion

        self.base_model_list = []

        # load base_model
        base_model_config = AutoConfig.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    num_labels=num_labels,
                    finetuning_task=data_args.task_name,
                    cache_dir=model_args.cache_dir,
                )
        tokenizer = AutoTokenizer.from_pretrained(
                    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                    cache_dir=model_args.cache_dir,
                    )
        base_model= RobertaForMultipleChoiceCustom.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=base_model_config,
                    cache_dir=model_args.cache_dir,
                )
        base_model.to(self.device)
        self.base_model_list.append(base_model)

        # load sub_models
        self.sub_model_list = []
        for adapter_name in adapter_names[0]:
            sub_model_config = AutoConfig.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    num_labels=num_labels,
                    finetuning_task=data_args.task_name,
                    cache_dir=model_args.cache_dir,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                    cache_dir=model_args.cache_dir,
                    )
            sub_model = RobertaForMultipleChoiceCustom.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=sub_model_config,
                    cache_dir=model_args.cache_dir,
                )
            adapter_path = os.path.join(fusion_args.pretrained_adapter_dir_path, adapter_name)
            adapter = sub_model.load_adapter(adapter_path)
            sub_model.set_active_adapters(adapter)
            sub_model.to(self.device)
            self.sub_model_list.append(sub_model) 

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        adapter_names=None,
        ):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        sub_model_hidden_states = []

        self.base_model_list[0].eval()
        base_model_hidden_states = self.base_model_list[0](input_ids,
                                    token_type_ids,
                                    attention_mask,
                                    labels,
                                    position_ids,
                                    head_mask,
                                    inputs_embeds,
                                    output_attentions,
                                    output_hidden_states,
                                    return_dict,
                                    adapter_names,)
        base_model_hidden_states = base_model_hidden_states.hidden_states


        for sub_model in self.sub_model_list:
            sub_model.eval() 
            raw_outputs = sub_model(input_ids,
                                    token_type_ids,
                                    attention_mask,
                                    labels,
                                    position_ids,
                                    head_mask,
                                    inputs_embeds,
                                    output_attentions,
                                    output_hidden_states,
                                    return_dict,
                                    adapter_names,)
            hidden_states = raw_outputs.hidden_states
            sub_model_hidden_states.append(hidden_states)

        # attention between LM representation and adapter representations
        hidden_states = self._adapter_fusion(hidden_states = sub_model_hidden_states,
                            query = base_model_hidden_states
                            )


        pooled_output = self.dropout(hidden_states)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # if not return_dict:
        #     output = (reshaped_logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

    def _adapter_fusion(self, hidden_states, query,):
        """
        Performs adapter fusion with the given adapters for the given input.
        
        # hidden_states : list of adapter hidden states

        """
        # config of _last_ fused adapter is significant
        # adapter_config = self.config.adapters.get(adapter_setup.last())
        # hidden_states, query, residual = self.get_adapter_preparams(adapter_config, hidden_states, input_tensor)


        # up_list = []

        # for adapter_block in adapter_setup:
        #     # Case 1: We have a nested stack -> call stack method
        #     if isinstance(adapter_block, Stack):
        #         _, up, _ = self.adapter_stack(adapter_block, hidden_states, input_tensor, lvl=lvl + 1)
        #         if up is not None:  # could be none if stack is empty
        #             up_list.append(up)
        #     # Case 2: We have a single adapter which is part of this module -> forward pass
        #     elif adapter_block in self.adapters:
        #         adapter_layer = self.adapters[adapter_block]
        #         _, _, up = adapter_layer(hidden_states, residual_input=residual)
        #         up_list.append(up)
        #     # Case 3: nesting other composition blocks is invalid
        #     elif isinstance(adapter_block, AdapterCompositionBlock):
        #         raise ValueError(
        #             "Invalid adapter setup. Cannot nest {} in {}".format(
        #                 adapter_block.__class__.__name__, adapter_setup.__class__.__name__
        #             )
        #         )
        #     # Case X: No adapter which is part of this module -> ignore

        up_list = hidden_states
        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 0, 2)

            hidden_states = self.adapter_fusion_layer(
                query,
                up_list,
                up_list,
            )

        return hidden_states

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.



    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments, FusionArguments))
    model_args, data_args, training_args, adapter_args, fusion_args = parser.parse_args_into_dataclasses()

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
    model = AttentionFusion(config, 
                        training_args=training_args,
                        data_args=data_args, 
                        model_args=model_args, 
                        fusion_args=fusion_args, 
                        num_labels=num_labels)

    # def compute_metrics(p: EvalPrediction) -> Dict:
    #     preds = np.argmax(p.predictions, axis=1)
    #     return {"acc": simple_accuracy(preds, p.label_ids)}
    
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
        if training_args.do_train or data_args.do_select
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
        if training_args.do_eval or data_args.do_select or training_args.do_train
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
        do_save_full_model=training_args.do_train,
        do_save_adapters=not training_args.do_train,
        do_save_adapter_fusion=not training_args.do_train,
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

    if data_args.do_select:
        # select best fusion
        df = pd.DataFrame(columns=["checkpoint","dev_acc","test_acc"])
        folder_path = os.path.join(training_args.output_dir)
        sub_folders = [ f.path for f in os.scandir(folder_path) if f.is_dir() ]
        sub_folders = [ x for x in sub_folders if "checkpoint" in x]
        for sub_folder in sub_folders:
            checkpoint = sub_folder.split("/")[-1]
            checkpoint_fusion = os.path.join(sub_folder, fusion_args.pretrained_adapter_names)
            print(checkpoint_fusion)

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

            adapter_names = [fusion_args.pretrained_adapter_names.split(',')]
            for adapter_name in adapter_names[0]:
                print("pooh adapter_name", adapter_name)
                adapter_path = os.path.join(fusion_args.pretrained_adapter_dir_path, adapter_name)
                print("pooh adapter_path", adapter_path)
                
                # adapter = model.load_adapter("/" + adapter_path)
                adapter=checkpoint_model.load_adapter(adapter_path)
                checkpoint_model.set_active_adapters(adapter) 

            checkpoint_model.load_adapter_fusion(checkpoint_fusion)
            checkpoint_model.set_active_adapters(adapter_names)

            temp_trainer = Trainer(
                model=checkpoint_model,
                args=training_args,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                # do_save_full_model=not adapter_args.train_adapter,
                do_save_adapters=adapter_args.train_adapter,
            )

            eval_result = temp_trainer.evaluate()
            if test_dataset is not None:
                test_result = temp_trainer.predict(test_dataset).metrics
                df = df.append({"checkpoint": checkpoint, "dev_acc": eval_result["eval_acc"], "test_acc":test_result["test_acc"]}, ignore_index=True)
            else:
                df = df.append({"checkpoint": checkpoint, "dev_acc": eval_result["eval_acc"]}, ignore_index=True)
        
        # select the best and move
        df.to_csv(os.path.join(training_args.output_dir, "accuracy.csv"))
        best_checkpoint = df.loc[df.dev_acc.idxmax(), 'checkpoint']
        best_dev_acc = df.loc[df.dev_acc.idxmax(), 'dev_acc']
        best_test_acc = df.loc[df.dev_acc.idxmax(), 'test_acc']


        best_path = os.path.join(training_args.output_dir, best_checkpoint, fusion_args.pretrained_adapter_names)
        to_best_path = os.path.join(data_args.best_model_path, fusion_args.pretrained_adapter_names)
        if os.path.isdir(to_best_path):
            shutil.rmtree(to_best_path)
        shutil.copytree(best_path, to_best_path)
        print("the best checkpoint is : {}".format(best_checkpoint) )
        print("accuracy at the best check point is dev: {} / test: {} ".format(best_dev_acc, best_test_acc))
        print("the best fusion is saved at : {}".format(to_best_path))
        print(df)


    # Eval
    if training_args.do_eval:
        results = {}
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "{}_eval_results.txt".format(data_args.task_name))
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

    # Predict
    if training_args.do_predict:
        results = {}
        logger.info("*** Test ***")

        result = trainer.predict(test_dataset)
        result = result.metrics

        output_test_file = os.path.join(training_args.output_dir, "{}_test_results.txt".format(data_args.task_name))
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