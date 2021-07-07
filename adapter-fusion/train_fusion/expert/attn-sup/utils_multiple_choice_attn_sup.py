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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import uuid

from transformers import (
    # AdapterConfig,
    # AdapterFusionConfig,
    # AdapterType,
    # AutoConfig,
    # AutoModelForMultipleChoice,
    # AutoTokenizer,
    EvalPrediction,
    # HfArgumentParser,
    # MultiLingAdapterArguments,
    Trainer,
    # TrainingArguments,
    # set_seed,
)

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

@dataclass(frozen=True)
class InputAttentionFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
    attention_label: Optional[List[int]]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            sub_model_list=None,
        ):
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )

            if sub_model_list is not None:
                cached_attention_features_file = os.path.join(
                    data_dir,
                    "cached_{}_{}_{}_{}_{}_adapter_att_label".format(
                        mode.value,
                        tokenizer.__class__.__name__,
                        str(max_seq_length),
                        task,
                        len(sub_model_list)
                    ),
                )
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            if sub_model_list is None:
                lock_path = cached_features_file + ".lock"
                with FileLock(lock_path):
                    if os.path.exists(cached_features_file) and not overwrite_cache:
                        logger.info(f"Loading features from cached file {cached_features_file}")
                        self.features = torch.load(cached_features_file)
                    else:
                        logger.info(f"Creating features from dataset file at {data_dir}")
                        label_list = processor.get_labels()
                        if mode == Split.dev:
                            examples = processor.get_dev_examples(data_dir)
                        elif mode == Split.test:
                            examples = processor.get_test_examples(data_dir)
                        else:
                            examples = processor.get_train_examples(data_dir)

                        logger.info("Training examples: %s", len(examples))
                        self.features = convert_examples_to_features(
                            examples,
                            label_list,
                            max_seq_length,
                            tokenizer,
                        )
                        logger.info("Saving features into cached file %s", cached_features_file)
                        torch.save(self.features, cached_features_file)

            else:
                attention_lock_path = cached_attention_features_file + ".lock"
                with FileLock(attention_lock_path):
                    if os.path.exists(cached_attention_features_file) and not overwrite_cache:
                        logger.info(f"Loading features from cached file {cached_attention_features_file}")
                        self.features = torch.load(cached_attention_features_file)
                    else:
                        lock_path = cached_features_file + ".lock"
                        with FileLock(lock_path):
                            if os.path.exists(cached_features_file) and not overwrite_cache:
                                logger.info(f"Loading features from cached file {cached_features_file}")
                                self.features = torch.load(cached_features_file)
                            else:
                                logger.info(f"Creating features from dataset file at {data_dir}")
                                label_list = processor.get_labels()
                                if mode == Split.dev:
                                    examples = processor.get_dev_examples(data_dir)
                                elif mode == Split.test:
                                    examples = processor.get_test_examples(data_dir)
                                else:
                                    examples = processor.get_train_examples(data_dir)

                                logger.info("Training examples: %s", len(examples))
                                self.features = convert_examples_to_features(
                                    examples,
                                    label_list,
                                    max_seq_length,
                                    tokenizer,
                                )
                                logger.info("Saving features into cached file %s", cached_features_file)
                                torch.save(self.features, cached_features_file)

                        logger.info("Training attention features: %s", len(self.features))

                        self.features = convert_examples_to_attention_features(
                            self.features,
                            sub_model_list
                        )

                        logger.info("Saving attention_features into cached file %s", cached_attention_features_file)
                        torch.save(self.features, cached_attention_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFMultipleChoiceDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            logger.info("Training examples: %s", len(examples))

            self.features = convert_examples_to_features(
                examples,
                label_list,
                max_seq_length,
                tokenizer,
            )

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": 0,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class SocialIQAProcessor(DataProcessor):
    """Processor for the SocialIQA data set from YJ."""

    def get_train_examples(self, data_dir, sub_model_list=None):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")), "train", sub_model_list)

    def get_dev_examples(self, data_dir, sub_model_list=None):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")), "dev", sub_model_list)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "socialIQa_v1.4_tst.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []

        for line in tqdm.tqdm(lines, desc="read social_iqa data"):
            data_raw = json.loads(line.strip("\n"))
            examples.append(
                InputExample(
                example_id= str(uuid.uuid1),
                question=data_raw['question'],  
                contexts=[data_raw['context'], data_raw['context'], data_raw['context']],
                endings=[data_raw['answerA'], data_raw['answerB'], data_raw['answerC']],
                label=data_raw['correct'],
                )
            )

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)


    return features

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}

def convert_examples_to_attention_features(
    example_features: List[InputFeatures],
    sub_model_list
) -> List[InputAttentionFeatures]:
    attention_features = []

    logit_list = []
    c_labels = None
    for sub_model in sub_model_list:
        trainer = Trainer(
            model=sub_model,
            compute_metrics=compute_metrics,
        )
        predictions, label_ids, metrics = trainer.predict(example_features)
        logits = torch.from_numpy(predictions)
        labels = torch.from_numpy(label_ids)
        s_logits = nn.Softmax(dim=-1)(logits)
        logit_list.append(s_logits)

        if c_labels is not None:
            assert (torch.equal(c_labels, labels)), "labels between sub models are different."
        c_labels = labels

    stack_all = torch.stack(logit_list)
    attention_label_list = []
    for i in range(stack_all.shape[1]):
        answer_index = None
        best_var = 0
        for j in range(stack_all.shape[0]):
            if torch.argmax(stack_all[j][i], dim=-1) == c_labels[i].item():
                if torch.std(stack_all[j][i]).item() > best_var:
                    best_var = torch.std(stack_all[j][i]).item()
                    answer_index = j
    
        attention_label_list.append(answer_index)

    attention_label = []
    for answer_label in attention_label_list:
        exp_label = []
        for choice in range(stack_all.shape[0]):
            if answer_label == choice:
                exp_label.append(1)
            else:
                exp_label.append(0)
        attention_label.append(exp_label)

    # attention_label = torch.tensor(attention_label)


    for (ex_f_idx, example_feature) in tqdm.tqdm(enumerate(example_features), desc="convert features to attention features"):
        attention_features.append(
            InputAttentionFeatures(
                example_id=example_feature.example_id,
                input_ids=example_feature.input_ids,
                attention_mask=example_feature.attention_mask,
                token_type_ids=example_feature.token_type_ids,
                label=example_feature.label,
                attention_label=attention_label[ex_f_idx]
            )
        )
        
    for f in attention_features[:2]:
        logger.info("*** Example ***")
        logger.info("attention_features: %s" % f)
    
    return attention_features


processors = {"siqa":SocialIQAProcessor}
# MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "syn", 5}

