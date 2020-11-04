from __future__ import absolute_import, division, print_function

from typing import List
import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys

import numpy as np
import tensorflow as tf
from seqeval.metrics import classification_report

from bert_ner.model import BertNer
from bert_ner.optimization import AdamWeightDecay, Warmup
from bert_ner.tokenization import FullTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('tensorflow')
# 设置日志提醒的等级
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class InputExample:
    """a single train/test example for sequence labeling classification"""

    def __init__(self, guid, text_a: str, text_b: str = None, label: List[str] = None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label: List[dict] = label


class InputFeature:
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class DataProcessor:
    """base class for different data processors"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_train_examples(self) -> List[InputExample]:
        raise NotImplementedError

    def get_test_examples(self) -> List[InputExample]:
        raise NotImplementedError

    def get_dev_examples(self) -> List[InputExample]:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        raise NotImplementedError


class NerProcessor(DataProcessor):
    """ner data process"""

    def get_labels(self) -> List[str]:
        label_file = os.path.join(self.data_dir, 'labels.txt')
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()
            return labels

    def get_train_examples(self) -> List[InputExample]:
        """get th train examples"""
        return self._create_example(
            os.path.join(self.data_dir, 'train.txt'), "train"
        )

    def get_test_examples(self) -> List[InputExample]:
        """get the test examples"""
        return self._create_example(
            os.path.join(self.data_dir, 'test.txt'), "test"
        )

    def get_dev_examples(self) -> List[InputExample]:
        """get the validation examples"""
        return self._create_example(
            os.path.join(self.data_dir, 'dev.txt'), "dev"
        )

    def _read_file(self, input_file):
        pass

    @staticmethod
    def _create_example(file: str, data_type: str = "train") -> List[InputExample]:
        """create examples from file"""
        examples: List[InputExample] = []
        with open(file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                data = json.loads(line)
                examples.append(InputExample(
                    guid=f"id-{data_type}-{index}",
                    text_a=data['text'],
                    text_b=None,
                    label=data['entities']
                ))
        return examples


def convert_examples_to_features(
        examples: List[InputExample],
        all_labels: List[str], max_sequence_length: int,
        tokenizer: FullTokenizer):
    """
    load examples to features

    :param examples:
    :param all_labels:
    :param max_sequence_length:
    :param tokenizer:
    :return:
    """
    label_map = {label: index for index, label in enumerate(all_labels, 1)}
    features = []
    for index, example in enumerate(examples):

        # 1. prepare the base data
        # this is for english language
        text_list = example.text_b.split(' ')
        # TODO -> should be refactor
        label_list = example.label

        # 2. build validation mask and label mask
        tokens, labels, valid, label_mask = [], [], [], []
        for i, word in text_list:

            # tokenizer("bupt") -> [bu, ##pt] to two sub tokens
            sub_tokens = tokenizer.tokenize(word)
            tokens.extend(sub_tokens)

            for j in range(len(sub_tokens)):
                if j == 0:
                    labels.append(label_list[i])
                    valid.append(1)
                    label_mask.append(True)
                else:
                    valid.append(0)

            if len(tokens) >= max_sequence_length - 1:
                tokens = tokens[0: max_sequence_length - 2]
                labels = labels[0: max_sequence_length - 2]
                valid = valid[0: max_sequence_length - 2]
                label_mask = label_mask[0: max_sequence_length - 2]

        n_tokens, segment_ids, label_ids = [], [], []
        n_tokens.append('[CLS]')
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, True)
        label_ids.insert(0, label_map['O'])
        tf.boolean_mask