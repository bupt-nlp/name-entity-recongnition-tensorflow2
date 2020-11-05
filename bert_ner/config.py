import argparse
from typing import List, Optional
from dataclasses import dataclass
from typing_extensions import Literal
from argtyped import Arguments

train_file: str = '/Users/wujingwujing/PycharmProjects/name_entity_recongnization/data/task1_public/new_train.json'
train_pickle_file: str = '/Users/wujingwujing/PycharmProjects/name_entity_recongnization/data/task1_public/new_train.json.pkl'
val_file: str = '/Users/wujingwujing/PycharmProjects/name_entity_recongnization/data/task1_public/new_val.json'
val_pickle_file: str = '/Users/wujingwujing/PycharmProjects/name_entity_recongnization/data/task1_public/new_val.json.pkl'
label_file: str = '/Users/wujingwujing/PycharmProjects/name_entity_recongnization/data/task1_public/labels.txt'
model_name: str = 'bert-base-uncased'


class ModelArguments(Arguments):
    data_dir: str
    bert_model: str
    output_dir: str

    # optional field
    num_labels: int = 6
    adam_epsilon: float = 0.3

    num_train_epochs: int = 2
    max_seq_length: int = 128
    do_train: bool = True
    do_eval: bool = True
    eval_on: Literal['dev', 'test'] = 'dev'
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 5e3
    warmup_proportion: float = 0.1
    weight_decay: float = 0.01

    seed: int = 42
    multi_gpu: bool = False
    gpu: str = 'cpu'
    do_lower_case: bool = False


def get_arguments() -> ModelArguments:
    args = ModelArguments()
    return args
