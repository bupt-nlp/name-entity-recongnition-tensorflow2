from __future__ import absolute_import, division, print_function

# refer: https://github.com/kamalkraj/BERT-NER-TF/blob/master/run_ner.py

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

from fastprogress import master_bar, progress_bar

from bert_ner.model import BertNer
from bert_ner.optimization import AdamWeightDecay, Warmup
from bert_ner.tokenization import FullTokenizer
from bert_ner.config import get_arguments, ModelArguments

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
        # TODO -> should be refactored
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

        # make sure the max_sequence_length
        if len(tokens) >= max_sequence_length - 1:
            tokens = tokens[0: max_sequence_length - 2]
            labels = labels[0: max_sequence_length - 2]
            valid = valid[0: max_sequence_length - 2]
            label_mask = label_mask[0: max_sequence_length - 2]

        n_tokens, segment_ids, label_ids = [], [], []
        n_tokens.append('[CLS]')
        segment_ids.insert(0, 0)
        valid.insert(0, 1)
        # 主要是用来处理 bupt -> bu ##pt 这样数据的结果的。
        label_mask.insert(0, True)
        label_ids.insert(0, label_map['O'])

        # 3. generate label_ids info
        for i, token in enumerate(tokens):
            n_tokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

        n_tokens.append('[SEP]')
        segment_ids.append(0)
        valid.append(1)
        # sentence end mask
        label_mask.append(True)

        input_ids = tokenizer.convert_tokens_to_ids(n_tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

            # this will remove ##endfix which will influence the result.
            # this is the key skills to resolve mutli-token problem
            valid.append(1)
            label_mask.append(False)

        while len(label_ids) < max_sequence_length:
            label_ids.append(0)
            label_mask.append(False)

        assert len(input_ids) == max_sequence_length
        assert len(input_mask) == max_sequence_length

        assert len(label_ids) == max_sequence_length
        assert len(label_mask) == max_sequence_length

        assert len(valid) == max_sequence_length
        assert len(segment_ids) == max_sequence_length

        if index < 5:
            logger.info('*** Example ***')
            logger.info(f'guid: {example.guid}')
            logger.info(f'tokens: {" ".join(n_tokens)}')
            logger.info(f'input mask: {" ".join([str(mask) for mask in input_mask])}')
            logger.info(f'segment ids: {" ".join([str(segment_id) for segment_id in segment_ids])}')

        features.append(InputFeature(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            valid_ids=valid,
            label_mask=label_mask
        ))
    return features


def main():
    args: ModelArguments = get_arguments()
    process = NerProcessor(args.data_dir)

    label_list = process.get_labels()
    num_labels = len(label_list) + 1

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError('Output directory ({}) already exist and the dir is not empty')
    if not args.output_dir:
        os.makedirs(args.output_dir)

    if args.do_train:
        tokenizer = FullTokenizer(os.path.join(args.bert_model, "vocab.txt"), args.do_lower_case)

    if args.multi_gpu:
        if len(args.gpu.split(',')) == 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            # build the gpu device name arr
            gpus = [f'/gpu:{gpu}' for gpu in args.gpu.split(',')]
            strategy = tf.distribute.MirroredStrategy(devices=gpus)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device=args.gpu)

    if args.do_train:
        train_examples = process.get_train_examples()

        # optimization total steps -> learning_rate scheduler, weight decay, warmup learning rate
        num_train_optimization_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)

        # keep the final learning should be zero
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=num_train_optimization_steps,
            end_learning_rate=0.
        )

        if warmup_steps:
            # layer norm and bias should not weight decay
            learning_rate_fn = AdamWeightDecay(
                learning_rate=args.learning_rate,
                weight_decay_rate=args.weight_decay,
                beta_1=0.9,
                beta_2=0.99,
                epsilon=args.adam_epsilon,
                exclude_from_weight_decay=['layer_norm', 'bias']
            )

        with strategy.scope():
            ner = BertNer(args.bert_model, tf.float32, args.num_labels, args.max_seq_length)
            # can define the specific meaning of the reduction
            loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    label_map = {label: index for index, label in enumerate(label_list, 1)}

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer
        )
        logger.info('*** Running training ***')
        logger.info('  Num Examples = %d', len(train_examples))
        logger.info('  Batch Size = %d', len(args.train_batch_size))
        logger.info('  Num Steps = %d', len(num_train_optimization_steps))
        all_input_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_ids for f in train_features])
        )
        all_input_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_mask for f in train_features])
        )

        all_label_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_ids for f in train_features])
        )
        all_label_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_mask for f in train_features])
        )

        all_valid_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.valid_ids for f in train_features])
        )
        all_segment_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.segment_ids for f in train_features])
        )

        train_data = tf.data.Dataset.zip((
            all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_label_mask
        ))

        # set the shuffle buffer size, reshuffle the train data in each iteration
        shuffled_train_data = train_data.shuffle(
            buffer_size=int(len(train_features) * 0.1),
            seed=args.seed,
            reshuffle_each_iteration=True
        ).batch(args.train_batch_size)

        distributed_data = strategy.experimental_distribute_dataset(shuffled_train_data)
        loss_metric = tf.keras.metrics.Mean()

        epoch_bar = master_bar(range(1))
        optimizer: tf.keras.optimizers.Optimizer = None

        def train_steps(input_ids, input_mask, segment_id, valid_ids, label_ids, label_mask):
            def step_fn(_input_ids, _input_mask, _segment_id, _valid_ids, _label_ids, _label_mask):
                with tf.GradientTape() as tape:
                    # _input_ids, one-axis, which will be run on pattern
                    output = ner(_input_ids, _input_mask, _segment_id, _label_ids, training=True)

                    # flatten all of the outputs
                    _label_mask = tf.reshape(_label_mask, (-1))
                    output = tf.reshape(output, (-1, num_labels))
                    output = tf.boolean_mask(output, _input_mask)

                    _label_ids = tf.reshape(_label_ids, (-1,))
                    _label_ids = tf.boolean_mask(_label_ids, _label_mask)

                    cross_entropy = loss_fct(_label_ids, output)

                    # this is for single one train data
                    loss = tf.reduce_sum(cross_entropy) * 1. / args.train_batch_size

                gradients = tape.gradient(loss, ner.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(gradients, ner.trainable_variables))

                return cross_entropy

            # 在多个gpu上并行跑训练数据
            per_example_loss = strategy.experimental_run_v2(step_fn, args=(
                input_ids, input_mask, segment_id, valid_ids, label_ids, label_mask
            ))
            mean_loss = strategy.reduce(tf.distribute)
            return mean_loss

        pb_max_length = math.ceil(
            len(train_features) / args.train_batch_size
        )
        for epoch in epoch_bar:
            with strategy.scope():
                for (input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask) in progress_bar(
                        distributed_data, total=pb_max_length, parent=epoch_bar):
                    loss = train_steps(input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask)
                    loss_metric(loss)
                    epoch_bar.child.comment = f'loss: {loss}'
                loss_metric.reset_states()

        ner.save_weights(os.path.join(args.output_dir, 'model.h5'))

    if args.do_eval:
        tokenizer = FullTokenizer(os.path.join(args.bert_model, 'vocab.txt'), do_lower_case=args.do_lower_case)
        ner = BertNer(args.bert_model, tf.float32, args.num_labels, args.max_seq_length)

        # create example eval data to build the model
        ids = tf.ones((1, 128), dtype=tf.float32)
        ner(ids, ids, ids, ids, ids, training=False)
        ner.load_weights(os.path.join(args.output_dir, 'model.h5'))

        # load the data
        if args.eval_on == 'dev':
            eval_examples = process.get_dev_examples()
        elif args.eval_on == 'test':
            eval_examples = process.get_test_examples()
        else:
            raise KeyError(f'eval_on arguments is expected in [dev, test]')

        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
        )

        # print the eval info
        logger.info('*** Eval Examples ***')
        logger.info('  Num Examples = %d', len(eval_features))
        logger.info('  Batch Size = %d', args.eval_batch_size)

        all_input_ids = tf.data.Dataset.from_tensor_slices(
            [f.input_ids for f in eval_features]
        )
        all_input_mask = tf.data.Dataset.from_tensor_slices(
            [f.input_mask for f in eval_features]
        )
        all_segment_ids = tf.data.Dataset.from_tensor_slices(
            [f.segment_ids for f in eval_features]
        )
        all_valid_ids = tf.data.Dataset.from_tensor_slices(
            [f.valid_ids for f in eval_features]
        )
        all_label_ids = tf.data.Dataset.from_tensor_slices(
            [f.label_ids for f in eval_features]
        )
        all_label_mask = tf.data.Dataset.from_tensor_slices(
            [f.label_mask for f in eval_features]
        )

        eval_data = tf.data.Dataset.zip((
            all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids, all_label_mask
        )).batch(args.eval_batch_size)

        loss_metric = tf.metrics.Mean()
        epoch_bar = master_bar(range(1))
        processor_bar_length = math.ceil(
            len(eval_features) / args.eval_batch_size
        )
        y_true, y_predict = [], []
        for epoch in epoch_bar:
            for (input_ids, input_mask, segment_ids, valid_ids, label_ids, label_mask) in progress_bar(eval_data,
                                                                                                       total=processor_bar_length,
                                                                                                       parent=epoch_bar):
                logits = ner(input_ids, input_mask, segment_ids, valid_ids, training=False)
                logits = tf.argmax(logits, axis=-1)
                label_predict = tf.boolean_mask(logits, label_mask)
                y_true.append(label_ids)
                y_predict.append(label_predict)

        report = classification_report(y_true, y_predict, digits=4)
        output_eval_file = os.path.join(args.output_dir, 'eval_result.txt')
        with open(output_eval_file, 'w', encoding='utf-8') as f:
            logger.info('*** Eval Result ***')
            logger.info(report)
            f.write(report)


if __name__ == '__main__':
    main()
