from typing import Dict, List

from tf_slim import xavier_initializer
from tqdm import tqdm
import pickle
import os
import json

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from transformers import (
    BertTokenizer, TFBertForSequenceClassification, TFBertForTokenClassification,
    TFBertModel, BertConfig, BatchEncoding,
)
from transformers.modeling_tf_bert import TFBaseModelOutputWithPooling
from ner.data_process import load_data
from ner.config import train_file, train_pickle_file, model_name


class BertCrfModel(tf.keras.Model):
    """CRF with Bert model, output the standard"""
    def get_config(self):
        return super(BertCrfModel, self).get_config()

    def __init__(self, num_class: int = 5, pretrained_model_name: str = 'bert-base-uncased'):
        super().__init__()
        # 1. 加载bert 模型
        self.bert = TFBertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, cache_dir='./model_cache'
        )

        # 2. 初始化crf转移矩阵
        self.crf_transition = self.add_weight(
            name='crf_transition',
            shape=(num_class, num_class),
            initializer=tf.keras.initializers.he_normal(seed=1234)
        )

    def call(self, inputs: BatchEncoding, training=None, mask: tf.Tensor = None):
        # 1. encoding by bert module
        labels = inputs.pop('labels')
        bert_outputs: TFBaseModelOutputWithPooling = self.bert(inputs)

        # 1.1 get the pooling result => (batch_size, sequence_length, hidden_size)
        # eg: (64, 128, 768)
        output: tf.Tensor = bert_outputs.last_hidden_state

        # 1.2 compute the mask

        # 2. log_likelihood by crf
        sequence_lengths = tf.cast(tf.reduce_sum(mask, axis=-1), tf.int32)
        log_likelihood, trans = tfa.text.crf_log_likelihood(
            inputs=output,
            tag_indices=labels,
            transition_params=self.crf_transition,
            sequence_lengths=sequence_lengths
        )
        loss = tf.reduce_mean(-log_likelihood)

        predicted_ids, _ = tfa.text.crf_decode(
            potentials=output,
            transition_params=trans,
            sequence_length=sequence_lengths)
        if training:
            return loss
        return predicted_ids


class CrfLoss(tf.losses.Loss):
    """get the final crf loss"""
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return y_true


def epoch_lr_schedule(epoch, lr):
    if epoch < 10:
        return 0.001
    if epoch < 20:
        return 0.002
    return lr * tf.math.exp(-0.1)


def train():
    # 1. define the model and base class
    model = BertCrfModel(
        num_class=7,
    )
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = CrfLoss()
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics='acc',
    )

    # 2. load the data
    inputs, labels = load_data(file=train_file, pickle_file=train_pickle_file)

    model.fit(
        inputs, labels, batch_size=4, validation_split=0.3,
        callbacks=[tf.keras.callbacks.TensorBoard(), tf.keras.callbacks.LearningRateScheduler(epoch_lr_schedule)]
    )


if __name__ == '__main__':
    train()
