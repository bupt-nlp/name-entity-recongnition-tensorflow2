from __future__ import annotations
import os
import json
import tensorflow as tf
from nltk import word_tokenize

from bert_ner.tokenization import FullTokenizer
from bert_ner.model import BertNer


class NER:
    def __init__(self, model_dir: str):
        self.model, self.tokenizer, self.model_config = self.load_model(model_dir)
        self.label_map = self.model_config['label_map']
        self.max_seq_length = self.model_config['max_seq_length']
        self.label_map = {int(key): value for key, value in self.label_map.items()}

    @staticmethod
    def load_model(model_dir: str, model_config_file: str = 'model_config.json'):
        """load the model"""
        model_config_file: str = os.path.join(model_dir, model_config_file)
        model_config = json.load(open(model_config_file, 'r'))
        bert_config = json.load(open(os.path.join(model_dir, 'bert_config.json'), 'r'))
        model = BertNer(
            bert_model=bert_config, float_type=tf.float32, num_labels=model_config['num_labels'],
            max_seq_length=model_config['max_seq_length']
        )
        ids = tf.ones(shape=(1, 128))
        model(ids, ids, ids, ids, training=False)
        model.load_weights(os.path.join(model_dir, 'model.h5'))
        vocab = os.path.join(model_dir, 'vocab.txt')
        tokenizer = FullTokenizer(vocab_file=vocab, do_lower_case=True)
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """tokenize the text with full tokenizer"""
        words = word_tokenize(text)
        tokens, valid_positions = [], []
        for index, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(tokens)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """preprocess the text"""

        # 1. tokenize the text
        tokens, valid_positions = self.tokenize(text)

        # 2. insert CLS
        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)

        # 3. insert SEP
        tokens.append('[SEP]')
        valid_positions.append(1)

        # 4. build segment id
        segment_id = [0] * len(tokens)

        # 5. generate input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 6. input_mask
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_id.append(0)
            valid_positions.append(0)

        return input_ids, input_mask, segment_id, valid_positions

    def predict(self, text: str):
        """predict by text"""
        input_ids, input_mask, segment_id, valid_positions = self.preprocess(text)
        input_ids = tf.Variable([input_ids], dtype=tf.int32)
        input_mask = tf.Variable([input_mask], dtype=tf.int32)
        segment_id = tf.Variable([segment_id], dtype=tf.int32)
        valid_positions = tf.Variable([valid_positions], dtype=tf.int32)
        logits: tf.Tensor = self.model([input_ids, segment_id, input_mask, valid_positions])
        logits_labels: tf.Tensor = tf.argmax(logits, axis=-1)
        logits_labels = logits_labels.numpy().tolist()[0]
        # 此过程肯定会有更加高效的写法
        # 后续可针对此处进行优化
