from typing import Dict, List

from tqdm import tqdm
import pickle
import os
import json

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

import tensorflow_datasets as tfds
from transformers import (
    BertTokenizer, TFBertForSequenceClassification, TFBertForTokenClassification,
    TFBertModel, BertConfig, BatchEncoding
)
from ner.config import train_file, train_pickle_file, label_file, model_name


def load_data(file: str, pickle_file: str, tokenizer: BertTokenizer = None):
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    if os.path.exists(pickle_file):
        return pickle.load(open(pickle_file, 'rb'))
    with open(file, 'r', encoding='utf-8') as f:
        all_results, all_labels = [], []
        for line in tqdm(f.readlines()):
            # 1. 加载基础数据
            data = json.loads(line)
            text: str = data['text']

            if len(text) > 512:
                text = text[:512]

            # 2. 对数据进行分词，生成token_ids, mask 以及token_type
            inputs: BatchEncoding = tokenizer(text, return_token_type_ids=True, return_tensors='tf')

            # 3. 构造labels
            sequence_length = tf.shape(inputs['input_ids']).numpy()[-1]
            labels = ['O'] * sequence_length
            input_ids = inputs['input_ids']

            # 4. 开始线性搜索实体数据
            global_index = 0

            # 对所有的实体进行排序
            for entity_obj in sorted(data['entities'], key=lambda x: x['start']):

                entity: str = entity_obj['entity']
                # 4.1 将label数据编码成token_ids，然后在全局进行搜索

                tokenized_output: tf.Tensor = tokenizer.encode(entity, add_special_tokens=False, return_tensors='tf')
                entity_token_length: int = tf.shape(tokenized_output).numpy()[-1]

                # 4.2 全局线性搜索
                for search_index in range(global_index, sequence_length - entity_token_length):

                    # 如果搜索到对应的id
                    if tf.reduce_all(
                            tokenized_output == input_ids[:, search_index: search_index + entity_token_length]
                    ).numpy():
                        labels[search_index] = "B-" + entity_obj['type']
                        for index in range(search_index + 1, search_index + entity_token_length):
                            labels[index] = "I-" + entity_obj['type']
                        # 将下一轮的开始搜索索引设置成上一轮的结束字符
                        global_index = search_index + entity_token_length
                        break
            inputs['labels'] = labels
            all_labels.append(labels)
            all_results.append(inputs)
        pickle.dump(all_results, open(pickle_file, 'wb'))

        # 将labels转化成最后的数据
        labels = tf.data.Dataset()
        return all_results, labels


def generate_labels():
    """生成标签文件"""
    labels = ['O']
    with open(train_file, 'r') as f:
        for line in tqdm(f.readlines()):
            entities = json.loads(line)['entities']
            for entity in entities:
                b_type = f'B-{entity["type"]}'
                if b_type not in labels:
                    labels.append(f'B-{entity["type"]}')
                    labels.append(f'I-{entity["type"]}')
    with open(label_file, 'w') as f:
        f.write('\n'.join(labels))


if __name__ == '__main__':
    # 1. generate train pickle file
    tokenizer = BertTokenizer.from_pretrained(model_name)
    load_data(
        train_file,
        train_pickle_file,
        tokenizer
    )

    # 2. generate labels file
    generate_labels()
