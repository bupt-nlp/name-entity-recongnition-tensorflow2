from typing import Union, List, Callable, Text
import unicodedata
import six
from collections import OrderedDict
from ner.config import model_name

import tensorflow as tf
from transformers import BertTokenizer

UNKNOWN_TOKEN = '<UNK>'


def convert_to_unicode(text: Union[str, bytes]):
    """convert the text to unicode, such as utf-8 encoding format"""
    if six.PY3:
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')


def load_vocabulary(vocab_file: str):
    """load vocabulary to dict data"""
    vocab = OrderedDict()
    vocab[UNKNOWN_TOKEN] = 0

    with tf.io.gfile.GFile(vocab_file, 'r') as reader:
        for index, line in enumerate(reader, start=1):
            token = line.strip('\n')
            vocab[token] = index
        return vocab


def convert_to_ids(items: List[str], vocab: OrderedDict) -> list:
    """convert tokens to ids"""
    token_ids = []
    for item in items:
        if item in vocab:
            token_ids.append(vocab[item])
        else:
            token_ids.append(vocab[UNKNOWN_TOKEN])
    return token_ids


class FullTokenizer:
    def __init__(self, vocab_file: str, do_lower_case: bool = True, tokenizer: Callable[[Text], List[str]] = None):

        # 1. load vocabulary from vocab_file
        self.vocab: OrderedDict = load_vocabulary(vocab_file)

        # 2. build index to vocabulary
        self.index2vocab: OrderedDict = OrderedDict(
            {index: token for token, index in self.vocab.items()}
        )

        if tokenizer:
            self.tokenizers: Callable[[Text], List[int]] = tokenizer
        else:
            self.tokenizers: Callable[[Text], List[int]] = BertTokenizer.from_pretrained(model_name).tokenize

    def tokenize(self, text: str) -> List[int]:
        """tokenize your text base your own corpus

        you can change the code here to use different tokenizer"""
        return self.tokenizers(text)

    def convert_tokens_to_ids(self, tokens: List[str]):
        return convert_to_ids(tokens, self.vocab)
