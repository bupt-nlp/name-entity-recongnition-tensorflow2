import os
import tensorflow as tf
from transformers import BertConfig, TFBertModel


class BertNer(tf.keras.Model):
    def __init__(self, bert_model: str, float_type, num_labels: int, max_seq_length: int, final_layer_initializer = None):
        super().__init__()

        # 1. define the inputs of the model
        input_word_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

        # 2. load the bert configuration
        if isinstance(bert_model, str):
            config_file = os.path.join(bert_model, 'bert_config.json')
            bert_config = BertConfig.from_json_file(config_file)
        elif isinstance(bert_model, dict):
            bert_config = BertConfig.from_dict(bert_model)

        # 3. build bert layer to get sequence output
        bert_layer = TFBertModel(config=bert_config, float_type=float_type)
        _, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)

        # 4. restore the bert model checkpoint from the disk
        self.bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[sequence_output])
        if isinstance(bert_model, str):
            init_checkpoint = os.path.join(bert_model, 'bert_model.ckpt')
            checkpoint = tf.train.Checkpoint(model=self.bert)
            checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

        # 5. init the initializer
        if final_layer_initializer:
            initializer = final_layer_initializer
        else:
            initializer = tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range
            )

        # 6. define the dropout layer
        self.dropout = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)

        # 7. define the final classifier layer to get logits
        self.classifier = tf.keras.layers.Dense(
            units=num_labels,
            kernel_initializer=initializer,
            activation='softmax',
            name='output_layer',
        )

    def call(self, input_word_ids, input_mask, input_type_ids, valid_mask, **kwargs):
        """apply the forward method on training data"""
        sequence_output: tf.Tensor = self.bert([input_word_ids, input_mask, input_type_ids], **kwargs)
        sequence_output = self.dropout(sequence_output, trainable=kwargs.get('training', False))
        logits = self.classifier(sequence_output)
        return logits

    def get_config(self):
        return super(BertNer, self).get_config()

