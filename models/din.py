from utils import *
from layers import DNN


class LocalAttentionUnit(Layer):
    """
    input_shape
        - A list of two 3D tensor with shape [(None, 1, embedding_size), (None, T, embedding_size))
    output shape
        - 3D tensor with shape [(None, T, 1)]
    """
    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        super(LocalAttentionUnit, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.supports_masking = True
        self.kernel = None
        self.bias = None
        self.dnn = None
        self.dense = None

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A LocalAttentionUnit layer should be called on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('A LocalAttentionUnit layer require inputs of two input with shape (batch_size, 1, embedding_size)'
                             'and (batch_size, T, embedding_size), got shape: %s, %s' % (input_shape[0], input_shape[1]))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))
        
        size = 4 * int(input_shape[0][-1]) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(name='kernel', shape=(size, 1), initializer=tf.initializers.glorot_normal(self.seed))
        self.bias = self.add_weight(name='bias', shape=(1,), initializer=tf.initializers.Zeros())
        self.dnn = DNN(hidden_units=self.hidden_units, activation=self.activation, l2_reg=self.l2_reg,
                       dropout_rate=self.dropout_rate, use_bn=self.use_bn, seed=self.seed)
        self.dense = tf.keras.layers.Lambda(lambda x: tf.nn.bias_add(tf.tensordot(x[0], x[1], axes=(-1, 0)), x[2]))
        
        super(LocalAttentionUnit, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, keys = inputs
        keys_len = keys.get_shape()[1]  # T
        queries = K.repeat_elements(query, keys_len, 1)  # (None, T, embedding_size)
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=1)  # (None, T, 4 * embedding_size)
        att_output = self.dnn(att_input)  # (None, T, hidden_units[-1])
        attention_score = self.dense([att_output, self.kernel, self.bias])  # (None, T, 1)
        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1, )

    def get_config(self):
        base_config = super(LocalAttentionUnit, self).get_config()
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'dropout_rate': self.dropout_rate, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config.update(config)
        return base_config


class AttentionSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in din
      inputs shape:
        - A list of three tensor: [query, keys, keys_len]
        - query: (None, 1, embedding_size)
        - keys: (None, T, embedding_size)
        - keys_len: (None, 1)
      output shape:
        - 3D tensor with shape: (None, 1, embedding_size)
    """
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False, return_score=False,
                 supports_masking=False, **kwargs):
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        self.supports_masking = supports_masking
        self.local_att = None

    def build(self, input_shape):
        if not self.supports_masking:
            assert isinstance(input_shape, list) and len(input_shape) == 3
            assert len(input_shape[0]) == 3 and len(input_shape[1]) == 3 and len(input_shape[2]) == 2
            assert input_shape[0][-1] == input_shape[1][-1] and input_shape[0][1] == 1 and input_shape[2][1] == 1

        self.local_att = LocalAttentionUnit(self.att_hidden_units, self.att_activation)

        super(AttentionSequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            if not mask:
                raise ValueError('when supports_masking=True, inputs must supports masking')

            queries, keys = inputs
            key_masks = tf.expand_dims(mask[-1], axis=1)

        else:
            queries, keys, keys_len = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_len, hist_len)  # (None, 1, T)

        attention_score = LocalAttentionUnit()([queries, keys])
        outputs = tf.transpose(attention_score, (0, 2, 1))  # （None, 1, T)

        if self.weight_normalization:
            padding = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            padding = tf.ones_like(outputs)

        outputs = tf.where(key_masks, outputs, padding)

        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        outputs._uses_learning_phase = training

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return None, 1, input_shape[1][1]
        else:
            return None, 1, input_shape[0][-1]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        base_config.update(config)
        return base_config


def din(feature_columns, dnn_feature_name, history_feature_list, dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu',
        att_hidden_units=(80, 40), att_activation='dice', att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6,
        dnn_dropout=0, seed=1024):

    dnn_feature_columns = [feat_col for feat_col in feature_columns if feat_col.name in dnn_feature_name]
    keras_inputs = build_keras_inputs(dnn_feature_columns)

