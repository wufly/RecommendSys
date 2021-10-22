from utils import *
from itertools import combinations
from tensorflow.keras import backend as K


class AFMLayer(Layer):
    """
        Input_shape
            - A list of 3D tensor with shape: (batch_size, 1, embedding_size)
        Output_shape
            - 2D tensor with shape: (batch_size, 1)
    """

    def __init__(self, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, **kwargs):
        super(AFMLayer, self).__init__(**kwargs)
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.attention_W = None
        self.attention_b = None
        self.projection_h = None
        self.projection_p = None
        self.dropout = None

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError("`AttentionalFM` layer should be called on a list of at least 2 inputs")
        shape_set = set()
        reduced_input_shape = [shape.as_list() for shape in input_shape]
        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_input_shape[i]))
        if len(shape_set) > 1:
            raise ValueError("AttentionalFM layer requires inputs with same shapes")
        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError("AttentionalFM layer require inputs of a list tensor with shape(None, 1, embedding_size)")
        embedding_size = int(input_shape[0][-1])

        self.attention_W = self.add_weight(
            name="attention_W",
            shape=(embedding_size, self.attention_factor),
            initializer=tf.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg_w),
            trainable=True
        )

        self.attention_b = self.add_weight(
            name="attention_b",
            shape=(self.attention_factor,),
            initializer=tf.initializers.Zeros(),
            trainable=True
        )

        self.projection_h = self.add_weight(
            name="projection_h",
            shape=(self.attention_factor, 1),
            initializer=tf.initializers.glorot_normal(),
        )

        self.projection_p = self.add_weight(
            name="projection_p",
            shape=(embedding_size, 1),
            initializer=tf.initializers.glorot_normal()
        )

        self.dropout = Dropout(rate=self.dropout_rate, seed=self.seed)

        super(AFMLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embedding_vec_list = inputs
        row = []
        col = []

        for r, c in combinations(embedding_vec_list, 2):
            row.append(r)
            col.append(c)

        p = tf.concat(row, axis=1)  # [None, n(n-1)/2, k)
        q = tf.concat(col, axis=1)  # [None, n(n-1)/2, k)
        inner_product = p * q  # [None, n(n-1)/2, k)

        attention_tmp = tf.nn.relu(tf.nn.bias_add(
            # self.attention_W:[k, self.attention_factor], inner_product: [None, n(n-1)/2, k]
            # [None, n(n-1)/2, self.attention_factor]
            tf.tensordot(inner_product, self.attention_W, axes=(-1, 0)), self.attention_b
        ))

        # [None, n(n-1)/2, 1]
        normalized_att_score = tf.nn.softmax(
            # [None, n(n-1)/2, self.attention_factor] * [self.attention_factor, 1]
            tf.tensordot(attention_tmp, self.projection_h, axes=(-1, 0)), axis=1
        )

        # [None,  embedding_size)
        attention_output = tf.reduce_sum(inner_product * normalized_att_score, axis=1)
        attention_output = self.dropout(attention_output, training=training)

        # [None, embedding_size]  [embedding_size, 1]
        afm_out = tf.matmul(attention_output, self.projection_p)
        return afm_out

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self):

        config = super(AFMLayer, self).get_config()
        config.update({'attention_factor': self.attention_factor, 'l2_reg_w': self.l2_reg_w, 'dropout_rate': self.dropout_rate})
        return config


def afm(inputs, linear_columns_name, dnn_columns_name, attention_factor, l2_reg_w=0, dropout_rate=0,  activation='relu', seed=1024):
    linear_feature_columns = [feat_col for feat_col in inputs if feat_col.name in linear_columns_name]
    dnn_feature_columns = [feat_col for feat_col in inputs if feat_col.name in dnn_columns_name]
    feature_columns = linear_feature_columns + dnn_feature_columns
    keras_input = build_keras_inputs(feature_columns)
    input_list = list(keras_input.values())

    # linear_part
    linear_embedding_dict = build_embedding_dict(linear_feature_columns, linear=True)
    linear_sparse_value_list, linear_dense_value_list = input_from_feature_columns(keras_input, linear_feature_columns, linear_embedding_dict)
    linear_logit = get_linear_logit(linear_sparse_value_list, linear_dense_value_list)

    # afm part
    embedding_dict = build_embedding_dict(dnn_feature_columns)
    sparse_value_list, _ = input_from_feature_columns(keras_input, dnn_feature_columns, embedding_dict)
    afm_out = AFMLayer(attention_factor, l2_reg_w, dropout_rate)(sparse_value_list)

    # combine
    final_logit = Add()([linear_logit, afm_out])

    output = tf.keras.layers.Activation(activation='sigmoid')(final_logit)
    return tf.keras.Model(input_list, output)
