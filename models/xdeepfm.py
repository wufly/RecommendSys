from utils import *


class CIN(Layer):

    def __init__(self, cin_size, **kwargs):
        super(CIN, self).__init__(**kwargs)
        self.cin_size = cin_size
        self.field_num = None
        self.cin_W = None

    def build(self, input_shape):
        """

        :param input_shape: (None, n, k)
        :return:
        """
        if len(input_shape) != 3:
            raise Exception('input_shape must be 3-dimensional')
        self.field_num = [input_shape[1]] + self.cin_size
        self.cin_W = [self.add_weight(name='w'+str(i),
                                      shape=(1, self.field_num[0] * self.field_num[i], self.field_num[i+1]),
                                      initializer=tf.initializers.glorot_uniform(),
                                      regularizer=tf.keras.regularizers.l1_l2(1e-5),
                                      trainable=True)
                      for i in range(len(self.field_num)-1)]

    def call(self, inputs, **kwargs):
        k = inputs.shape[-1]
        res_list = [inputs]
        x0 = tf.split(inputs, k, axis=-1)  # list: k * [None, field_num[0], 1]
        for i, size in enumerate(self.field_num[1:]):
            xi = tf.split(res_list[-1], k, axis=-1)  # list: k * [None, field_num[i], 1]
            x = tf.matmul(x0, xi, transpose_b=True)  # list: k * [None, field_num[0], field_num[i]]
            x = tf.reshape(x, shape=[k, -1, self.field_num[0] * self.field_num[i]])  # [k, None, self.field_num[0] * self.field_num[i]]
            x = tf.transpose(x, [1, 0, 2])  # [None, k, field_num[0] * self.field[1]]
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1, padding='VALID')  # [None, k, field_num[i+1]]
            x = tf.transpose(x, [0, 2, 1])  # [None, field_num[i+1], k]
            res_list.append(x)
        res_list = res_list[1:]
        res = tf.concat(res_list, axis=1)
        output = tf.reduce_sum(res, axis=-1)
        return output

    def get_config(self):

        config = super(CIN, self).get_config()
        config.update({'cin_size': self.cin_size})
        return config


def xdeepfm(inputs, linear_columns_name, dnn_columns_name, cin_size, hidden_unit=(128, 64, 32), activation='relu', seed=1024):
    linear_feature_columns = [feat_col for feat_col in inputs if feat_col.name in linear_columns_name]
    dnn_feature_columns = [feat_col for feat_col in inputs if feat_col.name in dnn_columns_name]
    feature_columns = linear_feature_columns + dnn_feature_columns
    keras_input = build_keras_inputs(feature_columns)
    input_list = list(keras_input.values())
    # linear_part

    linear_embedding_dict = build_embedding_dict(linear_feature_columns, linear=True)

    linear_sparse_value_list, linear_dense_value_list = input_from_feature_columns(keras_input, linear_feature_columns, linear_embedding_dict)

    linear_logit = get_linear_logit(linear_sparse_value_list, linear_dense_value_list)

    # cin_part
    embedding_dict = build_embedding_dict(dnn_feature_columns)
    sparse_value_list, _ = input_from_feature_columns(keras_input, dnn_feature_columns, embedding_dict)
    concat_embedding_values = Concatenate(axis=1)(sparse_value_list)
    cin = CIN(cin_size)
    cin_logit = Dense(1)(cin(concat_embedding_values))

    # dnn_part
    dnn_input = combine_dnn_inputs(sparse_value_list, [])
    dnn_output = 0
    for i in range(len(hidden_unit)):
        if i == len(hidden_unit) - 1:
            dnn_output = Dense(hidden_unit[i], activation=activation, use_bias=True, name='dnn_' + str(i))(dnn_input)
        else:
            dnn_input = Dense(hidden_unit[i], activation=activation, use_bias=True, name='dnn_' + str(i))(dnn_input)

    dnn_logit = Dense(1, activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    # combine
    final_layer = Add()([linear_logit, cin_logit, dnn_logit])
    output = tf.keras.layers.Activation(activation='sigmoid')(final_layer)
    return tf.keras.Model(input_list, output)
