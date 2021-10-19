from utils import *


class CrossNet(Layer):

    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4, **kwargs):
        super(CrossNet, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b
        self.cross_weights = None
        self.cross_bias = None

    def build(self, input_shape):
        self.cross_weights = [self.add_weight(name='w'+str(i),
                                              shape=(input_shape[1], 1),
                                              initializer=tf.initializers.glorot_normal(),
                                              regularizer=tf.keras.regularizers.l2(self.reg_w),
                                              trainable=True)
                              for i in range(self.layer_num)]
        self.cross_bias = [self.add_weight(name='b'+str(i),
                                           shape=(input_shape[1], 1),
                                           initializer=tf.initializers.glorot_normal(),
                                           regularizer=tf.keras.regularizers.l2(self.reg_b),
                                           trainable=True)
                           for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)   # [None, n, 1]
        x1 = x0
        for i in range(self.layer_num):
            x1 = tf.matmul(tf.transpose(x1, [0, 2, 1]), self.cross_weights[i])   # [None, 1, 1]
            x1 = tf.matmul(x0, x1) + self.cross_bias[i] + x1  # [None, n, 1]

        output = tf.squeeze(x1, axis=2)
        return output

    def get_config(self):
        config = super(CrossNet, self).get_config()
        config.update({'layer_num': self.layer_num})
        return config


def dcn(inputs, linear_columns_name, dnn_columns_name, layer_num, hidden_unit=(128, 64, 32), activation='relu'):
    linear_feature_columns = [feat_col for feat_col in inputs if feat_col.name in linear_columns_name]
    dnn_feature_columns = [feat_col for feat_col in inputs if feat_col.name in dnn_columns_name]
    feature_columns = linear_feature_columns + dnn_feature_columns
    keras_input = build_keras_inputs(feature_columns)
    input_list = list(keras_input.values())

    embedding_dict = build_embedding_dict(dnn_feature_columns)
    sparse_value_list, _ = input_from_feature_columns(keras_input, dnn_feature_columns, embedding_dict)
    dnn_input = combine_dnn_inputs(sparse_value_list, [])
    # cross_net part
    cross_net = CrossNet(layer_num)
    cross_logit = cross_net(dnn_input)
    # dnn part
    dnn_output = 0
    for i in range(len(hidden_unit)):
        if i == len(hidden_unit) - 1:
            dnn_output = Dense(hidden_unit[i], activation=None, use_bias=True, name='dnn_' + str(i))(dnn_input)
        else:
            dnn_input = Dense(hidden_unit[i], activation=activation, use_bias=True, name='dnn_' + str(i))(dnn_input)
    final_logit = tf.concat([cross_logit, dnn_output], axis=1)
    output = Dense(1, activation='sigmoid')(final_logit)
    return tf.keras.Model(input_list, output)
