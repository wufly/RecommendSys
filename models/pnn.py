from utils import *


class InnerProductLayer(Layer):

    def __init__(self, d1, reg=1e-4, **kwargs):
        """

        :param d1: 第一层mlp的units个数
        :param kwargs:
        """
        super(InnerProductLayer, self).__init__(**kwargs)
        self.d1 = d1
        self.reg = reg
        self.Wp = None
        self.b1 = None

    def build(self, input_shape):
        self.Wp = self.add_weight(
            name="Wp",
            shape=(self.d1, input_shape[1]),
            initializer=tf.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l2(self.reg),
            trainable=True
        )
        self.b1 = self.add_weight(
            name="b1",
            shape=(1, self.d1),
            initializer=tf.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l2(self.reg),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        """

        :param inputs: shape:[None, n, m]
        :param kwargs:
        :return:
        """

        lp = []
        for i in range(self.d1):
            signal = K.reshape(self.Wp[i], (len(self.Wp[i]), 1)) * inputs
            lp.append(0.5 * tf.reduce_sum(tf.pow(tf.reduce_sum(signal, axis=1), 2) - tf.reduce_sum(tf.pow(signal, 2), axis=1), axis=1, keepdims=True))
        lp = Concatenate(axis=-1)(lp)
        return lp # [None, d1]

    def get_config(self):
        config = super(InnerProductLayer, self).get_config()
        config.update({'d1': self.d1, 'reg': self.reg})


class OuterProductLayer(Layer):

    def __init__(self, **kwargs):
        super(OuterProductLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """

        :param inputs: shape:[None, n, k]
        :param kwargs:
        :return: [None, k * k]
        """
        sum_of_f = tf.reduce_sum(inputs, axis=1, keepdims=True)  # (None, 1, k)
        lp = 0.5 * (
                # (None, k, k)
                tf.matmul(
                    # (None, k, 1)
                    tf.transpose(sum_of_f, (0, 2, 1)),
                    sum_of_f)  #
                    -
                # (None, k, k)
                tf.matmul(
                    # (None, k, n)
                    tf.transpose(inputs, (0, 2, 1)),
                    # (None, n, k)
                    inputs))
        return Flatten()(lp)


def pnn(inputs, columns_name, product_type='inner', hidden_unit=(128, 64, 32), activation='relu', seed=1024):

    input_feature_columns = [feat_col for feat_col in inputs if feat_col.name in columns_name]
    keras_input = build_keras_inputs(input_feature_columns)
    input_list = list(keras_input.values())

    embedding_dict = build_embedding_dict(input_feature_columns)
    sparse_value_list, _ = input_from_feature_columns(keras_input, input_feature_columns, embedding_dict)
    concat_embedding_values = Concatenate(axis=1)(sparse_value_list)
    if product_type == 'inner':
        lp_output = InnerProductLayer(hidden_unit[0])(concat_embedding_values)
    elif product_type == 'outer':
        lp_output = OuterProductLayer()(concat_embedding_values)
        lp_output = Dense(hidden_unit[0])(lp_output)
    else:
        raise Exception('product_type must be inner or outer')
    # lz part
    lz_input = combine_dnn_inputs(sparse_value_list, [])
    lz_output = Dense(hidden_unit[0])(lz_input)
    inner_product_layer = tf.keras.activations.relu(Add()([lp_output, lz_output]))
    output = 0
    for i in range(1, len(hidden_unit)):
        if i == len(hidden_unit) - 1:
            output = Dense(hidden_unit[i], activation=activation)(inner_product_layer)
        else:
            inner_product_layer = Dense(hidden_unit[i], activation=activation)(inner_product_layer)

    final_output = Dense(1, activation='sigmoid')(output)
    return tf.keras.Model(input_list, final_output)
