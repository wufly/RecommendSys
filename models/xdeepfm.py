import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K


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
        if K.ndim(input_shape) != 3:
            raise Exception('input_shape must be 3-dimensional')
        self.field_num = [input_shape[1]] + self.cin_size
        self.cin_W = [self.add_weight(name='w'+str(i),
                                      shape=(1, self.field_num[0] * self.field_num[i], self.field_num[i+1]),
                                      initializers=tf.initializers.glorot_uniform(),
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
            x = tf.transpose(x, [1, 0, 2])
