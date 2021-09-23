from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import backend as K


class VocabularyLayer(Layer):

    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabularyLayer, self).__init__(**kwargs)
        self.keys = keys
        self.mask_value = mask_value
        val = tf.constant(range(2, len(keys)+2), dtype=tf.int32)
        keys = tf.constant(self.keys)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, val), 1)

    def call(self, inputs, **kwargs):
        if inputs.dtype != 'string':
            inputs = tf.cast(inputs, tf.int32)
        idx = self.table.lookup(inputs)
        if self.mask_value:
            mask = tf.not_equal(inputs, tf.constant(self.mask_value, dtype=inputs.dtype))
            padding = tf.ones_like(idx) * 0
            idx = tf.where(mask, idx, padding)
        return idx

    def get_config(self):
        config = super(VocabularyLayer, self).get_config()
        config.update({'keys': self.keys, 'mask_value': self.mask_value})
        return config


class HashLayer(Layer):

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        super(HashLayer, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero

    def call(self, inputs, **kwargs):
        zeros = tf.zeros_like(inputs, dtype=tf.string)
        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets-1
        if inputs.dtype != 'string':
            inputs = tf.as_string(inputs)
        hash_x = tf.strings.to_hash_bucket_fast(inputs, num_buckets)
        if self.mask_zero:
            masks = tf.cast(tf.not_equal(zeros, inputs), tf.int64)
            hash_x = (hash_x + 1) * masks
        return hash_x

    def get_config(self):
        config = super(HashLayer, self).get_config()
        config.update({'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero})
        return config


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        elif len(inputs) == 0:
            return tf.constant([0])
        else:
            return add(inputs)


class EmbeddingLookup(Layer):

    def __init__(self, embed, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embed = embed

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.int32)
        return tf.nn.embedding_lookup(self.embed, inputs)

    def get_config(self):
        config = super(EmbeddingLookup, self).get_config()
        config.update({'embed': self.embed})
        return config


class EmbeddingLookupSparse(Layer):

    def __init__(self, embed, combiner='sum', has_weights=False, **kwargs):
        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.embed = embed
        self.combiner = combiner
        self.has_weights = has_weights

    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.has_weights:
            ids, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embed, sp_ids=ids, sp_weights=val, combiner=self.combiner)
        else:
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embed, sp_ids=inputs, sp_weights=None, combiner=self.combiner)
        return tf.expand_dims(combiner_embed, axis=1)

    def get_config(self):
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'embed': self.embed, 'combiner': self.combiner, 'has_weights': self.has_weights})


class DenseTensorToSparse(Layer):

    def __init__(self, mask_value=0, **kwargs):
        super(DenseTensorToSparse, self).__init__(**kwargs)
        self.mask_value = mask_value

    def build(self, input_shape):
        super(DenseTensorToSparse, self).build(input_shape)

    def call(self, inputs, **kwargs):
        masks = tf.not_equal(tf.constant(self.mask_value, dtype=inputs.dtype), inputs)
        idx = tf.where(masks)
        return tf.SparseTensor(idx, tf.gather_nd(inputs, idx), tf.shape(inputs, out_type=tf.int64))

    def get_config(self):
        config = super(DenseTensorToSparse, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config


class FMLayer(Layer):

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception('Unexpected inputs dimensions % d, expect to be 3 dimensions' % (len(input_shape)))
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise Exception('Unexpected inputs dimensions % d, expect to be 3 dimensions' % (K.ndim(inputs)))
        square_of_sum = tf.square(tf.reduce_sum(inputs, 1, keepdims=True))
        sum_of_square = tf.reduce_sum(tf.square(inputs), 1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        return 0.5 * tf.reduce_sum(cross_term, 2, keepdims=False)

    def compute_output_shape(self, input_shape):
        return None, 1
