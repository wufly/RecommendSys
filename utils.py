from collections import OrderedDict
import tensorflow as tf
from feature_columns import DenseFeat, SparseFeat, VarLenFeat, BucketFeat
from tensorflow.keras.layers import *


def build_keras_inputs(feature_columns):
    """
    :param feature_columns:
    :return:
    """

    inputs_dict = OrderedDict()
    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenFeat):
            inputs_dict[feat_col.name] = Input(shape=(None,), name=feat_col.name, dtype=feat_col.sub_dtype)
            if feat_col.weight_name:
                inputs_dict[feat_col.weight_name] = Input(shape=(None,), name=feat_col.weight_name, dtype=tf.float32)

        elif isinstance(feat_col, (DenseFeat, SparseFeat, BucketFeat)):
            inputs_dict[feat_col.name] = Input(shape=(1,), name=feat_col.name, dtype=feat_col.dtype)

        else:
            raise Exception('unknown feature column type: %s' % feat_col.name)

    return inputs_dict


def build_embedding_matrix(feature_columns, linear_dim=False):

    embedding_matrix = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, (VarLenFeat, SparseFeat)):
            vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
            if vocab_name not in embedding_matrix:
                vocab_size = feat_col.vocab_size+2 if feat_col.vocab_size else feat_col.hash_size
                if not linear_dim:
                    emb = Embedding(vocab_size, feat_col.emb_dim, embeddings_initializer=tf.keras.initializers.truncated_normal(mean=0, stddev=0.001),
                                    embeddings_regularizer=tf.keras.regularizers.l2(0.001))
                    emb.trainable = True
                else:
                    emb = tf.Variable(initial_value=tf.random.truncated_normal((vocab_size, 1), mean=0, stddev=0.001, dtype=tf.float32),
                                      trainable=True, name=vocab_name+'embed_linear')
                embedding_matrix[vocab_name] = emb
        elif isinstance(feat_col, BucketFeat):
            vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
            vocab_size = len(feat_col.boundaries)
            if vocab_name not in embedding_matrix:
                if not linear_dim:
                    emb = Embedding(vocab_size, feat_col.emb_dim, embeddings_initializer=tf.keras.initializers.truncated_normal(
                        mean=0, stddev=0.001), embeddings_regularizer=tf.keras.regularizers.l2(0.001))
                    emb.trainable = True
                else:
                    emb = tf.Variable(initial_value=tf.random.truncated_normal((vocab_size, 1), mean=0, stddev=0.001), trainable=True,
                                      name=feat_col.name+'embed_linear')

                embedding_matrix[vocab_name] = emb
    return embedding_matrix


def build_embedding_dict(feature_columns):






