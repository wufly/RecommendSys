from collections import OrderedDict
from feature_columns import DenseFeat, SparseFeat, VarLenFeat, BucketFeat
from tensorflow.keras.layers import *
from layers import *
from config.config import MASK_VALUE, DICT_CATEGORICAL


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
            raise Exception('unknown feature column type: %s' % feat_col)

    return inputs_dict


def build_embedding_matrix(feature_columns, linear_dim=False):

    embedding_matrix = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, (VarLenFeat, SparseFeat)):
            vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
            if vocab_name not in embedding_matrix:
                vocab_size = feat_col.vocab_size+2 if feat_col.vocab_size else feat_col.hash_size
                if not linear_dim:
                    # emb = Embedding(vocab_size, feat_col.emb_dim, embeddings_initializer=tf.keras.initializers.truncated_normal(mean=0,
                    #                                                                                                             stddev=0.001),
                    #                 embeddings_regularizer=tf.keras.regularizers.l2(0.001))
                    # emb.trainable = True
                    emb = tf.Variable(initial_value=tf.random.truncated_normal((vocab_size, feat_col.emb_dim), mean=0, stddev=0.001,
                                                                               dtype=tf.float32),
                                      trainable=True, name=vocab_name+'embed')
                else:
                    emb = tf.Variable(initial_value=tf.random.truncated_normal((vocab_size, 1), mean=0, stddev=0.001, dtype=tf.float32),
                                      trainable=True, name=vocab_name+'embed_linear')
                embedding_matrix[vocab_name] = emb
        elif isinstance(feat_col, BucketFeat):
            vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
            vocab_size = len(feat_col.boundaries) + 1
            if vocab_name not in embedding_matrix:
                if not linear_dim:
                    # emb = Embedding(vocab_size, feat_col.emb_dim, embeddings_initializer=tf.keras.initializers.truncated_normal(
                    #     mean=0, stddev=0.001), embeddings_regularizer=tf.keras.regularizers.l2(0.001))
                    # emb.trainable = True
                    emb = tf.Variable(initial_value=tf.random.truncated_normal((vocab_size, feat_col.emb_dim), mean=0, stddev=0.001,
                                                                               dtype=tf.float32),
                                      trainable=True, name=vocab_name + 'embed')
                else:
                    emb = tf.Variable(initial_value=tf.random.truncated_normal((vocab_size, 1), mean=0, stddev=0.001), trainable=True,
                                      name=feat_col.name+'embed_linear')

                embedding_matrix[vocab_name] = emb
    return embedding_matrix


def build_embedding_dict(feature_columns, linear=False):
    embedding_dict = {}
    embedding_matrix = build_embedding_matrix(feature_columns, linear_dim=linear)
    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenFeat):
            vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
            if feat_col.weight_name:
                embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding_matrix[vocab_name], has_weights=True, combiner=feat_col.combiner)
            else:
                embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding_matrix[vocab_name], combiner=feat_col.combiner)

        elif isinstance(feat_col, (SparseFeat, BucketFeat)):
            vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding_matrix[vocab_name])

    return embedding_dict


def input_from_feature_columns(inputs, feature_columns, embedding_dict):
    sparse_value_list = []
    dense_value_list = []

    for feat_col in feature_columns:
        _input = inputs[feat_col.name]
        if isinstance(feat_col, VarLenFeat):
            if not feat_col.hash_size:
                vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
                keys = DICT_CATEGORICAL[vocab_name]
                ids = VocabularyLayer(keys, mask_value=MASK_VALUE[feat_col.name])(_input)
            else:
                ids = HashLayer(feat_col.hash_size)(_input)
            sparse_input = DenseTensorToSparse()(ids)
            if feat_col.weight_name:
                weights = DenseTensorToSparse()(inputs[feat_col.weight_name])
                emb = embedding_dict[feat_col.name]([sparse_input, weights])
            else:
                emb = embedding_dict[feat_col.name](sparse_input)
            sparse_value_list.append(emb)

        elif isinstance(feat_col, SparseFeat):
            if feat_col.hash_size:
                ids = HashLayer(feat_col.hash_size, mask_zero=True)(_input)
            else:
                vocab_name = feat_col.share_emb if feat_col.share_emb else feat_col.name
                keys = DICT_CATEGORICAL[vocab_name]
                ids = VocabularyLayer(keys, mask_value=MASK_VALUE[feat_col.name])(_input)

            sparse_value_list.append(embedding_dict[feat_col.name](ids))

        elif isinstance(feat_col, BucketFeat):
            sparse_value_list.append(embedding_dict[feat_col.name](_input))

        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(_input)

        else:
            raise Exception('unknown feature column type: %s' % feat_col.name)

    return sparse_value_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def get_linear_logit(sparse_value_list, dense_value_list):
    if sparse_value_list and dense_value_list:
        sparse_value = Flatten()(Add(sparse_value_list))
        dense_value = Dense(1)(Flatten()(concat_func(dense_value_list)))
        return Add([sparse_value, dense_value])
    elif sparse_value_list:
        return Flatten()(Add()(sparse_value_list))
    elif dense_value_list:
        return Dense(1)(Flatten()(concat_func(dense_value_list)))
    else:
        raise Exception('inputs is empty:get_linear_logit')


def combine_dnn_inputs(sparse_value_list, dense_value_list):
    if sparse_value_list and dense_value_list:
        sparse_value = Flatten()(concat_func(sparse_value_list))
        dense_value = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_value, dense_value])
    elif sparse_value_list:
        return Flatten()(concat_func(sparse_value_list))
    elif dense_value_list:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise Exception('inputs is empty: combine_dnn_inputs')
