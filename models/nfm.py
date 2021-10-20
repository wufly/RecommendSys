from utils import *


def nfm(inputs, linear_columns_name, dnn_columns_name, hidden_unit=(128, 64, 32), activation='relu', seed=1024):
    linear_feature_columns = [feat_col for feat_col in inputs if feat_col.name in linear_columns_name]
    dnn_feature_columns = [feat_col for feat_col in inputs if feat_col.name in dnn_columns_name]
    feature_columns = linear_feature_columns + dnn_feature_columns
    keras_input = build_keras_inputs(feature_columns)
    inputs_list = list(keras_input.values())

    # linear part
    linear_embedding_dict = build_embedding_dict(linear_feature_columns, linear=True)
    linear_sparse_value_list, linear_dense_value_list = input_from_feature_columns(keras_input, linear_feature_columns, linear_embedding_dict)
    linear_logit = get_linear_logit(linear_sparse_value_list, linear_dense_value_list)

    # nfm part
    embedding_dict = build_embedding_dict(dnn_feature_columns)
    sparse_value_list, _ = input_from_feature_columns(keras_input, dnn_feature_columns, embedding_dict)
    concat_embedding_values = Concatenate(axis=1)(sparse_value_list)

    bi_interaction_layer = 0.5 * (tf.pow(tf.reduce_sum(concat_embedding_values, axis=1), 2) -
                                  tf.reduce_sum(tf.pow(concat_embedding_values, 2), axis=1))  # (None, k)
    output = 0
    for i in range(len(hidden_unit)):
        if i == len(hidden_unit) - 1:
            output = Dense(units=hidden_unit[i], activation=activation, use_bias=True, name='dnn'+str(i))(bi_interaction_layer)
        else:
            bi_interaction_layer = Dense(units=hidden_unit[i], activation=activation, use_bias=True, name='dnn'+str(i))(bi_interaction_layer)
    dnn_logit = Dense(1, activation=None, use_bias=False, kernel_initializer=tf.initializers.glorot_normal(seed))(output)
    # combine
    final_layer = Add()([linear_logit, dnn_logit])
    output = tf.keras.layers.Activation(activation='sigmoid')(final_layer)
    return tf.keras.Model(inputs_list, output)

