from utils import *
from tensorflow.keras.layers import Dense
from layers import FMLayer, DNN


def deepfm(inputs, linear_columns_name, dnn_columns_name, hidden_unit=(128, 64, 32), l2_reg=0, activation='relu', dropout_rate=0, seed=1024):
    linear_feature_columns = [feat_col for feat_col in inputs if feat_col.name in linear_columns_name]
    dnn_feature_columns = [feat_col for feat_col in inputs if feat_col.name in dnn_columns_name]
    feature_columns = linear_feature_columns + dnn_feature_columns
    keras_input = build_keras_inputs(feature_columns)
    input_list = list(keras_input.values())
    # linear_part

    linear_embedding_dict = build_embedding_dict(linear_feature_columns, linear=True)

    linear_sparse_value_list, linear_dense_value_list = input_from_feature_columns(keras_input, linear_feature_columns, linear_embedding_dict)

    linear_logit = get_linear_logit(linear_sparse_value_list, linear_dense_value_list)

    # fm_part
    embedding_dict = build_embedding_dict(dnn_feature_columns)
    sparse_value_list, _ = input_from_feature_columns(keras_input, dnn_feature_columns, embedding_dict)
    concat_embedding_values = Concatenate(axis=1)(sparse_value_list)
    fm_logit = FMLayer()(concat_embedding_values)

    # dnn_part
    dnn_input = combine_dnn_inputs(sparse_value_list, [])
    dnn_output = DNN(hidden_unit, l2_reg=l2_reg, dropout_rate=dropout_rate)(dnn_input)
    # dnn_output = 0
    # for i in range(len(hidden_unit)):
    #     if i == len(hidden_unit) - 1:
    #         dnn_output = Dense(hidden_unit[i], activation=activation, use_bias=True, name='dnn_'+str(i))(dnn_input)
    #     else:
    #         dnn_input = Dense(hidden_unit[i], activation=activation, use_bias=True, name='dnn_'+str(i))(dnn_input)

    dnn_logit = Dense(1, activation=None, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    # combine
    final_layer = Add()([linear_logit, fm_logit, dnn_logit])
    output = tf.keras.layers.Activation(activation='sigmoid')(final_layer)
    return tf.keras.Model(input_list, output)
