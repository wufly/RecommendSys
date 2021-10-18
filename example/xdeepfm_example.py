import tensorflow as tf
from data_process import parse_data, padding_data
from feature_columns import feature_columns
from tensorflow.keras.layers import *
from models.xdeepfm import xdeepfm


if __name__ == "__main__":
    padding_shape, padding_value = padding_data()
    dataset = tf.data.TextLineDataset('../dataset/rank_test_data.tsv', num_parallel_reads=4).skip(1)
    test_data = dataset.take(500000).map(parse_data, num_parallel_calls=30).padded_batch(padded_shapes=padding_shape, padding_values=padding_value,
                                                                                         batch_size=1024)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)
    train_data = dataset.skip(500000).map(parse_data, num_parallel_calls=60).shuffle(102400).padded_batch(padded_shapes=padding_shape,
                                                                                                          padding_values=padding_value,
                                                                                                          batch_size=1024)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    linear_features_column_names = [
        'user_id', 'job_id', 'distance', 'salary_min', 'salary_max', 'company_id', 'gender',
        'age', 'new_channel_no', 'expect_salary_min', 'expect_salary_max', 'fast_job_status', 'expect_job', 'city_id', 'category_id'
    ]

    fm_feature_column_names = [
        'user_id', 'job_id', 'distance', 'salary_min', 'salary_max', 'company_id', 'gender',
        'age', 'new_channel_no', 'expect_salary_min', 'expect_salary_max', 'fast_job_status', 'expect_job', 'city_id', 'category_id'
    ]

    model = xdeepfm(feature_columns, linear_features_column_names, fm_feature_column_names, cin_size=[128, 128])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=tf.metrics.AUC(name='auc'))
    model.fit(train_data, epochs=5, validation_data=test_data, verbose=1)
