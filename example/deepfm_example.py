import tensorflow as tf
from data_process import parse_data, padding_data
from feature_columns import feature_columns
from models.deepfm import deepfm


if __name__ == "__main__":
    padding_shape, padding_value = padding_data()
    test_dataset = tf.data.TextLineDataset('../dataset/rank_test_data.tsv', num_parallel_reads=4).skip(1)
    test_data = test_dataset.map(parse_data, num_parallel_calls=30).padded_batch(padded_shapes=padding_shape,
                                                                                 padding_values=padding_value,
                                                                                 batch_size=1024)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)
    train_dataset = tf.data.TextLineDataset('../dataset/rank_train_data.tsv', num_parallel_reads=20).skip(1)
    train_data = train_dataset.map(parse_data, num_parallel_calls=60).shuffle(1024000).padded_batch(padded_shapes=padding_shape,
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

    model = deepfm(feature_columns, linear_features_column_names, fm_feature_column_names, l2_reg=1e-4, dropout_rate=0.2)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.8, staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='binary_crossentropy', metrics=tf.metrics.AUC(name='auc'))
    model.fit(train_data, epochs=10, validation_data=test_data, verbose=1)
