import tensorflow as tf
from feature_columns import feature_columns, VarLenFeat, SparseFeat, DenseFeat, BucketFeat
from config.config import COL_NAME, DEFAULT_VALUE
from tensorflow.python.ops.gen_math_ops import bucketize


def parse_data(line):

    csv_data = tf.io.decode_csv(line, record_defaults=DEFAULT_VALUE, field_delim='\t')
    parsed_data = dict(zip(COL_NAME, csv_data))
    feature_dict = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenFeat):
            if feat_col.weight_name:
                kvpairs = tf.strings.split([parsed_data[feat_col.name]], ',').values[:feat_col.max_len]
                kvpairs = tf.strings.split(kvpairs, ':')
                ids, val = tf.split(kvpairs, num_or_size_splits=2)
                ids = tf.reshape(ids, shape=[-1])
                val = tf.reshape(val, shape=[-1])
                if feat_col.sub_dtype != 'string':
                    ids = tf.strings.to_number(ids, out_type=tf.int32)
                feature_dict[feat_col.name] = ids
                feature_dict[feat_col.weight_name] = tf.strings.to_number(val, out_type='float32')
            else:
                ids = tf.strings.split([parsed_data[feat_col.name]], ',').values[:feat_col.max_len]
                ids = tf.reshape(ids, shape=[-1])
                if feat_col.sub_dtype != 'string':
                    ids = tf.strings.to_number(ids, out_type=tf.int32)
                feature_dict[feat_col.name] = ids

        elif isinstance(feat_col, (DenseFeat, SparseFeat)):
            feature_dict[feat_col.name] = parsed_data[feat_col.name]

        elif isinstance(feat_col, BucketFeat):
            bucket_num = bucketize(parsed_data[feat_col.name], feat_col.boundaries)
            feature_dict[feat_col.name] = bucket_num

        else:
            raise Exception('unknown feature column in parse_data {}'.format(feat_col.name))

    label = parsed_data['label']
    return feature_dict, label


def padding_data():
    pad_shape = {}
    pad_value = {}

    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenFeat):
            pad_shape[feat_col.name] = tf.TensorShape([feat_col.max_len])
            pad_value[feat_col.name] = '0' if feat_col.sub_dtype == 'string' else 0
            if feat_col.weight_name:
                pad_shape[feat_col.weight_name] = tf.TensorShape([feat_col.max_len])
                pad_value[feat_col.weight_name] = tf.constant(0, dtype=tf.float32)

        elif isinstance(feat_col, (DenseFeat, SparseFeat, BucketFeat)):
            pad_shape[feat_col.name] = tf.TensorShape([])
            pad_value[feat_col.name] = '0' if feat_col.dtype == 'string' else 0

        else:
            raise Exception('unknown feature column in padding_data {}'.format(feat_col.name))
        
    pad_shape = (pad_shape, tf.TensorShape([]))
    pad_value = (pad_value, 0)

    return pad_shape, pad_value


if __name__ == '__main__':
    dataset = tf.data.TextLineDataset('./dataset/rank_test_data.tsv', num_parallel_reads=4).skip(1)
    dataset = dataset.take(5).map(parse_data, num_parallel_calls=20)
    print(list(dataset))
