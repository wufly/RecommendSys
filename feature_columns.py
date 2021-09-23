from collections import namedtuple
from config.config import BOUNDARIES, DICT_CATEGORICAL


SparseFeat = namedtuple('SparseFeat', ['name', 'vocab_size', 'hash_size', 'share_emb', 'emb_dim', 'dtype'])
VarLenFeat = namedtuple('VarLenFeat', ['name', 'vocab_size', 'hash_size', 'share_emb', 'weight_name', 'emb_dim', 'max_len', 'combiner', 'dtype', 'sub_dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dim', 'share_emb', 'dtype'])
BucketFeat = namedtuple('BucketFeat', ['name', 'boundaries', 'share_emb', 'emb_dim', 'dtype'])

feature_columns = [
    SparseFeat(name='user_id', vocab_size=None, hash_size=100000, share_emb=None, emb_dim=16, dtype='int64'),
    SparseFeat(name='job_id', vocab_size=None, hash_size=20000, share_emb=None, emb_dim=16, dtype='int64'),
    BucketFeat(name='distance', boundaries=BOUNDARIES['distance'], share_emb=None, emb_dim=16, dtype='float64'),
    # VarLenFeat(name='employer_tag', vocab_size=len(DICT_CATEGORICAL['employer_tag']), hash_size=None, share_emb=None, weight_name=None, emb_dim=16, max_len=5,
    #            combiner='sum', dtype='string', sub_dtype='string'),
    # VarLenFeat(name='hightlighttag', vocab_size=len(DICT_CATEGORICAL['hightlighttag']), hash_size=None, share_emb=None, weight_name=None, emb_dim=16, max_len=5,
    #            combiner='sum', dtype='string', sub_dtype='string'),
    BucketFeat(name='salary_min', boundaries=BOUNDARIES['salary'], share_emb='salary', emb_dim=16, dtype='int64'),
    BucketFeat(name='salary_max', boundaries=BOUNDARIES['salary'], share_emb='salary', emb_dim=16, dtype='int64'),
    # SparseFeat(name='active_tag', vocab_size=len(DICT_CATEGORICAL['active_tag']), hash_size=None, share_emb=None, emb_dim=16, dtype='string'),
    SparseFeat(name='company_id', vocab_size=None, hash_size=10000, share_emb=None, emb_dim=16, dtype='int64'),
    SparseFeat(name='gender', vocab_size=3, hash_size=None, emb_dim=16, dtype='int64', share_emb=None),
    BucketFeat(name='age', boundaries=BOUNDARIES['age'], share_emb=None, emb_dim=16, dtype='int64'),
    SparseFeat(name='new_channel_no', vocab_size=len(DICT_CATEGORICAL['new_channel_no']), hash_size=None, share_emb=None, emb_dim=16, dtype='string'),
    BucketFeat(name='expect_salary_min', boundaries=BOUNDARIES['salary'], share_emb='salary', emb_dim=16, dtype='int64'),
    BucketFeat(name='expect_salary_max', boundaries=BOUNDARIES['salary'], share_emb='salary', emb_dim=16, dtype='int64'),
    SparseFeat(name='fast_job_status', vocab_size=2, hash_size=None, emb_dim=16, dtype='int64', share_emb=None),
    # VarLenFeat(name='expect_job', vocab_size=500, hash_size=None, share_emb='category_id', weight_name=None, emb_dim=16, max_len=6, combiner='sum',
    #            dtype='string', sub_dtype='int64'),
    SparseFeat(name='city_id', vocab_size=len(DICT_CATEGORICAL['city_id']), share_emb=None, emb_dim=16, dtype='int64', hash_size=None),
    # VarLenFeat(name='past_experience', vocab_size=500, hash_size=None, share_emb='category_id', weight_name=None, emb_dim=16, max_len=6, combiner='sum',
    #            dtype='string', sub_dtype='int64'),
    SparseFeat(name='category_id', vocab_size=500, hash_size=None, share_emb=None, emb_dim=16, dtype='int64')
]

