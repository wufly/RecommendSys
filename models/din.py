from utils import *


class LocalAttentionUnit(Layer):

    def __init__(self, hidden_units=(64, 32), activation='sigmoid', l2_reg=0, dropout=0, use_bn=False, seed=1024, **kwargs):
        super(LocalAttentionUnit, self).__init__(**kwargs)

