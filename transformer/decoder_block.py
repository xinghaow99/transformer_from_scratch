import numpy as np
from modules.dropout import Dropout
from modules.layer_normalization import LayerNormalization
from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward import PositionWiseFeedForward
class DecoderBlock():
    def __init__(self, optimizer, d_model=512, d_ff=2048, num_attention_heads=8, dropout_rate=0.1, mask1=None, mask2=None, data_type=np.float32):
        self.optimizer = optimizer
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.mask1 = mask1
        self.mask2 = mask2
        self.data_type = data_type
        self.dropout = Dropout(self.dropout_rate)
        self.layernorm1 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.layernorm2 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.layernorm3 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.multi_head_att1 = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.mask1, self.data_type)
        self.multi_head_att2 = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.mask2, self.data_type)
        self.ffn = PositionWiseFeedForward(self.optimizer, self.d_model, self.d_ff, self.dropout_rate, self.data_type)