import numpy as np
from modules.linear import Linear
from modules.dropout import Dropout
from modules.softmax import Softmax
class MultiHeadAttention():
    def __init__(self, d_model=512, num_attention_heads=8, dropout_date=0.1, data_type=np.float32):
        self.d_model = 512
        self.num_attention_heads = num_attention_heads
        self.data_type = data_type
        self.d_q = d_model // self.num_attention_heads
        self.d_k = self.d_q
        self.d_v = self.d_q
        self.scale_factor = np.sqrt(self.d_k)
        self.W_q = Linear(in_features=self.d_model, out_features=self.d_q, use_bias=False, data_type=np.float32)
        self.W_k = Linear(in_features=self.d_model, out_features=self.d_k, use_bias=False, data_type=np.float32)
        self.W_v = Linear(in_features=self.d_model, out_features=self.d_v, use_bias=False, data_type=np.float32)
        self.W_o = Linear(in_features=self.d_q*self.num_attention_heads*self.d_v, out_features=self.d_model, use_bias=True, data_type=np.float32)
        self.dropout = Dropout(dropout_date)
        self.softmax = Softmax()
    
    def attention(self, q_in, k_in, v_in, training=True):
        x = q_in @ k_in.T / np.sqrt(self.d_k)
        x = self.softmax.forward(x)
        x = self.dropout.forward(x, training)
        x = x @ v_in
        return x

    def forward(self, q, k, v, training=True):
        q = self.W_q.forward(q)
        k = self.W_k.forward(k)
        v = self.W_v.forward(v)