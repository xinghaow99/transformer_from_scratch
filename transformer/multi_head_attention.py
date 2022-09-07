import numpy as np
from modules.linear import Linear
from modules.dropout import Dropout
from modules.softmax import Softmax
class MultiHeadAttention():
    def __init__(self, optimizer, d_model=512, num_attention_heads=8, dropout_rate=0.1, data_type=np.float32):
        self.optimizer = optimizer
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.data_type = data_type
        self.d_q = d_model // self.num_attention_heads
        self.d_k = self.d_q
        self.d_v = self.d_q
        self.scale_factor = np.sqrt(self.d_k)
        self.W_q = Linear(in_features=self.d_model, out_features=self.d_q*self.num_attention_heads, optimizer=self.optimizer, use_bias=False, data_type=np.float32)
        self.W_k = Linear(in_features=self.d_model, out_features=self.d_k*self.num_attention_heads, optimizer=self.optimizer, use_bias=False, data_type=np.float32)
        self.W_v = Linear(in_features=self.d_model, out_features=self.d_v*self.num_attention_heads, optimizer=self.optimizer, use_bias=False, data_type=np.float32)
        self.W_o = Linear(in_features=self.d_model, out_features=self.d_model, optimizer=self.optimizer, use_bias=True, data_type=np.float32)
        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax()
    
    def attention_forward(self, q, k, v, mask, training=True):
        attention_score = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        if mask is not None:
            mask = np.asarray(mask)
            mask = mask[:, np.newaxis, ...]
        attention_score = np.where(mask == 0, float('-inf'), attention_score)
        softmax_output = self.softmax.forward(attention_score)
        self.dropout_output = self.dropout.forward(softmax_output, training)
        attention_output = self.dropout_output @ v
        return attention_output

    def attention_backward(self, mask, grad):
        self.grad_v = self.dropoutout_output.transpose(0, 1, 3, 2) @ grad
        grad = grad @ self.v.tranpose(0, 1, 3, 2)
        grad = self.dropout.backward(grad)
        grad = self.softmax.backward(grad)
        if mask is not None:
            grad = np.where(mask == 0, 0, grad)
        self.grad_q = grad @ self.k / np.sqrt(self.d_k)
        self.grad_k = (self.q.transpose(0, 1, 3, 2) @ grad / np.sqrt(self.d_k)).transpose(0, 1, 3, 2)
        return grad

    def forward(self, q, k, v, mask=None, training=True):
        self.mask = mask
        self.batch_size = q.shape[0]
        # [batch_size, seq_len, d_k*num_attention_heads]
        q = self.W_q.forward(q)
        k = self.W_k.forward(k)
        v = self.W_v.forward(v)
        # [batch_size, num_attention_heads, seq_len, d_k]
        self.q = q.reshape(self.batch_size, -1, self.num_attention_heads, self.d_q).transpose(0, 2, 1, 3)
        self.k = k.reshape(self.batch_size, -1, self.num_attention_heads, self.d_k).transpose(0, 2, 1, 3)
        self.v = v.reshape(self.batch_size, -1, self.num_attention_heads, self.d_v).transpose(0, 2, 1, 3)
        attention_output = self.attention_forward(self.q, self.k, self.v, self.mask, training)
        # concatenating
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.num_attention_heads*self.d_k)
        output = self.W_o.forward(attention_output)
        return output

    def backward(self, grad):
        grad = self.W_o.backward(grad)
        grad = grad.reshape(self.batch_size, -1, self.num_attention_heads, self.d_k).transpose(0, 2, 1, 3)
        grad = self.attention_backward(self.mask, grad)
        self.grad_q = self.grad_q.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.num_attention_heads*self.d_q)
        self.grad_k = self.grad_k.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.num_attention_heads*self.d_k)
        self.grad_v = self.grad_v.transpose(0, 2, 1, 3).reshape(self.batch_size, -1, self.num_attention_heads*self.d_v)
        self.grad_q = self.W_q.backward(self.grad_q)
        self.grad_k = self.W_k.backward(self.grad_k)
        self.grad_v = self.W_v.backward(self.grad_v)
        return self.grad_q, self.grad_k, self.grad_v

    def update_weights(self):
        self.W_o.update_weights()
        self.W_v.update_weights()
        self.W_k.update_weights()
        self.W_q.update_weights()