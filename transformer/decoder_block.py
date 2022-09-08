import numpy as np
from modules.dropout import Dropout
from modules.layer_normalization import LayerNormalization
from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward import PositionWiseFeedForward
class DecoderBlock():
    def __init__(self, optimizer, d_model=512, d_ff=2048, num_attention_heads=8, dropout_rate=0.1, data_type=np.float32):
        self.optimizer = optimizer
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.data_type = data_type
        self.dropout = Dropout(self.dropout_rate)
        self.layernorm1 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.layernorm2 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.layernorm3 = LayerNormalization(self.optimizer, self.d_model, 1e-5, self.data_type)
        self.multi_head_att1 = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.data_type)
        self.multi_head_att2 = MultiHeadAttention(self.optimizer, self.d_model, self.num_attention_heads, self.dropout_rate, self.data_type)
        self.ffn = PositionWiseFeedForward(self.optimizer, self.d_model, self.d_ff, self.dropout_rate, self.data_type)

    def forward(self, target, source, target_mask, source_mask, training=True):
        q, k, v = target, target, target
        attention_output1 = self.multi_head_att1.forward(q, k, v, target_mask, training)
        attention_output1 = self.dropout.forward(attention_output1, training)
        target_ = self.layernorm1.forward(target + attention_output1)
        q, k, v = target_, source, source
        attention_output2 = self.multi_head_att2.forward(q, k, v, source_mask, training)
        attention_output2 = self.dropout.forward(attention_output2, training)
        x = self.layernorm2.forward(target_ + attention_output2)
        x_ = self.ffn.forward(x, training)
        x_ = self.dropout.forward(x_, training)
        x = self.layernorm3.forward(x + x_)
        return x

    def backward(self, grad):
        grad_res1 = self.layernorm3.backward(grad)
        grad = self.dropout.backward(grad_res1)
        grad = self.ffn.backward(grad)
        grad = grad_res1 + grad
        grad_res2 = self.layernorm2.backward(grad)
        grad = self.dropout.backward(grad_res2)
        grad, grad_src_k , grad_src_v = self.multi_head_att2.backward(grad)
        grad = grad_res2 + grad
        grad_res3 = self.layernorm1.backward(grad)
        grad = self.dropout.backward(grad_res3)
        grad_tgt_q, grad_tgt_k, grad_tgt_v = self.multi_head_att1.backward(grad)
        return grad_res3 + grad_tgt_q + grad_tgt_k + grad_tgt_v, grad_src_k + grad_src_v

    def update_weights(self):
        self.layernorm3.update_weights()
        self.ffn.update_weights()
        self.layernorm2.update_weights()
        self.multi_head_att2.update_weights()
        self.layernorm1.update_weights()
        self.multi_head_att1.update_weights()

