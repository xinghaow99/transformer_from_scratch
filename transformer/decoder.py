import imp
import numpy as np
from modules.embedding import Embedding
from modules.dropout import Dropout
from modules.linear import Linear
from modules.softmax import Softmax
from transformer.positional_encoding import PositionalEncoding
from transformer.decoder_block import DecoderBlock
class Decoder():
    def __init__(self, optimizer, vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, optimizer, data_type)
        self.dropout = Dropout(dropout_rate, data_type)
        self.positional_enconding = PositionalEncoding(max_len, d_model, data_type)
        self.decoder_layers = [DecoderBlock(optimizer, d_model, d_ff, num_attention_heads, dropout_rate, data_type) for _ in range(block_num)]
        self.fc = Linear(d_model, vocab_size, optimizer, True, data_type)
        self.softmax = Softmax()

    def forward(self, target, source, target_mask, source_mask, training):
        target = self.embedding.forward(target) * np.sqrt(self.d_model)
        target = self.positional_enconding.forward(target)
        target = self.dropout.forward(target, training)
        for decoder_layer in self.decoder_layers:
            target =  decoder_layer.forward(target, source, target_mask, source_mask, training)
        target = self.fc.forward(target)
        target = self.softmax.forward(target)
        return target

    def backward(self, grad):
        grad = self.softmax.backward(grad)
        grad = self.fc.backward(grad)
        grad_source_sum = 0
        for decoder_layer in reversed(self.decoder_layers):
            grad, grad_source = decoder_layer.backward(grad)
            grad_source_sum += grad_source
        grad = self.dropout.backward(grad)
        grad = self.positional_enconding.backward(grad) * np.sqrt(self.d_model)
        grad = self.embedding.backward(grad)
        return grad, grad_source_sum

    def update_weights(self):
        self.fc.update_weights()
        for decoder_layer in reversed(self.decoder_layers):
            decoder_layer.update_weights()
        self.embedding.update_weights()

