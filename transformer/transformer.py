import numpy as np
from .encoder import Encoder
from .decoder import Decoder

class Transformer():
    def __init__(self, optimizer, source_vocab_size, target_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.encoder = Encoder(optimizer, source_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.decoder = Decoder(optimizer, target_vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.data_type = data_type

    def forward(self, source_ids, target_ids, source_mask, target_mask, training=True):
        encoder_output = self.encoder.forward(source_ids, source_mask, training)
        decoder_output = self.decoder.forward(target_ids, encoder_output, target_mask, source_mask, training)
        return decoder_output

    def update_weights(self):
        self.decoder.update_weights()
        self.encoder.update_weights()