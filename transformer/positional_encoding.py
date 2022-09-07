import numpy as np

class PositionalEncoding():
    def __init__(self, max_len, d_model, data_type=np.float32):
        self.pe = np.zeros((max_len, d_model), dtype=data_type)
        ev_cln = np.arange(0, d_model, 2)
        diff = 1.0 / (10000) ** (ev_cln / d_model)
        pos = np.arange(0, max_len)[:, np.newaxis]
        self.pe[:, 0::2] = np.sin(pos * diff)
        self.pe[:, 1::2] = np.cos(pos * diff)
        self.pe = self.pe[:, np.newaxis, :]

    def forward(self, x):
        batch_size = x.shape[0]
        return x + self.pe[:batch_size, :, :]

    def backward(self, grad):
        return grad
