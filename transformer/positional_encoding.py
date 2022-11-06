from turtle import pen
import cupy as cp
class PositionalEncoding():
    def __init__(self, max_len, d_model, data_type=cp.float32):
        self.pe = cp.zeros((max_len, d_model), dtype=data_type)
        ev_cln = cp.arange(0, d_model, 2)
        diff = 1.0 / (10000) ** (ev_cln / d_model)
        pos = cp.arange(0, max_len)[:, cp.newaxis]
        self.pe[:, 0::2] = cp.sin(pos * diff)
        self.pe[:, 1::2] = cp.cos(pos * diff)
        self.pe = self.pe[cp.newaxis, :, :]
        del ev_cln, diff, pos

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

    def backward(self, grad):
        return grad

# max_len=200
# d_model=512
# pe = cp.zeros((max_len, d_model))  # (max_len, d_model)
# position = cp.arange(0, max_len)[:, cp.newaxis]# (max_len, 1)
# div_term = cp.exp(cp.arange(0, d_model, 2) * (-cp.log(10000.0) / d_model))  # (d_model,)

# pe[:, 0::2] = cp.sin(position * div_term)
# pe[:, 1::2] = cp.cos(position * div_term)

# pe = pe[:, cp.newaxis, :]
# # print(pe)

# mype = PositionalEncoding(max_len, d_model)
# # print(mype.pe)

# x = cp.random.randn(32, max_len, d_model)
# x_pos1 = x + pe[:x.shape[0], :]
# x_pos2 = mype.forward(x)
# print(x_pos1[-1])
# print(x_pos2[-1])