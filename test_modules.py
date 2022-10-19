from re import X
from tkinter import Y
from modules.linear import Linear
from modules.relu import ReLU
from modules.layer_normalization import LayerNormalization
from transformer.encoder import Encoder
from transformer.encoder_block import EncoderBlock
from optimizer import Adam
from loss import MSE
from train import get_padding_mask
import cupy as cp
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, in_features, out_features, hidden_features, optimizer, layer_norm=True):
        self.layer_norm = layer_norm
        self.linear1 = Linear(in_features, hidden_features, optimizer)
        self.linear2 = Linear(hidden_features, hidden_features, optimizer)
        self.linear3 = Linear(hidden_features, out_features, optimizer)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        if self.layer_norm:
            self.layer_norm1 = LayerNormalization(optimizer, hidden_features)
            self.layer_norm2 = LayerNormalization(optimizer, hidden_features)


    def forward(self, x):
        x = self.linear1.forward(x)
        if self.layer_norm:
            x = self.layer_norm1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear2.forward(x)
        if self.layer_norm:
            x = self.layer_norm2.forward(x)
        x = self.relu2.forward(x)
        x = self.linear3.forward(x)
        return x

    def backward(self, grad):
        grad = self.linear3.backward(grad)
        grad = self.relu2.backward(grad)
        if self.layer_norm:
            grad = self.layer_norm2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        if self.layer_norm:
            grad = self.layer_norm1.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

    def update_weights(self):
        self.linear3.update_weights()
        if self.layer_norm:
            self.layer_norm1.update_weights()
        self.linear2.update_weights()
        if self.layer_norm:
            self.layer_norm2.update_weights()
        self.linear1.update_weights()

class EncoderTest():
    def __init__(self, optimizer, vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type):
        self.max_len = max_len
        self.d_model = d_model
        self.encoder = Encoder(optimizer, vocab_size, max_len, d_model, d_ff, num_attention_heads, block_num, dropout_rate, data_type)
        self.linear = Linear(d_model*max_len, max_len, optimizer)

    def forward(self, x, mask, training=True):
        x = self.encoder.forward(x, mask, training)
        self.bz, _, __ = x.shape
        x = x.reshape(self.bz, self.max_len*self.d_model)
        x = self.linear.forward(x)
        return x

    def backward(self, grad):
        grad = self.linear.backward(grad)
        grad = grad.reshape(self.bz, self.max_len, self.d_model)
        grad = self.encoder.backward(grad)
        return grad

    def update_weights(self):
        self.linear.update_weights()
        self.encoder.update_weights()

class EncoderBlockTest():
    def __init__(self, optimizer, in_features, out_features, d_model, d_ff, num_attention_heads, dropout_rate, data_type):
        self.encoder_block = EncoderBlock(d_model, d_ff, optimizer, num_attention_heads, dropout_rate, data_type)
        self.linear1 = Linear(in_features, d_model, optimizer)
        self.linear2 = Linear(d_model, out_features, optimizer)

    def forward(self, x, mask, training):
        x = self.linear1.forward(x)
        x = self.encoder_block.forward(x, mask, training)
        x = self.linear2.forward(x)
        return x

    def backward(self, grad):
        grad = self.linear2.backward(grad)
        grad = self.encoder_block.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

    def update_weights(self):
        self.linear2.update_weights()
        self.encoder_block.update_weights()
        self.linear1.update_weights()



BATCH_SIZE = 32
MAX_LEN = 1
MODEL_DIM = 128
FF_DIM = 512
ATTENTION_HEADS_NUM = 4
BLOCK_NUM = 3
DROPOUT_RATE = 0.1
DATA_TYPE = cp.float32
LR = 0.001
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
WARMUP_STEPS = 1000
MAX_STEPS = 2000
VOCAB_SIZE = 100

def test_mlp(layer_norm=True):  
    in_dim = 50
    adam = Adam(LR, BETA1, BETA2, EPS, WARMUP_STEPS, MODEL_DIM)
    model = MLP(in_dim, in_dim, 1024, adam, layer_norm=layer_norm)
    progress = trange(MAX_STEPS)
    for step in progress:
        x = cp.random.rand(32, 1, in_dim)
        y = x ** 2
        y_pred = model.forward(x)

        criterion = MSE(y_pred, y)
        loss = criterion.forward()
        progress.set_postfix({'loss': loss})
        grad = criterion.grad()
        # print(grad)
        model.backward(grad)
        model.update_weights()
        adam._step()
    x = cp.ones((1, 1, in_dim)) * 0.5
    y_pred = model.forward(x)
    y_true = x ** 2
    print(y_pred)
    print(y_true)

def test_transformer_encoder():
    adam = Adam(LR, BETA1, BETA2, EPS, WARMUP_STEPS, MODEL_DIM)
    model = EncoderTest(adam, VOCAB_SIZE, MAX_LEN, MODEL_DIM, FF_DIM, ATTENTION_HEADS_NUM, BLOCK_NUM, DROPOUT_RATE, DATA_TYPE)
    progress = trange(MAX_STEPS)
    loss_history = []
    for step in progress:
        a = cp.random.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
        # b = cp.random.randint(low=1, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
        y = (a + 1) ** 2
        # input = cp.concatenate([a, b], -1)
        mask = get_padding_mask(a, 0)
        y_pred = model.forward(a, mask=mask, training=True)
        criterion = MSE(y_pred, y)
        loss = criterion.forward()
        loss_history.append(loss)
        progress.set_postfix({'loss': loss})
        grad = criterion.grad()
        model.backward(grad)
        model.update_weights()
        adam._step()
    a = cp.ones((1, MAX_LEN), dtype=cp.int8)
    # b = cp.ones((1, MAX_LEN), dtype=cp.int8) * 2
    # input = cp.concatenate((a, b), -1)
    mask = get_padding_mask(a, 0)
    y_pred = model.forward(a, mask=mask, training=False)
    y = (a + 1) ** 2
    print(y_pred)
    print(y)

def test_encoder_block():
    in_dim = 5
    adam = Adam(LR, BETA1, BETA2, EPS, WARMUP_STEPS, MODEL_DIM)
    model = EncoderBlockTest(adam, 2*in_dim, in_dim, MODEL_DIM, FF_DIM, ATTENTION_HEADS_NUM, DROPOUT_RATE, DATA_TYPE)
    progress = trange(MAX_STEPS)
    loss_history = []
    for step in progress:
        a = cp.random.rand(BATCH_SIZE, MAX_LEN, in_dim)
        b = cp.random.rand(BATCH_SIZE, MAX_LEN, in_dim)
        y = (a + 1) ** 2 + b
        input = cp.concatenate([a, b], -1)
        # mask = get_padding_mask(input, 0)
        # print(input.shape)
        # print(mask.shape)
        mask = cp.ones((BATCH_SIZE, MAX_LEN), cp.int8)[:, cp.newaxis, :]
        y_pred = model.forward(input, mask=mask, training=True)
        criterion = MSE(y_pred, y)
        loss = criterion.forward()
        loss_history.append(loss)
        progress.set_postfix({'loss': loss})
        grad = criterion.grad()
        model.backward(grad)
        model.update_weights()
        adam._step()
    print('Mean MSE: ', cp.stack(loss_history).mean())
    a = cp.ones((1, MAX_LEN, in_dim)) * 0.5
    b = cp.ones((1, MAX_LEN, in_dim)) * 0.5
    input = cp.concatenate((a, b), -1)
    # mask = get_padding_mask(a, 0)
    mask = cp.ones((1, MAX_LEN), cp.int8)[:, cp.newaxis, :]

    y_pred = model.forward(input, mask=mask, training=False)
    y = (a + 1) ** 2 + b
    print(y_pred)
    print(y)

def plot_lr():
    base_lr = MODEL_DIM ** -0.5
    step = np.arange(MAX_STEPS)
    
    a = (step+1) ** -0.5
    b = (step+1) * WARMUP_STEPS ** -1.5
    lr = np.where(a<b, a, b) * base_lr

    plt.plot(step, lr)
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.title('MODEL_DIM = {}, WARMUP_STEPS = {}'.format(MODEL_DIM, WARMUP_STEPS))
    plt.savefig('plots/lr-{}-{}'.format(MODEL_DIM, WARMUP_STEPS))

if __name__ == "__main__":
    # test_mlp()
    test_encoder_block()
    # test_transformer_encoder()
    # plot_lr()