import torch
from torch import nn
from torch.optim import Adam
from tqdm import trange
LR = 0.001
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
MAX_STEPS = 2000
MODEL_DIM = 256
FF_DIM = 512
ATTENTION_HEADS_NUM = 4
BATCH_SIZE = 32
MAX_LEN = 1
def test_mlp():
    in_dim = 50
    hidden_dim = 1024
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),

        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),

        nn.ReLU(),
        nn.Linear(hidden_dim, in_dim)
    ).cuda()
    optimizer = Adam(model.parameters(), LR, (BETA1, BETA2), EPS)
    progress = trange(MAX_STEPS)
    for step in progress:
        x = torch.rand(32, 1, in_dim).cuda()
        y = x ** 2
        y_pred = model(x)
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        progress.set_postfix({'loss': loss})
        loss.backward()
        optimizer.step()
    x = torch.ones(1, 1, in_dim).cuda() * 0.5
    y = x ** 2
    y_pred = model(x)
    print(y)
    print(y_pred)

def test_encoder_block():
    in_dim = 5
    model = nn.Sequential(
        nn.Linear(2*in_dim, MODEL_DIM),
        nn.TransformerEncoderLayer(MODEL_DIM, ATTENTION_HEADS_NUM, FF_DIM,),
        nn.Linear(MODEL_DIM, in_dim)
    ).cuda()
    optimizer = Adam(model.parameters(), LR, (BETA1, BETA2), EPS)
    progress = trange(MAX_STEPS)
    loss_history = []
    for step in progress:
        a = torch.rand(BATCH_SIZE, MAX_LEN, in_dim).cuda()
        b = torch.rand(BATCH_SIZE, MAX_LEN, in_dim).cuda()
        y = (a+1)**2+b
        input = torch.cat((a,b), -1)
        y_pred = model(input)
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss_history.append(loss.item())
        progress.set_postfix({'loss': loss})
        loss.backward()
        optimizer.step()
    print('Mean MSE: ', sum(loss_history)/len(loss_history))
    model.eval()
    a = (torch.ones(1, MAX_LEN, in_dim) * 0.5).cuda()
    b = (torch.ones(1, MAX_LEN, in_dim) * 0.5).cuda()
    y = (a+1)**2+b
    input = torch.cat((a,b), -1)
    y_pred = model(input)
    print(y)
    print(y_pred)

if __name__ == "__main__":
    # test_mlp()
    test_encoder_block()