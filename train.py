import enum
import numpy as np
from dataloader import Dataloader
from optimizer import Adam
from transformer.transformer import Transformer
from tqdm import tqdm
# Hyperparameters
DATASET_NAME = 'news_commentary'
LANG_PAIR = 'en-zh'
BATCH_SIZE = 32
MAX_LEN = 5000
MODEL_DIM = 512
FF_DIM = 2048
ATTENTION_HEADS_NUM = 8
BLOCK_NUM = 6
DROPOUT_RATE = 0.1
DATA_TYPE = np.float32

LR = 0.001
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
WARMUP_STEPS = 4000
# 1 pass
def get_padding_mask(ids, padding_id):
    mask = (ids != padding_id).astype(int)[:, np.newaxis, :]
    return mask

def get_subsequent_mask(ids):
    seq_len = ids.shape[1]
    mask = np.tril(np.ones((seq_len, seq_len)), k=0).astype(int)
    return mask

def train_epoch(source_ids, target_ids, model, padding_id):
    progress = tqdm(enumerate(zip(source_ids, target_ids)))
    for batch_id, (source, target) in progress:
        source_mask = get_padding_mask(source, padding_id)
        target_mask = get_padding_mask(target, padding_id) & get_subsequent_mask(target)
        output = model.forward(source, target, source_mask, target_mask)
        print(output)


if __name__=='__main__':
    dataloader = Dataloader(DATASET_NAME, LANG_PAIR, BATCH_SIZE)
    padding_id = dataloader.base_vocab[dataloader.PAD_TOKEN]
    train_source_ids, train_target_ids = dataloader.train_source_ids, dataloader.train_target_ids
    tedt_source_ids, test_target_ids = dataloader.test_source_ids, dataloader.test_target_ids
    optimizer = Adam(lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, warmup_steps=WARMUP_STEPS)
    transformer = Transformer(
        optimizer=optimizer,
        source_vocab_size=len(dataloader.source_vocab),
        target_vocab_size=len(dataloader.target_vocab),
        max_len=MAX_LEN,
        d_model=MODEL_DIM,
        d_ff=FF_DIM,
        num_attention_heads=ATTENTION_HEADS_NUM,
        block_num=BLOCK_NUM,
        dropout_rate=DROPOUT_RATE,
        data_type=DATA_TYPE
    )
    train_epoch(train_source_ids, train_target_ids, transformer, padding_id)
