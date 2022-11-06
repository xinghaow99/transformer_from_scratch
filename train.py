import numpy as np
import cupy as cp
from dataloader import Dataloader
from optimizer import Adam
from transformer.transformer import Transformer
from loss import CrossEntropy
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import _release_memory
from bleu import compute_bleu
import wandb
# Hyperparameters
DATASET_NAME = 'news_commentary'
LANG_PAIR = 'en-zh'
BATCH_SIZE = 32
MAX_LEN = 200
MODEL_DIM = 512
FF_DIM = 2048
ATTENTION_HEADS_NUM = 8
BLOCK_NUM = 6
DROPOUT_RATE = 0.1
DATA_TYPE = np.float32
SEED = 12
LR = MODEL_DIM ** -0.5
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
WARMUP_STEPS = 6000
BLEU_NUM_SENTENCES = 100
NUM_EPOCH = 60
GRADIENT_ACCUMULATION_STEPS = 10

wandb.init(
    project="transformer_from_scratch",
    config = {
        'learning_rate': LR,
        'epochs': NUM_EPOCH,
        'warmup_steps': WARMUP_STEPS,
        'accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'batch_size': BATCH_SIZE,
        'max_len': MAX_LEN,
        'd_model': MODEL_DIM,
        'd_ff': FF_DIM,
        'num_attention_head': ATTENTION_HEADS_NUM,
        'num_blocks': BLOCK_NUM,
        'dropout_rate': DROPOUT_RATE,
        'beta1': BETA1,
        'beta2': BETA2,
        'eps': EPS
            })

mempool = cp.get_default_memory_pool()

def set_seed(seed):
    np.random.seed(seed)
    cp.random.seed(seed)

def limit_memory_pool(size):
    mempool = cp.get_default_memory_pool()
    with cp.cuda.Device(0):
        mempool.set_limit(size=size)
# 1 pass
def get_padding_mask(ids, padding_id):
    # [batch_size, seq_len, seq_len]
    batch_size, seq_len = ids.shape
    mask1d = (ids != padding_id).astype(int)
    mask_cnt = mask1d.sum(-1)
    mask = cp.zeros((batch_size, seq_len, seq_len), cp.int8)
    for i in range(batch_size):
        mask[i, :mask_cnt[i], :mask_cnt[i]] = 1
    # mask = (ids != padding_id).astype(int)[:, cp.newaxis, :]
    return mask

def get_subsequent_mask(ids):
    # [batch_size, seq_len, seq_len]
    seq_len = ids.shape[1]
    mask = cp.tril(cp.ones((seq_len, seq_len)), k=0).astype(int)
    return mask

def get_src_tgt_mask(src_ids, tgt_ids, padding_id):
    batch_size, src_seq_len = src_ids.shape
    _, tgt_seq_len = tgt_ids.shape
    src_mask_cnt = (src_ids != padding_id).astype(int).sum(-1)
    tgt_mask_cnt = (tgt_ids != padding_id).astype(int).sum(-1)
    mask = cp.zeros((batch_size, tgt_seq_len, src_seq_len), cp.int8)
    for i in range(batch_size):
        mask[i, :tgt_mask_cnt[i], :src_mask_cnt[i]] = 1
    return mask

def train_epoch(optimizer, source_ids, target_ids, model, padding_id, criterion, epoch):
    print('Training Epoch', epoch, 'Learning Rate =', optimizer.lr)
    progress = tqdm(enumerate(zip(source_ids, target_ids)), total=len(source_ids))
    loss_history = []
    perplexity_history = []
    for batch_id, (source, target) in progress:
        _release_memory()
        source = cp.array(source)
        target = cp.array(target)
        target_in = target[:, :-1]
        # print('source.shape: {}, target_in.shape: {}'.format(source.shape, target_in.shape))
        source_mask = get_padding_mask(source, padding_id)
        target_mask = get_padding_mask(target_in, padding_id) & get_subsequent_mask(target_in)
        src_tgt_mask = get_src_tgt_mask(source, target_in, padding_id)
        # print('source_mask.shape: {}, target_mask.shape: {}, src_tgt_mask.shape: {}'.format(source_mask.shape, target_mask.shape, src_tgt_mask.shape))
        output = model.forward(source, target_in, source_mask, target_mask, src_tgt_mask, training=True)
        output_shape = output.shape
        output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
        loss = criterion.forward(output, target[:, 1:].flatten()).mean().get()
        del source, target_in, source_mask, target_mask, src_tgt_mask
        _release_memory()
        perplexity = np.exp(loss)
        wandb.log({
            'train_loss': loss,
            'train_perplexity': perplexity
        })
        progress.set_postfix({'train_loss': loss, 'train_perplexity': perplexity})
        loss_history.append(loss)
        perplexity_history.append(perplexity)
        grad = criterion.grad(output, target[:, 1:].flatten()).reshape(output_shape)
        model.backward(grad)
        _release_memory()
        if batch_id % GRADIENT_ACCUMULATION_STEPS == 0:
            model.update_weights()
        optimizer._step()
    _release_memory()
    return loss_history

def train(optimizer, train_source_ids, train_target_ids, test_source_ids, test_target_ids, source_vocab, target_vocab, raw_test_set, model, padding_id, criterion, epochs):
    train_loss_history = []
    eval_loss_history = []
    bleu_history = []
    for epoch in range(epochs):
        train_loss_history.extend(train_epoch(optimizer, train_source_ids, train_target_ids, model, padding_id, criterion, epoch))
        eval_loss_epoch, bleu_epoch = eval_epoch(test_source_ids, test_target_ids, source_vocab, target_vocab, raw_test_set, model, padding_id, criterion, epoch)
        # eval_loss_history.extend(eval_loss_epoch)
        eval_loss_history.append(sum(eval_loss_epoch) / len(eval_loss_epoch))
        bleu_history.append(bleu_epoch)
        wandb.log({
            'eval_loss': sum(eval_loss_epoch) / len(eval_loss_epoch),
            'bleu': bleu_epoch
        })
    # print(eval_loss_epoch)
    # plot_loss(train_loss_history, 'Step', 'Loss', 'train_loss')  
    # plot_loss(eval_loss_history, 'Epoch', 'Loss', 'eval_loss')
    # plot_bleu(bleu_history, 'bleu')
    plot_result(train_loss_history, eval_loss_history, bleu_history)

def plot_result(train_loss, eval_loss, bleu):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Result')
    ax1.plot(train_loss)
    ax1.set_ylabel('train_loss')
    ax1.set_xlabel('step')
    ax2.plot(eval_loss)
    ax2.set_ylabel('eval_loss')
    ax2.set_xlabel('epoch')
    ax3.plot(bleu)
    ax3.set_ylabel('bleu')
    ax3.set_xlabel('epoch')
    fig.savefig('plots/result.png')

def plot_loss(loss_history, x, y, filename):
    plt.figure()
    plt.plot(loss_history)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.savefig('{}.png'.format(filename))

def plot_bleu(bleu_history, filename):
    plt.figure()
    plt.plot(bleu_history)
    plt.ylabel('BLEU')
    plt.xlabel('Epoch')
    plt.savefig('{}.png'.format(filename))

def decode(ids, vocab, special_token_ids=range(3), sep=""):
    ids = [id for id in ids if id not in special_token_ids]
    tokens = [vocab[i] for i in ids]
    sentence = sep.join(tokens)
    # print(sentence)
    return sentence

def predict(model: Transformer, source_ids, reversed_target_vocab, padding_id, bos_id, eos_id, max_length = 100):
    target_ids = [bos_id]
    source_ids = cp.asarray(source_ids).reshape(1, -1)
    source_mask = get_padding_mask(source_ids, padding_id)
    encoded = model.encoder.forward(source_ids, source_mask, False)

    for i in range(max_length-1):
        target_input = cp.asarray(target_ids).reshape(1, -1)
        target_mask = get_padding_mask(target_input, padding_id) & get_subsequent_mask(target_input)
        src_tgt_mask = get_src_tgt_mask(source_ids, target_input, padding_id)
        outputs = model.decoder.forward(target_input, encoded, target_mask, src_tgt_mask, False)
        pred_id = outputs.argmax(axis=-1)[:, -1].item()
        target_ids.append(pred_id)
        if pred_id == eos_id or len(target_ids) >= max_length:
            break
    sentence = decode(target_ids, reversed_target_vocab)
    return sentence
        

def eval_epoch(source_ids, target_ids, source_vocab, target_vocab, raw_test_set, model, padding_id, criterion, epoch):
    print('Evaluating Epoch', epoch)
    loss_history = []
    perplexity_history = []
    progress = tqdm(enumerate(zip(source_ids, target_ids)), total=len(source_ids))
    for batch_id, (source, target) in progress:
        _release_memory()
        source = cp.array(source)
        target = cp.array(target)
        target_in = target[:, :-1]
        source_mask = get_padding_mask(source, padding_id)
        target_mask = get_padding_mask(target_in, padding_id) & get_subsequent_mask(target_in)
        src_tgt_mask = get_src_tgt_mask(source, target_in, padding_id)
        output = model.forward(source, target_in, source_mask, target_mask, src_tgt_mask, training=False)
        output_shape = output.shape
        output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
        loss = criterion.forward(output, target[:, 1:].flatten()).mean().get()
        del source, target_in, source_mask, target_mask, src_tgt_mask
        perplexity = np.exp(loss)
        progress.set_postfix({'eval_loss': loss, 'eval_perplexity': perplexity})
        loss_history.append(loss)
        perplexity_history.append(perplexity)
    bleu_score = compute_test_set_bleu(source_ids, target_vocab, raw_test_set, model, BLEU_NUM_SENTENCES)[0]
    print('Epoch {} Bleu Score: '.format(epoch), bleu_score)
    sample_sentences_and_translate(model, source_ids, target_ids, source_vocab, target_vocab, raw_test_set, 8, special_token_ids=range(3))
    return loss_history, bleu_score
    

def compute_test_set_bleu(test_source_ids, target_vocab, raw_test_dataset, model, num_sentences=100):
    reversed_target_vocab = dict((v, k) for k, v in target_vocab.items())
    progress = tqdm(enumerate(test_source_ids), total=len(test_source_ids))
    predictions = []
    references = []
    step = 0
    for batch_id, ids in progress:
        for _ids in ids:
            predictions.append(predict(model, _ids, reversed_target_vocab, 0, 1, 2, MAX_LEN))
            # print(''.join(raw_test_dataset[step]['zh'][1:-1]))
            # references.append([''.join(raw_test_dataset[step]['zh'][1:-1])])
            references.append([raw_test_dataset[step]['zh']])
            step += 1
            if step >= num_sentences:
                break
        if step >= num_sentences:
            break
    bleu_score = compute_bleu(references, predictions)
    # print(references[:3])
    # print(predictions[:3])
    # print(bleu_score)
    return bleu_score



def sample_sentences_and_translate(model, test_source_ids, test_target_ids, source_vocab, target_vocab, raw_test_dataset, num, special_token_ids):
    test_source_ids = test_source_ids[0].tolist()
    test_target_ids = test_target_ids[0].tolist()
    reversed_source_vocab = dict((v, k) for k, v in source_vocab.items())
    reversed_target_vocab = dict((v, k) for k, v in target_vocab.items())

    random_indices = np.random.randint(0, len(test_source_ids), num)
    sample_source_ids = [test_source_ids[i] for i in random_indices]
    # sample_source_sentences = [decode(sample_source_ids[i], reversed_source_vocab, special_token_ids, sep=" ") for i in range(num)]
    # sample_target_ids = [test_target_ids[i] for i in random_indices]
    # sample_target_sentences = [decode(sample_target_ids[i], reversed_target_vocab, special_token_ids) for i in range(num)]
    sample_source_sentences = []
    sample_target_sentences = []
    for id in random_indices:
        sample_source_sentences.append(raw_test_dataset[id]['en'])
        sample_target_sentences.append(raw_test_dataset[id]['zh'])
    predict_sentences = [predict(model, sample_source_ids[i], reversed_target_vocab, 0, 1, 2, MAX_LEN) for i in range(num)]
    for id, (src, tgt, pred) in enumerate(zip(sample_source_sentences, sample_target_sentences, predict_sentences)):
        print('====Sentence {}===='.format(id))
        print('Source sentence: ', src)
        print('Target sentence: ', tgt)
        print('Predicted sentence: ', pred)
        print('==================')




if __name__=='__main__':
    set_seed(SEED)
    # limit_memory_pool(1*1024**3) # 1GB
    dataloader = Dataloader(DATASET_NAME, LANG_PAIR, BATCH_SIZE, MAX_LEN, SEED)
    padding_id = dataloader.base_vocab[dataloader.PAD_TOKEN]
    train_source_ids, train_target_ids = dataloader.train_source_ids, dataloader.train_target_ids
    test_source_ids, test_target_ids = dataloader.test_source_ids, dataloader.test_target_ids
    optimizer = Adam(lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, warmup_steps=WARMUP_STEPS, d_model=MODEL_DIM)
    optimizer.set_lr()
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
    target_vocab_size = len(dataloader.target_vocab)
    criterion = CrossEntropy(padding_id, target_vocab_size)
    train(optimizer, train_source_ids, train_target_ids, test_source_ids, test_target_ids, dataloader.source_vocab, dataloader.target_vocab, dataloader.raw_test_dataset, transformer, padding_id, criterion, NUM_EPOCH)
    print('Sentences from training set')
    sample_sentences_and_translate(transformer, train_source_ids, train_target_ids, dataloader.source_vocab, dataloader.target_vocab, dataloader.raw_train_dataset, 3, special_token_ids=range(3))
    print('Sentences from testing  set')
    sample_sentences_and_translate(transformer, test_source_ids, test_target_ids, dataloader.source_vocab, dataloader.target_vocab, dataloader.raw_test_dataset, 3, special_token_ids=range(3))
    wandb.finish()
