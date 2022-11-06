from datasets import load_dataset
import numpy as np
import re
import copy
import matplotlib.pyplot as plt
class Dataloader():
    def __init__(self, dataset_name='news_commentary', lang_pair='en-zh', batch_size=32, max_len=256, seed=42):
        self.dataset_name = dataset_name
        self.lang_pair = lang_pair
        self.batch_size = batch_size
        self.source_lang, self.target_lang = self.lang_pair.split('-')
        if self.dataset_name != 'news_commentary':
            raise NotImplementedError
        self.dataset = load_dataset(self.dataset_name, self.lang_pair, cache_dir='data')['train'].train_test_split(test_size=0.2, seed=seed)
        self.raw_train_dataset = self.dataset['train']['translation']
        self.raw_test_dataset = self.dataset['test']['translation']
        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        self.OOV_TOKEN = '<oov>'
        self.split_word_punc_str = r"[\w']+|[.,!?;]"
        self.base_vocab = {self.PAD_TOKEN: 0, self.BOS_TOKEN: 1, self.EOS_TOKEN: 2, self.OOV_TOKEN: 3}
        self.source_vocab = self.base_vocab.copy()
        self.target_vocab = self.base_vocab.copy()
        self.build_vocab()
        self.train_src_length_cnt = []
        self.test_src_length_cnt = []
        self.train_tgt_length_cnt = []
        self.test_tgt_length_cnt = []
        self.train_dataset = self.batch_examples_and_add_special_tokens(self.raw_train_dataset, max_len, True)
        self.test_dataset = self.batch_examples_and_add_special_tokens(self.raw_test_dataset, max_len, False)
        self.train_source_ids, self.train_target_ids = self.convert_to_ids(self.train_dataset)
        self.test_source_ids, self.test_target_ids = self.convert_to_ids(self.test_dataset)
        self.print_vocab_size()
        self.print_sentence_length()

    def print_vocab_size(self):
        print('Source Vocab Size: ', len(self.source_vocab))
        print('Target Vocab Size: ', len(self.target_vocab))

    def print_sentence_length(self):
        print('Total Sentences in Training Set: ', len(self.train_src_length_cnt))
        print('Total Sentences in Testing  Set: ', len(self.test_src_length_cnt))
        print('Mean Length of Source Sentences in Training Set: ', sum(self.train_src_length_cnt)/len(self.train_src_length_cnt))
        print('Max  Length of Source Sentences in Training Set: ', max(self.train_src_length_cnt))
        print('Mean Length of Target Sentences in Training Set: ', sum(self.train_tgt_length_cnt)/len(self.train_tgt_length_cnt))
        print('Max  Length of Target Sentences in Training Set: ', max(self.train_tgt_length_cnt))
        print('Mean Length of Source Sentences in Testing  Set: ', sum(self.test_src_length_cnt)/len(self.test_src_length_cnt))
        print('Max  Length of Source Sentences in Testing  Set: ', max(self.test_src_length_cnt))
        print('Mean Length of Target Sentences in Testing  Set: ', sum(self.test_tgt_length_cnt)/len(self.test_tgt_length_cnt))
        print('Max  Length of Target Sentences in Testing  Set: ', max(self.test_tgt_length_cnt))
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Sequence Length Frequency')
        ax1.hist(self.train_src_length_cnt, bins=100, color='g', label='src')
        ax1.hist(self.train_tgt_length_cnt, bins=100, color='b', label='tgt')
        ax1.set_xlim(0, 500)
        ax1.set_ylabel('Train Freq')
        ax1.legend()
        ax2.hist(self.test_src_length_cnt, bins=100, color='r', label='src')
        ax2.hist(self.test_tgt_length_cnt, bins=100, color='y', label='tgt')
        ax2.set_xlim(0, 500)
        ax2.set_ylabel('Test Freq')
        ax2.legend()
        fig.savefig('plots/seq_len.png')

    def build_vocab(self, min_token_freq=1):
        source_token_freqs = {}
        target_token_freqs = {}
        for data in self.raw_train_dataset:
            for token in re.findall(self.split_word_punc_str, data[self.source_lang]):
                if token not in source_token_freqs:
                    source_token_freqs[token] = 1
                else:
                    source_token_freqs[token] += 1
            for token in list(data[self.target_lang]):
                if token not in target_token_freqs:
                    target_token_freqs[token] = 1
                else:
                    target_token_freqs[token] += 1
        for data in self.raw_train_dataset:
            for token in re.findall(self.split_word_punc_str, data[self.source_lang]):
                if token not in self.source_vocab and source_token_freqs[token] > min_token_freq:
                    self.source_vocab[token] = len(self.source_vocab)
            for token in list(data[self.target_lang]):
                if token not in self.target_vocab and target_token_freqs[token] > min_token_freq:
                    self.target_vocab[token] = len(self.target_vocab)

    def batch_examples_and_add_special_tokens(self, raw_dataset, max_len, is_train_set):
        dataset = copy.deepcopy(raw_dataset)
        for example in dataset:
            example[self.source_lang] = [self.BOS_TOKEN] + re.findall(self.split_word_punc_str, example[self.source_lang]) + [self.EOS_TOKEN]
            example[self.target_lang] = [self.BOS_TOKEN] + list(example[self.target_lang]) + [self.EOS_TOKEN]
            if is_train_set:
                self.train_src_length_cnt.append(len(example[self.source_lang])-2)
                self.train_tgt_length_cnt.append(len(example[self.target_lang])-2)
            else:
                self.test_src_length_cnt.append(len(example[self.source_lang])-2)
                self.test_tgt_length_cnt.append(len(example[self.target_lang])-2)
            if len(example[self.source_lang]) > max_len:
                example[self.source_lang] = example[self.source_lang][:max_len-1] + [self.EOS_TOKEN]
            if len(example[self.target_lang]) > max_len:
                example[self.target_lang] = example[self.target_lang][:max_len-1] + [self.EOS_TOKEN]
        batched_data = np.array_split(dataset, np.arange(self.batch_size, len(dataset), self.batch_size))
        for batch in batched_data:
            batch_source_max_len = 0
            batch_target_max_len = 0
            for example in batch:
                batch_source_max_len = max(batch_source_max_len, len(example[self.source_lang]))
                batch_target_max_len = max(batch_target_max_len, len(example[self.target_lang]))
            # max_len = max(batch_source_max_len, batch_target_max_len)
            for example in batch:
                example[self.source_lang] += [self.PAD_TOKEN] * (batch_source_max_len - len(example[self.source_lang]))
                example[self.target_lang] += [self.PAD_TOKEN] * (batch_target_max_len - len(example[self.target_lang]))
        return batched_data

    def convert_to_ids(self, dataset):
        source_batched_ids = []
        target_batched_ids = []
        for batch in dataset:
            source_ids_list = []
            target_ids_list = []
            for example in batch:
                source_tokens = example[self.source_lang]
                target_tokens = example[self.target_lang]
                source_ids = [self.source_vocab[token] if token in self.source_vocab else self.source_vocab[self.OOV_TOKEN] for token in source_tokens]
                target_ids = [self.target_vocab[token] if token in self.target_vocab else self.target_vocab[self.OOV_TOKEN] for token in target_tokens]
                source_ids_list.append(source_ids)
                target_ids_list.append(target_ids)
            source_batched_ids.append(np.asarray(source_ids_list))
            target_batched_ids.append(np.asarray(target_ids_list))
        return source_batched_ids, target_batched_ids
