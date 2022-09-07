from datasets import load_dataset
import numpy as np
import re

class Dataloader():
    def __init__(self, dataset_name='news_commentary', lang_pair='en-zh', batch_size=32):
        self.dataset_name = dataset_name
        self.lang_pair = lang_pair
        self.batch_size = batch_size
        self.source_lang, self.target_lang = self.lang_pair.split('-')
        if self.dataset_name != 'news_commentary':
            raise NotImplementedError
        self.dataset = load_dataset(self.dataset_name, self.lang_pair)['train'].train_test_split(0.2)
        self.train_dataset = self.dataset['train']['translation']
        self.test_dataset = self.dataset['test']['translation']
        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        self.OOV_TOKEN = '<oov>'
        self.split_word_punc_str = r"[\w']+|[.,!?;]"
        self.base_vocab = {self.PAD_TOKEN: 0, self.BOS_TOKEN: 1, self.EOS_TOKEN: 2, self.OOV_TOKEN: 3}
        self.source_vocab = self.base_vocab.copy()
        self.target_vocab = self.base_vocab.copy()
        self.build_vocab()
        self.batch_examples_and_add_special_tokens(self.train_dataset)
        self.batch_examples_and_add_special_tokens(self.test_dataset)
        self.train_source_ids, self.train_target_ids = self.convert_to_ids(self.train_dataset)
        self.test_source_ids, self.test_target_ids = self.convert_to_ids(self.test_dataset)

    def build_vocab(self, min_token_freq=1):
        source_token_freqs = {}
        target_token_freqs = {}
        for data in self.train_dataset:
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
        for data in self.train_dataset:
            for token in re.findall(self.split_word_punc_str, data[self.source_lang]):
                if token not in self.source_vocab and source_token_freqs[token] > min_token_freq:
                    self.source_vocab[token] = len(self.source_vocab)
            for token in list(data[self.target_lang]):
                if token not in self.target_vocab and target_token_freqs[token] > min_token_freq:
                    self.target_vocab[token] = len(self.target_vocab)

    def batch_examples_and_add_special_tokens(self, dataset):
        for example in dataset:
            example[self.source_lang] = [self.BOS_TOKEN] + re.findall(self.split_word_punc_str, example[self.source_lang]) + [self.EOS_TOKEN]
            example[self.target_lang] = [self.BOS_TOKEN] + list(example[self.target_lang]) + [self.EOS_TOKEN]
        batched_data = np.array_split(dataset, np.arange(self.batch_size, len(dataset), self.batch_size))
        for batch in batched_data:
            batch_source_max_len = 0
            batch_target_max_len = 0
            for example in batch:
                batch_source_max_len = max(batch_source_max_len, len(example[self.source_lang]))
                batch_target_max_len = max(batch_target_max_len, len(example[self.target_lang]))
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
