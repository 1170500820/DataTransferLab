"""
为T5模型准备数据集
"""
import os
import glob
import pickle
import re
from loguru import logger
from tqdm import tqdm
import json
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5TokenizerFast, BertTokenizer, BertTokenizerFast


class ImdbDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.pos_file_path = os.path.join(data_dir, type_path, 'pos')
        self.neg_file_path = os.path.join(data_dir, type_path, 'neg')

        self.pos_files = glob.glob("%s/*.txt" % self.pos_file_path)
        self.neg_files = glob.glob("%s/*.txt" % self.neg_file_path)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        self._buil_examples_from_files(self.pos_files, 'positive')
        self._buil_examples_from_files(self.neg_files, 'negative')

    def _buil_examples_from_files(self, files, sentiment):
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        for path in files:
            with open(path, 'r') as f:
                text = f.read()

            line = text.strip()
            line = REPLACE_NO_SPACE.sub("", line)
            line = REPLACE_WITH_SPACE.sub("", line)
            line = line + ' </s>'

            target = sentiment + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


# 用于ie实验的混合数据集。混合方式会在里面确定
class IeDataset(Dataset):
    def __init__(self, tokenizer, data_type: str, max_len=512):
        print('    loading raw data')
        self.raw_file = list(json.loads(x) for x in open(f'../data/prompted/ner_weibo_{data_type}.jsonl', 'r', encoding='utf-8').read().strip().split('\n'))

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs, self.targets = [], []
        print('    _build() running')
        self._build()

    def _build(self):
        for elem in tqdm(self.raw_file):
            inp = elem['input'] + ' </s>'
            tgt = elem['target'] + ' </s>'

            tokenized_inp = self.tokenizer.batch_encode_plus(
                [inp], max_length=self.max_len, pad_to_max_length=True, return_tensors='pt'
            )
            tokenized_tgt = self.tokenizer.batch_encode_plus(
                [tgt], max_length=self.max_len, pad_to_max_length=True, return_tensors='pt'
            )

            self.inputs.append(tokenized_inp)
            self.targets.append(tokenized_tgt)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class DuIE_Dataset(Dataset):
    def __init__(self, tokenizer, data_type: str, prompt_type='', max_len=512, lazy_tokenize=True, no_store=True):
        if prompt_type == '':  fname = f'../data/prompted/duie_{data_type}.jsonl'
        else:  fname = f'../data/prompted/duie_{prompt_type}_{data_type}.jsonl'

        self.raw_file = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lazy = lazy_tokenize
        self.no_store = no_store
        if self.lazy:  # 若lazy_tokenize，则需要填充list
            self.inputs, self.targets = [None] * len(self.raw_file), [None] * len(self.raw_file)
        else:
            self.inputs, self.targets = [], []
        if not self.lazy:
            self._build()

    def _build(self):
        for elem in tqdm(self.raw_file):
            inp = elem['input']
            tgt = elem['target']

            tokenized_inp = self.tokenizer.batch_encode_plus(
                [inp], max_length=self.max_len, padding='max_length', return_tensors='pt', truncation=True
            )
            tokenized_tgt = self.tokenizer.batch_encode_plus(
                [tgt], max_length=self.max_len, padding='max_length', return_tensors='pt', truncation=True
            )

            self.inputs.append(tokenized_inp)
            self.targets.append(tokenized_tgt)


    def __len__(self):
        return len(self.raw_file)

    def __getitem__(self, index):
        if self.lazy and self.inputs[index] is None:
            elem = self.raw_file[index]
            inp, tgt = elem['input'], elem['target']
            tokenized_inp = self.tokenizer.batch_encode_plus(
                [inp], max_length=self.max_len, padding='max_length', return_tensors='pt', truncation=True
            )
            tokenized_tgt = self.tokenizer.batch_encode_plus(
                [tgt], max_length=self.max_len, padding='max_length', return_tensors='pt', truncation=True
            )
            self.inputs[index] = tokenized_inp
            self.targets[index] = tokenized_tgt

        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        if self.no_store:
            self.inputs[index] = None
            self.targets[index] = None

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


def get_dataset(tokenizer, data_type='train', prompt_type=''):
    # return IeDataset(tokenizer=tokenizer, data_type=data_type)
    return DuIE_Dataset(tokenizer=tokenizer, data_type=data_type, prompt_type=prompt_type)


if __name__ == '__main__':
    print('building tokenizer')
    bert_tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
    print('building dataset')
    duie = DuIE_Dataset(bert_tokenizer, 'train', 'find_object')