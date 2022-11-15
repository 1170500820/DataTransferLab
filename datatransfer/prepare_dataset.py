"""
为T5模型准备数据集
"""
import os
import glob
import random
import pickle
from typing import List
import re
from bisect import bisect
from loguru import logger
from tqdm import tqdm
import json
from torch.utils.data import Dataset, IterableDataset
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
    def __init__(self, tokenizer, data_type: str, prompt_type='', max_len=512, lazy_tokenize=True, no_store=True, add_extra=True):
        if prompt_type == '':  fname = f'../data/prompted/duie_{data_type}.jsonl'
        else:  fname = f'../data/prompted/duie_{prompt_type}_{data_type}.jsonl'

        self.raw_file = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lazy = lazy_tokenize
        self.no_store = no_store
        self.add_extra = add_extra
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
            if self.add_extra:
                inp += ' <extra_id_1>'

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
            if self.add_extra:
                inp += ' <extra_id_1>'
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


class DuIE_RE_Direct_Dataset(Dataset):
    """
    Direct，直接加载预先处理好的数据结果
    """
    def __init__(self, data_type: str, fname: str):
        self.raw_file = pickle.load(open(fname, 'rb'))
        self.data_type = data_type

    def __len__(self):
        return len(self.raw_file)

    def __getitem__(self, index):
        return self.raw_file[index]

class DuIE_RE_Dataset(Dataset):
    def __init__(self):
        pass


class DatasetCompactor:
    def __init__(self, lengths: List[int], max_length: int = 512, seed: int = 42):
        random.seed(seed)
        self.lengths = lengths
        self.l = len(self.lengths)
        self.max_length = max_length
        self.idx = list(range(self.l))

        comp = list(zip(self.lengths, self.idx))
        comp.sort()
        self.lengths, self.idx = list(zip(*comp))
        self.lengths, self.idx = list(self.lengths), list(self.idx)

        self.extra_size = 2

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.idx) == 0:
            raise StopIteration()
        total_l = 0  # 累积的序列长度。序列长度>=token序列长度
        chosen = []
        # 先随机选一个
        first = random.randint(0, len(self.idx) - 1)
        first_i, first_l = self.idx.pop(first), self.lengths.pop(first)
        total_l = first_l + 1 + self.extra_size
        chosen.append(first_i)

        while total_l < self.max_length and self.idx:
            left_l = self.max_length - total_l - 1 - self.extra_size
            border = bisect(self.lengths, left_l)
            if border == 0:
                break

            another = random.randint(0, border - 1)
            another_i, another_l = self.idx.pop(another), self.lengths.pop(another)
            total_l += another_l + 1 + self.extra_size
            chosen.append(another_i)
        return chosen


class DuIECompactDataset(Dataset):
    def __init__(self, tokenizer, data_type: str='train', prompt_type: str='', max_len: int=512, seed: int=42):
        super(DuIECompactDataset, self).__init__()
        if prompt_type == '':  fname = f'../data/prompted/duie_{data_type}.jsonl'
        else:  fname = f'../data/prompted/duie_{prompt_type}_{data_type}.jsonl'

        self.raw_file = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
        lengths = list(len(x['input']) for x in self.raw_file)
        self.comp = list(DatasetCompactor(lengths, max_len, seed))
        self.prompt_type = prompt_type
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.extra_id_0 = 250099

    def __len__(self):
        return len(self.comp)

    def __getitem__(self, index):
        ci = self.comp[index]
        inp, tgt = self._concat(ci)
        tokenized_inp = self.tokenizer.batch_encode_plus(
            [inp], max_length=self.max_len, padding='max_length', return_tensors='pt', truncation=True
        )
        tokenized_tgt = self.tokenizer.batch_encode_plus(
            [tgt], max_length=self.max_len, padding='max_length', return_tensors='pt', truncation=True
        )
        source_ids = tokenized_inp['input_ids'].squeeze()
        target_ids = tokenized_tgt['input_ids'].squeeze()
        src_mask = tokenized_inp['attention_mask'].squeeze()
        tgt_mask = tokenized_tgt['attention_mask'].squeeze()
        return {
            'source_ids': source_ids,
            'source_mask': src_mask,
            'target_ids': target_ids,
            'target_mask': tgt_mask
        }

    def _concat(self, compact_idxs):
        inputs, targets = [], []
        for i in compact_idxs:
            inputs.append(self.raw_file[i]['input'])
            targets.append(self.raw_file[i]['target'])
        input_str, target_str = '', ''
        for i, (e_input, e_target) in enumerate(zip(inputs, targets)):
            input_str = input_str + e_input + f'<extra_id_{i + 1}>' + '. '
            target_str = target_str + f' <extra_id_{i + 1}> ' + e_target
        return input_str, target_str




def get_dataset(tokenizer = None, model_type: str = 't5', data_type='train', prompt_type='', compact=False, direct=True):
    """

    :param tokenizer:
    :param model_type: 数据所要用于的模型。t5, casrel, bart
    :param data_type: 数据是训练集还是开发集
    :param prompt_type: 仅针对t5与bart模型的prompt式数据。prompt的类型
    :param compact: 仅针对t5的预训练数据。是否使用堆叠的数据集
    :param direct: 仅针对casrel的数据集。是否直接加载预处理的结果
    :return:
    """
    # return IeDataset(tokenizer=tokenizer, data_type=data_type)
    if model_type in ['t5', 'bart']:
        if compact:
            return DuIECompactDataset(tokenizer=tokenizer, data_type=data_type, prompt_type=prompt_type)
        else:
            return DuIE_Dataset(tokenizer=tokenizer, data_type=data_type, prompt_type=prompt_type)
    elif model_type == 'casrel':
        if direct:
            if data_type == 'train':
                fname = f'Models/RE/CASREL/temp_data/{data_type}.duie.ro_labeled.pk'
            elif data_type in ['dev', 'valid']:
                fname = 'Models/RE/CASREL/temp_data/valid.duie.eval_final.pk'
            return DuIE_RE_Direct_Dataset(data_type=data_type, fname=fname)
        else:
            pass



if __name__ == '__main__':
    d = get_dataset(model_type='casrel', data_type='train')
    d2 = get_dataset(model_type='casrel', data_type='valid')
