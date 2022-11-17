"""
关系抽取数据集
"""
import sys
sys.path.append('..')

import os, glob, random, re, json
from typing import List
from tqdm import tqdm
from bisect import bisect

from torch.utils.data import Dataset
from transformers import T5Tokenizer, BertTokenizerFast

from datatransfer.utils import io_tools


class DuIE_CASREL_Dataset(Dataset):
    def __init__(self, data_type: str):
        if data_type == 'dev':
            data_type = 'valid'
        fname = f'../../data/processed/duie_indexed_{data_type}.jsonl'
        self.raw_file = io_tools.load_jsonl(fname)

    def __len__(self):
        return len(self.raw_file)

    def __getitem__(self, index):
        return self.raw_file[index]


if __name__ == '__main__':
    d = DuIE_CASREL_Dataset('train')
