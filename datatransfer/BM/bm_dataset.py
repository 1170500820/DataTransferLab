import sys
sys.path.append('..')

import bmtrain as bmt
bmt.init_distributed(seed=0)

from model_center.dataset.bertdataset import DATASET
from model_center.dataset import DistributedDataLoader
from model_center.tokenizer import BertTokenizer


path_to_dataset = '../../data'

def prepare_train_dataloader():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cur_split = 'train'

    dataset = DATASET['BoolQ'](path_to_dataset, cur_split, bmt.rank(), bmt.world_size(), tokenizer, max_encoder_length=512)
    batch_size = 64
    train_dataloader = DistributedDataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def prepare_dev_dataloader():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cur_split = 'dev'

    dataset = DATASET['BoolQ'](path_to_dataset, cur_split, bmt.rank(), bmt.world_size(), tokenizer, max_encoder_length=512)
    batch_size = 64
    dev_dataloader = DistributedDataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dev_dataloader

