import json
import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datatransfer.naive_t5_2 import T5FineTuner
from datatransfer.prepare_dataset import IeDataset


conf = dict(
    # 随机种子
    seed=42,

    # 模型参数
    model_name='/mnt/huggingface_models/mt5-small',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_epsilon=1e-3,
    warmup_steps=0,

    # 训练参数
    train_batch_size=4,
    eval_batch_size=2,
    max_epochs=6,
    gpus=1,
    accumulate_grad_batches=4,

    # 日志控制
    logger_dir='tb_log/',
    dirpath='t5-checkpoints/',
    every_n_epochs=1,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(conf):
    logger = TensorBoardLogger(
        save_dir=conf.logger_dir,
        name='t5-small-weiboner'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=conf.ckp_dir,
        every_n_epochs=conf.every_n_epochs
    )
    trainer = pl.Trainer(
        **conf,
        logger=logger,
        callbacks=[checkpoint_callback])
    model = T5FineTuner(conf)
    trainer.fit(model)


def predict(conf):
    pass


parser = ArgumentParser()

# 环境参数
parser.add_argument('--seed', type=int, default=conf.seed)

# 日志参数
parser.add_argument('--logger_dir', type=str, default=conf.logger_dir)
parser.add_argument('--every_n_epochs', type=int, default=conf.every_n_epochs)

# 模型参数
# 没有模型参数

args = parser.parse_args()
args = vars(args)
conf.update(args)

set_seed(conf.seed)


# def main(args_dict):
#     """
#     训练模型相关的代码
#     :return:
#     """
#     # 首先使用argparser来处理参数
#     args = argparse.Namespace(**args_dict)
#     logger = TensorBoardLogger(save_dir=logger_dir, name='t5-small-weiboner')
#     checkpoint_callback = ModelCheckpoint(
#             dirpath=ckp_dir,
#             every_n_epochs=1
#             )
#     train_params = dict(
#         accumulate_grad_batches=args.accumulate_grad_batches,
#         gpus=args.n_gpus,
#         max_epochs=args.num_train_epochs,
#         precision=32,
#         logger=logger, # 加入tensorboard logger,
#         callbacks=[checkpoint_callback]
#     )
#
#     # 定义模型与训练器，然后开始训练
#     model = T5FineTuner(model_name, args)
#     trainer = pl.Trainer(**train_params)
#     trainer.fit(model)
#     # model.model.save_pretrained(f't5_exp.{model_name}.weiboner')
#
#     # 评价
#     dataset = IeDataset(model.tokenizer, 'test')
#     loader = DataLoader(dataset, batch_size=args.eval_batch_size)
#     model.model.eval()
#     outputs, targets = [], []
#     for batch in tqdm(loader):
#         outs = model.model.generate(
#             input_ids=batch['source_ids'].cuda(),
#             attention_mask=batch['source_mask'].cuda(),
#             #max_length=30
#         )
#         dec = [model.tokenizer.decode(ids) for ids in outs]
#         target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]
#
#         outputs.extend(dec)
#         targets.extend(target)
#
#     json.dump([outputs, targets], open('model_output.json', 'w', encoding='utf-8'), ensure_ascii=False)


# def pred(ckp_path):
#     args = argparse.Namespace(**args_dict)
#     model = T5FineTuner.load_from_checkpoint('t5-checkpoints/epoch=5-step=498.ckpt', params=args)
#     dataset = IeDataset(model.tokenizer, 'test')
#     loader = DataLoader(dataset, batch_size=4)
#     model.model.eval()
#     outputs, targets = [], []
#     print(model.model.device)
#     model.model.to('cuda')
#     for batch in tqdm(loader):
#         outs = model.model.generate(
#                 input_ids=batch['source_ids'].cuda(),
#                 attention_mask=batch['source_mask'].cuda()
#                 )
#         dec = [model.tokenizer.decode(ids) for ids in outs]
#         target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]
#         outputs.extend(dec)
#         targets.extend(target)
#     json.dump([outputs, targets], open('model_output.json', 'w', encoding='utf-8'), ensure_ascii=False)

    # 随机数种子与logger的配置
#    logger = logging.getLogger(__name__)
