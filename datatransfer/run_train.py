import sys
sys.path.append('..')

import random
import numpy as np
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datatransfer.naive_t5_2 import T5FineTuner


# 一些重要参数需要单独列出
name = 't5-small-duie'
batchsize = 4
epochs = 6
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
    train_batch_size=batchsize,
    eval_batch_size=2,
    max_epochs=epochs,
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


def get_logger(config):
    return TensorBoardLogger(
        save_dir=config['logger_dir'],
        name=name
    )


def get_callbacks(config):
    return [ModelCheckpoint(
        dirpath=config['ckp_dir'],
        every_n_epochs=config['every_n_epochs']
    )]


def train(config):
    logger = get_logger(config)
    checkpoint_callback = get_callbacks(config)

    train_params = dict(
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gpus=config['n_gpus'],
        max_epochs=config['num_train_epochs'],
        precision=32,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    model = T5FineTuner(config)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)





if __name__ == '__main__':
    train(conf)
