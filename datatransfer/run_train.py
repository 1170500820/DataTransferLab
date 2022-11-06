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
batchsize = 3
n_gpus = 4
epochs = 4
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
    eval_batch_size=batchsize,
    max_epochs=epochs,
    n_gpus=n_gpus,
    accumulate_grad_batches=4,
    strategy='ddp',
    accelerator='gpu',

    # 日志控制
    logger_dir='tb_log/',
    dirpath='t5-checkpoints/',
    every_n_epochs=1,

    # checkpoint
    save_top_k=4,
    monitor='val_loss',
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
        dirpath=config['dirpath'],
    )]


def train(config):
    logger = get_logger(config)
    callbacks = get_callbacks(config)

    train_params = dict(
        accumulate_grad_batches=config['accumulate_grad_batches'],
        accelerator=config['accelerator'],
        devices=config['n_gpus'],
        max_epochs=config['max_epochs'],
        strategy=config['strategy'],
        precision=32,
        logger=logger,
        callbacks=callbacks
    )
    model_params = dict(
        weight_decay=config['weight_decay'],
        model_name=config['model_name'],
        learning_rate=config['learning_rate'],
        adam_epsilon=config['adam_epsilon'],
        max_seq_length=config['max_seq_length'],
        warmup_steps=config['warmup_steps'],
        train_batch_size=config['train_batch_size'],
        n_gpus=config['n_gpus'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        num_train_epochs=config['max_epochs'],
        eval_batch_size=config['eval_batch_size'] 
    )

    model = T5FineTuner(model_params)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)





if __name__ == '__main__':
    train(conf)
