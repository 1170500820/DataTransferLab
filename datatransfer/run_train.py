import sys
sys.path.append('..')

import random
import numpy as np
from loguru import logger as ru_logger
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datatransfer.conditional_generation_model import T5FineTuner, BartFineTuner


def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, choices=['bart', 't5'], default='t5', help='用于训练的模型。BART与T5的结构，tokenizer均不同，需注意。')
    parser.add_argument('--bsz', type=int, default=4, help='单卡的batch size。实际的batch size为bsz * n_gpus * grad_acc')
    parser.add_argument('--n_gpus', type=int, default=1, help='用于训练的显卡数量')
    parser.add_argument('--epoch', type=int, default=5, help='训练的epoch数')
    parser.add_argument('--name', type=str, default='TaskTransfer_default', help='用于标识该次训练的名字，将用于对checkpoint进行命名。')
    parser.add_argument('--prompt_type', type=str, choices=['find_object', 'find_subject', 'find_relation',
                                                            'hybrid_find'], default='', help='如果使用默认，则采用粗糙处理的数据集')
    parser.add_argument('--grad_acc', type=int, default=4, help='梯度累积操作，可用于倍增batch size')
    parser.add_argument('--compact', action='store_true', help='是否使用堆叠的数据集。该选项只能在模型为t5的时候使用！')

    args = vars(parser.parse_args())

    if args['model'] != 't5' and args['compact']:
        raise Exception('compact参数只能在模型为t5的时候设定。')
    if args['model'] == 't5':
        model_name = '/mnt/huggingface_models/mt5-small'
    else:  # args['model'] == 'bart'
        model_name = '/mnt/huggingface_models/bart-base-chinese'

    # 一些重要参数需要单独列出

    conf = dict(
        # 随机种子
        seed=42,

        # 本次训练的任务名字（也就是模型名字））
        name=args['name'],

        # 模型参数
        model=args['model'],
        model_name=model_name,
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-3,
        warmup_steps=0,

        # 训练数据
        prompt_type=args['prompt_type'],
        compact=args['compact'],

        # 训练参数
        train_batch_size=args['bsz'],
        eval_batch_size=args['bsz'],
        max_epochs=args['epoch'],
        n_gpus=args['n_gpus'],
        accumulate_grad_batches=args['grad_acc'],
        strategy='ddp',
        accelerator='gpu',

        # 日志控制
        logger_dir='tb_log/',
        dirpath='t5-checkpoints/',
        every_n_epochs=1,

        # checkpoint
        save_top_k=-1,
        monitor='val_loss',
    )
    return conf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(config):
    return TensorBoardLogger(
        save_dir=config['logger_dir'],
        name=config['name']
    )


def get_callbacks(config):
    return [ModelCheckpoint(
        dirpath=config['dirpath'],
        save_top_k=config['save_top_k'],
        filename=config['name'] + '.' + '{epoch}-{val_loss:.2f}'
    )]


def train(config):
    logger = get_logger(config)
    callbacks = get_callbacks(config)

    if config['n_gpus'] == 1:
        train_params = dict(
            accumulate_grad_batches=config['accumulate_grad_batches'],
            accelerator=config['accelerator'],
            max_epochs=config['max_epochs'],
            precision=32,
            logger=logger,
            callbacks=callbacks
        )
    else:  # config['n_gpus'] > 1

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
        eval_batch_size=config['eval_batch_size'],
        prompt_type=config['prompt_type'],
        compact=config['compact']
    )
    ru_logger.info(f'正在加载模型{config["model_name"]}')
    if config['model'] == 't5':
        model = T5FineTuner(model_params)
    else:  # config['model'] == 'bart'
        model = BartFineTuner(model_params)
    ru_logger.info('模型加载完毕')
    ru_logger.info('正在加载Trainer')
    trainer = pl.Trainer(**train_params)
    ru_logger.info('Trainer加载完毕，开始fit！')
    trainer.fit(model)



if __name__ == '__main__':
    conf = handle_cli()
    # logger.info(conf)
    train(conf)
