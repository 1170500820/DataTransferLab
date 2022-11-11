import sys
sys.path.append('..')

from tqdm import tqdm
import json
from loguru import logger
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datatransfer.conditional_generation_model import T5FineTuner, BartFineTuner
from datatransfer.prepare_dataset import DuIE_Dataset, get_dataset


def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, choices=['bart', 't5'], default='t5')
    parser.add_argument('--ckp_file', type=str)
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--prompt_type', type=str, choices=['find_object', 'find_subject', 'find_relation',
                                                            'hybrid_find'], default='')
    parser.add_argument('--length', type=int, default=20)

    args = vars(parser.parse_args())

    if args['model'] == 't5':
        model_name = '/mnt/huggingface_models/mt5-small'
    else:  # args['model'] == 'bart'
        model_name = '/mnt/huggingface_models/bart-base-chinese'

    conf = dict(
        # 需要读取的ckp文件，以及需要预测的数据
        checkpoint_path=args['ckp_file'],
        name=args['name'],
        length=args['length'],
        prompt_type=args['prompt_type'],

        # 随机种子
        seed=42,

        # 模型参数
        model=args['model'],
        model_name=model_name,
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-3,
        warmup_steps=0,

        # 训练参数
        train_batch_size=args['bsz'],
        eval_batch_size=32,
        max_epochs=1,
        n_gpus=1,
        accumulate_grad_batches=4,
        strategy='ddp',
        accelerator='gpu',

        # 日志控制
        logger_dir='tb_log/',
        dirpath='t5-checkpoints/',
        every_n_epochs=1,

        # checkpoint
        save_top_k=4,
        monitor='val_loss'
    )
    return conf


def predict(config):
    logger.info(f'使用的checkpoint文件：{config["checkpoint_path"]}')
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
    if config['model'] == 't5':
        model = T5FineTuner.load_from_checkpoint(config['checkpoint_path'], params=model_params)
    elif config['model'] == 'bart':
        model = BartFineTuner.load_from_checkpoint(config['checkpoint_path'], params=model_params)
    else:
        logger.info(f'不存在{config["model"]}模型！')
        return
    logger.info(f'{config["model"]}模型加载完成')

    # dataset = DuIE_Dataset(model.tokenizer, data_type='dev')
    dataset = get_dataset(model.tokenizer, data_type='dev', prompt_type=config['prompt_type'], compact=False)
    loader = DataLoader(dataset, batch_size=config['eval_batch_size'])
    logger.info(f'数据集加载完成。prompt_type:{config["prompt_type"]}, data_type:dev')

    model.model.eval()
    outputs, targets = [], []
    model.model.to('cuda')
    logger.info('模型已迁移到CUDA，开始预测')
    for batch in tqdm(loader):
        outs = model.model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=config['length']
        )
        dec = [model.tokenizer.decode(ids) for ids in outs]
        target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]
        outputs.extend(dec)
        targets.extend(target)
    json.dump([outputs, targets], open(f'{config["name"]}.dev.{config["prompt_type"]}.json', 'w', encoding='utf-8'),
              ensure_ascii=False)


if __name__ == '__main__':
    logger.info('预测流程已启动')
    conf = handle_cli()

    logger.info('参数读取完成，开始预测')
    predict(conf)

    logger.info('预测结束')
