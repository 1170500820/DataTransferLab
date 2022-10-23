import json
import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datatransfer.naive_t5_2 import T5FineTuner
from datatransfer.prepare_dataset import IeDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info(" -- -- -- Validation results -- -- -- ")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ['log', 'progress_bar']:
                    logger.info(f'{key} = {str(metrics[key])}')


# 一些全局配置参数
global_seed = 42
model_name = 'google/mt5-medium'

# 用于训练与模型的参数
args_dict = dict(
    # 模型参数
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_epsilon=1e-8,
    warmup_steps=0,

    # 训练参数
    train_batch_size=2,
    eval_batch_size=2,
    num_train_epochs=5,
    accumulate_grad_batches=1,
    n_gpus=1,

    # 随机数种子配置
    seed = global_seed,
)

def main(args_dict):
    """
    训练模型相关的代码
    :return:
    """
    # 首先使用argparser来处理参数
    args = argparse.Namespace(**args_dict)
    train_params = dict(
        accumulate_grad_batches=args.accumulate_grad_batches,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=32,
    )

    # 定义模型与训练器，然后开始训练
    model = T5FineTuner(model_name, args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    # model.model.save_pretrained(f't5_exp.{model_name}.weiboner')

    # 评价
    dataset = IeDataset(model.tokenizer, 'test')
    loader = DataLoader(dataset, batch_size=args_dict.eval_batch_size)
    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(loader):
        outs = model.model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=30
        )
        dec = [model.tokenizer.decode(ids) for ids in outs]
        target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]

        outputs.extend(dec)
        targets.extend(target)

    json.dump([outputs, targets], open('model_output.json', 'w', encoding='utf-8'))



if __name__ == '__main__':
    # 随机数种子与logger的配置
    logger = logging.getLogger(__name__)
    set_seed(global_seed)

    # 开始执行训练流程
    main(args_dict)


