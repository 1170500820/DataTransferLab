import sys
sys.path.append('..')

from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datatransfer.naive_t5_2 import T5FineTuner
from datatransfer.prepare_dataset import DuIE_Dataset


name = 't5-small-duie'
batchsize = 16
n_gpus = 1
epochs = 4
conf = dict(
    dev_path='../data/prompted/duie_dev.jsonl',
    checkpoint_path='t5-checkpoints/epoch=1-step=7132.ckpt',

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
    eval_batch_size=32,
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
    monitor='val_loss'
)

def predict(config):
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
    model = T5FineTuner.load_from_checkpoint(config['checkpoint_path'], params=model_params)
    dataset = DuIE_Dataset(model.tokenizer, data_type='dev')
    loader = DataLoader(dataset, batch_size=config['eval_batch_size'])
    model.model.eval()
    outputs, targets = [], []
    print(f'the model device: {str(model.model.device)}')
    model.model.to('cuda')
    for batch in tqdm(loader):
        outs = model.model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda(),
            max_length=100
        )
        dec = [model.tokenizer.decode(ids) for ids in outs]
        target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]
        outputs.extend(dec)
        targets.extend(target)
    json.dump([outputs, targets], open('model_output_duie.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    predict(conf)
