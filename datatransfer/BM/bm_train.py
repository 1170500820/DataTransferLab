import sys
sys.path.append('../..')

import bmtrain as bmt
bmt.init_distributed(seed=0)

import torch
import torch.nn as nn
from datatransfer.BM.bm_dataset import prepare_dev_dataloader, prepare_train_dataloader
from datatransfer.BM.bmtrain_try import BertModel, BertConfig
from sklearn.metrics import accuracy_score, recall_score, f1_score

from loguru import logger
from argparse import ArgumentParser


def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--epoch', type=int, default=10)

    args = vars(parser.parse_args())
    return args


def do_train(config: dict):
    """

    :param config:
        - epoch
    :return:
    """
    logger.info('正在初始化模型与数据集')
    model = BertModel(BertConfig.from_pretrained('bert-base-uncased'))
    train_dataloader = prepare_train_dataloader()
    dev_dataloader = prepare_dev_dataloader()

    logger.info('正在初始化optimizer、lr_scheduler与loss')
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())
    lr_scheduler = bmt.lr_scheduler.Noam(
        optimizer,
        start_lr = 1e-5,
        warmip_iter = 100,
        end_iter = -1
    )
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    logger.info('开始训练')
    for epoch in range(config['epoch']):
        model.train()
        for data in train_dataloader:
            input_ids, attention_mask = data['input_ids'], data['attention_mask']
            labels = data['labels']

            optimizer.zero_grad()

            # model forward
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # calculate loss
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

            # scale loss to avoid precision underflow of fp16
            loss = optimizer.loss_scale(loss)

            # model backward
            loss.backward()

            # clip gradient norm
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale=optimizer.scale, norm_type=2)

            bmt.optim_step(optimizer, lr_scheduler)

            bmt.print_rank(
                f'losss: {bmt.sum_loss(loss).item():.4f} | lr: {lr_scheduler.current_lr:.4e}, scale: {int(optimizer.scale):.4f} | grad_norm: {grad_norm:.4f}'
            )

        # evaluate model
        model.eval()
        with torch.no_grad():
            pd, gt = [], [] # prediction, groud_truth
            for data in dev_dataloader:
                input_ids, attention_mask = data['input_ids'], data['attention_mask']
                labels = data['labels']

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

                logits = logits.argmax(dim=-1)

                pd.extend(logits.cpu().tolist())
                gt.extend(labels.cpu().tolist())

            pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
            gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

            # calculate metric
            acc = accuracy_score(gt, pd)
            bmt.print_rank(f'accuracy: {acc * 100:.2f}')


if __name__ == '__main__':
    args = handle_cli()
    do_train(args)
