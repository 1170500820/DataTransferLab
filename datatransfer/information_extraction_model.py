"""
关系抽取相关的非T5模型
"""
import sys
sys.path.append('..')

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl

from transformers import BertModel, BertTokenizerFast


class CASREL(nn.Module):
    hparams = {
        'relation_cnt': 1,
        'plm_path': 'bert-base-chinese',
        'plm_lr': 2e-5,
        'others_lr': 3e-4,
        'weight_decay': 0.0
    }

    def __init__(self, hparams: dict = None):
        """

        :param hparams:
            relation_cnt
            plm_path
            plm_lr
            others_lr
            weight_decay
        """
        super(CASREL, self).__init__()
        self.hparams.update(hparams)
        self.relation_cnt = self.hparams['relation_cnt']
        self.plm_lr = self.hparams['plm_lr']
        self.others_lr = self.hparams['others_lr']

        # BERT本体
        self.bert = BertModel.from_pretrained(self.hparams['plm_path'])
        self.hidden_size = self.bert.config.hidden_size

        # 分类器
        self.subject_start_cls = nn.Linear(self.hidden_size, 1)
        self.subject_end_cls = nn.Linear(self.hidden_siz, 1)
        self.object_start_cls = nn.Linear(self.hidden_size, self.relation_cnt)
        self.object_end_cls = nn.Linear(self.hidden_size, self.relation_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.subject_start_cls.weight)
        self.subject_start_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.subject_end_cls.weight)
        self.subject_end_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.object_start_cls.weight)
        self.object_start_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.object_end_cls.weight)
        self.object_end_cls.bias.data.fill_(0)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                subject_gt_start: torch.Tensor = None,
                subject_gt_end: torch.Tensor = None):
        """
        在train模式下，
            - subject_gt_start/end均不为None，将用于模型第二步的训练
            - 返回值
                - subject_start_result
                - subject_end_result
                - object_start_result
                - object_end_result
        在eval模式下，subject_gt_start/end为None（就算不为None，也不会用到）
        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :param subject_gt_start: (bsz, seq_l) 只包含1和0的向量。
        :param subject_gt_end: (bsz, seq_l)
        :return:
        """
        def calculate_embed_with_subject(bert_embed, subject_start, subject_end):
            """
            计算subject的表示向量subject_repr
            :param bert_embed: (bsz, seq_l, hidden)
            :param subject_start: (bsz, 1, seq_l)
            :param subject_end: (bsz, 1, seq_l)
            :return:
            """
            start_repr = torch.bmm(subject_start, bert_embed)  # (bsz, 1, hidden)
            end_repr = torch.bmm(subject_end, bert_embed)  # (bsz, 1, hidden)
            subject_repr = (start_repr + end_repr) / 2  # (bsz, 1, hidden)

            # h_N + v^k_{sub}
            embed = bert_embed + subject_repr
            return embed

        # 获取BERT embedding
        result = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output, _ = result[0], result[1]  # output: (bsz, seq_l, hidden)

        mask = (1 - attention_mask).bool()  # (bsz, seq_l)
        # 获取subject的预测
        subject_start_result = self.subject_start_cls(output)  # (bsz, seq_l, 1)
        subject_end_result = self.subject_end_cls(output)  # (bsz, seq_l, 1)
        subject_start_result = torch.sigmoid(subject_start_result)  # (bsz, seq_l, 1)
        subject_end_result = torch.sigmoid(subject_end_result)  # (bsz, seq_l, 1)
        subject_start_result = subject_start_result.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)  # (bsz, seq_l, 1)
        subject_end_result = subject_end_result.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)  # (bsz, seq_l, 1)
        if self.training:
            gt_start, gt_end = subject_gt_start.unsqueeze(dim=1), subject_gt_end.unsqueeze(dim=1)
            # both (bsz, 1, seq_l)
            embed = calculate_embed_with_subject(output, gt_start, gt_end)  # (bsz, seq_l, hidden)

            # 计算object的预测
            object_start_result = self.object_start_cls(embed)  # (bsz, seq_l, relation_cnt)
            object_end_result = self.object_end_cls(embed)  # (bsz, seq_l, relation_cnt)

            object_start_result = torch.sigmoid(object_start_result)  # (bsz, seq_l, relation_cnt)
            object_end_result = torch.sigmoid(object_end_result)  # (bsz, seq_l, relation_cnt)
            relation_cnt = object_end_result.shape[-1]
            ro_mask = mask.unsqueeze(dim=-1)  # (bsz, seq_l, 1)
            ro_mask = ro_mask.repeat(1, 1, relation_cnt)  # (bsz, seq_l, relation_cnt)
            ro_attention_mask = (1 - ro_mask.float())  # (bsz, seq_l, relation_cnt)
            object_start_result = object_start_result.masked_fill(mask=ro_mask, value=0)  # (bsz, seq_l, relation_cnt)
            object_end_result = object_end_result.masked_fill(mask=ro_mask, value=0)  # (bsz, seq_l, relation_cnt)
            return {
                "subject_start_result": subject_start_result,  # (bsz, seq_l, 1)
                "subject_end_result": subject_end_result,  # (bsz, seq_l, 1)
                "object_start_result": object_start_result,  # (bsz, seq_l, relation_cnt)
                "object_end_result": object_end_result,  # （bsz, seq_l, relation_cnt)
                'subject_mask': attention_mask,  # (bsz, seq_l)
                'ro_mask': ro_attention_mask,  # (bsz, seq_l, relation_cnt)
            }
        else:  # eval模式。该模式下，bsz默认为1
            # 获取subject的预测：cur_spans - SpanList
            subject_start_result = subject_start_result.squeeze()  # (seq_l)
            subject_end_result = subject_end_result.squeeze()  # (seq_l)
            subject_start_int = (subject_start_result > 0.5).int().tolist()
            subject_end_int = (subject_end_result > 0.5).int().tolist()
            seq_l = len(subject_start_int)
            subject_spans = tools.argument_span_determination(subject_start_int, subject_end_int, subject_start_result, subject_end_result)  # SpanList

            # 迭代获取每个subject所对应的object预测
            object_spans = []  #  List[List[SpanList]]  (subject, relation, span number)
            for elem_span in subject_spans:
                object_spans_for_current_subject = []  # List[SpanList]  (relation, span number)
                temporary_start, temporary_end = torch.zeros(1, 1, seq_l).cuda(), torch.zeros(1, 1, seq_l).cuda()  # both (1, 1, seq_l)
                temporary_start[0][0][elem_span[0]] = 1
                temporary_end[0][0][elem_span[1]] = 1
                embed = calculate_embed_with_subject(output, temporary_start, temporary_end)  # (1, seq_l, hidden)

                # 计算object的预测
                object_start_result = self.object_start_cls(embed)  # (1, seq_l, relation_cnt)
                object_end_result = self.object_end_cls(embed)  # (1, seq_l, relation_cnt)

                object_start_result = object_start_result.squeeze().T  # (relation_cnt, seq_l)
                object_end_result = object_end_result.squeeze().T  # (relation_cnt, seq_l)

                object_start_result = torch.sigmoid(object_start_result)
                object_end_result = torch.sigmoid(object_end_result)
                for (object_start_r, object_end_r) in zip(object_start_result, object_end_result):
                    o_start_int = (object_start_r > 0.5).int().tolist()
                    o_end_int = (object_end_r > 0.5).int().tolist()
                    cur_spans = tools.argument_span_determination(o_start_int, o_end_int, object_start_r, object_end_r)  # SpanList
                    object_spans_for_current_subject.append(cur_spans)
                object_spans.append(object_spans_for_current_subject)

            return {
                "pred_subjects": subject_spans,  # SpanList
                "pred_objects": object_spans  # List[List[SpanList]]
            }

    def get_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        plm_params = self.bert.named_parameters()
        ss_params = self.subject_start_cls.named_parameters()
        se_params = self.subject_end_cls.named_buffers()
        os_params = self.object_start_cls.named_parameters()
        oe_params = self.object_end_cls.named_buffers()
        others_params = chain(ss_params, se_params, os_params, oe_params)
        # plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        # linear_optimizer = AdamW(params=chain(ss_params, se_params, os_params, oe_params), lr=self.others_lr)
        optimizer = AdamW(
            [{
                'params': [p for n, p in plm_params if not any(nd in n for nd in no_decay)],
                'lr': self.plm_lr,
                'weight_decay': self.hparams['weight_decay']
            }, {
                'params': [p for n, p in plm_params if any(nd in n for nd in no_decay)],
                'lr': self.plm_lr,
                'weight_decay': 0.0
            }, {
                'params': [p for n, p in chain(ss_params, se_params, os_params, oe_params) if
                           not any(nd in n for nd in no_decay)],
                'lr': self.others_lr,
                'weight_decay': self.hparams['weight_decay']
            }, {
                'params': [p for n, p in chain(ss_params, se_params, os_params, oe_params) if
                           any(nd in n for nd in no_decay)],
                'lr': self.others_lr,
                'weight_decay': 0.0
            }],
            eps=self.hparams['adam_epsilon']
        )
        return optimizer


class CASREL_Loss(nn.Module):
    def __init__(self, lamb: float = 0.6):
        super(CASREL_Loss, self).__init__()
        self.lamb = lamb
        # self.focal_weight = tools.FocalWeight()

    def forward(self,
                subject_start_result: torch.Tensor,
                subject_end_result: torch.Tensor,
                object_start_result: torch.Tensor,
                object_end_result: torch.Tensor,
                subject_mask: torch.Tensor,
                ro_mask: torch.Tensor,
                subject_start_label: torch.Tensor,
                subject_end_label: torch.Tensor,
                object_start_label: torch.Tensor,
                object_end_label: torch.Tensor):
        """

        :param subject_start_result: (bsz, seq_l, 1)
        :param subject_end_result: (bsz, seq_l, 1)
        :param object_start_result: (bsz, seq_l, relation_cnt)
        :param object_end_result: (bsz, seq_l, relation_cnt)
        :param subject_mask: (bsz, seq_l) 用于处理subject的mask
        :param ro_mask: (bsz, seq_l) 用于处理relation-object的mask
        :param subject_start_label: (bsz, seq_l)
        :param subject_end_label: (bsz, seq_l)
        :param object_start_label: (bsz, seq_l, relation_cnt)
        :param object_end_label: (bsz, seq_l, relation_cnt)
        :return:
        """
        # 将subject_result的形状与label对齐
        subject_start_result = subject_start_result.squeeze(-1)  # (bsz, seq_l)
        subject_end_result = subject_end_result.squeeze(-1)  # (bsz, seq_l)

        # 计算weight
        # subject_start_focal_weight = self.focal_weight(subject_start_label, subject_start_result)
        # subject_end_focal_weight = self.focal_weight(subject_end_label, subject_end_result)
        # object_start_focal_weight = self.focal_weight(object_start_label, object_start_result)
        # object_end_focal_weight = self.focal_weight(object_end_label, object_end_result)

        # 计算loss
        subject_start_loss = F.binary_cross_entropy(subject_start_result, subject_start_label, reduction='none')
        subject_end_loss = F.binary_cross_entropy(subject_end_result, subject_end_label, reduction='none')
        object_start_loss = F.binary_cross_entropy(object_start_result, object_start_label, reduction='none')
        object_end_loss = F.binary_cross_entropy(object_end_result, object_end_label, reduction='none')

        ss_loss = torch.sum(subject_start_loss * subject_mask) / torch.sum(subject_mask)
        se_loss = torch.sum(subject_end_loss * subject_mask) / torch.sum(subject_mask)
        os_loss = torch.sum(object_start_loss * ro_mask) / torch.sum(ro_mask)
        oe_loss = torch.sum(object_end_loss * ro_mask) / torch.sum(ro_mask)

        loss = self.lamb * (ss_loss + se_loss) + (1 - self.lamb) * (os_loss + oe_loss)
        return loss


class DuIE_FineTuner(pl.LightningModule):

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                subject_gt_start: torch.Tensor = None,
                subject_gt_end: torch.Tensor = None):
        output = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            subject_gt_start=subject_gt_start,
            subject_gt_end=subject_gt_end)
        return output

    def _step(self, batch):
        outputs = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            subject_gt_start=batch['subject_gt_start'],
            subject_gt_end=batch['subject_gt_end']
        )
        loss = self.Loss(
            # model output
            subject_start_result=outputs['subject_start_result'],
            subject_end_result=outputs['subject_end_result'],
            object_start_result=outputs['object_start_result'],
            object_end_result=outputs['object_end_result'],

            # preprocessed and label
            subject_mask=batch['subject_mask'],
            ro_mask=batch['ro_mask'],
            subject_start_label=batch['subject_start_label'],
            subject_end_label=batch['subject_end_label'],
            object_start_label=batch['object_start_label'],
            object_end_label=batch['object_end_label']
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('loss', float(loss))
        return loss

    def validation_step(self, batch, batch_ids):
        loss = self._step(batch)
        return {'val_loss': loss}


    def configure_optimizers(self):
        optimizer = self.model.get_optimizers()
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None,
                       optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None
                       ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass