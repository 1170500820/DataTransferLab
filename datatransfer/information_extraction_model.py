"""
关系抽取相关的非T5模型
"""
import sys
sys.path.append('..')

from itertools import chain
from typing import List, Tuple
from bisect import bisect_left, bisect_right

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup


from datatransfer.prepare_dataset import get_dataset
from datatransfer.utils import tools, batch_tool
from datatransfer.Datasets.RE import DuIE_CASREL_Dataset, casrel_dev_collate_fn_2, casrel_train_collate_fn_2
from datatransfer.Models.RE.RE_utils import convert_lists_to_triplet_casrel, Triplet, convert_token_triplet_to_char_triplet
from datatransfer.Models.RE.RE_settings import duie_relations_idx

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
        self.hparams.update({
            'relation_cnt': hparams['class_cnt'],
            'plm_path': hparams['model_name'],
            'plm_lr': hparams['learning_rate'],
            'others_lr': hparams['linear_lr'],
            'weight_decay': hparams['weight_decay'],
            'adam_epsilon': hparams['adam_epsilon'],
            'model_type': 'casrel',
            'overfit': hparams['overfit']
        })
        self.relation_cnt = self.hparams['relation_cnt']
        self.plm_lr = self.hparams['plm_lr']
        self.others_lr = self.hparams['others_lr']

        # BERT本体
        self.bert = BertModel.from_pretrained(self.hparams['plm_path'])
        self.hidden_size = self.bert.config.hidden_size

        # 分类器
        self.subject_start_cls = nn.Linear(self.hidden_size, 1)
        self.subject_end_cls = nn.Linear(self.hidden_size, 1)
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


class CASREL2(nn.Module):
    """
    https://github.com/Onion12138/CasRelPyTorch/blob/master/model/casRel.py
    参考上述实现，将subject、object过程拆开
    todo 尝试用CLN？
    """
    hparams = {
        'relation_cnt': 1,
        'plm_path': 'bert-base-chinese',
        'plm_lr': 2e-5,
        'others_lr': 3e-4,
        'weight_decay': 0.0,
        'threshold': 0.5,
        'use_mask': False
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
        super(CASREL2, self).__init__()
        self.hparams.update({
            'relation_cnt': hparams['class_cnt'],
            'plm_path': hparams['model_name'],
            'plm_lr': hparams['learning_rate'],
            'others_lr': hparams['linear_lr'],
            'weight_decay': hparams['weight_decay'],
            'adam_epsilon': hparams['adam_epsilon'],
            'model_type': 'casrel',
            'overfit': hparams['overfit']
        })
        self.relation_cnt = self.hparams['relation_cnt']
        self.plm_lr = self.hparams['plm_lr']
        self.others_lr = self.hparams['others_lr']

        # BERT本体
        self.bert = BertModel.from_pretrained(self.hparams['plm_path'])
        self.hidden_size = self.bert.config.hidden_size

        # 分类器
        self.subject_start_cls = nn.Linear(self.hidden_size, 1)
        self.subject_end_cls = nn.Linear(self.hidden_size, 1)
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

    def get_encoded_text(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :return:
            encoded_text
            mask (bsz, seq_l) pad位置为True，其他位置为False
        """
        encoded_text = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        mask = (1 - attention_mask).bool()  # (bsz, seq_l)
        return encoded_text, mask

    def get_subjects(self, encoded_text: torch.Tensor, mask: torch.Tensor = None):
        """

        :param encoded_text: (bsz, seq_l, hidden)
        :param mask: 默认为None，如果提供了，则为(bsz, seq_l)的布尔向量，其中对应pad的位置为True，其他为False。
            如果提供了，就会用于对预测出的start_prob和end_prob进行mask
        :return:
        """
        start = self.subject_start_cls(encoded_text)
        end = self.subject_end_cls(encoded_text)  # both (bsz, seq_l, 1)
        start_prob, end_prob = torch.sigmoid(start), torch.sigmoid(end)  # (bsz, seq_l, 1)
        # todo 是否需要在这一步将pad部分mask掉？是的话，该方法就需要添加mask参数
        if mask is not None and self.hparams['use_mask']:
            start_prob = start_prob.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)
            end_prob = end_prob.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)
            # both (bsz, seq_l, 1)
        return start_prob, end_prob

    def get_objects_for_specific_subject(self, sub_start: torch.Tensor, sub_end: torch.Tensor, encoded_text: torch.Tensor, mask: torch.Tensor = None):
        """

        :param sub_start: (bsz, seq_l)
        :param sub_end: (bsz, seq_l) 均为只包含1和0的向量。将用于从encoded_text中提取embedding
        :param encoded_text: (bsz, seq_l, hidden)
        :param mask: 默认为None，如果提供了，则为(bsz, seq_l)的布尔向量，其中对应pad的位置为True，其他为False。
            如果提供了，就会用于对预测出的start_prob和end_prob进行mask
        :return:
        """
        start_repr = torch.bmm(sub_start.unsqueeze(dim=1), encoded_text)  # (bsz, 1, hidden)
        end_repr = torch.bmm(sub_end.unsqueeze(dim=1), encoded_text)  # (bsz, 1, hidden)
        subject_repr = (start_repr + end_repr) / 2  # (bsz, 1, hidden)
        # h_N + v^k_{sub}
        embed = subject_repr + encoded_text  # (bsz, seq_l, hidden) 广播

        object_start = self.object_start_cls(embed)  # (bsz, seq_l, relation_cnt)
        object_end = self.object_end_cls(embed)  # (bsz, seq_l, relation_cnt)

        return object_start, object_end

    def get_object_for_specific_index(self, indexes: List[Tuple[int, int]], encoded_text: torch.Tensor, mask: torch.Tensor = None):
        """
        通过subject的index来从encoded_text中抽取对应的object和relation
        对于每句话，只能有一个span
        :param indexes: len = current batch size
        :param encoded_text: (cbsz, seq_l, hidden), 其中cbsz是current batch size，该值不一定要与bsz相同
        :param mask: 默认为None，如果提供了，则为(bsz, seq_l)的布尔向量，其中对应pad的位置为True，其他为False。
            如果提供了，就会用于对预测出的start_prob和end_prob进行mask
        :return:
        """
        cbsz = len(indexes)  # assert cbsz == encoded_text.shape[0]
        seq_l = encoded_text.shape[1]
        # 有没有更好的实现？
        mapping = torch.zeros((cbsz, seq_l))  # (cbsz, seq_l)
        mapping = mapping.to(self.bert.device)
        for i, v in enumerate(indexes):
            mapping[i][v[0]] = 0.5
            mapping[i][v[1]] = 0.5
        embed = torch.bmm(mapping.unsqueeze(dim=1), encoded_text)  # (bsz, seq_l, hidden)
        object_start, object_end = torch.sigmoid(self.object_start_cls(embed)), torch.sigmoid(self.object_end_cls(embed))
        # both (bsz, seq_l, relation_cnt)
        return object_start, object_end

    def get_object_for_indexes(self, indexes: List[List[Tuple[int, int]]], encoded_text: torch.Tensor, mask: torch.Tensor = None):
        """
        对batch中不同句子的indexes，分别抽取出object+relation
        对于每句话，可以有多个span
        :param indexes:
        :param encoded_text:
        :param mask: 默认为None，如果提供了，则为(bsz, seq_l)的布尔向量，其中对应pad的位置为True，其他为False。
            如果提供了，就会用于对预测出的start_prob和end_prob进行mask
        :return:
        """
        bsz = len(indexes)

        # encode
        tagged_indexes = []
        start_results, end_results = [], []
        for i, e in enumerate(indexes):
            tagged_indexes.extend(list((i, x[0], x[1]) for x in e))
        # [[(1,2),(2,3)], [(7,8)]] -> [(0,1,2), (0,2,3), (0,7,8)] 加上index
        new_batch_count = ((len(tagged_indexes) - 1) // bsz) + 1
        for i in range(new_batch_count):
            cur_indexes = tagged_indexes[i * bsz: i * bsz + bsz]
            cur_batch_idx = list(x[0] for x in cur_indexes)
            cur_indexes = list((x[1], x[2]) for x in cur_indexes)
            cur_embed = torch.stack(list(encoded_text[x] for x in cur_batch_idx))
            if len(cur_embed.shape) == 1:  # 扩充出一个batch维度
                cur_embed = cur_embed.unsqueeze(0)
            result_start, result_end = self.get_object_for_specific_index(cur_indexes, cur_embed)
            start_results.append(result_start)
            end_results.append(result_end)

        #breakpoint()
        # decode
        start, end = torch.concat(start_results, dim=0), torch.concat(end_results, dim=0)
        # (bsz * cnt, seq_l, relation_cnt)

        index_list = list(x[0] for x in tagged_indexes)
        start_decoded, end_decoded = [], []
        for i in range(bsz):
            r = (bisect_left(index_list, i), bisect_right(index_list, i))
            start_decoded.append(start[r[0]: r[1]] if r[0] != r[1] else None)
            end_decoded.append(end[r[0]: r[1]] if r[0] != r[1] else None)

        return start_decoded, end_decoded

    def find_subject_spans(self, start: torch.Tensor, end: torch.Tensor):
        """
        根据subject的预测结果tensor，抽取出所预测的subject
        :param start: (bsz, seq_l)
        :param end: (bsz, seq_l)
        :return:
        """

        subject_start_mapping, subject_end_mapping = (start > self.hparams['threshold']).int().tolist(), \
                                                 (end > self.hparams[
                                                     'threshold']).int().tolist()  # (bsz, seq_l)
        bsz, seq_l = start.shape
        spans = []
        for i in range(bsz):
            cur_spans = tools.argument_span_determination(subject_start_mapping[i], subject_end_mapping[i],
                                                          start[i].tolist(), end[i].tolist())
            spans.append(cur_spans)
        return spans

    def find_object_spans(self, object_start: torch.Tensor, object_end: torch.Tensor):
        """
        根据object-relation的预测结果tensor，抽取出预测的object-span以及对应的relation-idx
        :param object_start:
        :param object_end: (bsz, seq_l, relation_cnt)
            这里的batch是对应于一句话的，不是数据处理阶段的batch。
            其中每一个(seq_l, rel_cnt)都是一个subject所对应的object和relation的所有预测
        :return:
        """
        if object_start is None:
            return []
        bsz, relation_cnt = object_start.shape[0], object_start.shape[2]
        object_start, object_end = object_start.permute([0, 2, 1]), object_end.permute([0, 2, 1])
        ostart_mapping, oend_mapping = (object_start > self.hparams['threshold']).int().tolist(), \
                                       (object_end > self.hparams['threshold']).int().tolist()
        # both (bsz, relation_cnt, seq_l)
        relation_n_object_s = []
        for i in range(bsz):
            cur_result = []
            for ir in range(relation_cnt):
                cur_spans = tools.argument_span_determination(ostart_mapping, oend_mapping, object_start.tolist(), object_end.tolist())
                if len(cur_spans) != 0:
                    for span in cur_spans:
                        cur_result.append((ir, span))
            relation_n_object_s.append(cur_result)
        return relation_n_object_s  # List[List[(rel_idx, span)]]

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """
        预测的部分要并行化，似乎需要对多个subject进行拆分等处理，总之会复杂一些。
        :param input_ids:
        :param token_type_ids:
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        encoded_text, mask = self.get_encoded_text(input_ids, token_type_ids, attention_mask)  # (bsz, seq_l, hidden)
        subject_start, subject_end = self.get_subjects(encoded_text)  # both (bsz, seq_l, 1)
        subject_start, subject_end = subject_start.squeeze(), subject_end.squeeze()  # both (bsz, seq_l)

        # 找出所有的合法span
        spans = self.find_subject_spans(subject_start, subject_end)  # List[spans in batch]

        # both (bsz, seq_l), 只包含0和1，dtype=torch.int32
        object_start, object_end = self.get_object_for_indexes(spans, encoded_text)

        return {
            'subject_spans': spans,  # List[spans in each batch]
            'object_start': object_start,
            'object_end': object_end,  # both [stacked object-relation prob]，list长度为bsz
            'mask': mask  # (bsz, seq_l)
        }


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
        object_start_label = object_start_label.permute([0, 2, 1])
        object_end_label = object_end_label.permute([0, 2, 1])
        object_start_loss = F.binary_cross_entropy(object_start_result, object_start_label, reduction='none')
        object_end_loss = F.binary_cross_entropy(object_end_result, object_end_label, reduction='none')

        ss_loss = torch.sum(subject_start_loss * subject_mask) / torch.sum(subject_mask)
        se_loss = torch.sum(subject_end_loss * subject_mask) / torch.sum(subject_mask)
        os_loss = torch.sum(object_start_loss * ro_mask) / torch.sum(ro_mask)
        oe_loss = torch.sum(object_end_loss * ro_mask) / torch.sum(ro_mask)

        loss = self.lamb * (ss_loss + se_loss) + (1 - self.lamb) * (os_loss + oe_loss)
        return loss


class CASREL_Loss2(nn.Module):
    def __init__(self, lamb: float = 0.6):
        super(CASREL_Loss2, self).__init__()
        self.lamb = lamb

    def forward(self,
                subject_start_result: torch.Tensor,
                subject_end_result: torch.Tensor,
                object_start_result: torch.Tensor,
                object_end_result: torch.Tensor,
                subject_start_label: torch.Tensor,
                subject_end_label: torch.Tensor,
                object_start_label: torch.Tensor,
                object_end_label: torch.Tensor,
                mask: torch.Tensor = None):
        """

        :param subject_start_result: 模型预测的subject结果
        :param subject_end_result: both (bsz, seq_l, 1)
        :param object_start_result: 模型预测的object-relation结果
        :param object_end_result: both (bsz, seq_l, relation_cnt)
        :param subject_start_label: subject的标签
        :param subject_end_label: both (bsz, seq_l)
        :param object_start_label: object-relation的标签
        :param object_end_label: both (bsz, relation_cnt, seq_l)
        :param mask: 对input_ids的mask,维度为(bsz, seq_l)的布尔向量，其中pad对应位置的值为True，其他为False
        :return:
        """
        relation_cnt = object_start_result.shape[2]

        # 将subject_result的形状与label对齐
        subject_start_result = subject_start_result.squeeze(-1)  # (bsz, seq_l)
        subject_end_result = subject_end_result.squeeze(-1)  # (bsz, seq_l)

        # 计算loss
        #   subject loss
        subject_start_loss = F.binary_cross_entropy(subject_start_result, subject_start_label, reduction='none')
        subject_end_loss = F.binary_cross_entropy(subject_end_result, subject_end_label, reduction='none')
        #       both (bsz, seq_l)
        #   object-relation label的维度转换
        object_start_label = object_start_label.permute([0, 2, 1])
        object_end_label = object_end_label.permute([0, 2, 1])
        #   object-relation loss
        object_start_loss = F.binary_cross_entropy(object_start_result, object_start_label, reduction='none')
        object_end_loss = F.binary_cross_entropy(object_end_result, object_end_label, reduction='none')
        #       both (bsz, seq_l, relation_cnt)

        # 用mask来去除无效loss
        ss_loss = torch.sum(subject_start_loss * mask) / torch.sum(mask)
        se_loss = torch.sum(subject_end_loss * mask) / torch.sum(mask)
        os_loss = torch.sum(object_start_loss * mask.unsqueeze(-1)) / (torch.sum(mask) * relation_cnt)
        oe_loss = torch.sum(object_end_loss * mask.unsqueeze(-1)) / (torch.sum(mask) * relation_cnt)

        loss = self.lamb * (ss_loss + se_loss) + (1 - self.lamb) * (os_loss + oe_loss)
        return loss


class CASREL_Loss_subject(nn.Module):
    def forward(
            self,
            subject_start_result: torch.Tensor,
            subject_end_result: torch.Tensor,
            subject_start_label: torch.Tensor,
            subject_end_label: torch.Tensor,
            mask: torch.Tensor):
        """
        只计算subject部分loss
        :param subject_start_result: 模型预测的subject结果
        :param subject_end_result: both (bsz, seq_l, 1)
        :param subject_start_label: subject的标签
        :param subject_end_label: both (bsz, seq_l)
        :param mask: (bsz, seq_l)的布尔向量，对input_ids进行mask，pad对应位置为True，其余为False
        :return:
        """
        # 将subject_result的形状与label对齐
        subject_start_result = subject_start_result.squeeze(-1)  # (bsz, seq_l)
        subject_end_result = subject_end_result.squeeze(-1)  # (bsz, seq_l)

        # 计算loss
        #   subject loss
        subject_start_loss = F.binary_cross_entropy(subject_start_result, subject_start_label, reduction='none')
        subject_end_loss = F.binary_cross_entropy(subject_end_result, subject_end_label, reduction='none')
        #       both (bsz, seq_l)

        # 用mask来去除无效loss
        ss_loss = torch.sum(subject_start_loss * mask) / torch.sum(mask)
        se_loss = torch.sum(subject_end_loss * mask) / torch.sum(mask)

        loss = ss_loss + se_loss
        return loss


class DuIE_FineTuner(pl.LightningModule):
    def __init__(self, params=None):
        super(DuIE_FineTuner, self).__init__()
        if params is not None:
            self.hparams.update(params)
        self.hparams.update({
            'model_type': 'casrel'
            })
        self.model = CASREL2(params)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.hparams['model_name'])

        if self.hparams['subject_only']:
            self.Loss = CASREL_Loss_subject()
        else:
            self.Loss = CASREL_Loss()

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        # todo 对模型对输出还需要进行一些处理
        return output

    def _step(self, batch):
        inp, tgt = batch
        outputs = self(
            input_ids=inp['input_ids'],
            token_type_ids=inp['token_type_ids'],
            attention_mask=inp['attention_mask'],
            subject_gt_start=inp['subject_gt_start'],
            subject_gt_end=inp['subject_gt_end']
        )
        loss = self.Loss(
            # model output
            subject_start_result=outputs['subject_start_result'],
            subject_end_result=outputs['subject_end_result'],
            object_start_result=outputs['object_start_result'],
            object_end_result=outputs['object_end_result'],

            # preprocessed and label
            subject_mask=outputs['subject_mask'],
            ro_mask=outputs['ro_mask'],
            subject_start_label=tgt['subject_start_label'],
            subject_end_label=tgt['subject_end_label'],
            object_start_label=tgt['object_start_label'],
            object_end_label=tgt['object_end_label']
        )
        return loss

    def training_step(self, batch, batch_idx):
        inp, tgt = batch
        encoded_text, mask = self.model.get_encoded_text(
            input_ids=inp['input_ids'],
            token_type_ids=inp['token_type_ids'],
            attention_mask=inp['attention_mask']
        )  # (bsz, seq_l, hidden)
        subject_start, subject_end = self.model.get_subjects(encoded_text, mask)  # both (bsz, seq_l, 1)
        object_start, object_end = self.model.get_object_for_specific_index(
            indexes=inp['subject_indexes'],
            encoded_text=encoded_text
        )  # (bsz, seq_l, relation_cnt)

        loss = self.Loss(
            subject_start_result=subject_start, subject_end_result=subject_end,
            object_start_result=object_start, object_end_result=object_end,
            subject_start_label=tgt['subject_start_label'], subject_end_label=tgt['subject_end_label'],
            object_start_label=tgt['object_start_label'], object_end_label=tgt['object_end_label'],
            mask=mask
        )

        self.log('loss', float(loss))
        return loss

    # def validation_step(self, batch, batch_ids):
    #     loss = self._step(batch)
    #     return {'val_loss': loss}
    def validation_step(self, batch, batch_ids):
        inp, tgt = batch

        output = self.model(
            input_ids=inp['input_ids'],
            token_type_ids=inp['token_type_ids'],
            attention_mask=inp['attention_mask']
        )
        subject_spans = output['subject_spans']
        object_relation = []
        for i in range(len(subject_spans)):
            cur_ro = self.model.find_object_spans(output['object_start'][i], output['object_end'][i])
            object_relation.append(cur_ro)

        # 都转换为triplet的格式
        # pred_triplets = convert_lists_to_triplet_casrel(pred_subjects, pred_objects)  # List[Triplets]
        pred_triplets = []
        for e_s, e_ro in zip(subject_spans, object_relation):
            # e_s 一个batch中的所有subject
            # e_ro 一个batch中，与subject对应的object
            cur_pred_triplets = []
            for ee_s, ee_ro in zip(e_s, e_ro):
                # ee_s 一个subject
                # ee_ro 与该subject对应的所有object-relation
                for eee_ro in ee_ro:
                    cur_pred_triplets.append((eee_ro[0], ee_s[0], ee_s[1], eee_ro[1][0], eee_ro[1][1]))
            pred_triplets.append(set(cur_pred_triplets))
        gt_triplets = []
        for elem in tgt['gt_triplets']:
            cur_gt_triplets = []
            for e in elem:
                rel = e['relation']
                sub = e['subject_token_span']
                obj = e['object_token_span']
                cur_gt_triplets.append((duie_relations_idx[rel], sub[0], sub[1], obj[0], obj[1]))
            gt_triplets.append(set(cur_gt_triplets))
        return {
            'pred': pred_triplets,
            'gt': gt_triplets
        }

    def validation_epoch_end(self, validation_step_outputs):
        total_outputs = []
        for e in validation_step_outputs:
            total_outputs.extend(e)
        predict, correct, total = 0, 0, 0
        for e in total_outputs:
            predict += len(e['pred'])
            total += len(e['gt'])
            correct += len(e['pred'].intersection(e['gt']))

        precision = correct / predict if predict != 0 else 0
        recall = correct / total if total != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
        self.log_dict({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }, prog_bar=True)

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
        train_dataset = DuIE_CASREL_Dataset(data_type='train', tokenizer=self.tokenizer, overfit=self.hparams['overfit'])
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, collate_fn=casrel_train_collate_fn_2)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = DuIE_CASREL_Dataset('dev', self.tokenizer)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.hparams['eval_batch_size'], collate_fn=casrel_dev_collate_fn_2)
        return val_dataloader


if __name__ == '__main__':
    model = DuIE_FineTuner()
