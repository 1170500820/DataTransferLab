"""
关系抽取数据集
"""
import sys
sys.path.append('..')

import os, glob, random, re, json, copy
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
from bisect import bisect
from rich.progress import track

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, BertTokenizerFast, BertTokenizer

from datatransfer.utils import io_tools, tools, batch_tool, tensor_tool
from datatransfer.Models.RE.RE_settings import duie_relations_idx


def rearrange_by_subject(data_dicts: List[Dict[str, Any]]):
    """
    按照subject重组
    :param data_dicts:
    :return:
    """
    new_data_dicts = []
    for d in data_dicts:
        cur_data_dicts = []
        triplets_dict, common_dict = tools.split_dict(d, ['triplets'])
        # triplets_dict只包含triplets，而common_dict包含text、input_ids等剩下的字段

        subject_dict = {}
        subject_related_dict = {}
        # subject_dict用于存放subject所对应的token_span等信息
        # 而subject_related_dict则用于存放对应的relation-object pair list
        for elem_triplet in triplets_dict['triplets']:
            subject = elem_triplet['subject']
            if subject in subject_dict:
                subject_related_dict[subject].append(tools.split_dict(elem_triplet, ['subject', 'subject_occur', 'subject_token_span'])[1])
            else:
                subject_part, relation_n_object_part = tools.split_dict(elem_triplet, ['subject', 'subject_occur', 'subject_token_span'])
                subject_dict[subject] = subject_part
                subject_related_dict[subject] = [relation_n_object_part]

        for subject, subject_value in subject_dict.items():
            relation_n_objects = subject_related_dict[subject]
            common_dict_copy = copy.deepcopy(common_dict)
            # text, input_ids, subject_label, ...
            common_dict_copy.update(subject_value)
            common_dict_copy['relation_object'] = relation_n_objects

            cur_data_dicts.append(common_dict_copy)

        # 为当前句子的每个subject，都将所有subject信息重复一遍
        all_subjects = []
        for e in cur_data_dicts:
            all_subjects.append({
                'subject': e['subject'],
                'subject_occur': copy.deepcopy(e['subject_occur']),
                'subject_span': copy.deepcopy(e['subject_token_span'])
            })
        for i in range(len(cur_data_dicts)):
            cur_data_dicts[i]['all_subjects'] = all_subjects
        new_data_dicts.extend(cur_data_dicts)
    return new_data_dicts


class DuIE_CASREL_Dataset(Dataset):
    def __init__(self, data_type: str, tokenizer, overfit: bool = False):
        if data_type == 'dev':
            data_type = 'valid'
        self.data_type = data_type
        fname = f'../data/processed/duie_indexed_{data_type}.jsonl'
        self.raw_file = io_tools.load_jsonl(fname)
        if overfit:
            self.raw_file = self.raw_file[:200]
        logger.info(f'DuIE_Dataset正在为{data_type}数据集进行tokenize')
        for e in self.raw_file:
            tokenized = tokenizer(e['text'], return_offsets_mapping=True)
            e.update({
                'input_ids': tokenized['input_ids'],
                'token_type_ids': tokenized['token_type_ids'],
                'attention_mask': tokenized['attention_mask'],
                'tokens': tokenizer.convert_ids_to_tokens(tokenized['input_ids']),
                'offset_mapping': tokenized['offset_mapping']
            })
        if self.data_type == 'train':
            self.expanded = rearrange_by_subject(self.raw_file)
        else:
            pass

    def __len__(self):
        if self.data_type == 'train':
            return len(self.expanded)
        else:
            return len(self.raw_file)

    def __getitem__(self, index):
        if self.data_type == 'train':
            return self.expanded[index]
        else:
            return self.raw_file[index]


def casrel_collate_fn(lst, padding=256):
    data_dict = tools.transpose_list_of_dict(lst)
    bsz = len(lst)

    # basic input
    input_ids = batch_tool.batchify_with_padding(data_dict['input_ids'], padding=padding).to(torch.long)
    token_type_ids = batch_tool.batchify_with_padding(data_dict['token_type_ids'], padding=padding).to(torch.long)
    attention_mask = batch_tool.batchify_with_padding(data_dict['attention_mask'], padding=padding).to(torch.long)
    max_length = input_ids.shape[1]
    # bsz, max_length

    # subject gt
    start_tensors, end_tensors = [], []
    for e in data_dict['all_subjects']:
        start_tensor, end_tensor = torch.zeros(max_length), torch.zeros(max_length)  # max_seq_l
        for e_subject in e:
            start_tensor[e_subject['subject_span'][0]] = 1
            end_tensor[e_subject['subject_span'][1]] = 1
        start_tensors.append(start_tensor)
        end_tensors.append(end_tensor)
    subject_gt_start = torch.stack(start_tensors)
    subject_gt_end = torch.stack(end_tensors)
    # both (bsz, max_seq_l)

    # subject label
    start_indexes, end_indexes = [], []
    for i, e in enumerate(data_dict['subject_token_span']):
        start_indexes.append([i, e[0]])
        end_indexes.append([i, e[1]])
    subject_label_start = tensor_tool.generate_label([bsz, max_length], start_indexes).to(torch.float)
    subject_label_end = tensor_tool.generate_label([bsz, max_length], end_indexes).to(torch.float)

    # object-relation label based on current subject
    rel_start_indexes, rel_end_indexes = [], []
    for i, e in enumerate(data_dict['relation_object']):
        for e_subrel in e:
            rel_idx = duie_relations_idx[e_subrel['relation']]
            rel_start_indexes.append([i, rel_idx, e_subrel['object_token_span'][0]])
            rel_end_indexes.append([i, rel_idx, e_subrel['object_token_span'][1]])
    relation_label_start = tensor_tool.generate_label([bsz, len(duie_relations_idx), max_length], rel_start_indexes).to(torch.float)
    relation_label_end = tensor_tool.generate_label([bsz, len(duie_relations_idx), max_length], rel_end_indexes).to(torch.float)
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'subject_gt_start': subject_gt_start,
        'subject_gt_end': subject_gt_end
           }, {
        'subject_start_label': subject_label_start,
        'subject_end_label': subject_label_end,
        'object_start_label': relation_label_start,
        'object_end_label': relation_label_end
    }


def casrel_dev_collate_fn(lst):
    data_dict = tools.transpose_list_of_dict(lst)

    # generate basic input
    input_ids = torch.tensor(data_dict['input_ids'][0], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(data_dict['token_type_ids'][0], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(data_dict['attention_mask'][0], dtype=torch.long).unsqueeze(0)
    # all (1, seq_l)

    gt_triplets = data_dict['triplets'][0]
    tokens = data_dict['tokens'][0]

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }, {
        'gt_triplets': gt_triplets,
        'tokens': tokens,
        'offset_mapping': data_dict['offset_mapping']
    }



if __name__ == '__main__':
    logger.info('正在加载tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    logger.info('正在加载数据集')
    d = DuIE_CASREL_Dataset('dev', tokenizer)
    # dataloader = DataLoader(d, batch_size=4, shuffle=False, collate_fn=casrel_collate_fn)
    logger.info(f'数据集的大小:{str(len(d))}, 扩展前大小:{str(len(d.raw_file))}')
