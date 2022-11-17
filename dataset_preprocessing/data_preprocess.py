import sys
sys.path.append('..')

import json
from loguru import logger
from tqdm import tqdm

from transformers import BertTokenizerFast

from datatransfer.Models.RE.CASREL.preprocess import naive_char_to_token_span
from datatransfer.utils.io_tools import load_jsonl, dump_jsonl
from datatransfer.utils import tokenize_tools, tools


def add_index_sample(d: dict, tokenizer) -> dict:
    text, triplets = d['text'], d['triplets']

    # 1, tokenize
    tokenized = tokenizer(
        d['text'],
        return_offsets_mapping=True)
    token_seq = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    offset_mapping = tokenized['offset_mapping']
    # d['tokens'] = token_seq
    # d.update(tokenized)

    # 2, 找到每个词在句子中的出现
    #   遇到空的词，直接删除
    new_triplets = []
    for elem_triplet in triplets:
        if elem_triplet['subject'] == '' or elem_triplet['object'] == '':
            continue
        sub_occur = tools.get_word_occurrences_in_sentence(text, elem_triplet['subject'])
        obj_occur = tools.get_word_occurrences_in_sentence(text, elem_triplet['object'])
        elem_triplet['subject_occur'] = sub_occur[0]
        elem_triplet['object_occur'] = obj_occur[0]
        elem_triplet['subject_token_span'] = naive_char_to_token_span(elem_triplet['subject_occur'], offset_mapping)
        elem_triplet['object_token_span'] = naive_char_to_token_span(elem_triplet['object_occur'], offset_mapping)
        # if 'other_objects' in elem_triplet:
        #     for k, v in elem_triplet['other_objects']:
        #         v_occur = tools.get_word_occurrences_in_sentence(text, v)
        #         elem_triplet['other_objects'][k + '_occur'] = v_occur[0]
        new_triplets.append(elem_triplet)
    d['triplets'] = new_triplets
    return d




def add_index_to_duie():
    """
    通过tokenizer先进行处理，然后计算出词语的span位置

    - 大于tokenizer max len的部分会被裁剪，因此无法计算
    :return:
    """
    filenames = {
        'train': '../data/processed/duie_train.json',
        'valid': '../data/processed/duie_valid.json'
    }
    train = json.load(open(filenames['train'], 'r', encoding='utf-8'))
    valid = json.load(open(filenames['valid'], 'r', encoding='utf-8'))
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    for t in ['train', 'valid']:
        logger.info(f'正在处理{t}数据集')
        result = []
        for d in tqdm({'train': train, 'valid': valid}[t]):
            r = add_index_sample(d, tokenizer)
            result.append(r)
        dump_jsonl(result, f'../data/processed/duie_indexed_{t}.jsonl')
    return result




if __name__ == '__main__':
    train = add_index_to_duie()
