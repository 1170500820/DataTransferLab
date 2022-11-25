"""
主要是对一些数据集的信息进行统计，
- 标签类别
- 类别比例
- ...
"""
import sys
sys.path.append('..')

from datatransfer.utils import *
from collections import Counter
from get_dataset import *
from tqdm import tqdm
from rich.progress import track
import re
import xml.etree.ElementTree as ET
from datatransfer.temp_utils import print_dict_as_table, load_jsonl, dump_jsonl
from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast


def find_weibo_ner_types():
    result = get_weiboner()
    full_result = result['train'] + result['dev']
    tag_set = set()
    tag_dict = {}
    for e in full_result:
        for k in e['ners'].keys():
            tag_set.add(k)
            if k not in tag_dict: tag_dict[k] = []
            tag_dict[k].extend(e['ners'][k])
    print(tag_set)
    return tag_dict


def find_weibo_ner_property():

    def count_empty(l):
        cnt = 0
        for e in l:
            if len(e['ners']) == 0:
                cnt += 1
        return cnt

    def count_length_range(l):
        minl, maxl = float('inf'), float('-inf')
        for e in l:
            curl = len(e['sentence'])
            minl = min(minl, curl)
            maxl = max(maxl, curl)
        return minl, maxl

    def cal_length_avg(l):
        lengths = []
        for e in l:
            lengths.append(len(e['sentence']))
        return sum(lengths) / len(lengths)

    result = get_weiboner()
    train = result['train']
    dev = result['dev']
    test = result['test']

    for splt in ['train', 'dev', 'test']:
        d = result[splt]
        total, empty_count, lrange, lavg = len(d), count_empty(d), count_length_range(d), cal_length_avg(d)
        print(f'split {splt}: \ntotal: {total}, empty: {empty_count}, range: {lrange}, avg: {lavg}')


def find_AdvertiseGen_content():

    def check_format(s):
        infos = s.split('*')
        for e in infos:
            et, ei = e.split('#')
            if not et or not ei:
                return False
        return True

    dev_path = '../data/AdvertiseGen/dev.json'
    train_path = '../data/AdvertiseGen/train.json'
    dev = list(json.loads(x) for x in open(dev_path, 'r', encoding='utf-8').read().strip().split('\n'))
    train = list(json.loads(x) for x in open(train_path, 'r', encoding='utf-8').read().strip().split('\n'))
    contents = []
    for e in dev + train:  contents.append(e['content'])
    print('opened')

    # s = set()
    # for e in contents: s.add(check_format(e))
    # print(s)
    content_dict = {}
    for e in tqdm(contents):
        # infos = e.split('*#')
        infos = re.split('[*|#]', e)
        for i in range(len(infos) // 2):
            et, ei = infos[2 * i], infos[2 * i + 1]
            if et not in contents:
                content_dict[et] = [ei]
            else:
                content_dict[et].append(ei)
    print('parsed')
    content_count = {}
    for k, v in tqdm(content_dict.items()):
        content_count[k] = Counter(v)
    print('counted')
    return content_count


def parse_CEC_XML():
    cec_path = '../data/CEC/CEC-Corpus-master/CEC/交通事故/17岁少女殒命搅拌车下.xml'
    tree = ET.parse(cec_path)
    return tree


def clear_tokenized_result(text):
    while '##' in text:
        text = text.replace('##', '')
    return text


def clean_text(text):
    while ' ' in text:
        text = text.replace(' ', '')
    return text


def check_duie_indexed():
    train_file = '../data/processed/duie_indexed_train.jsonl'
    valid_file = '../data/processed/duie_indexed_valid.jsonl'
    tokenizer, tokenizerfast = BertTokenizer.from_pretrained('bert-base-chinese'), \
                               BertTokenizerFast.from_pretrained('bert-base-chinese')
    train_data, valid_data = load_jsonl(train_file), load_jsonl(valid_file)

    example = {
        "text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下",
        "triplets":
            [
                {
                    "subject": "邪少兵王",
                    "object": "冰火未央",
                    "relation": "作者",
                    "other_objects": {"@value": "冰火未央"},
                    "subject_occur": [1, 4],
                    "object_occur": [7, 10],
                    "subject_token_span": [2, 5],
                    "object_token_span": [8, 11]
                }
            ]
    }
    result = []
    for data_type in ['train', 'valid']:
        d = {'train': train_data, 'valid': valid_data}[data_type]
        logger.info(f'正在检查{data_type}数据')
        for i, e in enumerate(track(d)):
            text, triplets = e['text'], e['triplets']
            tokenized = tokenizer(text)
            fasttokenized = tokenizerfast(text)
            tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
            fasttokens = tokenizerfast.convert_ids_to_tokens(fasttokenized['input_ids'])
            cur_results = []
            for it, triplet in enumerate(triplets):
                subject, soccur, sspan = triplet['subject'], triplet['subject_occur'], triplet['subject_token_span']
                object, ooccur, ospan = triplet['object'], triplet['object_occur'], triplet['object_token_span']
                sentence_subject = text[soccur[0]: soccur[1] + 1]
                sentence_object = text[ooccur[0]: ooccur[1] + 1]
                sentence_subject, sentence_object = clean_text(sentence_subject), clean_text(sentence_object)
                token_subject = ''.join(tokens[sspan[0]: sspan[1] + 1])
                token_object = ''.join(tokens[ospan[0]: ospan[1] + 1])
                fast_token_subject = ''.join(fasttokens[sspan[0]: sspan[1] + 1])
                fast_token_object = ''.join(fasttokens[ospan[0]: ospan[1] + 1])
                token_subject, token_object = clear_tokenized_result(token_subject), clear_tokenized_result(token_object)
                fast_token_subject, fast_token_object = clear_tokenized_result(fast_token_subject), \
                                                        clear_tokenized_result(fast_token_object)
                if not subject == sentence_subject == token_subject == fast_token_subject or not object == sentence_object == token_object == fast_token_object:
                    # logger.warning(f'inequal at {data_type}-{i}-{it}')
                    cur_results.append({
                        'data_type': data_type,
                        'index': i,
                        'triplet_index': it,
                        'words': [(subject, sentence_subject, token_subject, fast_token_subject),
                                  (object, sentence_object, token_object, fast_token_object)]
                    })
            result.extend(cur_results)

    dump_jsonl(result, 'duie_indexed_check_result.jsonl')


def analyze_duie_indexed():
    """
    对duie——indexed文件的信息进行统计
    :return:
    """
    train_file = '../data/processed/duie_indexed_train.jsonl'
    valid_file = '../data/processed/duie_indexed_valid.jsonl'
    train_data, valid_data = load_jsonl(train_file), load_jsonl(valid_file)
    for data_type in ['train', 'valid']:
        d = {'train': train_data, 'valid': valid_data}[data_type]
        logger.info(f'{data_type}的样本数为:{len(d)}')
        cnt = 0
        for e in track(d):
            triplets = e['triplets']
            cnt += len(triplets)
        logger.info(f'{data_type}中的总triplets个数为:{cnt}')

def show_processed_dataset():
    files = os.listdir('../data/processed/')
    counts = {}
    for e in tqdm(files):
        d = json.load(open(os.path.join('../data/processed/', e), 'r', encoding='utf-8'))
        counts[e] = len(d)
    print_dict_as_table(counts)


def show_prompted_dataset():
    """
    如果报错了，尝试在终端输入rm .DS_store
    :return:
    """
    files = os.listdir('../data/prompted/')
    counts = {}
    for e in tqdm(files):
        if '.cache' in e:
            continue
        d = list(json.loads(x) for x in open(os.path.join('../data/prompted', e), 'r', encoding='utf-8').read().strip().split('\n'))
        counts[e] = len(d)
    print_dict_as_table(counts)


if __name__ == '__main__':
    # tag_dict = find_weibo_ner_types()
    # find_weibo_ner_property()
    # ct = find_AdvertiseGen_content()
    # tree = parse_CEC_XML()
    # show_processed_dataset()
    # show_prompted_dataset()
    # check_duie_indexed()
    analyze_duie_indexed()