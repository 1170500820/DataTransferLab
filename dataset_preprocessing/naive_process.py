import os
from typing import List, Dict, Tuple, Any
import json
from tqdm import tqdm

dataset_path = '../data'
weibo_dataset_path = 'Weibo'
weibo_ner_dev = 'simple_weiboNER_2nd_conll.dev'
weibo_ner_train = 'simple_weiboNER_2nd_conll.train'
weibo_ner_test = 'simple_weiboNER_2nd_conll.test'
fewfc_path = 'FewFC-main'
duee_path = 'duee'
duie_path = 'duie'


def check_BIO_string(BIO_string: List[str]):
    """
    检查一个BIO的strlist是否合法。如果合法，则直接返回。否则报错

    - 长度为0是合法的
    - I-type前面要么是B-type，要么是I-type，否则非法
    :param BIO_string:
    :return:
    """
    if len(BIO_string) == 0:
        return True
    last = 'O'
    for idx, elem in enumerate(BIO_string):
        if elem[0] == 'I':
            if last == 'B' + elem[1:] or last == elem:
                last = elem
                continue
            else:
                # raise Exception(f'[check_BIO_string]非法的I位置:{idx}！')
                return False
        last = elem
    return True


def BIO_to_spandict(BIO_string: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """
    将BIO格式标注，转化为span的dict。
    不会检查BIO_string合法性。如果BIO_string存在非法标注，抛出异常

    其中BIO格式：
        如果词长为1，则为B-type
        词长>=2，则为 B-type, I-type, ...
        否则为O

    span格式：
        第一个字的坐标为0
        start为该词第一个字的坐标
        end为该词右侧的第一个字的坐标
    :param BIO_string:
    :return:
    """
    check_BIO_string(BIO_string)
    tag_types = list(set(x[2:] for x in set(BIO_string) - {'O'}))
    spandict = {x: [] for x in tag_types}  # Dict[type name, list of span tuple]

    # 根据每一个tag以及其上一个tag，判断该tag是否是标注内
    # 外->内 内作为start ,内->外 外作为end
    # 例外，若最后一个仍在标注内，end=len(BIO_string)
    last_tag = 'O'
    span_list = []
    for idx, tag in enumerate(BIO_string):
        if tag[0] == 'B':
            if last_tag[0] != 'O':
                span_list.append(idx)
                cur_tag_type = last_tag[2:]
                spandict[cur_tag_type].append(tuple(span_list))
                span_list = []
            span_list.append(idx)
            continue
        elif last_tag != 'O' and tag == 'O':
            span_list.append(idx)
            cur_tag_type = last_tag[2:]
            spandict[cur_tag_type].append(tuple(span_list))
            span_list = []
        last_tag = tag
    if len(span_list) == 1:
        span_list.append(len(BIO_string))
        cur_tag_type = last_tag[2:]
        spandict[cur_tag_type].append(tuple(span_list))
        span_list = []

    return spandict


# 上面的BIO_to_spandict有bug，代码逻辑也比较混乱，下面是一个重写的版本
def BIO2span(s: List[str]):
    status, temp, postfix = 0, -1, ''
    d = {}
    for i, c in enumerate(s):
        if c == 'O':
            if status == 0:
                pass
            elif status == 1 or status == 2:
                d[postfix].append((temp, i))
            status = 0
        elif c[0] == 'B':
            if status == 0:
                temp, postfix = i, c[2:]
                if postfix not in d: d[postfix] = []
            elif status == 1 or status == 2:
                temp, postfix = i, c[2:]
                if postfix not in d: d[postfix] = []
                d[postfix].append((temp, i))
                temp, postfix = i, c[2:]
            status = 1
        else:  # c[0] == 'I'
            if status == 0: raise Exception('[BIO2span]由0状态不可能直接遇到I开头的标签')
            status = 2
    return d


def conll2dict(lines):
    """
    默认不同的标注序列由空格来切分
    默认标注格式为BIO
    :param lines:
    :return:
    """
    # 首先去除开头和结尾的空格
    while lines and lines[0] == '': lines = lines[1:]
    while lines and lines[-1] == '': lines = lines[:-1]

    # 读取出每个句子
    sentences, taggings = [], []
    temp_sent, temp_tag = [], []
    for e in lines:
        if e == '':
            sentences.append(''.join(temp_sent))
            taggings.append(temp_tag.copy())
            temp_sent, temp_tag = [], []
        else:
            token, tag = e.strip().split()
            temp_sent.append(token)
            temp_tag.append(tag)
    result = []
    for s, t in zip(sentences, taggings):
        if check_BIO_string(t):
            spandict = BIO2span(t)
            new_dict = {}
            for k, v in spandict.items():
                new_dict[k] = []
                for e in v:
                    new_dict[k].append(s[e[0]: e[1]])
            result.append({
                'sentence': s,
                'ners': new_dict
            })

    return result


def convert_jsonl_to_json(name, d: Dict[str, str]):
    """
    :param d: {'train': path}
    :return:
    """
    for key, value in d.items():
        lst = list(json.loads(x) for x in open(value, 'r', encoding='utf-8').read().strip().split('\n'))
        json.dump(lst, open(f'../data/processed/{name}-{key}.json', 'w', encoding='utf-8'), ensure_ascii=False)


def convert_CEC_XML(tree):
    root = tree.getroot()
    node = list(root)


def process_weibo():
    train_filename = os.path.join(dataset_path, weibo_dataset_path, weibo_ner_train)
    dev_filename = os.path.join(dataset_path, weibo_dataset_path, weibo_ner_dev)
    test_filename = os.path.join(dataset_path, weibo_dataset_path, weibo_ner_test)
    train_lines = list(open(train_filename).read().strip().split('\n'))
    dev_lines = list(open(dev_filename).read().strip().split('\n'))
    test_lines = list(open(test_filename).read().strip().split('\n'))
    train_result = conll2dict(train_lines)
    dev_result = conll2dict(dev_lines)
    test_result = conll2dict(test_lines)

    json.dump(train_result, open('../data/processed/weiboner_train.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(dev_result, open('../data/processed/weiboner_dev.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(test_result, open('../data/processed/weiboner_test.json', 'w', encoding='utf-8'), ensure_ascii=False)


def load_Duee_ee(file_dir: str):
    if file_dir[-1] != '/':
        file_dir += '/'
    test_file = 'duee_test2.json/duee_test2.json'
    train_file = 'duee_train.json/duee_train.json'
    valid_file = 'duee_dev.json/duee_dev.json'
    test_data = list(json.loads(x) for x in open(file_dir + test_file, 'r', encoding='utf-8').read().strip().split('\n'))
    train_data = list(json.loads(x) for x in open(file_dir + train_file, 'r', encoding='utf-8').read().strip().split('\n'))
    val_data = list(json.loads(x) for x in open(file_dir + valid_file, 'r', encoding='utf-8').read().strip().split('\n'))
    return {
        "train": train_data,
        "test": test_data,
        "valid": val_data
    }


def load_FewFC_ee(file_dir: str, splitted=True):
    if file_dir[-1] != '/':
        file_dir += '/'
    if splitted:
        test_file = 'test.json'
        train_file = 'AFQMC_train.json'
        val_file = 'val.json'
        test_data = list(json.loads(x) for x in open(file_dir + test_file, 'r', encoding='utf-8').read().strip().split('\n'))
        train_data = list(json.loads(x) for x in open(file_dir + train_file, 'r', encoding='utf-8').read().strip().split('\n'))
        val_data = list(json.loads(x) for x in open(file_dir + val_file, 'r', encoding='utf-8').read().strip().split('\n'))
        return {
            "train": train_data,
            "test": test_data,
            "valid": val_data
        }
    else:
        raise NotImplementedError


def convert_Duee_to_FewFC_format_sample(data_dict: Dict[str, Any]):
    """
    将一个Duee的sample的key值转化为FewFC格式，对于FewFC中不包含的key值则直接忽略
    :param data_dict:
    :return:
    """
    new_dict = {}
    new_dict['content'] = data_dict['text']
    new_dict['id'] = data_dict['id']
    if 'event_list' not in data_dict:
        return new_dict
    events = []
    for elem_event in data_dict['event_list']:
        # 初始化一个dict，作为FewFC格式的一个event
        # FewFC格式的一个event dict，包括type与mentions
        new_event = {}

        # 装载 type
        new_event['type'] = elem_event['event_type']

        # 装载 mentions
        # 先构造trigger，然后放入mentions列表
        trigger_span = [elem_event['trigger_start_index'], elem_event['trigger_start_index'] + len(elem_event['trigger'])]
        new_mentions = [{
            "word": elem_event['trigger'],
            "span": trigger_span,
            "role": 'trigger'
        }]
        # 接下来构造每一个argument，然后放入mentions列表
        for elem_arg in elem_event['arguments']:
            arg = elem_arg['argument']
            role = elem_arg['role']
            span = [elem_arg['argument_start_index'], elem_arg['argument_start_index'] + len(arg)]
            new_mentions.append({
                "word": arg,
                'span': span,
                'role': role,
            })
        new_event['mentions'] = new_mentions
        events.append(new_event)
    new_dict['events'] = events
    return new_dict


def convert_Duee_to_FewFC_format(duee_dicts: Dict[str, List[Dict[str, Any]]]):
    """
    Duee的每个dict的key与FewFC不同，这里直接改成FewFC的形式
    对于FewFC中不包含的key，直接忽略
    :param data_dicts:
    :return:
    """
    new_duee_dicts = {}
    for elem_dataset_type in ['train', 'valid', 'test']:
        new_data_dicts = []
        for elem_dict in duee_dicts[elem_dataset_type]:
            new_data_dicts.append(convert_Duee_to_FewFC_format_sample(elem_dict))
        new_duee_dicts[elem_dataset_type] = new_data_dicts
    return new_duee_dicts


def split_dict(d: dict, key_lst: List[str], keep_origin=False):
    """
    将一个dict分成两个，一个是key_lst中的key组成的，另一个是不包含在key_lst中的
    :param d:
    :param key_lst:
    :param keep_origin: 是否保留原dict，如果False，那么会真的把原dict拆开
    :return:
    """

    new_dict = {}
    for elem_key in key_lst:
        if elem_key not in d:
            raise Exception(f'[split_dict]{elem_key}不在输入的dict当中！')
        if keep_origin:
            new_dict[elem_key] = d[elem_key]
        else:
            new_dict[elem_key] = d.pop(elem_key)
    return new_dict, d


def process_duee_and_fewfc():
    """
    生成duee与fewfc的数据
    :return:
    """
    fewfc_dir = os.path.join(dataset_path, fewfc_path)
    duee_dir = os.path.join(dataset_path, duee_path)
    duee_dicts = load_Duee_ee(duee_dir)
    fewfc_dicts = load_FewFC_ee(fewfc_dir)
    converted_duee = convert_Duee_to_FewFC_format(duee_dicts)

    # FewFC
    #   train 7419
    #   valid 927
    #   test 928
    # DuEE
    #   train 11908
    #   valid 1492
    #   test 34904 (无标签)
    fewfc_train, fewfc_valid = fewfc_dicts['train'], fewfc_dicts['valid']
    duee_train, duee_valid = converted_duee['train'], converted_duee['valid']
    json.dump(fewfc_train, open('../data/processed/fewfc_train.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(fewfc_valid, open('../data/processed/fewfc_valid.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(duee_train, open('../data/processed/duee_train.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(duee_valid, open('../data/processed/duee_valid.json', 'w', encoding='utf-8'), ensure_ascii=False)


def load_duie_re(file_dir: str):
    """
    与WebNLG和NYT相比，duie有两点不同
    1，duie的test数据不包含标签。由于duie是一个仍在进行的比赛的数据集，因此test部分没有提供标签。本函数也会读取，不够triplets字段就为空了
    2，包含复杂object。duie的object部分有时候会包含多个，比如关系"票房"的object包括一个object本体和一个"inArea"
    {
    'text': 原句，
    'triplets': [
            {
            "subject": str,
            'object': str,
            'relation': str,
            'other_objects': {
                "type": str
            }
            }, ...
        ] or None
    }
    :param file_dir:
    :return:
    """
    if file_dir[-1] != '/':
        file_dir += '/'
    filenames = {
        "valid": 'duie_dev.json/duie_dev.json',
        "train": 'duie_train.json/duie_train.json',
        "test": "duie_test2.json/duie_test2.json"
    }
    loaded = {}
    for k, v in filenames.items():
        data_lst = []
        dicts = list(map(json.loads, open(file_dir + v, 'r').read().strip().split('\n')))
        if k == 'test':
            for elem in dicts:
                text = elem['text']
                text = text.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                data_lst.append({
                    "text": text,
                    "triplets": None
                })
            loaded[k] = data_lst
        else:
            for elem in dicts:
                triplets = []  # 当前句子所对应的所有triplet
                text = elem['text']
                text = text.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                relations = elem['spo_list']
                for elem_rel in relations:
                    sub, obj, rel = elem_rel['subject'], elem_rel['object']['@value'], elem_rel['predicate']
                    sub = sub.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                    obj = obj.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                    _, other_objects = split_dict(elem_rel['object'], ['@value'], keep_origin=True)
                    for elem_k in other_objects.keys():
                        other_objects[elem_k] = other_objects[elem_k].replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                    triplets.append({
                        "subject": sub,
                        "object": obj,
                        "relation": rel,
                        "other_objects": other_objects
                    })
                data_lst.append({
                    "text": text,
                    "triplets": triplets
                })
            loaded[k] = data_lst
    return loaded


def process_duie():
    duie_dir = os.path.join(dataset_path, duie_path)
    duie_result = load_duie_re(duie_dir)
    duie_train, duie_valid = duie_result['train'], duie_result['valid']
    json.dump(duie_train, open('../data/processed/duie_train.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(duie_valid, open('../data/processed/duie_valid.json', 'w', encoding='utf-8'), ensure_ascii=False)


def parse_AdvertiseGen():
    dev_path = '../data/AdvertiseGen/dev.json'
    train_path = '../data/AdvertiseGen/train.json'

    dev = list(json.loads(x) for x in open(dev_path, 'r', encoding='utf-8').read().strip().split('\n'))
    train = list(json.loads(x) for x in open(train_path, 'r', encoding='utf-8').read().strip().split('\n'))

    for dataname, dataset in zip(['dev', 'train'], [dev, train]):
        print(f'processing {dataname}')
        result = []
        for e in tqdm(dataset):
            contentd = {}
            content = e['content']
            for einfo in content.split('*'):
                et, ei = einfo.split('#')
                if et not in contentd:
                    contentd[et] = [ei]
                else:
                    contentd[et].append(ei)
            result.append({
                'content': contentd,
                'summary': e['summary']
            })
        json.dump(result, open(f'../data/processed/AdvertiseGen_{dataname}.json', 'w', encoding='utf-8'), ensure_ascii=False)


def process_AFQMC():
    dev_path = '../data/AFQMC/dev.json'
    train_path = '../data/AFQMC/train.json'
    dev = list(json.loads(x) for x in open(dev_path).read().strip().split('\n'))
    train = list(json.loads(x) for x in open(train_path).read().strip().split('\n'))

    json.dump(dev, open('../data/processed/AFQMC_dev.json', 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(train, open('../data/processed/AFQMC_train.json', 'w', encoding='utf-8'), ensure_ascii=False)


def process_C3():
    convert_jsonl_to_json('C3d', {
        'train': '../data/C3/new_d-train.json',
        'dev': '../data/C3/new_d-dev.json'
    })
    convert_jsonl_to_json('C3m', {
        'train': '../data/C3/new_m-train.json',
        'dev': '../data/C3/new_m-dev.json'
    })


def process_CCPM():
    convert_jsonl_to_json('CCPM', {
        'train': '../data/CCPM/train.jsonl',
        'dev': '../data/CCPM/dev.jsonl'
    })


def process_CHID():
    convert_jsonl_to_json('CHID', {
        'train': '../data/CHID/new_train.json',
        'dev': '../data/CHID/new_dev.json'
    })


def process_CLUENER():
    convert_jsonl_to_json('CLUENER', {
        'train': '../data/CLUENER/new_train.json',
        'dev': '../data/CLUENER/new_dev.json'
    })


def process_CMeEE():
    convert_jsonl_to_json('CMeEE', {
        'train': '../data/CMeEE/train.jsonl',
        'dev': '../data/CMeEE/dev.jsonl'
    })


def process_CMNLI():
    convert_jsonl_to_json('CMNLI', {
        'train': '../data/CMNLI/train.json',
        'dev': '../data/CMNLI/dev.json'
    })





if __name__ == '__main__':
    # process_weibo()
    # process_duee_and_fewfc()
    # process_duie()
    # parse_AdvertiseGen()
    process_AFQMC()
    process_C3()
    process_CCPM()
    process_CHID()
    process_CLUENER()
    process_CMeEE()
    process_CMNLI()
