"""
该代码负责 预处理好的数据->prompt型数据
prompt型数据大致可以分为：
1，自然语言形式的prompt
2，指令式的prompt
"""
import random
from get_dataset import *
ins_config = yaml.load(open('instruction_config.yml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
from typing import List
from scripts.file_tool import dump_jsonl


def weiboner_to_instruction_sample(sample: dict) -> List[dict]:
    sentence, ner = sample['sentence'], sample['ners']
    if len(ner) == 0:
        return []
    result = []
    for k, v in ner.items():
        label_name = ins_config['ner_weibo'][k]
        target = ','.join(v)
        input_sent = f'命名实体识别|请找出句子中的{label_name}:{sentence}'
        result.append({
            'input': input_sent,
            'target': target
        })
    return result


def duie_to_instruction_sample(sample: dict) -> List[dict]:
    """
    最简单的方案，直接生成三元组格式的数据
    :param sample:
    :return:
    """
    text, triplets = sample['text'], sample['triplets']
    if len(triplets) == 0:
        return []
    triplet_text = []
    for triplet in triplets:
        triplet_text.append(f'{triplet["subject"]}-{triplet["object"]}-{triplet["relation"]}')
    return [{
        'input': text,
        'target': ','.join(triplet_text)
    }]


def duie_to_instruction_find_object(sample: dict) -> List[dict]:
    """
    句子中subject的relation是：
    :param sample:
    :return:
    """
    text, triplets = sample['text'], sample['triplets']
    if len(triplets) == 0:
        return []
    find_object = []
    subject_and_relation = {}
    for triplet in triplets:
        sub_rel = (triplet['subject'], triplet['relation'])
        if sub_rel not in subject_and_relation:
            subject_and_relation[sub_rel] = [triplet['object']]
        else:
            subject_and_relation[sub_rel].append(triplet['object'])
    for k, v in subject_and_relation.items():
        find_object.append({
            'input': f'{text}|在这句话中，{k[0]}的{k[1]}是：',
            'target': ','.join(v)
        })
    return find_object


def duie_to_instruction_find_subject(sample: dict) -> List[dict]:
    """
    句子中什么的relation是object：
    :param sample:
    :return:
    """
    text, triplets = sample['text'], sample['triplets']
    if len(triplets) == 0:
        return []
    find_subject = []
    object_and_relation = {}
    for triplet in triplets:
        ob_rel = (triplet['object'], triplet['relation'])
        if ob_rel not in object_and_relation:
            object_and_relation[ob_rel] = [triplet['subject']]
        else:
            object_and_relation[ob_rel].append(triplet['subject'])
    for k, v in object_and_relation.items():
        find_subject.append({
            'input': f'{text}|在这句话中，什么的{k[1]}是{k[0]}：',
            'target': ','.join(v)
        })
    return find_subject


def duie_to_instruction_find_relation(sample: dict) -> List[dict]:
    """
    句子中subject的什么是object：
    :param sample:
    :return:
    """
    text, triplets = sample['text'], sample['triplets']
    if len(triplets) == 0:
        return []
    find_relation = []
    subject_and_object = {}
    for triplet in triplets:
        sub_ob = (triplet['subject'], triplet['object'])
        if sub_ob not in subject_and_object:
            subject_and_object[sub_ob] = [triplet['relation']]
        else:
            subject_and_object[sub_ob].append(triplet['relation'])
    for k, v in subject_and_object.items():
        find_relation.append({
            'input': f'{text}|在这句话中，{k[0]}与{k[1]}的关系是：',
            'target': ','.join(v)
        })
    return find_relation


def duie_to_instruction_hybrid_find(sample: dict) -> List[dict]:
    """
    句子中subject的relation是：
    :param sample:
    :return:
    """
    choice = random.sample([0, 1, 2], k=1)[0]
    if choice == 0:
        return duie_to_instruction_find_object(sample)
    elif choice == 1:
        return duie_to_instruction_find_subject(sample)
    else:
        return duie_to_instruction_find_relation(sample)

def weiboner_to_instruction():
    datas = get_weiboner()
    train = datas['train'] + datas['dev']  # 将训练集和开发集合并
    test = datas['test']

    train_modified, test_modified = [], []
    for e in train: train_modified.extend(weiboner_to_instruction_sample(e))
    for e in test: test_modified.extend(weiboner_to_instruction_sample(e))

    dump_jsonl(train_modified, os.path.join(path_config['prompted_path'], 'ner_weibo_train.jsonl'))
    dump_jsonl(test_modified, os.path.join(path_config['prompted_path'], 'ner_weibo_test.jsonl'))


def duie_to_instruction(option='instruction'):
    """

    :param option: instruction, find_object, find_subject, find_relation, hybrid_find, find_entity
    :return:
    """
    if option == 'instruction':
        func = duie_to_instruction_sample
    elif option == 'find_object':
        func = duie_to_instruction_find_object
    elif option == 'find_subject':
        func = duie_to_instruction_find_subject
    elif option == 'find_relation':
        func = duie_to_instruction_find_relation
    elif option == 'hybrid_find':
        func = duie_to_instruction_hybrid_find
    else:
        func = duie_to_instruction_sample
    datas = get_duie()
    train = datas['train']
    dev = datas['dev']

    train_modified, dev_modified = [], []
    for e in train: train_modified.extend(func(e))
    for e in dev: dev_modified.extend(func(e))

    dump_jsonl(train_modified, os.path.join(path_config['prompted_path'], f'duie_{option}_train.jsonl'))
    dump_jsonl(dev_modified, os.path.join(path_config['prompted_path'], f'duie_{option}_dev.jsonl'))

if __name__ == '__main__':
    # weiboner_to_instruction()
    duie_to_instruction('find_relation')
    duie_to_instruction('hybrid_find')
    print('')