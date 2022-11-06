"""
该代码负责 预处理好的数据->prompt型数据
prompt型数据大致可以分为：
1，自然语言形式的prompt
2，指令式的prompt
"""

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


def weiboner_to_instruction():
    datas = get_weiboner()
    train = datas['train'] + datas['dev']  # 将训练集和开发集合并
    test = datas['test']

    train_modified, test_modified = [], []
    for e in train: train_modified.extend(weiboner_to_instruction_sample(e))
    for e in test: test_modified.extend(weiboner_to_instruction_sample(e))

    dump_jsonl(train_modified, os.path.join(path_config['prompted_path'], 'ner_weibo_train.jsonl'))
    dump_jsonl(test_modified, os.path.join(path_config['prompted_path'], 'ner_weibo_test.jsonl'))


def duie_to_instruction():
    datas = get_duie()
    train = datas['train']
    dev = datas['dev']

    train_modified, dev_modified = [], []
    for e in train: train_modified.extend(duie_to_instruction_sample(e))
    for e in dev: dev_modified.extend(duie_to_instruction_sample(e))

    dump_jsonl(train_modified, os.path.join(path_config['prompted_path'], 'duie_train.jsonl'))
    dump_jsonl(dev_modified, os.path.join(path_config['prompted_path'], 'duie_dev.jsonl'))

if __name__ == '__main__':
    # weiboner_to_instruction()
    duie_to_instruction()
    print('')