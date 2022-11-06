"""
读取一些数据集
"""

import yaml
path_config = yaml.load(open('path_config.yml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)

import os
import json


def get_weiboner():
    processed_path = path_config['processed_path']
    weiboner_train = os.path.join(processed_path, 'weiboner_train.json')
    weiboner_dev = os.path.join(processed_path, 'weiboner_dev.json')
    weiboner_test = os.path.join(processed_path, 'weiboner_test.json')
    train_result = json.load(open(weiboner_train, 'r', encoding='utf-8'))
    dev_result = json.load(open(weiboner_dev, 'r', encoding='utf-8'))
    test_result = json.load(open(weiboner_test, 'r', encoding='utf-8'))
    return {
        'train': train_result,
        'dev': dev_result,
        'test': test_result
    }


def get_AdvertiseGen():
    pass

def get_duie():
    processed_path = path_config['processed_path']
    duie_train = os.path.join(processed_path, 'duie_train.json')
    duie_valid = os.path.join(processed_path, 'duie_valid.json')

    train_result = json.load(open(duie_train, 'r', encoding='utf-8'))
    valid_result = json.load(open(duie_valid, 'r', encoding='utf-8'))
    return {
        'train': train_result,
        'dev': valid_result
    }
