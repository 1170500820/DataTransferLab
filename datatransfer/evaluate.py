"""
实现一些评价方式
"""
import json
from argparse import ArgumentParser
from sklearn import metrics
from typing import List

def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--dataset', '-d', type=str, choices=['duie', 'fewfc'], default='duie')
    parser.add_argument('--prompt_type', '-p', type=str, choices=['find_object', 'find_subject', 'find_relation'], default='find_object')
    parser.add_argument('--metric', '-m', type=str, choices=['f1'], default='f1')

    args = vars(parser.parse_args())
    return args


def load_target(dataset: str = 'duie', data_type: str = 'dev', prompt_type: str = 'find_object'):
    fname = f'../data/prompted/{dataset}_{prompt_type}_{data_type}.jsonl'
    d = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
    targets = list(x['target'] for x in d)
    return targets


def calculate(output: List[str], target: List[str]):
    result = metrics.f1_score(target, output, average='micro')
    print(result)


if __name__ == '__main__':
    config = handle_cli()

    fname = config['file']
    output = open(fname, 'r', encoding='utf-8').read().strip().split('\n')
    target = load_target(config['dataset'], prompt_type=config['prompt_type'])
    calculate(output, target)