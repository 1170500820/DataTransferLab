"""
目的是能够快速地通过命令行判断一个文件的内容，而不需要读取文件来判断
"""
from argparse import ArgumentParser
from loguru import logger
import json


def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--task', '-t', type=str, choices=['cnt', 'key'], default='cnt')
    parser.add_argument('--file', '-f', type=str)

    args = vars(parser.parse_args())
    return args


def count_lines(fname: str):
    lines = open(fname, 'r', encoding='utf-8').readlines()
    logger.info(f'文件：[{fname}]共包含{str(len(lines))}条数据')


def print_key(fname: str):
    """

    :param fname: jsonl或者json格式的文件，包含list of dict
    :return:
    """
    if fname[-5:] == '.json':
        d = json.load(open(fname, 'r', encoding='utf-8'))
        logger.info(f'文件：[{fname}]包含的key为{str(list(d[0].keys()))}')
    else:
        d = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
        logger.info(f'文件：[{fname}]包含的key为{str(list(d[0].keys()))}')



if __name__ == '__main__':
    config = handle_cli()
    if config['task'] == 'cnt':
        count_lines(config['file'])
    elif config['task'] == 'key':
        print_key(config['file'])