import sys
sys.path.append('..')

from argparse import ArgumentParser

import json
import csv
from loguru import logger
import rich


def handle_cli():
    parser = ArgumentParser()

    parser.add_argument('--task', '-t', type=str, choices=['tellme', 'merge', 'rename'], default='tellme', help="""
    - tellme 展示文件的格式、行数、包括的键值
    - merge 将同类型文件合并
    - rename 将文件中的一个key值改为另外的名字
    """)
    part_args = vars(parser.parse_known_args()[0])

    if part_args['task'] == 'tellme':
        parser.add_argument('--file', '-f', type=str)
    elif part_args['task'] == 'merge':
        parser.add_argument('--file', '-f', type=str, nargs='+')
        parser.add_argument('--output', '-o', type=str)
    elif part_args['task'] == 'rename':
        parser.add_argument('--file', '-f', type=str)
        parser.add_argument('--origin', type=str)
        parser.add_argument('--new', type=str)


    args = vars(parser.parse_args())
    return args




def tellme(config):
    f = config['file']
    lines = list(open(f, 'r', encoding='utf-8').read().strip().split('\n'))
    cnt = len(lines)
    if lines[0][0] == '{':
        file_type = 'jsonl'
        d0 = json.loads(lines[0])
        keys = list(d0.keys())
        infostr = f"""
    文件名: {f}
    文件类型: {file_type}
    文件行数: {cnt}
    包含的key: {",".join(list(str(x) for x in keys))}
        """
    else:
        file_type = 'csv'
        keys = lines[0].split()
        infostr = f"""
    文件名: {f}
    文件类型: {file_type}
    文件行数：{cnt - 1}
    包含的可以: {",".join(keys)}
        """
    infostr += """
    通过python run_predict.py -t merge -f [file1] [file2] ... [filen]
    将多个同类型文件合并
    或者通过python run_predict.py -t [c2b|b2a|c2a]
    将任务结果转换为上级分类
    """
    logger.info(infostr)


def merge(config):
    """
    合并同类型的文件
    默认提供的是相同类型的文件
    :param config:
    :return:
    """
    filenames = config['file']
    if filenames[0][-5] == 'jsonl':
        ds = []
        for e in filenames:  ds.append(list(json.loads(x) for x in open(e, 'r', encoding='utf-8').read().strip().split('\n')))
        if len(set(len(x) for x in ds)) != 1:  raise Exception(f'文件大小不一致！长度分别为{",".join(str(len(x)) for x in ds)}')
        new_d = []
        for i in range(len(ds[0])):
            origin = ds[0][i]
            for j in range(1, len(ds)):
                for k, v in ds[j][i].items():
                    if k not in origin.keys():
                        origin[k] = v
            new_d.append(origin)
        f = open(config['output'], 'w', encoding='utf-8')
        for e in new_d: f.write(json.dumps(e, ensure_ascii=False) + '\n')
        f.close()



def rename(config):
    f = config['file']
    d = list(json.loads(x) for x in open(f, 'r', encoding='utf-8').read().strip().split('\n'))

    new_d = []
    for e in d:
        e[config['new']] = e.pop(config['origin'])
        new_d.append(e)


    f = open(config['file'], 'w', encoding='utf-8')
    for e in new_d: f.write(json.dumps(e, ensure_ascii=False) + '\n')
    f.close()




if __name__ == '__main__':
    config = handle_cli()

    if config['task'] == 'tellme':
        tellme(config)
    elif config['task'] == 'merge':
        merge(config)
    elif config['task'] == 'rename':
        rename(config)
    else:
        pass