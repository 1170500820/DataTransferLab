import os
import json
from rich.console import Console
from rich.table import Column, Table


"""
目录与文件操作
"""


def create_new_dir_if_not_exist(path_name: str):
    """
    如果目录不存在，则创建新目录
    :param path_name:
    :return:
    """
    if os.path.isdir(path_name):
        return
    else:
        os.mkdir(path_name)
        return


def load_jsonl(datapath: str):
    d = list(json.loads(x) for x in open(datapath, 'r', encoding='utf-8').read().strip().split('\n'))
    return d


def dump_jsonl(data: list, datapath: str):
    f = open(datapath, 'w', encoding='utf-8')
    for elem in data:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


def load_json(datapath: str, key: str = None):
    d = json.load(open(datapath, 'r', encoding='utf-8'))
    if key is not None:
        return d[key]
    return d


def load_tsv(datapath: str, first_line_tag: bool = True):
    """

    :param datapath:
    :param first_line_tag: 第一行是否为标签信息
    :return:
    """
    lines = open(datapath, 'r', encoding='utf-8').read().strip().split('\n')
    results = []
    if first_line_tag:
        tags = lines[0].split('\t')
        for elem in lines[1:]:
            parts = elem.split('\t')
            d = {x[0]: x[1] for x in zip(tags, parts)}
            results.append(d)
        return results
    else:
        for elem in lines:
            results.append(elem.split('\t'))
        return results


def print_dict_as_table(d: dict):
    console = Console()
    table = Table(show_header=True, header_style='bold magenta')
    table.add_column('key', style='dim')
    table.add_column('value')
    kv = sorted(list(d.items()))
    for (k, v) in kv:
        table.add_row(str(k), str(v))
    table.add_row(
        'Total', f'{sum(list(x[1] for x in kv))}'
    )
    console.print(table)

