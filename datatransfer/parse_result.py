"""
其实完全可以在generate阶段就完成parse这个操作，但是parse的方法可能会变
所以干脆单独拆出来作为一个部分
"""
from typing import List
from argparse import ArgumentParser
import json


def handle_cli():
   parser = ArgumentParser()

   parser.add_argument('--file', '-f', type=str)
   parser.add_argument('--type', '-t', type=str, choices=['welm', 't5', 'bart'], default='t5')
   parser.add_argument('--output', '-o', type=str)

   args = vars(parser.parse_args())

   return args


# 清理welm输出的句子。
def clean_welm_output(text: str):
    """
    通过简单的规则进行清洗
    - 删除开头与结尾的标点符号
    - 如果有多段标点符号隔开的短语，只保留第一段
    :param text:
    :return:
    """


def clean_welm_output_by_source(text: str, source_text: str):
    """
    对照原句，对抽取结果进行清洗
    :param text:
    :param source_text:
    :return:
    """


def clean_plm_output_sample(text: str, special_tokens: List[str] = None):
    """
    对预训练模型对输出结果进行初步清洗

    - 删除特殊token
    - 删除空格
    :param text:
    :param special_tokens:
    :return:
    """
    if special_tokens is None:
        special_tokens = []

    special = ['[PAD]', '[CLS]', '[SEP]']
    special.extend(special_tokens)

    # 删除特殊token
    for e in special:
        while e in text:
            text = text.replace(e, '')
    # 删除空格
    while ' ' in text:
        text = text.replace(' ', '')
    while '\t' in text:
        text = text.replace('\t', '')
    return text


def clean_plm_output(f: str, o: str):
    d = json.load(open(f, 'r', encoding='utf-8'))
    output, target = d
    result = []
    for e in output:
        r = clean_plm_output_sample(e)
        result.append(r)
    f = open(o, 'w', encoding='utf-8')
    for e in result:
        f.write(e + '\n')
    f.close()


if __name__ == '__main__':
    config = handle_cli()

    if config['type'] in {'t5', 'bart'}:
        clean_plm_output(config['file'], config['output'])
    elif config['type'] in {'welm'}:
        pass
