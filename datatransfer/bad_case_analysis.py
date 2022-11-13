"""
对预测结果进行坏例分析
"""
import sys
sys.path.append('..')

import pandas as pd
import json


def main():
    ftarget = '../data/prompted/duie_find_object_dev.tmp.jsonl'
    foutput = 'bart.epoch=2.result.txt'

    targets = list(json.loads(x) for x in open(ftarget, 'r', encoding='utf-8').read().strip().split('\n'))
    outputs = open(foutput, 'r', encoding='utf-8').read().strip().split('\n')
    for i in range(len(targets)):
        targets[i]['output'] = outputs[i]

    # 开始用pandas进行游戏
    frame = pd.DataFrame(targets)
    frame.info()  # 打印出基本信息

    return frame



if __name__ == '__main__':
    frame = main()
