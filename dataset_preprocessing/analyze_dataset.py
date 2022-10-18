"""
主要是对一些数据集的信息进行统计，
- 标签类别
- 类别比例
- ...
"""

from get_dataset import *


def find_weibo_ner_types():
    result = get_weiboner()
    full_result = result['train'] + result['dev']
    tag_set = set()
    tag_dict = {}
    for e in full_result:
        for k in e['ners'].keys():
            tag_set.add(k)
            if k not in tag_dict: tag_dict[k] = []
            tag_dict[k].extend(e['ners'][k])
    print(tag_set)
    return tag_dict


def find_weibo_ner_property():

    def count_empty(l):
        cnt = 0
        for e in l:
            if len(e['ners']) == 0:
                cnt += 1
        return cnt

    def count_length_range(l):
        minl, maxl = float('inf'), float('-inf')
        for e in l:
            curl = len(e['sentence'])
            minl = min(minl, curl)
            maxl = max(maxl, curl)
        return minl, maxl

    def cal_length_avg(l):
        lengths = []
        for e in l:
            lengths.append(len(e['sentence']))
        return sum(lengths) / len(lengths)

    result = get_weiboner()
    train = result['train']
    dev = result['dev']
    test = result['test']

    for splt in ['train', 'dev', 'test']:
        d = result[splt]
        total, empty_count, lrange, lavg = len(d), count_empty(d), count_length_range(d), cal_length_avg(d)
        print(f'split {splt}: \ntotal: {total}, empty: {empty_count}, range: {lrange}, avg: {lavg}')



if __name__ == '__main__':
    # tag_dict = find_weibo_ner_types()
    find_weibo_ner_property()

