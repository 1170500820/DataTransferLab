"""
主要是对一些数据集的信息进行统计，
- 标签类别
- 类别比例
- ...
"""
from datatransfer.utils import *
from collections import Counter
from get_dataset import *
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET


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


def find_AdvertiseGen_content():

    def check_format(s):
        infos = s.split('*')
        for e in infos:
            et, ei = e.split('#')
            if not et or not ei:
                return False
        return True

    dev_path = '../data/AdvertiseGen/dev.json'
    train_path = '../data/AdvertiseGen/train.json'
    dev = list(json.loads(x) for x in open(dev_path, 'r', encoding='utf-8').read().strip().split('\n'))
    train = list(json.loads(x) for x in open(train_path, 'r', encoding='utf-8').read().strip().split('\n'))
    contents = []
    for e in dev + train:  contents.append(e['content'])
    print('opened')

    # s = set()
    # for e in contents: s.add(check_format(e))
    # print(s)
    content_dict = {}
    for e in tqdm(contents):
        # infos = e.split('*#')
        infos = re.split('[*|#]', e)
        for i in range(len(infos) // 2):
            et, ei = infos[2 * i], infos[2 * i + 1]
            if et not in contents:
                content_dict[et] = [ei]
            else:
                content_dict[et].append(ei)
    print('parsed')
    content_count = {}
    for k, v in tqdm(content_dict.items()):
        content_count[k] = Counter(v)
    print('counted')
    return content_count


def parse_CEC_XML():
    cec_path = '../data/CEC/CEC-Corpus-master/CEC/交通事故/17岁少女殒命搅拌车下.xml'
    tree = ET.parse(cec_path)
    return tree



def show_processed_dataset():
    files = os.listdir('../data/processed/')
    counts = {}
    for e in tqdm(files):
        d = json.load(open(os.path.join('../data/processed/', e), 'r', encoding='utf-8'))
        counts[e] = len(d)
    print_dict_as_table(counts)


def show_prompted_dataset():
    """
    如果报错了，尝试在终端输入rm .DS_store
    :return:
    """
    files = os.listdir('../data/prompted/')
    counts = {}
    for e in tqdm(files):
        d = list(json.loads(x) for x in open(os.path.join('../data/prompted', e), 'r', encoding='utf-8').read().strip().split('\n'))
        counts[e] = len(d)
    print_dict_as_table(counts)

if __name__ == '__main__':
    # tag_dict = find_weibo_ner_types()
    # find_weibo_ner_property()
    # ct = find_AdvertiseGen_content()
    # tree = parse_CEC_XML()
    # show_processed_dataset()
    show_prompted_dataset()
