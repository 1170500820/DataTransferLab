import sys
sys.path.append('..')

import json
import os
from tqdm import tqdm
from loguru import logger
from welm.welm_request import get_result


status_file = '.status.json'
# {prompt_type: status}
# status: success_num, error_message
# success_num代表已经成果抽取的数目

"""
抽取结果的保存格式
以jsonl形式，通过追加来写入
dict包含的key为idx与text，分别代表当前的抽取序号，以及response的text
"""

def run(prompt_type: str='find_object'):
    """
    使用dev进行实验。为了后续能够进行对比
    :param prompt_type:
    :return:
    """
    logger.info('开始运行welm实验')

    if os.path.exists(status_file):  status = json.load(open(status_file, 'r', encoding='utf-8'))
    else:  status = {}
    if prompt_type not in status:
        logger.info(f'当前prompt类型={prompt_type}|没有实验记录')
        status[prompt_type] = {'success_num': 0, 'error_message': ''}
    else:
        logger.info(f'当前prompt类型={prompt_type}|已成功抽取的数目：{str(status["success_num"])}，最新的出错信息：{status["error_message"]}')
    success_num = status[prompt_type]['success_num']

    fname = f'../data/prompted/duie_{prompt_type}_dev.jsonl'
    d = list(json.loads(x) for x in open(fname, 'r', encoding='utf-8').read().strip().split('\n'))
    result_fname = f'result/{prompt_type}_result.jsonl'
    total = len(d)

    logger.info('正在提交中...')
    f = open(result_fname, 'a', encoding='utf-8')
    for i in tqdm(range(success_num, total)):
        sample = d[i]['input']
        res = get_result(prompt=sample)
        resdict = json.loads(res.text)
        f.write(json.dumps({'idx': i, 'output': resdict}, ensure_ascii=False) + '\n')
        success_num = i + 1
        status[prompt_type]['success_num'] = success_num
        json.dump(status, open(status_file, 'w', encoding='utf-8'))
    f.close()




if __name__ == '__main__':
    run()