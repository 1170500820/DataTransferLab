import requests
import json
import time


# 每分钟最多提交30个request
# 每24小时最多提交1000000个字符

"""
curl -H 'Content-Type: application/json' -H 'Authorization: cchi48mv9mc753cgstj0' https://welm.weixin.qq.com/v1/completions -d \
'{
    "prompt":"测试",
    "model":"large",
    "max_tokens":16,
    "stop":",，.。"
}'
"""

text = '命名实体识别。请找出句子中的人名（特指）:一节课的时间真心感动了李开复感动。人名：'

url = 'https://welm.weixin.qq.com/v1/completions'
d = {
    "prompt": text,
    "model": "large",
    "max_tokens": 16,
    "stop": ",. ，。",
    "n": -2
}
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'cchi48mv9mc753cgstj0'
}


def get_result(prompt: str, model: str='large', max_tokens: int=16, stop: str=',. ，。', n: int=2):
    time.sleep(1.95)
    # 限制每2.05秒才能提交一次，这样1分钟绝对不会有超过30次提交。
    response = requests.post(
        url=url,
        data=json.dumps({
            'prompt': prompt,
            'model': model,
            'max_tokens': max_tokens,
            'stop': stop,
            'n': n
        }),
        headers=headers
    )
    return response



if __name__ == '__main__':
    response = requests.post(url=url, data=json.dumps(d), headers=headers)
    result = json.loads(response.text)
    print(response)
    print(f'choices:')
    for e in result['choices']:
        print(e)