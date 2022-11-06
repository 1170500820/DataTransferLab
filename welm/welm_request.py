import requests
import json

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
    "n": 5
}
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'cchi48mv9mc753cgstj0'
}

if __name__ == '__main__':
    response = requests.post(url=url, data=json.dumps(d), headers=headers)
    result = json.loads(response.text)
    print(response)
    print(f'choices:')
    for e in result['choices']:
        print(e)