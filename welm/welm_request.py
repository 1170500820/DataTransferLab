import requests

"""
curl -H 'Content-Type: application/json' -H 'Authorization: cchi48mv9mc753cgstj0' https://welm.weixin.qq.com/v1/completions -d \
'{
    "prompt":"测试",
    "model":"xl",
    "max_tokens":16,
    "stop":",，.。"
}'
"""

url = 'https://welm.weixin.qq.com/v1/completions'
d = {
    "prompt": """请根据所学知识回答下面这个问题
问题：《百年孤独》的作者是？
答案：
""",
    "model": "large",
    "max_tokens": 16,
}
headers = {
    # 'Content-Type': 'application/json',
    'Authorization': 'cchi48mv9mc753cgstj0'
}

if __name__ == '__main__':
    response = requests.post(url=url, data=d, headers=headers)
    print(response)