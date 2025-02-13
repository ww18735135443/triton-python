import yaml

# 假设你的 YAML 文件名为 'config.yaml'
with open('/home/ww/work/project/triton_project/config/helmet_algo_config.yaml', 'r') as file:
    data = yaml.safe_load(file)

import requests
import json

# 假设你的 POST 请求的 URL 是 'http://example.com/api/task'
url = 'http://127.0.0.1:9010/start'

# 将 Python 字典转换为 JSON 字符串
json_data = json.dumps(data)

# 发送 POST 请求
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# 打印响应内容
print(response.status_code)
print(response.json())