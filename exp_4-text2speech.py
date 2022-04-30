#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/26
# project = exp_4
import json

from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.client.tts_client import TtsCustomizationClient

with open('./as_ak.json','r') as f:
    as_ak=json.load(f)
    app_key = as_ak['as']
    app_password = as_ak['ak']
project_id = "e1a62164540747f4934d30f8619a2765"
region = 'cn-north-4'

text = 'i like you, i like you'
path = './tmp/test.wav'
config = SisConfig()
config.set_connect_timeout(5)  # 设置超时5s
config.set_read_timeout(10)  # 设置读取超时10s
ttsc_client = TtsCustomizationClient(app_key, app_password, region, project_id, sis_config=config)
ttsc_request = TtsCustomRequest(text)
ttsc_request.set_saved(True)
ttsc_request.set_saved_path(path)
ttsc_request.set_property("chinese_xiaoyan_common")

result = ttsc_client.get_ttsc_response(ttsc_request)
print(json.dumps(result, indent=2, ensure_ascii=False))
