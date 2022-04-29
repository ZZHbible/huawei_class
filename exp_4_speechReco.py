#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/26
# project = exp_4_speechReco
import json

from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.client.asr_client import AsrCustomizationClient
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.asr_request import AsrCustomShortRequest
app_key = "AGOPE82VZ2JU50DUZSBN"
app_password = "ocXsUvRv9laBpLBpSiV3wPRGzzeQdM93iaWRPHTt"
project_id="e1a62164540747f4934d30f8619a2765"
region='cn-north-4'

path='./tmp/test.wav'
path_auto_format='wav'
path_property = "chinese_8k_common"
config=SisConfig()
config.set_connect_timeout(5)
config.set_read_timeout(10)
asr_client=AsrCustomizationClient(app_key,app_password,region,project_id,sis_config=config)
# 构造请求
data=io_utils.encode_file(path)
asr_request=AsrCustomShortRequest(path_auto_format,path_property,data)
asr_request.set_add_punc('yes')

result=asr_client.get_short_response(asr_request)
print(json.dumps(result,indent=2,ensure_ascii=False))

