# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License.  You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations under the License.
#
# sdk reference linking：https://support.huaweicloud.com/sdkreference-nlp/nlp_06_0001.html

import json

class MtClient(object):
    """
      Nlp client authenticated by token

      initializd by username,domainname,passwrod,region

      Attributes:
          ak: your ak
          sk: your sk
          region: region name for the Nlp user, such as cn-north-1,cn-east-2

      """
    def __init__(self, client):
        self.client = client
        self.text_translation_uri = "/v1/" + client.project_id + "/machine-translation/text-translation"
        self.language_detect_uri = "/v1/" + client.project_id + "/machine-translation/language-detection"

    def translate_text(self, text, fromLang, to, scene):
        req_obj = {"text":text, "from":fromLang, "to":to,"scene":scene}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.text_translation_uri, req_body)
        return response

    def detect_language(self, text):
        req_obj = {"text":text,}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.language_detect_uri, req_body)
        return response