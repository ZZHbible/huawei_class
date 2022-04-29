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

class NlgClient(object):
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
        self.summary_uri = "/v1/" + client.project_id + "/nlg/summarization"
        self.domain_summary_uri = "/v1/" + client.project_id + "/nlg/summarization/domain"
        self.poem_uri = "/v1/" + client.project_id + "/nlg/poem"
        self.doc_generation_uri = "/v1/" + client.project_id + "/nlg/data2doc/document/generation"

    def summary(self, content, title, length_limit, lang):
        req_obj = {"content":content, "title":title, "length_limit":length_limit,"lang":lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.summary_uri, req_body)
        return response

    def domain_summary(self, content, title, length_limit, lang, type):
        if length_limit is None:
            length_limit = 0.3
        if lang is None:
            lang = "zh"
        if type is None:
            type = 0
        req_obj = {"content":content, "title":title, "length_limit":length_limit, "lang":lang, "type":type}
        if title is None:
            req_obj = {"content": content, "length_limit": length_limit, "lang": lang, "type": type}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.domain_summary_uri, req_body)
        return response

    def generate_poem(self, title, type, acrostic):
        req_obj = {"title": title, "type": type, "acrostic": acrostic}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.poem_uri, req_body)
        return response
