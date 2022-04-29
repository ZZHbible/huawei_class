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

class NluClient(object):
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
        self.sentiment_uri = "/v1/" + client.project_id + "/nlu/sentiment"
        self.sentiment_domain_uri = "/v1/" + client.project_id + "/nlu/sentiment/domain"
        self.get_intent_uri = "/v1/" + client.project_id + "/nlu/semantic-parser"
        self.classify_uri = "/v1/" + client.project_id + "/nlu/classification"

    def get_sentiment(self, content):
        req_obj = {"content":content}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.sentiment_uri, req_body)
        return response;

    def get_domain_sentiment(self, content, type):
        if type is None:
            type = 0
        req_obj = {"content":content,"type":type}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.sentiment_domain_uri,req_body)
        return response

    def get_intent(self, text, lang):
        if lang is None:
            lang = "zh"
        req_obj = {"text":text, "lang":lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.get_intent_uri,req_body)
        return response

    def classify_text(self, content, domain):
        if domain is None :
            domain = 1
        req_obj = {"content":content,"domain":domain}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.classify_uri, req_body)
        return response
