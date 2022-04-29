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

class NlpfClient(object):
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
        self.segment_uri = "/v1/" + client.project_id + "/nlp-fundamental/segment"
        self.multi_grained_segment_uri = "/v1/" + client.project_id + "/nlp-fundamental/multi-grained-segment"
        self.dependency_parser_uri = "/v1/" + client.project_id + "/nlp-fundamental/dependency-parser"
        self.ner_uri = "/v1/" + client.project_id + "/nlp-fundamental/ner"
        self.ner_domain_uri = "/v1/" + client.project_id + "/nlp-fundamental/ner/domain"
        self.text_similar_uri = "/v1/" + client.project_id + "/nlp-fundamental/text-similarity"
        self.sentence_embedding_uri = "/v1/" + client.project_id + "/nlp-fundamental/sentence-embedding"
        self.entity_linking_uri = "/v1/" + client.project_id + "/nlp-fundamental/entity-linking"
        self.keyword_extraction = "/v1/" + client.project_id + "/nlp-fundamental/keyword-extraction"
        self.text_similar_advance_uri = "/v1/" + client.project_id + "/nlp-fundamental/text-similarity/advance"

    def segment(self, text, pos_switch, lang, criterion):
        req_obj = {"text":text, "pos_switch":pos_switch, "lang":lang,"criterion":criterion}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.segment_uri, req_body)
        return response

    def multi_grained_segment(self, text, lang):
        req_obj = {"text": text, "lang": lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.multi_grained_segment_uri, req_body)
        return response

    def ner(self, text, lang):
        req_obj = {"text":text, "lang":lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.ner_uri, req_body)
        return response

    def domain_ner(self, text, lang, domain):
        req_obj = {"text": text, "lang": lang, "domain":domain}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.ner_domain_uri, req_body)
        return response

    def get_text_similarity(self, text1, text2, lang):
        req_obj = {"text1": text1, "text2": text2, "lang": lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.text_similar_uri, req_body)
        return response

    def get_sentence_vectors(self, sentences, domain):
        req_obj = {"sentences": sentences, "domain": domain}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.sentence_embedding_uri, req_body)
        return response

    def get_entity_linking(self, text, lang):
        req_obj = {"text": text, "lang": lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.entity_linking_uri, req_body)
        return response

    def extract_keyword(self, text, limit, lang):
        req_obj = {"text":text, "limit":limit, "lang":lang}
        req_body = json.dumps(req_obj)
        response = self.client.request_nlp_service(self.keyword_extraction, req_body)
        return response