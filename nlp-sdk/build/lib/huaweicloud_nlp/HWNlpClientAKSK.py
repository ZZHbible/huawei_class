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

import requests
import json
from huaweicloud_nlp.NlpResponse import NlpResponse
from huaweicloud_nlp.apig_sdk import signer


class HWNlpClientAKSK(object):
    """
    NLP client authenticated by AK and SK

    initializd by ak,sk,endpoint

    Attributes:
        ak: Access key for the NLP user
        sk: Secret key for the NLP user
        region: Region for the NLP Server
        project_id: Project Id for the NLP user
    """
    def __init__(self, ak, sk, region, project_id):
        if ak == "" or sk == "" or region == "":
            raise ValueError("The parameter for the HWOcrClientAKSK constructor cannot be empty.")
        self.endpoint = "nlp-ext." + region + ".myhuaweicloud.com"
        self.sig = signer.Signer()
        self.sig.AppKey = ak
        self.sig.AppSecret = sk
        self.project_id = project_id
        self.httpschema = "https"  # Only HTTPS is supported.
        self.httpmethod = "POST"  # Only POST is supported.
        self.proxies = {"http": None, "https": None}
        session = requests.Session()
        session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=20, max_retries=3))
        self._session = session

    def set_proxy(self, proxies):
        self.proxies = proxies

    def set_endpoint(self, endpoint):
        self.endpoint = endpoint

    def request_nlp_service(self, uri, req_body):
        """
        :param uri: URI for the HTTP request to be called
        :param req_body: request body for the HTTP request
        """
        url = self.httpschema + "://" + self.endpoint + uri
        request = signer.HttpRequest()
        request.scheme = self.httpschema
        ma = self.endpoint.find("/")
        if ma > 0:
            request.host = self.endpoint[0:ma]
            request.uri = self.endpoint[ma:] + uri
        else:
            request.host = self.endpoint
            request.uri = uri
        request.method = self.httpmethod
        request.headers = {"Content-Type": "application/json"}
        request.body = req_body
        self.sig.Sign(request)
        proxies = self.proxies
        response = self._session.post(url, data=request.body, headers=request.headers, timeout=10,proxies=proxies
                                 , verify=False)
        result = NlpResponse()
        result.code = response.status_code;
        result.res = json.loads(response.content, encoding = "utf-8")
        return result
