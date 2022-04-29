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

import json
import requests
import time
from huaweicloud_nlp.NlpResponse import NlpResponse

class HWNlpClientToken(object):
    """
    Nlp client authenticated by token

    initializd by username,domainname,passwrod,region

    Attributes:
        domainname: domain name for the Nlp user. If not IAM user, it's the same as username
        password: password for the Nlp user
        region: region name for the Nlp user, such as cn-north-1,cn-east-2
        httpendpoint: HTTP endpoint for the Nlp request
        token: temporary authentication key for the Nlp user, which will expire after 24 hours
    """
    def __init__(self, domain_name, username, password, region, project_id):
        """
        Constructor for the HWNlpClientToken
        """
        if domain_name == "" or username == "" or password == "" or region == "":
            raise ValueError("The parameter for the HWNlpClientToken constructor cannot be empty.")
        self.domainname = domain_name
        self.username = username
        self.password = password
        self.region = region
        self.httpendpoint = "nlp-ext." + region + ".myhuaweicloud.com"
        self.httpschema = "https"
        self.token = None
        self.project_id = project_id
        self.refreshCount = 0
        self._RETRY_TIMES = 3
        self._POLLING_INTERVAL = 2.0
        self.proxies = {"http": None, "https": None}
        session = requests.Session()
        session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=20, max_retries=3))
        self._session = session
        self._iam_endpoint = "iam.%s.myhuaweicloud.com" % self.region

    def set_proxy(self, proxies):
        self.proxies = proxies

    def set_endpoint(self, endpoint):
        self.httpendpoint = endpoint

    def set_iam_endpoint(self, iam_endpoint):
        self._iam_endpoint = iam_endpoint

    def get_token(self):
        """
        Obtain the token for the Nlp user from the IAM server
        :return:
        """
        if self.token is not None:
            return
        retry_times = 0
        endpoint = self._iam_endpoint
        url = "https://%s/v3/auth/tokens" % endpoint
        headers = {"Content-Type": "application/json"}
        payload = {
          "auth": {
            "identity": {
              "methods": ["password"],
              "password": {
                "user": {
                  "name": self.username,
                  "password": self.password,
                  "domain": {
                    "name": self.domainname
                  }
                }
              }
            },
            "scope": {
              "project": {
                "name": self.region  # region name
              }
            }
          }
        }
        try:
            while True:
                response =  self._session.post(url, json=payload, headers=headers, proxies = self.proxies, verify=False
                                         , timeout=10)
                if 201 != response.status_code:
                    if retry_times < self._RETRY_TIMES:
                        retry_times += 1
                        print("Obtain the token again.")
                        time.sleep(self._POLLING_INTERVAL)
                        self.token = None
                        continue
                    else:
                        print("Failed to obtain the token.")
                        print(response.text)
                        self.token = None
                        return
                else:
                    print("Token obtained successfully.")
                    token = response.headers.get("X-Subject-Token", "")
                    self.token = token
                    return
        except Exception as e:
            print(e)
            print("Invalid token request.")

    def refresh_token(self):
        """
        Refresh the attribute token
        :return:None
        """
        print("The token expires and needs to be refreshed.")
        self.token = None
        self.get_token()

    def request_nlp_service(self, uri, req_body):
        """
        :param uri: the uri for the http request to be called
        :param req_body: the request body for the http request
        :param options: optional parameter in the Nlp http request
        :return:None
        """
        self.get_token()
        if self.token is not None:
            try:
                url = self.httpschema + "://"  + self.httpendpoint + uri
                headers = {
                    "Content-Type": "application/json",
                    "X-Auth-Token": self.token
                }
                proxies = self.proxies
                self._session.scheme = self.httpschema
                response =  self._session.post(url, req_body, headers=headers, proxies=proxies, verify=False)
                result = NlpResponse()
                result.code = response.status_code;
                result.res = json.loads(response.content, encoding="utf-8")
                if 401 == response.status_code and ("The token expires." in response.text):
                    # The token expires and needs to be refreshed.
                    self.refresh_token()
                    return self.request_nlp_service(uri, req_body)

                elif 403 == response.status_code and ("The authentication token is abnormal." in response.text):
                    # The token expires and needs to be refreshed.
                    self.refresh_token()
                    return self.request_nlp_service(uri, req_body)
                return result
            except Exception as e:
                print(e)
                return None
        return None
