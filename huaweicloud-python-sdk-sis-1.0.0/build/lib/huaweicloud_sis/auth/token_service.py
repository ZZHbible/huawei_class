# -*- coding: utf-8 -*-

from huaweicloud_sis.utils import http_utils


def get_token(user_name, password, domain_name, region, url=None):
    """
        获取token
    :param user_name:   用户名
    :param password:    密码
    :param domain_name: 账户名，一般等同用户名
    :param region:      区域，如cn-north-4
    :param url:         请求token的url，可使用默认值
    :return:            请求的token
    """
    if url is None:
        url = 'https://iam.' + region + '.myhuaweicloud.com/v3/auth/tokens'

    auth_data = {
        "auth": {
            "identity": {
                "password": {
                    "user": {
                        "name": user_name,
                        "password": password,
                        "domain": {
                            "name": domain_name
                        }
                    }
                },
                "methods": [
                    "password"
                ]
            },
            "scope": {
                "project": {
                    "name": region
                }
            }
        }
    }

    headers = {'Content-Type': 'application/json'}
    req = http_utils.post_connect(url, headers, auth_data)
    token = req.headers['X-Subject-Token']
    return token
