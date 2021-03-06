import requests
import json
from huaweicloud_sis.utils import logger_utils
from huaweicloud_sis.exception.exceptions import ClientException, ServerException

requests.packages.urllib3.disable_warnings()
my_logger = logger_utils.get_logger()
NUM_MAX_RETRY = 5


def post_connect(url, header, data, time_out=5, proxy=None):
    """
        post请求，带有header信息（用于认证）
    :param url: -
    :param header: 头部
    :param data: post数据
    :param time_out: 超时
    :param proxy: 代理
    :return: http请求的response
    """
    if isinstance(data, dict):
        data = json.dumps(data)
    if proxy is not None:
        proxy = _generate_request_proxy(proxy)
    # 加入重试机制
    count = 0
    resp = None
    while count < NUM_MAX_RETRY:
        try:
            resp = requests.post(url, headers=header, data=data, timeout=time_out, verify=False, proxies=proxy)
            break
        except requests.exceptions.RequestException as e:
            my_logger.error(e)
            count += 1
    if resp is None:
        my_logger.error(url + ' post 请求获取响应为空')
        raise ClientException(url + ' post 请求获取响应为空')
    return resp


def get_connect(url, header, data, time_out=50, proxy=None):
    """
        get请求，带有header信息（用于认证）
    :param url: -
    :param header: 头部
    :param data: 数据
    :param time_out: 超时
    :param proxy: 代理
    :return: get请求的response
    """
    if isinstance(data, dict):
        data = json.dumps(data)
    if proxy is not None:
        proxy = _generate_request_proxy(proxy)

    # 加入重试机制
    count = 0
    resp = None
    while count < NUM_MAX_RETRY:
        try:
            resp = requests.get(url, headers=header, params=data, timeout=time_out, verify=False, proxies=proxy)
            break
        except requests.exceptions.RequestException as e:
            my_logger.error(e)
            count += 1
    if resp is None:
        my_logger.error(url + ' get 请求获取响应为空')
        raise ClientException(url + ' get 请求获取响应为空')
    return resp


def parse_resp(resp):
    """
        requests响应转化为json格式
    :param resp: requests请求返回的响应
    :return: json
    """
    text = resp.text
    try:
        result = json.loads(text)
    except Exception as e:
        error_msg = text + ' 解析json出错'
        my_logger.error(error_msg)
        raise ClientException(error_msg)
    if 'error_code' in result and 'error_msg' in result:
        error_msg = json.dumps(result)
        my_logger.error(error_msg)
        raise ServerException(result['error_code'], result['error_msg'])
    return result


def generate_scheme_host_uri(url):
    if url.find('//') == -1 or url.find('com') == -1:
        error_msg = '%s 格式错误' % url
        my_logger.error(error_msg)
        raise ClientException(error_msg)
    split1s = url.split('//')
    split2s = split1s[1].split('com')
    scheme = split1s[0] + '//'
    host = split2s[0] + 'com'
    uri = split2s[1]
    return scheme, host, uri


def _generate_request_proxy(proxy):
    if proxy is None:
        return proxy
    if not isinstance(proxy, list) or (not len(proxy) == 2 and not len(proxy) == 4):
        my_logger.error('proxy must be list, the format is [host, port] or [host, port, username, password]')
        raise ClientException('proxy must be list, the format is [host, port] or [host, port, username, password]')
    proxy_str = str(proxy[0]) + ':' + str(proxy[1])
    if len(proxy) == 2:
        proxy = {
            'http': 'http://' + proxy_str,
            'https': 'https://' + proxy_str
        }
    else:
        proxy = {
            'http': 'http://' + str(proxy[2]) + ':' + str(proxy[3]) + '@' + proxy_str,
            'https': 'https://' + str(proxy[2]) + ':' + str(proxy[3]) + '@' + proxy_str
        }
    return proxy
