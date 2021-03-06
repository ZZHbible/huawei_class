# -*- coding: utf-8 -*-

from huaweicloud_sis.client.rasr_client import RasrClient
from huaweicloud_sis.bean.rasr_request import RasrRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.bean.callback import RasrCallBack
import json

# 鉴权信息
user_name = ''      # 用户登录华为云用户名
password = ''       # 用户登录华为云密码
domain_name = ''    # 账户名，一般等同用户名
region = ''         # region，如cn-north-4
project_id = ''     # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html

# 实时语音转写参数
path = ''           # 需要发送音频路径，同时sdk也支持byte流发送数据。
audio_format = ''   # 音频支持格式，详见api文档
property = ''       # 属性字符串，language_sampleRate_domain, 详见api文档


class MyCallback(RasrCallBack):
    """ 回调类，用户需要在对应方法中实现自己的逻辑，其中on_response必须重写 """
    def on_open(self):
        """ websocket连接成功会回调此函数 """
        print('websocket connect success')

    def on_start(self, trace_id):
        """
            websocket 开始识别回调此函数
        :param trace_id: 用与日志回溯，可忽略
        :return: -
        """
        print('webscoket start to recognize, trace_id is %s' % trace_id)

    def on_response(self, message):
        """
            websockert返回响应结果会回调此函数
        :param message: json格式
        :return: -
        """
        print(json.dumps(message, indent=2, ensure_ascii=False))

    def on_end(self, trace_id):
        """
            websocket 结束识别回调此函数
        :param trace_id: 用与日志回溯，可忽略
        :return: -
        """
        print('websocket is ended, trace_id is %s' % trace_id)

    def on_close(self):
        """ websocket关闭会回调此函数 """
        print('websocket is closed')

    def on_error(self, error):
        """
            websocket出错回调此函数
        :param error: 错误信息
        :return: -
        """
        print('websocket meets error, the error is %s' % error)


def rasr_example():
    """ 实时语音转写demo """
    # step1 初始化RasrClient, 暂不支持使用代理
    my_callback = MyCallback()
    rasr_client = RasrClient(user_name, password, domain_name, region, project_id, my_callback)

    # step2 构造请求
    request = RasrRequest(audio_format, property)
    # 所有参数均可不设置，使用默认值
    request.set_add_punc('yes')         # 设置是否添加标点， yes or no， 默认no
    request.set_vad_head(10000)         # 设置有效头部， [0, 60000], 默认10000
    request.set_vad_tail(500)           # 设置有效尾部，[0, 3000]， 默认500
    request.set_max_seconds(30)         # 设置一句话最大长度，[0, 60], 默认30
    request.set_interim_results('no')   # 设置是否返回中间结果，yes or no，默认no
    # request.set_vocabulary_id('')     # 设置热词表id，若不存在则不填写，否则会报错

    # step3 选择连接模式
    # rasr_client.short_stream_connect(request)       # 流式一句话模式
    # rasr_client.sentence_stream_connect(request)    # 实时语音转写单句模式
    rasr_client.continue_stream_connect(request)    # 实时语音转写连续模式

    # step4 发送音频
    rasr_client.send_start()
    # 连续模式下，可多次发送音频，发送格式为byte数组
    with open(path, 'rb') as f:
        data = f.read()
        rasr_client.send_audio(data)    # 可选byte_len和sleep_time参数，建议使用默认值
    rasr_client.send_end()

    # step5 关闭客户端，使用完毕后一定要关闭，否则服务端20s内没收到数据会报错并主动断开。
    rasr_client.close()


if __name__ == '__main__':
    try:
        rasr_example()
    except ClientException as e:
        print(e)
    except ServerException as e:
        print(e)


