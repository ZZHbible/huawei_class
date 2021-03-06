# -*- coding: utf-8 -*-

from huaweicloud_sis.client.asr_client import AsrCustomizationClient
from huaweicloud_sis.bean.asr_request import AsrCustomShortRequest
from huaweicloud_sis.bean.asr_request import AsrCustomLongRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.sis_config import SisConfig
import json
import time

# 鉴权参数
ak = ''  # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
sk = ''  # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
region = ''  # region，如cn-north-4
project_id = ''  # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html

# 一句话识别参数，以音频文件的base64编码传入，1min以内音频
path = ''  # 文件位置
path_audio_format = ''  # 音频格式，详见api文档
path_property = ''  # language_sampleRate_domain, 如chinese_8k_common，详见api文档

# 录音文件识别参数，音频文件以obs连接方式传入（即先需要将音频传送到华为云的obs）
obs_url = ''  # 音频obs连接
obs_audio_format = ''  # 音频格式，详见api文档
obs_property = ''  # language_sampleRate_domain, 如chinese_8k_common，详见api文档


def asrc_short_example():
    """ 一句话识别示例 """
    # step1 初始化客户端
    config = SisConfig()
    config.set_connect_timeout(5)  # 设置连接超时
    config.set_read_timeout(10)  # 设置读取超时
    # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
    # config.set_proxy(proxy)
    asr_client = AsrCustomizationClient(ak, sk, region, project_id, sis_config=config)

    # step2 构造请求
    data = io_utils.encode_file(path)
    asr_request = AsrCustomShortRequest(path_audio_format, path_property, data)
    # 所有参数均可不设置，使用默认值
    # 设置是否添加标点，yes or no，默认no
    asr_request.set_add_punc('yes')
    # 设置是否添加热词表id，没有则不填
    # asr_request.set_vocabulary_id(None)

    # step3 发送请求，返回结果,返回结果为json格式
    result = asr_client.get_short_response(asr_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def asrc_long_example():
    """ 录音文件识别示例 """
    # step1 初始化客户端
    config = SisConfig()
    config.set_connect_timeout(5)  # 设置连接超时
    config.set_read_timeout(10)  # 设置读取超时
    # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
    # config.set_proxy(proxy)
    asr_client = AsrCustomizationClient(ak, sk, region, project_id, sis_config=config)

    # step2 构造请求
    asrc_request = AsrCustomLongRequest(obs_audio_format, obs_property, obs_url)
    # 所有参数均可不设置，使用默认值
    # 设置是否添加标点，yes or no，默认no
    asrc_request.set_add_punc('yes')
    # 设置 是否需要分析信息，True or False, 默认False。 只有need_analysis_info生效，diarization、channel、emotion、speed才会生效
    # 目前仅支持8k模型，详见api文档
    asrc_request.set_need_analysis_info(True)
    # 设置是否需要话者分离，默认True，需要need_analysis_info设置为True才生效。
    asrc_request.set_diarization(True)
    # 设置声道信息, 一般都是单声道，默认为MONO，需要need_analysis_info设置为True才生效，详见api文档。
    asrc_request.set_channel('MONO')
    # 设置是否返回感情信息, 默认True，需要need_analysis_info设置为True才生效。
    asrc_request.set_emotion(True)
    # 设置是否需要返回语速信息，默认True，需要need_analysis_info设置为True才生效。
    asrc_request.set_speed(True)
    # 设置是否添加热词表id，没有则不填
    # asrc_request.set_vocabulary_id(None)

    # step3 发送请求，获取job_id
    job_id = asr_client.submit_job(asrc_request)

    # step4 根据job_id轮询，获取结果。
    status = 'WAITING'
    count = 0  # 每2s查询一次，最多尝试100次，即200s
    while status != 'FINISHED' and count < 100:
        print(count, ' query')
        result = asr_client.get_long_response(job_id)
        status = result['status']
        if status == 'ERROR':
            print('录音文件识别执行失败, %s' % json.dump(result))
            break
        time.sleep(2)
        count += 1
    if status == 'FINISHED':
        print(json.dumps(result, indent=2, ensure_ascii=False))
        # 也可以选择保存文件. indent参数可设置缩进，如果不需要也可以不设置
        # with open('2.json', 'w', encoding='utf-8') as f:
        #     json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print('查询时间不够，录音文件依然识别中，建议将count设置更大一些')


if __name__ == '__main__':
    try:
        asrc_short_example()  # 一句话识别
        asrc_long_example()  # 录音文件识别
    except ClientException as e:
        print(e)
    except ServerException as e:
        print(e)
