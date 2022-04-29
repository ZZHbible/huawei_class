import base64
import os
from huaweicloud_sis.utils import logger_utils
from huaweicloud_sis.exception.exceptions import ClientException
my_logger = logger_utils.get_logger()


def encode_file(file_path):
    if not os.path.exists(file_path):
        my_logger.error(file_path + '不存在')
        raise ClientException(file_path + '不存在')
    with open(file_path, 'rb') as f:
        data = f.read()
        base64_data = str(base64.b64encode(data), 'utf-8')
        return base64_data


def save_audio_from_base64str(base64_str, save_path):
    parent_path = os.path.dirname(save_path)
    if parent_path != '' and not os.path.exists(parent_path):
        os.makedirs(parent_path)
    with open(save_path, 'wb') as f:
        base64_data = base64.b64decode(base64_str)
        f.write(base64_data)
