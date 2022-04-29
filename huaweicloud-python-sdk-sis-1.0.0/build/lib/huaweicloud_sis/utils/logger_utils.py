import logging

# 获取一个日志logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(levelname)s] - [%(message)s]',
    datefmt='%Y-%m-%d %A %H:%M:%S',
    filename='../huaweicloud_sis.log',
    filemode='a'
)

# 添加console作为handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] - [%(levelname)s] - [%(message)s]")
handler.setFormatter(formatter)

# 设置handler
logging.getLogger().addHandler(handler)


def get_logger():
    return logging






