#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/24
# project = exp_2
import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt

from image_sdk.utils import encode_to_base64
from image_sdk.image_tagging import image_tagging_aksk
from image_sdk.utils import init_global_env

init_global_env('cn-north-4')

app_key = "AGOPE82VZ2JU50DUZSBN"
app_password = "ocXsUvRv9laBpLBpSiV3wPRGzzeQdM93iaWRPHTt"

# file_path = 'data/'
# labels = {}
# for file in os.listdir(file_path):
#     f = file_path + file
#     result = image_tagging_aksk(app_key, app_password, encode_to_base64(f), '', 'zh', 5, 30)
#     result_dict = json.loads(result)
#     labels[file] = result_dict['result']['tags']
#
# save_path = './label/'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# with open(save_path + 'label.json', 'w+') as f:
#     f.write(json.dumps(labels))

# 展示搜索词的图片
label_path = 'label/label.json'
with open(label_path, 'r') as f:
    labels = json.load(f)
# key_word = input('请输入搜索词\n')
# threshold = 60
# valid_list = set()
# for k, val in labels.items():
#     for item in val:
#         if float(item['confidence']) >= threshold and item['tag'] in key_word:
#             valid_list.add(k)

# valid_list = list(valid_list)
# for i, pic in enumerate(valid_list):
#     pic_path = './data/' + pic
#     img = cv2.imread(pic_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (640, 400))
#     plt.subplot(2, 2, i + 1)
#     plt.axis('off')
#     plt.imshow(img)
# plt.show()

# if not os.path.exists('./tmp/'):
#     os.mkdir('./tmp')
# gif_list=[]
# for i,pic in enumerate(valid_list):
#     pic_path='./data/'+pic
#     img=cv2.imread(pic_path)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=cv2.resize(img,(640,400))
#     gif_list.append(img)

# 根据opencv的image_list 制作gif
# import imageio
# gif= imageio.mimsave('./相册.gif',gif_list,'GIF',duration=1)


print(labels)
classes = [[val[0]['tag'], i] for i, val in labels.items()]
for cls in classes:
    if not os.path.exists('./data/' + cls[0]):
        os.mkdir('./data/' + cls[0])
    # 复制对应的图片
    shutil.copy('./data/' + cls[1], './data/' + cls[0] + '/' + cls[1])
