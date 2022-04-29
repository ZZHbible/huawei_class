#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/23
# project = exp_1

import matplotlib.pyplot as plt
import numpy as np
import cv2


# 使用plt 显示opencv图片，
def matshow(title='image', image=None, mode="RGB"):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            pass
        elif mode.lower() == "gray":
            # 转换opencv的颜色空间gray
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif mode.lower() == "hsv":
            # 转换opencv的颜色位RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    # 使用plt载入opencv图片
    plt.imshow(image, cmap='gray')

    plt.axis('off')  # 不显示坐标轴
    plt.title(title)

    plt.show()


# 平移函数
def translate(img, x, y):
    (h, w) = img.shape[:2]

    # 定义平移矩阵
    M = np.array([[1, 0, x], [0, 1, y]], dtype=np.float)
    shifted = cv2.warpAffine(img, M, (w, h))
    return shifted


# 旋转函数
def rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if not center:
        center = (w / 2, h / 2)
    # 调用旋转计算矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 使用opencv仿射变换函数
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


im = cv2.imread("./life_photo.jpg")


# matshow(image=im,mode='hsv')
#
# matshow("origin", im)
# shifted = translate(im, 0, 100)
# matshow("trans", shifted)
# rotated = rotate(im, 30, center=(20, 20))
# matshow("rotate", rotated)
#
# # 垂直镜像
# im_flip0 = cv2.flip(im, 0)
# matshow("flip0", im_flip0)
# # 水平镜像
# im_flip1 = cv2.flip(im, 1)
# matshow("flip1", im_flip1)
#
# # 缩放
# (h,w)=im.shape[:2]
# # 缩放的目标尺寸
# dst_size=(100,300)
# # 最邻近插值
# method=cv2.INTER_NEAREST
#
# # 进行缩放
# resized=cv2.resize(im,dst_size,interpolation=method)
# matshow("resize",resized)

# 定义线性灰度变化函数
# k>1时 实现灰度数值的拉伸
# 0<k<1 实现灰度数值的压缩
def linear_trans(img, k, b=0):
    # 计算灰度线性变化的映射表
    trans_list = [(np.float32(x) * k + b) for x in range(256)]
    trans_list = np.array(trans_list)
    trans_list[trans_list > 255] = 255
    trans_list[trans_list < 0] = 0
    trans_list = np.round(trans_list).astype(np.uint8)
    return cv2.LUT(img, trans_list)


im_inversion = linear_trans(im, -1, 255)  # 颜色反转 255-i
# matshow("im_inversion",im_inversion)

im_stretch = linear_trans(im, 1.2)
# matshow("im_stretch",im_stretch)

im_compress = linear_trans(im, 0.8)


# matshow("im_compress",im_compress)

def gamma_trans(img, gamma):
    gamma_list = [np.power(x / 255, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_list)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


# 使用gamma值为0.5的变化，实现对暗部的拉升，亮部的压缩
im_gama05 = gamma_trans(im, 0.5)
# matshow("gamma0.5",im_gama05)
# 使用gamma值为2的变化，实现对亮部的拉升，暗部的压缩
im_gama2 = gamma_trans(im, 2)
# matshow("gamma2",im_gama2)

# 直方图
# plt.hist(im.ravel(),256,[0,256])
# plt.show()

# 中值滤波
im_medianblur = cv2.medianBlur(im, 5)
# matshow('media_blur',im_medianblur)

# 均值滤波
im_meanblur1 = cv2.blur(im, (3, 3))
# matshow("mean_blur_1",im_meanblur1)

# 使用均值算子和fliter2D自定义滤波
# 均值算子
mean_blur = np.ones([3, 3], np.float32) / 9
# 使用fliter2D进行滤波操作
im_meanblur2 = cv2.filter2D(im, -1, mean_blur)
# matshow("mean_blur_2",im_meanblur2)

# 高斯模糊
im_gaussianblur1 = cv2.GaussianBlur(im, (5, 5), 0)
# matshow("gaussian_blur_1",im_gaussianblur1)

# 使用高斯算子和fliter2D自定义滤波操作
gaussianblur2 = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
], np.float32) / 273
im_gaussianblur2 = cv2.filter2D(im, -1, gaussianblur2)
# matshow("gaussian_blur_2",im_gaussianblur2)

# 锐化
# 锐化算子
sharpen_1=np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]
])
# 使用filter2D进行滤波
im_sharpen1=cv2.filter2D(im,-1,sharpen_1)
# matshow("sharpen_1",im_sharpen1)

# 锐化算子2
sharpen_2=np.array([
    [0,-1,0],
    [-1,8,-1],
    [0,1,0]
])/4.0

# 使用filter2D进行滤波
im_sharpen2=cv2.filter2D(im,-1,sharpen_2)
# matshow("sharpen_2",im_sharpen2)

