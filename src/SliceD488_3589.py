# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         SliceD488_3589
# Description:  本文件用于将  d488文件 和  3589两个文件的原始图像切片 然后保存到本地。
#
# Author:       Administrator
# Date:         2021/4/13
# -------------------------------------------------------------------------------
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse
# del train set 否则内存不够
import gc
import os
import albumentations as A
import cv2
import pandas as pd
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
import random
import numba
import pathlib
from datetime import datetime
import rasterio  # 由于新图像格式不太一致，使用rasterio会读不出某些图片 因此改为使用tiff. # 更新 tiff会爆内存 因此还是使用rasterio
from rasterio.windows import Window
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import glob
# aa05346ff   2ec3f1bb9  57512b7f1
# d488c759a  3589adb90


## 将图片切片 并保存到本地
# OutputPath = ["./aa05346ff_Vis/","./2ec3f1bb9_Vis/","./57512b7f1_Vis/"]
# slice_name = ["aa05346ff", "2ec3f1bb9","57512b7f1"]   #切片的图片
# DATA_PATH = pathlib.Path('./hubmap-kidney-segmentation') # 数据存放位置
# IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)
# WINDOW = 2048
# MIN_OVERLAP = 32
#
# def make_grid(shape, window=256, min_overlap=32):
#     """
#         Return Array of size (N,4), where N - number of tiles,
#         2nd axis represente slices: x1,x2,y1,y2
#     """
#     x, y = shape
#     nx = x // (window - min_overlap) + 1
#     x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
#     x1[-1] = x - window
#     x2 = (x1 + window).clip(0, x)
#     ny = y // (window - min_overlap) + 1
#     y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
#     y1[-1] = y - window
#     y2 = (y1 + window).clip(0, y)
#     slices = np.zeros((nx, ny, 4), dtype=np.int64)
#
#     for i in range(nx):
#         for j in range(ny):
#             slices[i, j] = x1[i], x2[i], y1[j], y2[j]
#     return slices.reshape(nx * ny, 4)
#
# for output_index,filename in enumerate(slice_name):
#     filepath = (DATA_PATH / 'train' / (filename + '.tiff')).as_posix()
#
#     with rasterio.open(filepath, transform=IDNT) as dataset:
#         slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
#         if dataset.count != 3:
#             print('Image file with subdatasets as channels:{}'.format(filename))
#             layers = [rasterio.open(subd) for subd in dataset.subdatasets]
#
#         for index, (slc) in enumerate(tqdm(slices)):
#             x1, x2, y1, y2 = slc
#             if dataset.count == 3:  # normal
#                 image = dataset.read([1, 2, 3],
#                                      window=Window.from_slices((x1, x2), (y1, y2)))
#                 image = np.moveaxis(image, 0, -1)
#             else:  # with subdatasets/layers
#                 image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
#                 for fl in range(3):
#                     image[:, :, fl] = layers[fl].read(window=Window.from_slices((x1, x2), (y1, y2)))
#             cv2.imwrite(OutputPath[output_index] + "{}_{}.jpg".format(filename,index),image)


## 提取出train中的几张图片 然后保存到本地进行查看  查看mask的标记方法
# ../input/hubmap-kidney-segmentation/train/0486052bb.tiff
# 读取并 查看 0486052bb 文件
# 设定随机种子，方便复现代码
def set_seeds(seed=23):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seeds()


# 根据数据格式 将字符串转为图片 将图片转为字符串
# used for converting the decoded image to rle mask
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# 这里是将预测生成的 01 mask 转变为可提交的 rle coding
@numba.njit()
def rle_numba(pixels):
    size = len(pixels)
    points = []
    if pixels[0] == 1: points.append(0)
    flag = True
    for i in range(1, size):
        if pixels[i] != pixels[i - 1]:
            if flag:
                points.append(i + 1)
                flag = False
            else:
                points.append(i + 1 - points[-1])
                flag = True
    if pixels[-1] == 1: points.append(size - points[-1] + 1)
    return points


def rle_numba_encode(image):
    pixels = image.flatten(order='F')
    points = rle_numba(pixels)
    return ' '.join(str(x) for x in points)

# 从原始图片拆分成不同的区域 并保证最小覆盖 返回的是一个由坐标构成的列表
def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

# tiff_ids = np.array(["aaa6a05cc", "26dc41664", "b9a3865fc",
#             "2f6ecfcdf", "c68fe75ea", "e79de561c",
#             "1e2425f28", "b2dc8411c", "afa5e8098",
#             "0486052bb", "cb2d976f4", "54f2eec69",
#             "4ef6695ce", "8242609fa", "095bf7a1f",
#             "2ec3f1bb9","3589adb90","57512b7f1",
#             "aa05346ff","d488c759a"])

OutputPath = ["./d488c759a_Vis/"]
OutputMask = ["./d488c759a_Msk/"]
slice_name = ["d488c759a"]   #切片的图片
DATA_PATH = pathlib.Path('./hubmap-kidney-segmentation') # 数据存放位置
IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)
WINDOW = 2048
MIN_OVERLAP = 32
csv_rle_coder = pd.read_csv((DATA_PATH / 'TrainPersudo.csv').as_posix(),
                               index_col=[0])

for output_index,filename in enumerate(slice_name):
    filepath = (DATA_PATH / 'train' / (filename + '.tiff')).as_posix()

    with rasterio.open(filepath, transform=IDNT) as dataset:
        total_mask = rle_decode(csv_rle_coder.loc[filename, 'encoding'], dataset.shape)
        slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
        if dataset.count != 3:
            print('Image file with subdatasets as channels:{}'.format(filename))
            layers = [rasterio.open(subd) for subd in dataset.subdatasets]

        for index, (slc) in enumerate(tqdm(slices)):
            x1, x2, y1, y2 = slc
            if dataset.count == 3:  # normal
                image = dataset.read([1, 2, 3],
                                     window=Window.from_slices((x1, x2), (y1, y2)))
                image = np.moveaxis(image, 0, -1)
            else:  # with subdatasets/layers
                image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
                for fl in range(3):
                    image[:, :, fl] = layers[fl].read(window=Window.from_slices((x1, x2), (y1, y2)))
            cv2.imwrite(OutputPath[output_index] + "{}_{}.jpg".format(filename,index),image)
            # 修正一下 将1 改为255
            mask = np.where(total_mask[x1:x2, y1:y2] == 1, 255, 0)
            cv2.imwrite(OutputMask[output_index] + "{}_{}.png".format(filename, index), mask)