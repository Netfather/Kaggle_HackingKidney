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
import glob
import json


##导入 权重 并保存模型

# model = smp.Unet(
#     encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
#     in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your dataset))
#     decoder_attention_type="scse",
#     encoder_depth=5,
#     decoder_channels=[256, 128, 64, 32, 16],
# )
#
# model.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\fsdownload\ChpAtMin_id_4GPU1_Val_diceloss.pth",map_location = "cuda:0"))
#
# torch.save(model, r"C:\Users\Administrator\Desktop\fsdownload\WholeModel_id_4GPU1.pth")

## 读入权重进行预测
SUBMISSION_NAME_1 = 'submission'
# SUBMISSION_NAME_2 = 'submission_U++EB3_E100_B32_0.8Focal_diceloss.csv'
Open_Parral_Trainning = False  # 是否开启并行化  True表示开启 False表示不开启
PTH_NAME = 'GPU1'
# TestWindow  和   Test_new_size
TEST_WINDOW = 1024 * 4
TEST_NEW_SIZE = 1024
TEST_MIN_OVERLAP = 512

Sigmoid_Threshold = 0.4

DATA_PATH = '../input/hubmap-kidney-segmentation'  # 数据存放位置

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 用gpu还是cpu
DEVICE = torch.device('cuda:0')
BATCH_SIZE = 32  # batch_size 大小
EPOCHES = 100

IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)


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


index = [0, 1, 2, 3, 4]

## 得到 结果最好前4个的索引


# Now begin test
submmsion = pd.read_csv((pathlib.Path(DATA_PATH) / 'sample_submission.csv').as_posix(), index_col='id')
submmsion_ids = submmsion.index.values
print(submmsion_ids)  # 测试成功 拿到了submission里的列表

trfm = T.Compose([
    T.ToPILImage(),
    # T.Resize(NEW_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

subm = {}
p = pathlib.Path(DATA_PATH)
best_choice_index_top4 = [0, 1, 2, 3, 4]

# 遍历前4个表现最好的模型的下标
# ChpAtMin_id_7Val_diceloss.pth#
# ../input/testforhumapsubmit
print(len(index))

for i, filename_id in enumerate(submmsion_ids):
    filename = os.path.join(p, "test", filename_id + '.tiff')
    print(filename)
    with rasterio.open(filename, transform=IDNT) as dataset:
        preds_models = np.zeros(dataset.shape, dtype=np.float32)
        print("Test {}".format(filename))
        for element in best_choice_index_top4:

            model.load("../input/testforhumapsubmit/WholeModel_id_{}".format(element) + PTH_NAME + ".pth")
            if Open_Parral_Trainning:
                model = torch.nn.DataParallel(model)
            model.to(DEVICE)
            model.eval()
            print("Now using:" + "WholeModel_id_{}".format(element) + PTH_NAME + ".pth")
            slices = make_grid(dataset.shape, window=TEST_WINDOW, min_overlap=TEST_MIN_OVERLAP)
            preds = np.zeros(dataset.shape, dtype=np.float32)
            if dataset.count != 3:
                # print('Image file with subdatasets as channels')
                layers = [rasterio.open(subd) for subd in dataset.subdatasets]

            for (x1, x2, y1, y2) in (slices):
                if dataset.count == 3:  # normal
                    image = dataset.read([1, 2, 3],
                                         window=Window.from_slices((x1, x2), (y1, y2)))
                    image = np.moveaxis(image, 0, -1)
                else:  # with subdatasets/layers
                    image = np.zeros((TEST_WINDOW, TEST_WINDOW, 3), dtype=np.uint8)
                    for fl in range(3):
                        image[:, :, fl] = layers[fl].read(window=Window.from_slices((x1, x2), (y1, y2)))

                #           print("Test {}-{}:Shape is:{}".format(filename,index,image.shape))
                image = cv2.resize(image, (TEST_NEW_SIZE, TEST_NEW_SIZE))
                image = trfm(image)
                with torch.no_grad():
                    image = image.to(DEVICE)[None]  # 这里加入的是batch维度 这里的测试是每张图的测试
                    score = model(image)[0][0]

                    score2 = model(torch.flip(image, [0, 3]))
                    score2 = torch.flip(score2, [3, 0])[0][0]

                    score3 = model(torch.flip(image, [1, 2]))
                    score3 = torch.flip(score3, [2, 1])[0][0]

                    score_mean = (score + score2 + score3) / 3.0
                    score_sigmoid = score_mean.sigmoid().cpu().numpy()
                    #                 score_sigmoid = score.sigmoid().cpu().numpy()

                    score_sigmoid = cv2.resize(score_sigmoid, (TEST_WINDOW, TEST_WINDOW))

                    # preds[x1:x2, y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)
                    preds[x1:x2, y1:y2] = np.where(preds[x1:x2, y1:y2] != 0, (preds[x1:x2, y1:y2] + score_sigmoid) / 2,
                                                   score_sigmoid)
            # preds_models += (preds >= 1).astype(np.uint8)
            preds_models += preds
            del preds, slices, image, score, score2, score3, score_mean, score_sigmoid
            gc.collect()
        preds_models = preds_models / len(best_choice_index_top4)
        preds_models = (preds_models > Sigmoid_Threshold).astype(np.uint8)
    subm[i] = {'id': filename_id, 'predicted': rle_numba_encode(preds_models)}
    del preds_models
    gc.collect()

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv(SUBMISSION_NAME_1 + ".csv", index=False)