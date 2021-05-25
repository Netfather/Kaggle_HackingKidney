# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Unet_EffieientnetB7_MainFlow
# Description:  这里把所有用的库函数全部放到了一个文件中 是用于kagglekernel线上训练的版本
# Author:       Administrator
# Date:         2021/3/15
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
import json
import matplotlib.pyplot as plt

SUBMISSION_NAME_1 = 'submission_UEb3_NewPersudo_0.937_512_NoFinetune'
HISTORY_JSON = 'ML1_GPU01_losshistory_'
BEST_CHOICE_JSON = 'ML1_GPU01_BestChoice.json'
PTH_NAME = 'ML1_GPU01_Val_diceloss'
# SUBMISSION_NAME_2 = 'submission_U++EB3_E100_B32_0.8Focal_diceloss.csv'
WINDOW = 512 * 2
MIN_OVERLAP = 32
NEW_SIZE = 512
DataSet_ThreShold = 100
Open_Parral_Trainning = True  # 是否开启并行化  True表示开启 False表示不开启

Open_External_Dataset = True  # 是否启用额外的数据

Open_Classifer = False  # 是否在所有数据中加入分类器 进行多监督和多任务学习

Open_Persudo = True # 是否开启伪标签

Sigmoid_Threshold = 0.4
# TestWindow  和   Test_new_size
TEST_WINDOW = 2048 * 2
TEST_NEW_SIZE = 2048
TEST_MIN_OVERLAP = 512

DATA_PATH = './hubmap-kidney-segmentation'  # 数据存放位置
# SKFold_Split_Path = './hubmap-kidney-segmentation/train/*.tiff'
EXTERNAL_DATA_IMAGE_PATH = './External_Kidney_Segmentation/images_1024'
EXTERNAL_DATA_MASK_PATH = "./External_Kidney_Segmentation/masks_1024"

if Open_Parral_Trainning:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 用gpu还是cpu
else:
    DEVICE = torch.device('cuda:1')
BATCH_SIZE = 10  # batch_size 大小
EPOCHES = 100

IDNT = rasterio.Affine(1, 0, 0, 0, 1, 0)


def get_model():
    if Open_Classifer:
        model = smp.Unet(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset))
            decoder_attention_type="scse",
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
            aux_params=
            {
                "classes": 1,
                "pooling": "avg",
                "dropout": 0.5,
                "activation": None
            }
        )
    else:
        model = smp.Unet(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset))
            decoder_attention_type="scse",
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
        )

    return model


# 设定随机种子，方便复现代码
def set_seeds(seed=97):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seeds()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2. * intersection) / (inputs.mean() + targets.mean() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2. * intersection) / (inputs.mean() + targets.mean() + smooth)

        return dice_loss


class LovaszBinarayLoss(nn.Module):
    def __init__(self):
        super(LovaszBinarayLoss, self).__init__()

    def forward(self, logits, labels, per_image=True, ignore=None):
        """
            Binary Lovasz hinge loss
              logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
              labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
              per_image: compute the loss per image instead of per batch
              ignore: void class id
            """
        if per_image:
            loss = LovaszBinarayLoss.mean(
                self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss

    # --------------------------- BINARY LOSSES ---------------------------

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    # --------------------------- HELPER FUNCTIONS ---------------------------
    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def isnan(x):
        return x != x

    @staticmethod
    def mean(l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(LovaszBinarayLoss.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n


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


# 定义及早停止类
# 2021年3月27日修正 增加适配 KFold
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, OpenEarlyStop=True, name="Defalut"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            OpenEarlyStop (bool): 如果为真，表明我们需要开启EarlyStop和自动保存最优监视器功能
                                  如果为假，表明我们只开启最优监视器功能
            name (string): 用于存储的checkpoint是哪个指标的
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor_metric_min = np.PINF
        self.monitor_metric_max = np.NINF
        self.OpenES = OpenEarlyStop
        self.name = name

    def __call__(self, monitor_metric, model, id, mode='min'):
        '''
        此函数用于给定一个检测指标，然后按照mode模式来进行模型的保存和 earlyStop
        :param monitor_metric:  需要监测的模型指标
        :param model: 目前正在训练要保存的模型
        :param mode: 需要检测的最优值是min还是max 默认为min
        :param idx: 提示这时第几个id  一般和KFold联动使用
        :return:
        '''
        if (mode == 'min'):
            score = -monitor_metric

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
            elif score < self.best_score:
                if self.OpenES:
                    self.counter += 1
                    print(f'Message From Early Stop: EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
                self.counter = 0
        elif (mode == 'max'):
            score = monitor_metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
            elif score < self.best_score:
                if self.OpenES:
                    self.counter += 1
                    print(f'Message From Early Stop: EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(monitor_metric, model, id, mode=mode)
                self.counter = 0

    def save_checkpoint(self, monitor_metric, model, id, mode='min'):

        '''Saves model when validation loss decrease.'''
        if mode == "min":
            if self.verbose:
                print(
                    f'Message From Early Stop: Monite metric:({self.name}) decreased! ({self.monitor_metric_min:.6f} --> {monitor_metric:.6f}).  Saving model ...')
            torch.save(model.state_dict(), './ChpAtMin' + "_id_" + str(id) + self.name + '.pth')
            self.monitor_metric_min = monitor_metric  # 保存完成后更新保存的最优值
        elif mode == "max":
            if self.verbose:
                print(
                    f'Message From Early Stop: Monite metric:({self.name}) increased! ({self.monitor_metric_max:.6f} --> {monitor_metric:.6f}).  Saving model ...')
            torch.save(model.state_dict(), './ChpAtMax' + "_id_" + str(id) + self.name + '.pth')
            self.monitor_metric_max = monitor_metric  # 保存完成后更新保存的最优值


# 用于返回最真实的dice得分。  2021年4月1日修正，原有的计算方式会导致我本地的val远小于实际的值 可能在最终结果选择上有所波动，应当完全按照标准进行
def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()
    dice = 1. - 2 * overlap / (uion + 0.001)
    return dice


def np_dice_score_perbatch(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()

    return overlap, uion


def loss_fn_train(y_pred, y_true):
    focal = loss_focal(y_pred, y_true)
    dicse = loss_dicse(y_pred, y_true)
    # lovaz = loss_lovaz(y_pred, y_true)

    bcsls = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="mean")

    return 0.8 * focal + 0.6 * bcsls + 0.2 * dicse


def loss_fn_classifier(y_pred, y_true):
    bcsls = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="mean")

    return bcsls


def loss_fn_val(y_pred, y_true):
    val_dice = loss_dicse(y_pred, y_true)
    return val_dice


# 修正 修正为对内存有好的方式
# @torch.no_grad()
# def validation(model, loader, loss_fn):
# val_probability, val_mask = [], []
# model.eval()

# for data in loader:
# if Open_Classifer:
# image, target,classif_label = data
# image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
# else:
# image, target = data
# image, target  = image.to(DEVICE), target.float().to(DEVICE)
# output = model(image)
# if Open_Classifer:
# output = output[0]
# output_ny = output.sigmoid().data.cpu().numpy()
# target_np = target.data.cpu().numpy()

###此地方存在内存问题   需要修正   不需要使用 这种方式  否则很容易导致内存爆炸

# val_probability.append(output_ny)
# val_mask.append(target_np)

# val_probability = np.concatenate(val_probability)
# val_mask = np.concatenate(val_mask)

# return np_dice_score(val_probability, val_mask)


# 这时统一验证 对内存友好的方式
# @torch.no_grad()
# def validation(model, loader, loss_fn):
#     Fenzi = 0.
#     Fenmu = 0.
#     model.eval()
#
#     for data in loader:
#         if Open_Classifer:
#             image, target, classif_label = data
#             image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
#         else:
#             image, target = data
#             image, target = image.to(DEVICE), target.float().to(DEVICE)
#         output = model(image)
#         if Open_Classifer:
#             output = output[0]
#         output_ny = output.sigmoid().data.cpu().numpy()
#         target_np = target.data.cpu().numpy()
#
#         res = np_dice_score_perbatch(output_ny, target_np)
#
#         Fenzi += res[0]
#         Fenmu += res[1]
#
#     return 1. - 2. * Fenzi / Fenmu


# 份文件验证  对内存友好
# 2021.4.21 修正
@torch.no_grad()
def validation(model, loader, loss_fn):
    model.eval()
    Score_Perfile_dice = []
    Score_Perfile = []
    for element in loader:
        Fenzi = 0.
        Fenmu = 0.
        for data in element:
            if Open_Classifer:
                image, target, classif_label = data
                image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
            else:
                image, target = data
                image, target = image.to(DEVICE), target.float().to(DEVICE)
            output = model(image)
            if Open_Classifer:
                output = output[0]
            output_ny = output.sigmoid().data.cpu().numpy()
            target_np = target.data.cpu().numpy()

            res = np_dice_score_perbatch(output_ny, target_np)

            Fenzi += res[0]
            Fenmu += res[1]
        Score_Perfile_dice.append(2. * Fenzi / Fenmu)
        Score_Perfile.append(1. - 2. * Fenzi / Fenmu)
    print(Score_Perfile_dice)
    return Score_Perfile


class ExternalDataset(D.Dataset):
    def __init__(self, image_dir, mask_dir, transform, threshold=DataSet_ThreShold):
        super(ExternalDataset, self).__init__()
        print("Now Processing External Data")
        self.imgpath = pathlib.Path(image_dir)
        self.mskpath = pathlib.Path(mask_dir)
        self.transform = transform
        self.threshold = threshold
        self.x, self.y, self.z = [], [], []  # x 存放图片 y存放标签 z存放分类器信息
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        for (image_path, mask_path) in zip(self.imgpath.glob('*.png'), self.mskpath.glob('*.png')):
            # 检测文件名是否一致
            image_name = str(image_path).split("/")[-1]
            mask_name = str(mask_path).split("/")[-1]
            # print("image_name:{}".format(image_name))
            # print("mask_name:{}".format(mask_name))
            if (image_name != mask_name):
                print("Error")
                raise ("Order error for external data!!!!!")
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # CHAGE color mode  读入的是  0 到  255 的图像
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 读入的只有  0 和1
            # test_mask = np.unique(mask)
            # if (len(test_mask) != 1):
            #     print(test_mask)
            # print(image.shape)
            # print(mask.shape)
            # Only For test
            if image.shape[-1] != 3:
                raise ("Error")
            if len(mask.shape) != 2:
                raise ("Mask shape error")
            if (mask.shape != image.shape[0:2]):
                raise ("Shape doesnt match!!!")

            ## 加入分类器判断  如果mask中有标记 说明 图中存在肾小球
            if Open_Classifer:
                if mask.sum() >= self.threshold:
                    # 说明该图中有肾小球 对应的分类器为1
                    self.z.append(torch.tensor(1.))
                else:
                    self.z.append(torch.tensor(0.))
            image = cv2.resize(image, (NEW_SIZE, NEW_SIZE))
            mask = np.array(mask, dtype=np.uint8)
            mask = cv2.resize(mask, (NEW_SIZE, NEW_SIZE))
            # ONly for test
            # test_mask = np.unique(mask)
            # if (len(test_mask) != 1):
            #     print(test_mask)
            # print(image.shape)
            # print(mask.shape)
            self.x.append(image)
            self.y.append(mask)

    # get data operation
    def __getitem__(self, index):
        if Open_Classifer:
            image, mask, classifier = self.x[index], self.y[index], self.z[index]
        else:
            image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        if Open_Classifer:
            return self.as_tensor(augments['image']), augments['mask'][None], classifier[None]  # 补入batch维度 方便进行检测
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class HubDataset(D.Dataset):

    def __init__(self, root_dir, tiff_ids, transform,
                 window=256, overlap=32, threshold=DataSet_ThreShold, isvalid=False):
        self.path = pathlib.Path(root_dir)
        self.tiff_ids = tiff_ids  # 输入的是一个列表 表示这次处理的是哪个文件
        self.overlap = overlap
        self.window = window
        self.transform = transform
        self.csv = pd.read_csv((self.path / 'TrainPersudo_0.937_Mix.csv').as_posix(),
                               index_col=[0])
        self.threshold = threshold
        self.isvalid = isvalid
        self.x, self.y, self.z = [], [], []
        self.build_slices()
        self.len = len(self.x)
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    def build_slices(self):
        print("Now Processing {}".format(self.tiff_ids))
        for i, filename in enumerate(self.csv.index.values):
            if not filename in self.tiff_ids:
                continue
            filepath = (self.path / 'train' / (filename + '.tiff')).as_posix()

            with rasterio.open(filepath, transform=IDNT) as dataset:
                total_mask = rle_decode(self.csv.loc[filename, 'encoding'], dataset.shape)
                slices = make_grid(dataset.shape, window=self.window, min_overlap=self.overlap)
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
                    image = cv2.resize(image, (NEW_SIZE, NEW_SIZE))
                    mask = cv2.resize(total_mask[x1:x2, y1:y2], (NEW_SIZE, NEW_SIZE))
                    # print(image.shape)
                    # print(mask.shape)
                    # print("Processing {} / {} in {} : \n ImageShape:{} MaskShape:{}".format(index + 1, len(slices), filename,image.shape,mask.shape))
                    # 对于测试集，我们应当包括所有可能的边界，因为测试部分不需要进行数据提炼
                    if self.isvalid:
                        self.x.append(image)
                        self.y.append(mask)
                        if Open_Classifer:
                            if total_mask[x1:x2, y1:y2].sum() >= self.threshold:
                                # 说明该图中有肾小球 对应的分类器为1
                                self.z.append(torch.tensor(1.))
                            else:
                                self.z.append(torch.tensor(0.))
                    else:
                        # 阈值判定， 对于训练集，包括边界的图片 或者几乎没有标签的数据 并不是我们需要关注的对象，因此这里我们需要做一个过滤 避免背景数据集过多导致训练失衡
                        # 确保mask中的 标签和
                        # 这里的后一句话 也将某些边界加入了图片中
                        if total_mask[x1:x2, y1:y2].sum() >= self.threshold or (
                                image > 40).mean() > 0.99:  # 4月19日修正 不再限制输入的图片
                            # if total_mask[x1:x2, y1:y2].sum() >= self.threshold:
                            self.x.append(image)
                            self.y.append(mask)
                            if Open_Classifer:
                                if total_mask[x1:x2, y1:y2].sum() >= self.threshold:
                                    # 说明该图中有肾小球 对应的分类器为1
                                    self.z.append(torch.tensor(1.))
                                else:
                                    self.z.append(torch.tensor(0.))

    # get data operation
    def __getitem__(self, index):
        if Open_Classifer:
            image, mask, classifier = self.x[index], self.y[index], self.z[index]
        else:
            image, mask = self.x[index], self.y[index]
        augments = self.transform(image=image, mask=mask)
        if Open_Classifer:
            return self.as_tensor(augments['image']), augments['mask'][None], classifier[None]
        return self.as_tensor(augments['image']), augments['mask'][None]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


trfm = A.Compose([
    # A.Resize(NEW_SIZE, NEW_SIZE),  # 在这里确认了
    # 取消Transpose
    A.RandomRotate90(),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),  # 修正加入
        A.RandomGamma(p=1),
        A.GaussNoise(p=1)
    ], p=0.25),

    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),  # 修正加入
        A.OpticalDistortion(p=1),
        A.GridDistortion(p=1),
    ], p=0.25),  # 修正加入

    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MotionBlur(blur_limit=3, p=1),
        A.MedianBlur(blur_limit=3, p=1)
    ], p=0.25),

    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=30, p=1),
        # A.CLAHE(clip_limit=4),
        A.RandomBrightnessContrast(brightness_limit=0.4, p=1),
    ], p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
])

# 用于验证集的 变形操作
val_trfm = A.Compose([
    # A.CenterCrop(NEW_SIZE, NEW_SIZE),
    # A.Resize(NEW_SIZE, NEW_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

# 每个file单独做一个验证集

# 获得每个图的id

if Open_Persudo:
    tiff_ids = np.array(["aaa6a05cc", "26dc41664", "b9a3865fc",
    "2f6ecfcdf", "c68fe75ea", "e79de561c",
    "1e2425f28", "b2dc8411c", "afa5e8098",
    "0486052bb", "cb2d976f4", "54f2eec69",
    "4ef6695ce", "8242609fa", "095bf7a1f"])
    tiff_8_ids = np.array(["2ec3f1bb9","3589adb90"])
    tiff_6_ids = np.array(["57512b7f1","aa05346ff"])
    tiff_4_ids = np.array(["d488c759a"])
                         # "2ec3f1bb9", "3589adb90", "57512b7f1",
                         # "aa05346ff", "d488c759a"])
    train_ds_8 = HubDataset(DATA_PATH, tiff_8_ids, window=WINDOW, overlap=MIN_OVERLAP,
                            threshold=DataSet_ThreShold, transform=trfm)
    train_ds_6 = HubDataset(DATA_PATH, tiff_6_ids, window=WINDOW, overlap=MIN_OVERLAP,
                            threshold=DataSet_ThreShold, transform=trfm)
    train_ds_4 = HubDataset(DATA_PATH, tiff_4_ids, window=WINDOW, overlap=MIN_OVERLAP,
                            threshold=DataSet_ThreShold, transform=trfm)
    train_loader_8 = D.DataLoader(
        train_ds_8, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    train_loader_6 = D.DataLoader(
        train_ds_6, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    train_loader_4 = D.DataLoader(
        train_ds_4, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
else:
    tiff_ids = np.array(["aaa6a05cc", "26dc41664", "b9a3865fc",
    "2f6ecfcdf", "c68fe75ea", "e79de561c",
    "1e2425f28", "b2dc8411c", "afa5e8098",
    "0486052bb", "cb2d976f4", "54f2eec69",
    "4ef6695ce", "8242609fa", "095bf7a1f"])


skf = KFold(n_splits=5, shuffle=True)  # 用Kfold方法进行分割  开启shuffle

Choice_Vector = []

# Only For Test
if Open_External_Dataset:
    train_ds_external = ExternalDataset(image_dir=EXTERNAL_DATA_IMAGE_PATH, mask_dir=EXTERNAL_DATA_MASK_PATH,
                                        transform=trfm, threshold=DataSet_ThreShold)
    print(len(train_ds_external))
    # 然后将数据额外分成5份

    lengths = [int(len(train_ds_external) * 0.2),
               int(len(train_ds_external) * 0.2),
               int(len(train_ds_external) * 0.2),
               int(len(train_ds_external) * 0.2),
               int(len(train_ds_external)) - int(len(train_ds_external) * 0.8)]
    External_Subset = torch.utils.data.random_split(train_ds_external, lengths,
                                                    generator=torch.Generator().manual_seed(23))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tiff_ids, tiff_ids)):
    print(tiff_ids[val_idx])  # 拿到用于测试的数据集大小

    train_ds = HubDataset(DATA_PATH, tiff_ids[train_idx], window=WINDOW, overlap=MIN_OVERLAP,
                          threshold=DataSet_ThreShold, transform=trfm)

    valid_ds_mul = []
    for element in tiff_ids[val_idx]:
        valid_ds_mul.append(HubDataset(DATA_PATH, element, window=WINDOW, overlap=MIN_OVERLAP,
                                       threshold=0, transform=val_trfm, isvalid=True))
    # 2021.4.22 修正   将val设定为每个文件
    # valid_ds = HubDataset(DATA_PATH, tiff_ids[val_idx], window=WINDOW, overlap=MIN_OVERLAP,
    # threshold=0, transform=val_trfm, isvalid=True)
    print(len(train_ds), len(valid_ds_mul))

    if Open_External_Dataset:  # 4月19日修正  只有当  idx为 0 或4时才会加入额外数据 否则不加入
        train_ds = D.ConcatDataset([train_ds, External_Subset[fold_idx]])
        print("With External Data The total size is {}".format(len(train_ds)))

    train_loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    val_loader_mul = []
    for element in valid_ds_mul:
        val_loader_mul.append(D.DataLoader(
            element, batch_size=BATCH_SIZE, shuffle=False, num_workers=12))


    # 2021.4.22 修正   将val设定为每个文件
    # val_loader = D.DataLoader(
    #     valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    model = get_model()

    if Open_Parral_Trainning:
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)

    # optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,
    # lr=5e-3, weight_decay=1e-3)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=0.2)

    # lambda2 = lambda epoch: 0.9 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2])
    # 对于 SAUnet需要使用scheduler

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4)

    loss_focal = FocalLoss(logits=True).to(DEVICE)
    loss_dicse = DiceLoss().to(DEVICE)

    Trainlosses_Perepoch = []
    Vallosses_Perepoch = []
    Valdicelosses_Perepoch = []
    # early_stopping_Val_trainloss = EarlyStopping(patience=6, verbose=True, OpenEarlyStop=True, name="Val_trainloss")  # 没有必要再保存这个ckpt了
    early_stopping_Val_Diceloss = EarlyStopping(patience=8, verbose=True, OpenEarlyStop=True,
                                                name=PTH_NAME)  # 只考察Val_Diceloss的结果

    for epoch in range(1, EPOCHES + 1):
        print('Epoch is [{}/{}]'.format(epoch, EPOCHES + 1))

        prec_time = datetime.now()  # 记录用时

        sum_losses = 0
        model.train()

        if Open_Persudo:
            # 8分位
            print("Persudo Open！")
            for i, data in enumerate(train_loader_8):
                if Open_Classifer:
                    image, target, classif_label = data
                    image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
                else:
                    image, target = data
                    image, target = image.to(DEVICE), target.float().to(DEVICE)
                # print(classif_label.shape)
                optimizer.zero_grad()
                if Open_Classifer:
                    output, output_classifer = model(image)
                    loss_seg = loss_fn_train(output, target)
                    loss_cls = loss_fn_classifier(output_classifer, classif_label)
                    loss =  0.9 *  (0.6 * loss_seg + 0.4 * loss_cls)
                else:
                    output = model(image)
                    loss = 0.9 * loss_fn_train(output, target)
                loss.backward()
                optimizer.step()
                sum_losses += loss.item()
            # 6分位
            for i, data in enumerate(train_loader_6):
                if Open_Classifer:
                    image, target, classif_label = data
                    image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
                else:
                    image, target = data
                    image, target = image.to(DEVICE), target.float().to(DEVICE)
                # print(classif_label.shape)
                optimizer.zero_grad()
                if Open_Classifer:
                    output, output_classifer = model(image)
                    loss_seg = loss_fn_train(output, target)
                    loss_cls = loss_fn_classifier(output_classifer, classif_label)
                    loss =  0.8 *  (0.6 * loss_seg + 0.4 * loss_cls)
                else:
                    output = model(image)
                    loss = 0.8 * loss_fn_train(output, target)
                loss.backward()
                optimizer.step()
                sum_losses += loss.item()
            #4分位
            for i, data in enumerate(train_loader_4):
                if Open_Classifer:
                    image, target, classif_label = data
                    image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
                else:
                    image, target = data
                    image, target = image.to(DEVICE), target.float().to(DEVICE)
                # print(classif_label.shape)
                optimizer.zero_grad()
                if Open_Classifer:
                    output, output_classifer = model(image)
                    loss_seg = loss_fn_train(output, target)
                    loss_cls = loss_fn_classifier(output_classifer, classif_label)
                    loss =  0.7 *  (0.6 * loss_seg + 0.4 * loss_cls)
                else:
                    output = model(image)
                    loss = 0.7 * loss_fn_train(output, target)
                loss.backward()
                optimizer.step()
                sum_losses += loss.item()

        
        for i, data in enumerate(train_loader):
            if Open_Classifer:
                image, target, classif_label = data
                image, target, classif_label = image.to(DEVICE), target.float().to(DEVICE), classif_label.to(DEVICE)
            else:
                image, target = data
                image, target = image.to(DEVICE), target.float().to(DEVICE)
            # print(classif_label.shape)
            optimizer.zero_grad()
            if Open_Classifer:
                output, output_classifer = model(image)
                loss_seg = loss_fn_train(output, target)
                loss_cls = loss_fn_classifier(output_classifer, classif_label)
                loss = 0.6 * loss_seg + 0.4 * loss_cls
            else:
                output = model(image)
                loss = loss_fn_train(output, target)

            loss.backward()
            optimizer.step()
            sum_losses += loss.item()
            # print('Training [{}/{}]||batch[{}/{}]|batch_loss {: .8f}|'.format(epoch, EPOCHES + 1, i + 1,
            # len(train_loader),
            # loss.item()))
        # 检测 train_val 并且越小越好
        #     early_stopping_Train_loss(sum_losses / len(loader), model, mode='min')
        
        Trainlosses_Perepoch.append(sum_losses / len(train_loader))

        model.eval()

        vloss_dics = validation(model, val_loader_mul, [loss_fn_train, loss_fn_val])
        # Vallosses_Perepoch.append(vloss_train)
        Valdicelosses_Perepoch.append((vloss_dics))  # 修正 将diceloss作为考察学习计划考察对象

        # 修正加入
        vloss_dics = np.mean(vloss_dics)
        scheduler.step(vloss_dics)  # 修正 将diceloss作为考察学习计划考察对象
        #     scheduler.step()
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print("Finish One epoch Consuming time：" + time_str)

        # early_stopping_Val_trainloss(vloss_train, model,fold_idx ,mode="min")
        early_stopping_Val_Diceloss(vloss_dics, model, fold_idx, mode="min")
        if early_stopping_Val_Diceloss.early_stop:
            print("Message From Early Stop: Early stopping at {} epoch!!!!".format(epoch))
            break

    with open(HISTORY_JSON + str(fold_idx) + '_continue.json', 'w') as outfile:
        json.dump([Trainlosses_Perepoch, Valdicelosses_Perepoch], outfile)
    Choice_Vector.append(1 - np.min(Valdicelosses_Perepoch))
    # 回收loader vloader
    del train_loader, val_loader_mul, train_ds, valid_ds_mul
    gc.collect()

with open(BEST_CHOICE_JSON, 'w') as outfile:
    json.dump(Choice_Vector, outfile)

## 得到 结果最好前4个的索引
best_choice = np.array(Choice_Vector)
best_choice_index_top4 = np.argpartition(best_choice, -5)[-5:]

# Now begin test
submmsion = pd.read_csv((pathlib.Path(DATA_PATH) / 'sample_submission.csv').as_posix(), index_col='id')
submmsion_ids = submmsion.index.values
print(submmsion_ids)  # 拿到了submission里的列表

trfm = T.Compose([
    T.ToPILImage(),
    # T.Resize(NEW_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

subm = {}
p = pathlib.Path(DATA_PATH)

model = get_model()

if Open_Parral_Trainning:
    model = torch.nn.DataParallel(model)
model.to(DEVICE)
## 遍历前4个表现最好的模型的下标
# ChpAtMin_id_7Val_diceloss.pth#

for i, filename_id in enumerate(submmsion_ids):
    filename = os.path.join(p, "test", filename_id + '.tiff')
    print(filename)
    with rasterio.open(filename, transform=IDNT) as dataset:
        preds_models = np.zeros(dataset.shape, dtype=np.float32)
        print("Test {}".format(filename))
        for element in best_choice_index_top4:

            model.load_state_dict(torch.load("./ChpAtMin_id_{}".format(element) + PTH_NAME + ".pth"))
            model.eval()
            print("Now using:" + "./ChpAtMin_id_{}".format(element) + PTH_NAME + ".pth")
            slices = make_grid(dataset.shape, window=TEST_WINDOW, min_overlap=TEST_MIN_OVERLAP)
            preds = np.zeros(dataset.shape, dtype=np.float32)
            if dataset.count != 3:
                print('Image file with subdatasets as channels')
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
                    if Open_Classifer:
                        image = image.to(DEVICE)[None]  # 这里加入的是batch维度 这里的测试是每张图的测试
                        score = model(image)[0][0][0]

                        score2 = model(torch.flip(image, [0, 3]))[0]
                        score2 = torch.flip(score2, [3, 0])[0][0]

                        score3 = model(torch.flip(image, [1, 2]))[0]
                        score3 = torch.flip(score3, [2, 1])[0][0]
                    else:
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
        submit_preds = (preds_models > Sigmoid_Threshold).astype(np.uint8)
    subm[i] = {'id': filename_id, 'predicted': rle_numba_encode(submit_preds)}
    del preds_models
    gc.collect()

submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv(SUBMISSION_NAME_1 + ".csv", index=False)

# # 第二个模型的测试
# model.load_state_dict(torch.load("./ChpAtMinVal_diceloss.pth"))
# model.eval()
# for i, filename_id in enumerate(submmsion_ids):
#     filename = os.path.join(p,"test",filename_id+'.tiff')
#     print(filename)
#     with rasterio.open(filename) as dataset:
#         slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)
#         preds = np.zeros(dataset.shape, dtype=np.uint8)
#         if dataset.count != 3:
#             print('Image file with subdatasets as channels')
#             layers = [rasterio.open(subd) for subd in dataset.subdatasets]
#         print("Test {}".format(filename))
#
#         for index, (x1, x2, y1, y2) in enumerate(slices):
#             if dataset.count == 3:  # normal
#                 image = dataset.read([1, 2, 3],
#                                      window=Window.from_slices((x1, x2), (y1, y2)))
#                 image = np.moveaxis(image, 0, -1)
#             else:  # with subdatasets/layers
#                 image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
#                 for fl in range(3):
#                     image[:, :, fl] = layers[fl].read(window=Window.from_slices((x1, x2), (y1, y2)))
#
#             #           print("Test {}-{}:Shape is:{}".format(filename,index,image.shape))
#             image = trfm(image)
#             with torch.no_grad():
#                 image = image.to(DEVICE)[None]  # 这里加入的是batch维度 这里的测试是每张图的测试
#                 score = model(image)[0][0]
#
#                 score2 = model(torch.flip(image, [0, 3]))
#                 score2 = torch.flip(score2, [3, 0])[0][0]
#
#                 score3 = model(torch.flip(image, [1, 2]))
#                 score3 = torch.flip(score3, [2, 1])[0][0]
#
#                 score_mean = (score + score2 + score3) / 3.0
#                 score_sigmoid = score_mean.sigmoid().cpu().numpy()
#                 #                 score_sigmoid = score.sigmoid().cpu().numpy()
#                 score_sigmoid = cv2.resize(score_sigmoid, (WINDOW, WINDOW))
#
#                 preds[x1:x2, y1:y2] = (score_sigmoid > 0.5).astype(np.uint8)
#
#     subm[i] = {'id': filename_id, 'predicted': rle_numba_encode(preds)}
#     del preds
#     gc.collect()
#
# submission = pd.DataFrame.from_dict(subm, orient='index')
# submission.to_csv(SUBMISSION_NAME_2, index=False)
#

