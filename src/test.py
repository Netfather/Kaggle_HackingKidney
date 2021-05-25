# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         test
# Description:  为pytorch.kernel进行测试
# Author:       Administrator
# Date:         2021/3/8
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



# def get_model():
#     model = torchvision.models.segmentation.fcn_resnet50(False)
#
#     # pth = torch.load("./fcn_resnet50_coco-1167a1af.pth")
#     # for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias",
#     #             "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked",
#     #             "aux_classifier.4.weight", "aux_classifier.4.bias"]:
#     #     del pth[key]
#     #
#     # model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
#     return model
#
# model = get_model()
# print(model)
# test = model.backbone.layer1[0].conv1
#
#
# print(test.weight)



# model = get_model()
# input = torch.randn(size= [2,3,512,512])
# net = model.eval()
# output = net(input)
# # print(len(output))
# print(output["out"])   # torch.Size([2, 1, 512, 512])

## 测试从v.resize 是否会创建新的空间用于存储数据
# 确认cv2 会创建一块全新的内存空间用于存储图像
# import cv2
# import gc
# import numpy as np
#
# a = np.ones(shape= [1024,1024])
# b =cv2.resize(a,(256,256))
# del a
# gc.collect()
#
# print(b)
# print(b.shape)


##
import segmentation_models_pytorch as smp

# class Model(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


# model = smp.Unet(
#     encoder_name="efficientnet-b1",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
#     in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your dataset))
#     decoder_attention_type="scse",
#     encoder_depth=5,
#     decoder_channels=[256, 128, 64, 32, 16],
#     aux_params =
#     {
#         "classes":1,
#         "pooling":"avg",
#         "dropout": 0.2,
#         "activation": None
#     }
# )
# DEVICE = "cuda:0"
# model.to(DEVICE)
# model.eval()
#
# test = torch.ones(size=(3,3,1024,1024)).to(DEVICE)
# output = model(test)
# print(output[0].cpu().detach().numpy().shape)
# print(output[1].cpu().detach().numpy().shape)
# print(output)
#
# # print(model)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 用gpu还是cpu
# model.to(DEVICE)
# summary(model,(1,240,320))


### 测试Kfold

# def set_seeds(seed=23):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# set_seeds()
#
#
# DATA_PATH = './hubmap-kidney-segmentation'  # 数据存放位置
# tiff_ids = np.array([x.split('/')[-1][:-5] for x in glob.glob('./hubmap-kidney-segmentation/train/*.tiff')])
# skf = KFold(n_splits=8,shuffle= True) # 用Kfold方法进行分割
# Trainlosses_Perepoch = []
# Vallosses_Perepoch = []
# print(tiff_ids)
# for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tiff_ids, tiff_ids)):
#         print("This {} times is split into files below".format(fold_idx))
#         print(train_idx)
#         print(val_idx)


# submmsion = pd.read_csv((pathlib.Path(DATA_PATH) / 'sample_submission.csv').as_posix(),index_col='id')
# submmsion_ids = submmsion.index.values
# print(submmsion_ids)  # 测试成功 拿到了submission里的列表
#
# subm = {}
# p = pathlib.Path(DATA_PATH)
#
#
# for i, filename_id in enumerate(submmsion_ids):
#     filename = os.path.join(p,"test",filename_id+'.tiff')
#     print(filename)
#     with rasterio.open(filename) as dataset:
#         preds = np.zeros(dataset.shape, dtype=np.uint8)
#         if dataset.count != 3:
#             print('Image file with subdatasets as channels')
#             layers = [rasterio.open(subd) for subd in dataset.subdatasets]
#         print("Test {}".format(filename))
#     subm[i] = {'id': filename_id, 'predicted': 0}
#     del preds
#     gc.collect()
#
# submission = pd.DataFrame.from_dict(subm, orient='index')
# submission.to_csv("test_predict.csv", index=False)

###

# a  = np.array([2,4,3,2,2])
# print(np.min(a))
# Choice_Vector = [1,4,5,6,9,9,5,6,7]
# best_choice = np.array(Choice_Vector)
# best_choice_index_top4 = np.argpartition(best_choice, -5)[-5:]
# print(best_choice_index_top4)


## c而是 persudo代码  生成persudo使用的mask文件
# 如下代码主要用于合成 伪标签

#1. 把train中的encoding 和  训练完成的persudo 两个编码部分合成到一个表单中
# train_ids = ["aaa6a05cc", "26dc41664", "b9a3865fc",
#             "2f6ecfcdf", "c68fe75ea", "e79de561c",
#             "1e2425f28", "b2dc8411c", "afa5e8098",
#             "0486052bb", "cb2d976f4", "54f2eec69",
#             "4ef6695ce", "8242609fa", "095bf7a1f"]
# persudo_ids = ["2ec3f1bb9","3589adb90","57512b7f1","aa05346ff","d488c759a"]
#
#
# # 4月9日修正   将d488图的Persudo更替为别人提供的
# # persudo_ids = ["2ec3f1bb9","3589adb90","57512b7f1","aa05346ff"]
# # zhao_persudo_d488_id = ["d488c759a"]
#
#
# tiff_ids = ["aaa6a05cc", "26dc41664", "b9a3865fc",
#             "2f6ecfcdf", "c68fe75ea", "e79de561c",
#             "1e2425f28", "b2dc8411c", "afa5e8098",
#             "0486052bb", "cb2d976f4", "54f2eec69",
#             "4ef6695ce", "8242609fa", "095bf7a1f",
#             "2ec3f1bb9","3589adb90","57512b7f1",
#             "aa05346ff","d488c759a"]
#
# subm = {}
# # # 2. 尝试拿出各自的编码
# root_dir = r'./hubmap-kidney-segmentation'
# path = pathlib.Path(root_dir)
#
# train_csv = pd.read_csv((path / 'train.csv').as_posix(),
#                                index_col=[0])
# print(train_csv.index.values)
# for i,filename in enumerate(train_ids,0):
#     print(i)
#     subm[i] = {'id': filename, 'encoding': train_csv.loc[filename, 'encoding']}
#     # print(train_csv.loc[filename, 'encoding'])
#
#
# persudo_csv = pd.read_csv((path / 'Persudo_Mix_0.937.csv').as_posix(),index_col=[0])
# for i,filename in enumerate(persudo_ids,15):
#     print(i)
#     subm[i] = {'id': filename, 'encoding': persudo_csv.loc[filename, 'predicted']}
#     # print(persudo_csv.loc[filename, 'predicted'])
#
#
# # zhao_persudo_csv = pd.read_csv((path / 'zhao_persudo.csv').as_posix(),index_col=[0])
# # for i,filename in enumerate(zhao_persudo_d488_id,19):
# #     print(i)
# #     subm[i] = {'id': filename, 'encoding': zhao_persudo_csv.loc[filename, 'predicted']}
# #     # print(persudo_csv.loc[filename, 'predicted'])
#
# submission = pd.DataFrame.from_dict(subm, orient='index')
# submission.to_csv("TrainPersudo_0.937_Mix.csv", index=False)
#
# test_csv = pd.read_csv(('./TrainPersudo_0.937_Mix.csv'),
#                                index_col=[0])
# print(test_csv.index.values)
# print("Complete!")


##
# 删除掉文件中的其他部分 只测试 d488文件 查看提升幅度
# test_id =["2ec3f1bb9","3589adb90","57512b7f1","aa05346ff","d488c759a"]  # 只填写你需要保留的ID
# csv_path = r"C:\Users\Administrator\Desktop\labserver_Humap\D_0.933.csv"
# subm = {}
#
# input_csv = pd.read_csv(csv_path,
#                                index_col=[0])
# print(input_csv.index.values)
# for i,filename in enumerate(test_id,0):
#     print(i)
#     if filename == "d488c759a":
#         subm[i] = {'id': filename, 'predicted': input_csv.loc[filename, 'predicted']}
#     # print(train_csv.loc[filename, 'encoding'])
#     else:
#         subm[i] = {'id': filename, 'predicted': ""}
#
#
# submission = pd.DataFrame.from_dict(subm, orient='index')
# submission.to_csv("./D_933_d488c759a.csv", index=False)

### 尝试修正图片中的overlap部分直接赋值  而不是求平均的问题

# a = np.array([[1,2,0],
#               [0,3,4],
#               [5,0,6]
#               ])
#
# b = np.array([[3,2,6],
#               [8,3,4],
#               [5,12,6]
#               ])
#
# c = np.where(a != 0,(a + b) / 2,b)
#
# print(c)
###
# list = [1,2,3,4]
# res = np.mean(list)
# print(res)
### 测试将  smp  Unet结构中的  Encoder部分提取出来 自己手动做 Decoder
# model = smp.Unet(
#     encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
#     in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#     classes=1,  # model output channels (number of classes in your dataset))
#     decoder_attention_type="scse",
#     encoder_depth=5,
#     decoder_channels=[256, 128, 64, 32, 16],
# )
# DEVICE = "cuda:0"
# model.to(DEVICE)
# model.eval()
#
# # from torchsummary import summary
# # summary(model,input_size= [3,256,256],device= "CUDA")
# test = torch.ones(size=(3,3,512,512)).to(DEVICE)
# output = model(test)
# print(output.shape)
### 测试dice loss评分是否可以按patch 分块计算 然后求平均值
# def np_dice_score(probability, mask):
#     p = probability.reshape(-1)
#     t = mask.reshape(-1)
#
#     p = p > 0.5
#     t = t > 0.5
#     uion = p.sum() + t.sum()
#
#     overlap = (p * t).sum()
#
#     if (uion == 0):
#         return 0.
#     dice = 2 * overlap / (uion)
#     return dice
#
# def np_dice_score_perbatch(probability, mask):
#     p = probability.reshape(-1)
#     t = mask.reshape(-1)
#
#     p = p > 0.5
#     t = t > 0.5
#     uion = p.sum() + t.sum()
#
#     overlap = (p * t).sum()
#
#     return overlap,uion
#
# predict = np.array([
#     [1,1,0,0],
#     [1,1,0,0],
#     [1,0,0,1],
#     [0,1,1,0]
# ])
#
# mask = np.array([
#     [0,0,0,0],
#     [0,0,0,0],
#     [0,0,0,1],
#     [0,1,1,0]
# ])
#
# dice_true =np_dice_score(predict,mask)
# print("dice_true:", dice_true)
#
# # 下面按照每个patch进行分数的计算
# dice1 = 0.
# p = predict[0:2,0:2]
# m = mask[0:2,0:2]
# dice1 += np_dice_score(p,m)
#
# p = predict[2:4,0:2]
# m = mask[2:4,0:2]
# dice1 += np_dice_score(p,m)
#
# p = predict[0:2,2:4]
# m = mask[0:2,2:4]
# dice1 += np_dice_score(p,m)
#
# p = predict[2:4,2:4]
# m = mask[2:4,2:4]
# dice1 += np_dice_score(p,m)
# print("dice_false:", dice1 / 4.)
#
#
# # 下面是修正写法 按照每个batch  我们统计 sum 和  然后最后统一相加
# Fenzi = 0.
# Fenmu = 0.
# dice1 = 0.
# p = predict[0:2,0:2]
# m = mask[0:2,0:2]
# res = np_dice_score_perbatch(p,m)
# Fenzi += res[0]
# Fenmu += res[1]
#
# p = predict[2:4,0:2]
# m = mask[2:4,0:2]
# res = np_dice_score_perbatch(p,m)
# Fenzi += res[0]
# Fenmu += res[1]
#
# p = predict[0:2,2:4]
# m = mask[0:2,2:4]
# res = np_dice_score_perbatch(p,m)
# Fenzi += res[0]
# Fenmu += res[1]
#
# p = predict[2:4,2:4]
# m = mask[2:4,2:4]
# res = np_dice_score_perbatch(p,m)
# Fenzi += res[0]
# Fenmu += res[1]
# print("dice_xiuzheng:", 2.* Fenzi / Fenmu )

# 测试成功！

### 此文件用于抽调各个文件中表现最好的模块
test_id =["2ec3f1bb9","3589adb90","57512b7f1","aa05346ff","d488c759a"]  # 只填写你需要保留的ID
SW_csv_path = r"C:\Users\Administrator\Desktop\labserver_Humap\persudo_UEb1_PersudoItr2_0.925_ExData_1.0HSV_512.csv"
Vas_csv_path = r"C:\Users\Administrator\Desktop\labserver_Humap\Vas_submission_0.933.csv"
D_csv_path = r"C:\Users\Administrator\Desktop\labserver_Humap\D_0.933.csv"
subm = {}

input_csv = pd.read_csv(SW_csv_path,
                               index_col=[0])
input_csv_vas = pd.read_csv(Vas_csv_path,
                               index_col=[0])
input_csv_512 = pd.read_csv(D_csv_path,
                               index_col=[0])

print(input_csv.index.values)
for i,filename in enumerate(test_id,0):
    print(i)
    if filename == "d488c759a":
        subm[i] = {'id': filename, 'predicted': input_csv_vas.loc[filename, 'predicted']}
    # print(train_csv.loc[filename, 'encoding'])
    elif filename == "57512b7f1":
        subm[i] = {'id': filename, 'predicted': input_csv_512.loc[filename, 'predicted']}
    elif filename == "2ec3f1bb9":
        subm[i] = {'id': filename, 'predicted': input_csv_512.loc[filename, 'predicted']}
    else:
        subm[i] = {'id': filename, 'predicted': input_csv.loc[filename, 'predicted']}


submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv("./submission_mix_Itr3.csv", index=False)

###  将nn para 转换为 普通能用的
# def get_model():
#     model = smp.Unet(
#         encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights=None,  # use `imagenet` pretreined weights for encoder initialization
#         in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#         classes=1,  # model output channels (number of classes in your dataset))
#         decoder_attention_type="scse",
#         encoder_depth=5,
#         decoder_channels=[256, 128, 64, 32, 16],
#     )
#     return model
#
# # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = "cuda:1"
# model = get_model()
#
# # 如下代码进行权重的load
# # model = torch.nn.DataParallel(model)
#
# model.load_state_dict(
#                         torch.load("ChpAtMin_id_1ML1_GPU01_Val_diceloss.pth",map_location= DEVICE))

# model.to(DEVICE)
#
# torch.save(model.module.state_dict(), "ChpAtMin_id_1ML1_GPU01_Val_diceloss_Single.pth")
###



