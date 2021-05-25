# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         Unet_EffieientnetB7_MainFlow
# Description:  这里实现Unet 通过efficientNet作为特征提取器的网络
# Author:       Administrator
# Date:         2021/3/15
# -------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES']  ="0"

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
from torchvision import transforms as T
import random
import pathlib
from datetime import datetime
import rasterio  # 由于新图像格式不太一致，使用rasterio会读不出某些图片 因此改为使用tiff. # 更新 tiff会爆内存 因此还是使用rasterio
from rasterio.windows import Window
from sklearn.model_selection import KFold
import json
from toolbox.early_box.earlystop import *
from toolbox.loss_box.binaray_loss import *
from toolbox.rle_str_box.rel_enco_deco import *
from toolbox.model_effiUnet import *
from toolbox.util import *

###################################################################################
# 关键参数定义
SUBMISSION_NAME_1 = 'submission_UEb1_NewPersudo_0.937_512_Finalsub'
HISTORY_JSON = 'ML3_GPU1_losshistory_'
BEST_CHOICE_JSON = 'ML3_GPU1_BestChoice.json'
PTH_NAME = 'ML3_GPU1_Val_diceloss'
# SUBMISSION_NAME_2 = 'submission_U++EB3_E100_B32_0.8Focal_diceloss.csv'
WINDOW = 512 * 2
MIN_OVERLAP = 32
NEW_SIZE = 512
DataSet_ThreShold = 100
Open_Parral_Trainning = False  # 是否开启并行化  True表示开启 False表示不开启

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

##########################################################################################
# 设定随机种子，方便复现代码
def set_seeds(seed=9048):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seeds()


############################################################################################
#相关函数
# 2021.4.21 修正
@torch.no_grad()
def validation(model, loader):
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
    return Score_Perfile


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

    model = get_model(name="efficientnet-b1",input_channels=3,output_channels=1)

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

    loss_focal = FocalLoss().to(DEVICE)
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

model = get_model(name="efficientnet-b1",input_channels=3,output_channels=1)

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


