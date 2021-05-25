# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# Name:         model_effiUnet
# Description:  此文件返回训练所需的网络结构
# Author:       Administrator
# Date:         2021/5/20
# -------------------------------------------------------------------------------
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp


def get_model(name:str,input_channels:int,output_channels:int):
    if name != "efficientnet-b1" and name != "efficientnet-b3":
        raise("model_name error! Only support efficientnetb1 or efficientnetb3")

    model = smp.Unet(
        encoder_name=name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
        in_channels=input_channels,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=output_channels,  # model output channels (number of classes in your dataset))
        decoder_attention_type="scse",
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    return model

# check #################################################################

def run_check_net():
    batch_size = 7
    C,H,W = 3, 224, 224
    image = torch.randn((batch_size,C,H,W))

    net = get_model(name="efficientnet-b1",input_channels=3,output_channels=1)
    net.train()

    logit = net(image)
    print('')
    print(image.shape)
    print('---')

    print(logit.shape)
    print('---')
    print("Forward OK!")


# main #################################################################
if __name__ == '__main__':
     run_check_net()

