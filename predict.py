import numpy as np
import cv2 as cv
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from dataset_CXR import train_data,test_data
from resnet50 import model
import sys

data_transforms = transforms.Compose([
    transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
])
def predict(model,path):
    img=Image.open(path).convert("RGB")
    img=data_transforms(img)
    img = img.unsqueeze(0).repeat(1, 1, 1, 1)
    output=model(img.cuda())
    _, predicted = torch.max(output.data, 1)
    return predicted

state_dict=torch.load('pth/model_epoch_19.pth')
model.load_state_dict(state_dict)
a=predict(model,'C:/Users/armstrong/Desktop/sth/COVID_CXR/dataset/CXR/images/P_173.jpeg')
print(a)