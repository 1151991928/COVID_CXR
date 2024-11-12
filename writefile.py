import math
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from dataset_CXR_2label import train_data, test_data
from resnet50_2label import model
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import datetime

batch_size = 64
epochs = 5
data_transforms = transforms.Compose([
    transforms.Resize(256),  # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),  # 将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
])

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

#file and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_filename = str(current_time)+"pig.txt"
file=open(log_filename, mode='w')
file.write("Time\tepochs\tLoss\tAccuracy\tlearning_rate\n")

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

tb_writer = SummaryWriter('./log')

total_train_step = 0
total_test_step = 0


for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    # 训练阶段
    model.train()
    train_bar = tqdm(train_dataloader, file=sys.stdout)
    for data in train_bar:
        imgs, targets = data
        targets = targets.long()
        imgs, targets = imgs.cuda(), targets.cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1

        train_bar.set_description(f'Train Step: {total_train_step} Loss: {loss.item():.4f}')

    # 验证阶段
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    test_bar = tqdm(test_dataloader, file=sys.stdout)
    with torch.no_grad():
        for data in test_bar:
            imgs, targets = data
            imgs, targets = imgs.cuda(), targets.long().cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            test_bar.set_description(
                f'Test Loss: {total_test_loss.item() / (total_test_step + 1):.4f} Accuracy: {total_accuracy / test_data_size}')

    print(f'Epoch [{epoch + 1}/{epochs}] Train Step: {total_train_step} Loss: {loss.item():.4f}')
    print(
        f'Epoch [{epoch + 1}/{epochs}] Test Loss: {total_test_loss.item() / (total_test_step + 1):.4f} Accuracy: {total_accuracy / test_data_size}')
    tags = ["loss", "accuracy", "learning_rate"]
    tb_writer.add_scalar(tags[0], total_test_loss.item() / (total_test_step + 1), epoch)
    tb_writer.add_scalar(tags[1], total_accuracy / test_data_size, epoch)
    tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
    file.write()

    torch.save(model.state_dict(), 'pth/' + f'model_epoch_{epoch}.pth')
    total_test_step += 1
tb_writer.close()
print('Finished Training')