from torchvision import models
from torchvision.models import ResNet34_Weights
from torch import nn
# 使用新的权重方式
resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
num_ftrs = resnet34.fc.in_features

# 保持in_features不变，修改out_features=2
resnet34.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.LogSoftmax(dim=1))

model = resnet34
model.cuda()

if __name__ == '__main__':
    print(model)