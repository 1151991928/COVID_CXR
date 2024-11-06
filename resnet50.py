from torch import nn
import torchvision.models as models


resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features 

 
#保持in_features不变，修改out_features=4
resnet50.fc = nn.Sequential(nn.Linear(num_ftrs,4),nn.LogSoftmax(dim=1))


model=resnet50
model.cuda()
if __name__=='__main__':
    print(model)