
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image

class mydataset(Dataset):
    def __init__(self, path,transform=None):
        filenames=[]
        labels1 = []
        labels2 = []
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split()  # 移除行尾的换行符并按空格分割
                filename = parts[0]
                label1 = int(parts[1])
                label2 = int(parts[2])
                filenames.append(filename)
                labels1.append(label1)
                labels2.append(label2)
        mapping = {
                            '00': 0,
                            '10': 1,
                            '11': 2,
                            '01': 3
                        }        
        combined_labels_str = np.core.defchararray.add(np.array(labels1).astype(str), np.array(labels2).astype(str))
        new_combined_labels = np.vectorize(mapping.get)(combined_labels_str)
        self.all_image_paths = filenames
        self.all_image_labels = new_combined_labels
        self.transform=transform
    

    def __getitem__(self, index):
        img = Image.open('dataset/CXR/images/'+self.all_image_paths[index]).convert('RGB')
        img=self.transform(img)
        label=self.all_image_labels[index]
        label=torch.tensor(label,dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.all_image_paths)
    
data_transforms = transforms.Compose([
    transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
])

train_path='dataset/CXR/train.txt'
test_path='dataset/CXR/test.txt'

train_data=mydataset(train_path,data_transforms)
test_data=mydataset(test_path,data_transforms)

# train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
# test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

# train_data_size=len(train_data)
# test_data_size=len(test_data)
