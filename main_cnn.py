import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from resnet50 import model

def main():
    

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    state_dict=torch.load('pth/model_epoch_19.pth')
    model.load_state_dict(state_dict)
    target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
    
])
    # load image
    img_path = "dataset\CXR\images\P_1_126.jpeg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img_a=img
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)
    print(img.shape)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    
    transform_seq= transforms.Compose([
        transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
        transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
     ])
    img=transform_seq(img_a)
    img=np.array(img)
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
