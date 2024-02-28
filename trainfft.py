
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from PIL import Image
import os
import random




class fftSiameseNetwork(nn.Module):
    def __init__(self):
        super(fftSiameseNetwork, self).__init__()
        # 使用ImageNet预训练的ResNet18作为backbone
        self.backbone = models.resnet18(pretrained=True)
        # 移除原始ResNet18的最后一个全连接层
        self.backbone.fc = nn.Identity()
        # 定义一个新的全连接层来处理合并后的特征
        self.fc = nn.Linear(self.backbone.fc.in_features * 2, 128)  # 假设原始特征和FFT特征拼接

    def forward(self, x1, x2):
        # 对输入图像x1和x2进行FFT变换
        fft1 = self.fft_to_image(x1)
        fft2 = self.fft_to_image(x2)
        
        # 分别处理原始图像和FFT图像，通过backbone
        output1_orig = self.backbone(x1)
        output2_orig = self.backbone(x2)
        output1_fft = self.backbone(fft1)
        output2_fft = self.backbone(fft2)
        
        # 将原始图像特征和FFT图像特征进行concat
        output1_concat = torch.cat((output1_orig, output1_fft), dim=1)
        output2_concat = torch.cat((output2_orig, output2_fft), dim=1)

        # 通过一个全连接层进行特征降维
        output1 = self.fc(output1_concat)
        output2 = self.fc(output2_concat)

        return output1, output2

    def fft_to_image(self, x):
        # 实现FFT变换，并将结果转换为可用于模型的格式
        # 注意：这里简化处理，实际应用中需要根据输入图像的具体格式调整
        x_fft = torch.fft.fft2(x)
        x_fft_shift = torch.fft.fftshift(x_fft)
        x_fft_mag = torch.log(torch.abs(x_fft_shift) + 1)
        # 保持与原图像相同的维度
        x_fft_mag = torch.unsqueeze(x_fft_mag, 1)
        return x_fft_mag

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SiameseNetworkDataset(Dataset):
    # 这里需要实现数据集的加载，包括transform等
    pass


model = SiameseNetwork()
criterion = ContrastiveLoss()
bce_loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


def train(model, dataloader, criterion, bce_loss, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            img1, img2, label = data
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            contrastive_loss = criterion(output1, output2, label)
            bce_loss = bce_loss(output1, label.float())
            loss = contrastive_loss + bce_loss
            loss.backward()
            optimizer.step()


dataset = SiameseNetworkDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train(model, dataloader, criterion, bce_loss, optimizer)

