import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

    def forward(self, x1, x2):
        output1 = self.backbone(x1)
        output2 = self.backbone(x2)
        return output1, output2


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
    """
    Siamese Network Dataset for loading pairs of images.
    """
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # 我们需要确保第二张图片属于同一类别（label=1）或不同类别（label=0）
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                # 在相同类别中选择另一张图片
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                # 在不同类别中选择另一张图片
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img0_tuple[1]!=img1_tuple[1])],dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


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

