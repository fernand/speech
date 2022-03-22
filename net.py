import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet.resnet18(pretrained=False, progress=False, num_classes=1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.resnet(x)
