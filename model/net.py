import torch
import torch.nn as nn

import torch
from torch.nn import Module, Conv2d, Parameter, Softmax


class Encoder(nn.Module):
    """(convolution => [BN] => ReLU) 2 times"""
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.flatten = nn.Flatten(start_dim=1)
        # hw = (patch_size // 2) **2
        # self.linear = nn.Sequential(
        #     nn.Linear(out_channels*hw, 20)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.pam(x)
        # x = self.flatten(x)
        # x = self.linear(x)
        return x

class Decoder(nn.Module):
    """(convolution => [BN] => ReLU) 2次"""
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        # self.linear = nn.Sequential(
        #     nn.Linear(20, in_channels * ((patch_size // 2) ** 2))
        # )
        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(in_channels, (patch_size // 2), (patch_size // 2)))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=(3, 3), padding=1),
        )
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x = self.linear(x)
        # x = self.unflatten(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.up(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x).squeeze()
        x = self.fc(x)
        x = self.softmax(x)
        return x

class COAE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels)
        )

    def forward(self, x):
        x = self.avg(x).squeeze()
        Rx = self.fc(x)
        return x, Rx