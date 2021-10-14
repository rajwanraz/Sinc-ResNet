

import torchvision.models as models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import SincConv


class Resnet(nn.Module):
    def __init__(self, pretrained=True, channel_output=462):
        super(Resnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, channel_output, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class ResincNet(nn.Module):
    def __init__(self, sample_rate=16000, chunk_len=3200, n_classes=462):
        super(ResincNet, self).__init__()
        self.layerNorm = nn.LayerNorm([1, chunk_len])
        kernel1 = 251
        kernel1 = 501
        kernel1 = 1001
        out_channels = 80
        self.sincNet1 = nn.Sequential(
            SincConv(1, out_channels, kernel1, padding=(kernel1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(256))
        self.sincNet2 = nn.Sequential(
            SincConv(1, out_channels, kernel1, padding=(kernel1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(256))
        self.sincNet3 = nn.Sequential(
            SincConv(1, out_channels, kernel1, padding=(kernel1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(256))
        self.resnet = Resnet(pretrained=True)

    def forward(self, x):
        x = self.layerNorm(x)

        feat1 = self.sincNet1(x)
        feat2 = self.sincNet2(x)
        feat3 = self.sincNet3(x)

        x = torch.cat((feat1.unsqueeze_(dim=1),
                       feat2.unsqueeze_(dim=1),
                       feat3.unsqueeze_(dim=1)), dim=1)
        x = self.resnet(x)
        return x
