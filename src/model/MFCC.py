

from numpy.lib.npyio import mafromtxt
from torch.nn.modules.conv import Conv2d
import torchvision.models as models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC
from model.model import CNNBlock, MLPBlock


class CONV2dblock(nn.Module):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 2, dropout_p: float = 0.0):
        super().__init__()
        conv_block = Conv2d(in_channels, out_channels,
                            kernel_size, padding='same')

        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        dropout = nn.Dropout(p=dropout_p)
        pooling = nn.MaxPool2d(pool_size)

        self.net = nn.Sequential(conv_block, bn, relu, dropout, pooling)

    def forward(self, x):
        return self.net(x)


class mfcc(nn.Module):
    def __init__(self, chunk_len, device, n_classes):
        super().__init__()
        self.num_mfcc = 33
        self.ln1 = nn.LayerNorm(chunk_len)
        self.mfcc = MFCC(n_mfcc=33).to(device)
        self.cnn_blocks = nn.Sequential(CONV2dblock(1, 32, 3),
                                        CONV2dblock(32, 32*2, 3),
                                        CONV2dblock(32*2, 32*3, 3))
        self.flatten = nn.Flatten(start_dim=1)
        self.ln2 = nn.LayerNorm((32*32)//64*96)
        self.mlp_blocks = nn.Sequential(MLPBlock((32*32)//64*96, 2048),
                                        MLPBlock(2048, 2048),
                                        MLPBlock(2048, 2048))
        self.classification_head = nn.Linear(2048, n_classes)

    def forward(self, wavs):
        x = self.ln1(wavs)
        x = self.mfcc(x)
        x = self.cnn_blocks(x)
        x = self.flatten(x)
        x = self.ln2(x)
        x = self.mlp_blocks(x)
        return self.classification_head(x)
