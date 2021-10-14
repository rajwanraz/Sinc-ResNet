import typing as tp
import torchvision.models as models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import CNNBlock, MLPBlock, SincConv


class SincNet(nn.Module):
    def __init__(self, chunk_len, n_classes, first_block):
        super().__init__()
        ln1 = nn.LayerNorm(chunk_len)
        cnn_blocks = nn.Sequential(CNNBlock(chunk_len, first_block, 1, 80, 251),
                                   CNNBlock(chunk_len // 3,
                                            nn.Conv1d, 80, 60, 5),
                                   CNNBlock(chunk_len // 9, nn.Conv1d, 60, 60, 5))
        flatten = nn.Flatten(start_dim=1)
        ln2 = nn.LayerNorm(chunk_len // 27 * 60)
        mlp_blocks = nn.Sequential(MLPBlock(chunk_len // 27 * 60, 2048),
                                   MLPBlock(2048, 2048),
                                   MLPBlock(2048, 2048))
        self.backbone = nn.Sequential(
            ln1, cnn_blocks, flatten, ln2, mlp_blocks)
        self.classification_head = nn.Linear(2048, n_classes)

    def forward(self, wavs):
        return self.classification_head(self.backbone(wavs))
