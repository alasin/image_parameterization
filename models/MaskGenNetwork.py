import torch
import torch.nn as nn
from .common import *
import torch.nn.functional as F

'''
SR-GAN based model
'''

def swish(x):
    return x * F.sigmoid(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class MaskGenNetwork(nn.Module):
    def __init__(self, n_residual_blocks, sr_factor, out_channels=128, up_channels = 128, half_scale=2):
        super(MaskGenNetwork, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = int(np.log2(sr_factor//half_scale))

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor):
            self.add_module('upsample' + str(i+1), nn.UpsamplingBilinear2d(scale_factor=2))

        self.conv3 = nn.Conv2d(64, out_channels, 9, stride=1, padding=4)


    def forward(self, x):
        if not isinstance(x, list):
            x = swish(self.conv1(x))

            y = x.clone()
            for i in range(self.n_residual_blocks):
                y = self.__getattr__('residual_block' + str(i+1))(y)

            x = self.bn2(self.conv2(y)) + x

            for i in range(self.upsample_factor):
                x = self.__getattr__('upsample' + str(i+1))(x)

            output_1 = nn.functional.sigmoid(self.conv3(x))
            return output_1
        else:
            pass