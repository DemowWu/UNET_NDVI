import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ndvi_models.base_networks import *
import skimage as sk
import math

from torch.autograd import Variable
import skimage.metrics as measure


class SRCNN(nn.Module):

    def __init__(self, params):
        super(SRCNN, self).__init__()
        self.num_channels = params

        self.layers = torch.nn.Sequential(
            ConvBlock(3, self.num_channels, 9, 1, 4, norm=None),  # 144*144*64 # conv->batchnorm->activation
            ConvBlock(self.num_channels, self.num_channels // 2, 1, 1, 0, norm=None),  # 144*144*32
            ConvBlock(self.num_channels // 2, 3, 5, 1, 2, activation=None, norm=None)  # 144*144*1
        )

    def forward(self, s):
        out = self.layers(s)
        return out


def loss_fn(outputs, labels):
    N, C, H, W = outputs.shape

    mse_loss = torch.sum((outputs - labels) ** 2) / N / C  # each photo, each channel
    mse_loss *= 255 * 255
    mse_loss /= H * W
    # average loss on each pixel(0-255)
    return mse_loss


def accuracy(outputs, labels):
    N, _, _, _ = outputs.shape
    psnr = 0
    for i in range(N):
        psnr += measure.peak_signal_noise_ratio(labels[i], outputs[i])
    return psnr / N


def ssim(outputs, labels):
    N, _, _, _ = outputs.shape
    ssim = 0
    for i in range(N):
        ssim += measure.structural_similarity(labels[i], outputs[i], data_range=3, multichannel=True)
    return ssim / N


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'PSNR': accuracy,
    'SSIM': ssim,
    # could add more metrics such as accuracy for each token type
}
