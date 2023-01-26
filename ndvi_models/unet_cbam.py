from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

from ndvi_models.attention.CBAM import CBAMBlock


def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Unet6_512_CBAM1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        p5 = self.down_pooling(x5)
        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        ux5 = self.att5(x5)  # add attetion 5
        ux5 = torch.cat([up5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = torch.cat([up4, x4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)
        return output


class Unet6_512_CBAM2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        p5 = self.down_pooling(x5)
        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        ux5 = self.att5(x5)
        ux5 = torch.cat([up5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = self.att4(x4)  # add attetion 4
        ux4 = torch.cat([up4, ux4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.att3 = CBAMBlock(channel=64, reduction=8, kernel_size=3)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)

        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        ux5 = self.att5(x5)
        ux5 = torch.cat([up5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = self.att4(x4)
        ux4 = torch.cat([up4, ux4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = self.att3(x3)  # add attetion 3
        ux3 = torch.cat([up3, ux3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.att3 = CBAMBlock(channel=64, reduction=8, kernel_size=3)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.att2 = CBAMBlock(channel=32, reduction=8, kernel_size=3)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)

        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        ux5 = self.att5(x5)
        ux5 = torch.cat([up5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = self.att4(x4)
        ux4 = torch.cat([up4, ux4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = self.att3(x3)
        ux3 = torch.cat([up3, ux3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = self.att2(x2)  # add attetion 4
        ux2 = torch.cat([up2, ux2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        p5 = self.down_pooling(x5)
        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        up5 = self.att5(up5)  # add attetion 5 in up route
        ux5 = torch.cat([up5, x5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = torch.cat([up4, x4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)
        return output


class Unet6_512_CBAM6(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        p5 = self.down_pooling(x5)
        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        up5 = self.att5(up5)
        ux5 = torch.cat([up5, x5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        up4 = self.att4(up4)  # add attetion 4
        ux4 = torch.cat([up4, x4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM7(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.att3 = CBAMBlock(channel=64, reduction=8, kernel_size=3)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)

        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        up5 = self.att5(up5)
        ux5 = torch.cat([up5, x5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        up4 = self.att4(up4)
        ux4 = torch.cat([up4, x4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        up3 = self.att3(up3)  # add attetion 3
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM8(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(512, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.att3 = CBAMBlock(channel=64, reduction=8, kernel_size=3)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.att2 = CBAMBlock(channel=32, reduction=8, kernel_size=3)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)

        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        up5 = self.att5(up5)
        ux5 = torch.cat([up5, x5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        up4 = self.att4(up4)
        ux4 = torch.cat([up4, x4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        up3 = self.att3(up3)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        up2 = self.att2(up2)  # add attetion 4
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM9(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(256 * 2, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.up_conv4 = conv_bn_leru(256, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        p5 = self.down_pooling(x5)
        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        ux5 = self.att5(x5)  # add attetion 5
        uat5 = self.att5(up5)  # add attetion 5
        ux5 = torch.cat([up5 + uat5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = torch.cat([up4, x4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)
        return output


class Unet6_512_CBAM10(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(256 * 2, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(128 * 2, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.up_conv3 = conv_bn_leru(128, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)

        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)

        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)

        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        p5 = self.down_pooling(x5)
        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        ux5 = self.att5(x5)
        uat5 = self.att5(up5)
        ux5 = torch.cat([up5 + uat5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        ux4 = self.att4(x4)  # add attetion 4
        uat4 = self.att4(up4)  # add attetion 4
        ux4 = torch.cat([up4 + uat4, ux4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        ux3 = torch.cat([up3, x3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM11(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(256 * 2, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(128 * 2, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.att3 = CBAMBlock(channel=64, reduction=8, kernel_size=3)
        self.up_conv3 = conv_bn_leru(64 * 2, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.up_conv2 = conv_bn_leru(64, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)

        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        uat5 = self.att5(up5)
        ux5 = self.att5(x5)
        ux5 = torch.cat([up5 + uat5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        uat4 = self.att4(up4)
        ux4 = self.att4(x4)
        ux4 = torch.cat([up4 + uat4, ux4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        uat3 = self.att3(up3)  # add attetion 3
        ux3 = self.att3(x3)  # add attetion 3
        ux3 = torch.cat([up3 + uat3, ux3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output


class Unet6_512_CBAM12(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.conv6 = conv_bn_leru(256, 512)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool5 = up_pooling(512, 256)
        self.att5 = CBAMBlock(channel=256, reduction=16, kernel_size=3)
        self.up_conv5 = conv_bn_leru(256 * 2, 256)

        self.up_pool4 = up_pooling(256, 128)
        self.att4 = CBAMBlock(channel=128, reduction=8, kernel_size=3)
        self.up_conv4 = conv_bn_leru(128 * 2, 128)

        self.up_pool3 = up_pooling(128, 64)
        self.att3 = CBAMBlock(channel=64, reduction=8, kernel_size=3)
        self.up_conv3 = conv_bn_leru(64 * 2, 64)

        self.up_pool2 = up_pooling(64, 32)
        self.att2 = CBAMBlock(channel=32, reduction=8, kernel_size=3)
        self.up_conv2 = conv_bn_leru(32 * 2, 32)

        self.up_pool1 = up_pooling(32, 16)
        self.up_conv1 = conv_bn_leru(32, 16)

        self.conv_out = nn.Conv2d(16, out_ch, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
        p5 = self.down_pooling(x5)

        x6 = self.conv6(p5)

        up5 = self.up_pool5(x6)
        uat5 = self.att5(up5)
        ux5 = self.att5(x5)
        ux5 = torch.cat([up5 + uat5, ux5], dim=1)
        ux5 = self.up_conv5(ux5)

        up4 = self.up_pool4(ux5)
        uat4 = self.att4(up4)
        ux4 = self.att4(x4)
        ux4 = torch.cat([up4 + uat4, ux4], dim=1)
        ux4 = self.up_conv4(ux4)

        up3 = self.up_pool3(ux4)
        uat3 = self.att3(up3)
        ux3 = self.att3(x3)
        ux3 = torch.cat([up3 + uat3, ux3], dim=1)
        ux3 = self.up_conv3(ux3)

        up2 = self.up_pool2(ux3)
        uat2 = self.att2(up2)  # add attetion 4
        ux2 = self.att2(x2)  # add attetion 4
        ux2 = torch.cat([up2 + uat2, ux2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output
