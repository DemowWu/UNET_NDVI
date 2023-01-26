import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_leru(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_pooling():
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    # 1024
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 64)
        self.conv2 = conv_bn_leru(64, 128)
        self.conv3 = conv_bn_leru(128, 256)
        self.conv4 = conv_bn_leru(256, 512)
        self.conv5 = conv_bn_leru(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_leru(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_leru(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_leru(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_leru(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

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

        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)
        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)
        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)
        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = F.sigmoid(output)

        return output


class UNet6_512(nn.Module):
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


class UNet5_256(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.conv5 = conv_bn_leru(128, 256)
        self.down_pooling = nn.MaxPool2d(2)

        # up
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

        up4 = self.up_pool4(x5)
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


class UNet5_64(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 16)
        self.conv3 = conv_bn_leru(16, 32)
        self.conv4 = conv_bn_leru(32, 32)
        self.conv5 = conv_bn_leru(32, 64)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool4 = up_pooling(64, 32)
        self.up_conv4 = conv_bn_leru(64, 32)

        self.up_pool3 = up_pooling(32, 32)
        self.up_conv3 = conv_bn_leru(64, 32)

        self.up_pool2 = up_pooling(32, 16)
        self.up_conv2 = conv_bn_leru(32, 16)

        self.up_pool1 = up_pooling(16, 16)
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

        up4 = self.up_pool4(x5)
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


class UNet4_128(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 32)
        self.conv3 = conv_bn_leru(32, 64)
        self.conv4 = conv_bn_leru(64, 128)
        self.down_pooling = nn.MaxPool2d(2)

        # up
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

        up3 = self.up_pool3(x4)
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


class UNet4_32(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 16)
        self.conv3 = conv_bn_leru(16, 32)
        self.conv4 = conv_bn_leru(32, 32)
        self.down_pooling = nn.MaxPool2d(2)

        # up

        self.up_pool3 = up_pooling(32, 32)
        self.up_conv3 = conv_bn_leru(64, 32)

        self.up_pool2 = up_pooling(32, 16)
        self.up_conv2 = conv_bn_leru(32, 16)

        self.up_pool1 = up_pooling(16, 16)
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

        up3 = self.up_pool3(x4)
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


class UNet3_32(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # down
        self.conv1 = conv_bn_leru(in_ch, 16)
        self.conv2 = conv_bn_leru(16, 16)
        self.conv3 = conv_bn_leru(16, 32)
        self.down_pooling = nn.MaxPool2d(2)

        # up
        self.up_pool2 = up_pooling(32, 16)
        self.up_conv2 = conv_bn_leru(32, 16)

        self.up_pool1 = up_pooling(16, 16)
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

        up2 = self.up_pool2(x3)
        ux2 = torch.cat([up2, x2], dim=1)
        ux2 = self.up_conv2(ux2)

        up1 = self.up_pool1(ux2)
        ux1 = torch.cat([up1, x1], dim=1)
        ux1 = self.up_conv1(ux1)

        output = self.conv_out(ux1)
        output = torch.sigmoid(output)

        return output
