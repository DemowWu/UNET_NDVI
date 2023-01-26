import numpy as np
import torch
from torch import nn
from torch.nn import init


class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        #
        self.conv1 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (1) + 1, padding=0 + 1)
        self.conv2 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (2) + 1, padding=1 + 1)
        self.conv3 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (3) + 1, padding=2 + 1)
        self.conv4 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (4) + 1, padding=3 + 1)
        self.conv5 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (5) + 1, padding=4 + 1)
        self.conv6 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (6) + 1, padding=5 + 1)
        self.conv7 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (7) + 1, padding=6 + 1)
        self.conv8 = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (8) + 1, padding=7 + 1)

        self.seq1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq7 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.seq8 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # self.convs = []
        # for i in range(S):
        #     self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        # self.se_blocks = []
        # for i in range(S):
        #     self.se_blocks.append(nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
        #         nn.Sigmoid()
        #     ))

        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # Step1:SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs,s,ci,h,w
        a1 = self.conv1(SPC_out[:, 0, :, :, :])
        a2 = self.conv2(SPC_out[:, 1, :, :, :])
        a3 = self.conv3(SPC_out[:, 2, :, :, :])
        a4 = self.conv4(SPC_out[:, 3, :, :, :])
        a5 = self.conv5(SPC_out[:, 4, :, :, :])
        a6 = self.conv6(SPC_out[:, 5, :, :, :])
        a7 = self.conv7(SPC_out[:, 6, :, :, :])
        a8 = self.conv8(SPC_out[:, 7, :, :, :])

        # for idx, conv in enumerate(self.convs):
        #     SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step2:SE weight
        se_out = []
        # for idx, se in enumerate(self.se_blocks):
        #     se_out.append(se(SPC_out[:, idx, :, :, :]))

        se_out.append(self.seq1(a1))
        se_out.append(self.seq2(a2))
        se_out.append(self.seq3(a3))
        se_out.append(self.seq4(a4))
        se_out.append(self.seq5(a5))
        se_out.append(self.seq6(a6))
        se_out.append(self.seq7(a7))
        se_out.append(self.seq8(a8))

        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    psa = PSA(channel=512, reduction=8)
    output = psa(input)
    a = output.view(-1).sum()
    a.backward()
    print(output.shape)
