from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_c, out_c, 5, 2, padding=2, output_padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convt(x)
        out = self.norm(out)
        return self.relu(out)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 5, 2, 2, bias=False)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.lrelu(out)

# Adapted from https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch
class Generator(nn.Module):

    def __init__(self, in_dim, dim=128):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        self.convts = nn.Sequential(
            UpBlock(dim * 8, dim * 4),
            UpBlock(dim * 4, dim * 2),
            UpBlock(dim * 2, dim),
            UpBlock(dim, dim // 2),
            nn.ConvTranspose2d(dim // 2, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

        self.init_weight()

    def forward(self, x):
        b = x.size(0)
        out = self.linear(x)
        out = out.view(b, -1, 4, 4)
        out = self.convts(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #print('here')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self, in_dim, dim=128):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2),
            DownBlock(dim//2, dim),
            DownBlock(dim, dim * 2),
            DownBlock(dim * 2, dim * 4),
            DownBlock(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

        self.init_weight()

    def forward(self, x):
        y = self.convs(x)
        y = y.view(-1)
        return y

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


# Adapted from https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch
class Generator1(nn.Module):

    def __init__(self, in_dim, dim=128):
        super().__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            dconv_bn_relu(dim, dim // 2),
            nn.ConvTranspose2d(dim // 2, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator1(nn.Module):

    def __init__(self, in_dim, dim=128):
        super().__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim // 2, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim//2, dim),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
