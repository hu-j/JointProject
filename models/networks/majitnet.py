import torch
from torch import nn
import torch.nn.functional as F

from models import BaseNet, UNetIllumi
from models.networks.modules import *


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

class LoosenMSE(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(F.mse_loss(x, y, reduction='none') - self.eps)


class pyramid_demo(nn.Module):
    def __init__(self, pyr_num=1, block_size=2):
        super(pyramid_demo, self).__init__()
        self.pyr_num = pyr_num
        self.block_size = block_size

    def forward(self, image):
        x = image
        pyramids = []
        pyramids.append(x)

        for i in range(self.pyr_num):
            p_down = space_to_depth(x, self.block_size)
            pyramids.append(p_down)
            x = p_down

        return pyramids


class residual_block(nn.Module):
    def __init__(self, channels=64, stride=1, norm=nn.InstanceNorm2d):
        super(residual_block, self).__init__()
        self.inbo_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * 5,
                kernel_size=1,
                stride=stride,
                padding=0
            ),
            norm(channels * 5, affine=True),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=channels * 5,
                out_channels=channels * 5,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels * 5
            ),
            norm(channels * 5),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=channels * 5,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                padding=0
            ),
            norm(channels),
            nn.PReLU()
        )

    def forward(self, x):
        y = x
        outputs = y + self.inbo_block(x)
        return outputs


class kernel_est(nn.Module):
    def __init__(self, in_channels=64, channel=64, n_in=1, n_out=2, gz=1, stage=3, norm=nn.InstanceNorm2d):
        super(kernel_est, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.gz = gz

        self.pre = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channel,
                kernel_size=3,
                padding=1
            ),
            nn.PReLU()
        )
        self.block1 = nn.Sequential(
            residual_block(channel),
            residual_block(channel),
            residual_block(channel)
        )
        if stage > 1:
            self.pool1 = nn.MaxPool2d(2, 2)
            self.block2 = nn.Sequential(
                residual_block(channel),
                residual_block(channel),
                residual_block(channel)
            )
        if stage > 2:
            self.pool2 = nn.MaxPool2d(2, 2)
            self.block3 = nn.Sequential(
                residual_block(channel),
                residual_block(channel),
                residual_block(channel)
            )
        if stage > 3:
            self.pool3 = nn.MaxPool2d(2, 2)
            self.block4 = nn.Sequential(
                residual_block(channel),
                residual_block(channel),
                residual_block(channel)
            )

        self.after = nn.Conv2d(
            in_channels=channel,
            out_channels=self.gz*self.n_in*self.n_out,
            kernel_size=1,
            padding=0
        )

    def forward(self, inputs):
        x = self.pre(inputs)
        x = self.block1(x)
        if self.stage > 1:
            x = self.pool1(x)
            x = self.block2(x)
        if self.stage > 2:
            x = self.pool2(x)
            x = self.block3(x)
        if self.stage > 3:
            x = self.pool3(x)
            x = self.block4(x)

        out = self.after(x)
        out = torch.stack(
            torch.split(out, self.n_in * self.n_out, dim=1),
            dim=4
        )
        return out


class ma_conv_block(nn.Module):
    def __init__(self, channel=64, gz=1, stage=3, group=64, norm=nn.InstanceNorm2d):
        super(ma_conv_block, self).__init__()
        self.channel = channel
        self.gz = gz
        self.n_in = 1
        self.n_out = 2
        self.group = group

        self.block = nn.Sequential(
                nn.AvgPool2d(2, 2),
                kernel_est(self.channel, self.channel, self.n_in, self.n_out, self.gz * self.group, self.stage)
            )

    def forward(self, x):

        return


class illumiNet(BaseNet):
    def __init__(self, in_channels=3, out_channels=1, norm=True):
        super(illumiNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

        self.ma_conv = ma_conv_block()

        self.up1 = Up(256, 64, bilinear=True, norm=norm)
        self.up2 = Up(128, 32, bilinear=True, norm=norm)
        self.up3 = Up(64, 32, bilinear=True, norm=norm)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# class preNet(nn.Module):
#     def __init__(self, channel=64):
#         super(preNet, self).__init__()
#         self.pyrs_demo = pyramid_demo(3)
#         self.


class denoNet(nn.Module):
    def __init__(self, channel=64, gz=1, stage=3, group=64, norm=nn.InstanceNorm2d):
        super(denoNet, self).__init__()
        self.channel = channel
        self.gz = gz
        self.n_in = 1
        self.n_out = 2
        self.group = group

        self.blocks = [
            nn.Sequential(
                nn.AvgPool2d(2, 2),
                kernel_est(self.channel, self.channel, self.n_in, self.n_out, self.gz * self.group, self.stage)
            ) for i in range(2)
        ]
        self.blocks.append(
            nn.Sequential(
                kernel_est(self.channel, self.channel, self.n_in, self.n_out, self.gz * self.group * 2, self.stage)
            )
        )
        self.blocks.append(
            nn.Sequential(
                kernel_est(self.channel, self.channel, self.n_in, self.n_out, self.gz * self.group * 4, self.stage)
            )
        )


    def forward(self, inputs):


        return

class majitNet(nn.Module):
    def __init__(self, channel=64):
        super(majitNet, self).__init__()

        self.ill_branch = UNetIllumi()
        self.eps = 1e-3
        self.channel = channel

        self.pyr_pre = pyramid_demo(3)

        self.res_block = residual_block()

        self.ill_loss = LoosenMSE()

    def load_ill_weight(self, weight_pth):
        state_dict = torch.load(weight_pth)
        ret = self.illumi_branch.load_state_dict(state_dict)
        print(ret)
        self.illumi_branch.requires_grad_(False)
        self.illumi_branch.eval()

    def forward(self, x_in, x_gt=None):
        ill = self.ill_branch(x_in)
        out0 = x_in / (ill + self.eps)
        restor_loss = self.ill_loss(out0, x_gt)
        restor_loss += self.ill_loss(x_in, x_gt * ill)

        # using ill to estimate kernel weights
        pyrs_i = self.pyr_pre(ill)

        return ill, out0