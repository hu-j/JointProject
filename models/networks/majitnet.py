import torch
from torch import nn
import torch.nn.functional as F

from models import BaseNet


# class depth_wise_block(nn.Module):
#     def __init__(self):
#         super(depth_wise_block, self).__init__()
#

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
            self.pool1 = nn.MaxPool2d(2)
            self.block2 = nn.Sequential(
                residual_block(channel),
                residual_block(channel),
                residual_block(channel)
            )
        if stage > 2:
            self.pool2 = nn.MaxPool2d(2)
            self.block3 = nn.Sequential(
                residual_block(channel),
                residual_block(channel),
                residual_block(channel)
            )
        if stage > 3:
            self.pool3 = nn.MaxPool2d(2)
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


class illumiNet(BaseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(illumiNet, self).__init__(in_channels, out_channels, norm)


# class denoNet(nn.Module):
#     def __init__(self):



class majitNet(nn.Module):
    def __init__(self, channel=64):
        super(majitNet, self).__init__()

        self.ill_branch = illumiNet(3, 1)
        self.eps = 1e-3
        self.channel = channel

        self.pre = nn.Sequential(
            # nn.Conv2d
        )

        self.res_block = residual_block()

        self.ill_loss = LoosenMSE()

    def forward(self, x_in, x_gt=None):
        ill = self.ill_branch(x_in)
        out0 = x_in / (ill + self.eps)
        restor_loss = self.ill_loss(out0, x_gt)
        restor_loss += self.ill_loss(x_in, x_gt * ill)




        return ill, out0