from typing import Type

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F

import tools.saver as saver

affine = True
inplace = True


def conv3x3(inplanes, planes, stride=1, norm=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1),
        norm(planes, affine=affine),
        nn.LeakyReLU(inplace=inplace)
    )


def conv_sig3x3(inplanes, planes, stride=1, norm=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1),
        norm(planes, affine=affine),
        nn.Sigmoid()
    )


def conv1x1(inplanes, planes, stride=1, norm=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.Conv2d(inplanes, planes, stride=stride, kernel_size=1, padding=0),
        norm(planes, affine=affine),
        nn.LeakyReLU(inplace=inplace)
    )


def deconv3x3(inplanes, planes, stride=2, norm=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.ConvTranspose2d(inplanes, planes, stride=stride, kernel_size=2),
        norm(planes, affine=affine),
        nn.LeakyReLU(inplace=inplace)
    )


class DownConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv3x3(in_channels, out_channels),
            conv3x3(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            conv3x3(out_channels, out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(self, inplanes):
        super(UNetDenoiser, self).__init__()
        self.inc = conv1x1(inplanes, inplanes * 2)
        self.down1 = DownConv(inplanes * 2, inplanes * 4)
        self.down2 = DownConv(inplanes * 4, inplanes * 4)
        self.up1 = UpConv(inplanes * 8, inplanes * 2)
        self.up2 = UpConv(inplanes * 4, inplanes * 2)
        self.outc = conv1x1(inplanes * 2, inplanes)

    def forward(self, x):
        # saver.save_feature(x, 'identity')

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)

        # saver.save_feature(x, 'out', exit_flag=True)
        return x


class RUNetDenoiser(nn.Module):
    def __init__(self, inplanes):
        super(RUNetDenoiser, self).__init__()
        self.inc = conv1x1(inplanes, inplanes * 2)
        self.down1 = DownConv(inplanes * 2, inplanes * 4)
        self.down2 = DownConv(inplanes * 4, inplanes * 4)
        self.up1 = UpConv(inplanes * 8, inplanes * 2)
        self.up2 = UpConv(inplanes * 4, inplanes * 2)
        self.outc = conv1x1(inplanes * 2, inplanes)

    def forward(self, x):
        identity = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        out = self.up1(x3, x2)
        out = self.up2(out, x1)
        out = self.outc(out)

        out += identity

        return out

        # return out

        # saver.save_feature(identity, 'identity')
        # saver.save_feature(noise, 'noise')
        # saver.save_feature(out, 'out', exit_flag=True)


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class MUDNet(nn.Module):
    def __init__(self, dims=32):
        super(MUDNet, self).__init__()
        self.dims = dims
        self.pre = conv3x3(3, self.dims)

        self.stage1 = nn.Sequential(
            conv3x3(self.dims, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2)
        )

        self.stage2 = UNetDenoiser(self.dims * 2)

        self.stage3 = nn.Sequential(
            conv3x3(self.dims * 2, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2),
            conv3x3(self.dims * 2, self.dims),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.dims, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        if y is not None:
            z = torch.cat([x, y], dim=0)
            z = self.pre(z)
            z = self.stage1(z)
            x, y = torch.split(z, x.shape[0], dim=0)
            x_hat = self.stage2(x)

            z = torch.cat([x_hat, y], dim=0)
            z = self.stage3(z)
            out = self.post(z)

            return x_hat, y, out
        else:
            x = self.pre(x)
            x = self.stage1(x)
            x_hat = self.stage2(x)
            x_hat = self.stage3(x_hat)
            out = self.post(x_hat)
            return out


class MURDNet(nn.Module):
    def __init__(self, dims=32):
        global affine, inplace
        affine = True
        inplace = True
        super(MURDNet, self).__init__()

        self.dims = dims
        self.pre = conv3x3(3, self.dims)

        self.stage1 = nn.Sequential(
            conv3x3(self.dims, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2)
        )

        self.stage2 = RUNetDenoiser(self.dims * 2)

        self.stage3 = nn.Sequential(
            conv3x3(self.dims * 2, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2),
            conv3x3(self.dims * 2, self.dims),
        )

        self.post = conv_sig3x3(self.dims, 3)

    def forward(self, x, y=None):
        if y is not None:
            z = torch.cat([x, y], dim=0)
            z = self.pre(z)
            z = self.stage1(z)

            x = z[:x.shape[0]].contiguous()
            y = z[x.shape[0]:].contiguous()

            x_hat = self.stage2(x)

            z = torch.cat([x_hat, y], dim=0)
            saver.save_feature(x_hat, 'x_hat')
            saver.save_feature(y, 'y', exit_flag=True)

            z = self.stage3(z)
            out = self.post(z)

            return x_hat, y, out
        else:
            x = self.pre(x)
            x = self.stage1(x)
            x_hat = self.stage2(x)
            x_hat = self.stage3(x_hat)
            out = self.post(x_hat)
            return out


class RMUDNet(nn.Module):
    def __init__(self, dims=32):
        super(RMUDNet, self).__init__()
        self.dims = dims
        self.pre = conv3x3(3, self.dims)

        self.stage1_1 = nn.Sequential(
            conv3x3(self.dims, self.dims * 2),
            conv3x3(self.dims * 2, self.dims * 2),
        )

        self.stage1_2 = nn.Sequential(
            conv3x3(self.dims * 2, self.dims * 2)
        )

        self.stage2 = UNetDenoiser(self.dims * 2)

        self.stage3_1 = nn.Sequential(
            conv3x3(self.dims * 2, self.dims * 2)
        )

        self.stage3_2 = nn.Sequential(
            conv3x3(self.dims * 4, self.dims * 2),
            conv3x3(self.dims * 2, self.dims),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.dims, 3, kernel_size=1),
            nn.Sigmoid()

        )

    def forward(self, x, y=None, not_save=True):
        if y is not None:
            if not_save:
                z = torch.cat([x, y], dim=0)
                z = self.pre(z)
                z = self.stage1_1(z)
                res = z
                z = self.stage1_2(z)
                x_feat, y_feat = torch.split(z, x.shape[0], dim=0)
                x_feat = self.stage2(x_feat)

                z = torch.cat([x_feat, y_feat], dim=0)
                z = self.stage3_1(z)
                z = torch.cat([z, res], dim=1)
                z = self.stage3_2(z)
                out = self.post(z)
                x_out, y_out = torch.split(out, x.shape[0], dim=0)

                return x_feat, y_feat, x_out, y_out
            else:
                z = torch.cat([x, y], dim=0)
                z = self.pre(z)
                z = self.stage1_1(z)
                res = z
                z = self.stage1_2(z)
                x_feat, y_feat = torch.split(z, x.shape[0], dim=0)
                saver.save_feature(x_feat.detach(), 'x_feat1')
                x_feat = self.stage2(x_feat)

                saver.save_feature(x_feat.detach(), 'x_feat2')
                saver.save_feature(y_feat.detach(), 'y_feat')

                z = torch.cat([x_feat, y_feat], dim=0)
                z = self.stage3_1(z)
                z = torch.cat([z, res], dim=1)
                z = self.stage3_2(z)
                out = self.post(z)

                x_out, y_out = torch.split(out, x.shape[0], dim=0)

                saver.save_image(y_out.detach(), 'y_out')
                saver.save_image(x_out.detach(), 'x_out')

                return x_feat, y_feat, x_out, y_out

        else:
            x = self.pre(x)
            x = self.stage1_1(x)
            res = x
            x = self.stage1_2(x)
            x_feat = self.stage2(x)

            x_feat = self.stage3_1(x_feat)
            x_comp = torch.cat([x_feat, res], dim=1)
            x_comp = self.stage3_2(x_comp)
            out = self.post(x_comp)
            return out

    # def forward(self, x, y=None):
    #         if y is not None:
    #             z = torch.cat([x, y], dim=0)
    #             z = self.pre(z)
    #             z = self.stage1_1(z)
    #             res = z
    #             z = self.stage1_2(z)
    #             x_feat, y_feat = torch.split(z, x.shape[0], dim=0)
    #             x_feat = self.stage2(x_feat)
    #
    #             z = torch.cat([x_feat, y_feat], dim=0)
    #             z = self.stage3_1(z)
    #             z = torch.cat([z, res], dim=1)
    #             z = self.stage3_2(z)
    #             out = self.post(z)
    #             x_out, y_out = torch.split(out, x.shape[0], dim=0)
    #
    #             return x_feat, y_feat, x_out, y_out
    #         else:
    #             x = self.pre(x)
    #             x = self.stage1_1(x)
    #             res = x
    #             x = self.stage1_2(x)
    #             x_feat = self.stage2(x)
    #
    #             x_feat = self.stage3_1(x_feat)
    #             x_comp = torch.cat([x_feat, res], dim=1)
    #             x_comp = self.stage3_2(x_comp)
    #             out = self.post(x_comp)
    #             return out


class CPRMUDNet(RMUDNet):
    def __init__(self, dims=32):
        super(CPRMUDNet, self).__init__(dims=dims)
        self.stage3_2 = nn.Sequential(
            CALayer(self.dims * 4),
            PALayer(self.dims * 4),
            conv3x3(self.dims * 4, self.dims * 2),
            conv3x3(self.dims * 2, self.dims),
        )

# if __name__ == '__main__':
#     print(MaxRMUDNet())
