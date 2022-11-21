import kornia
import torch

from models.metrics import PSNR, SSIM1
from models.losses import *
from models.networks.modules import *


# from Bread.models.networks.modules import *


class BaseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(BaseNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

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


class IAN(BaseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(IAN, self).__init__(in_channels, out_channels, norm)


class ANSN(BaseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(ANSN, self).__init__(in_channels, out_channels, norm)
        self.outc = OutConv(32, out_channels, act=False)


class FuseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=False):
        super(FuseNet, self).__init__()
        self.inc = AttentiveDoubleConv(in_channels, 32, norm=norm, leaky=False)
        self.down1 = AttentiveDown(32, 64, norm=norm, leaky=False)
        self.down2 = AttentiveDown(64, 64, norm=norm, leaky=False)
        self.up1 = AttentiveUp(128, 32, bilinear=True, norm=norm, leaky=False)
        self.up2 = AttentiveUp(64, 32, bilinear=True, norm=norm, leaky=False)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits


class JTN(FuseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(JTN, self).__init__(in_channels, out_channels, norm)
        self.inc = AttentiveDoubleConv(in_channels, 32, norm=norm, leaky=False)
        self.down1 = AttentiveDown(32, 64, norm=norm, leaky=False)
        self.down2 = AttentiveDown(64, 128, norm=norm, leaky=False)
        self.down3 = AttentiveDown(128, 128, norm=norm, leaky=False)

        self.up1 = AttentiveUp(256, 64, bilinear=True, norm=norm, leaky=False)
        self.up2 = AttentiveUp(128, 32, bilinear=True, norm=norm, leaky=False)
        self.up3 = AttentiveUp(64, 32, bilinear=True, norm=norm, leaky=False)
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


class LoosenMSE(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(F.mse_loss(x, y, reduction='none') - self.eps)


class EnhanceNet(nn.Module):
    def __init__(self, in_channels=3):
        super(EnhanceNet, self).__init__()
        self.feature_num = 64

        self.illumi_branch = BaseNet(in_channels=3, out_channels=1)

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            Residual_Block(64, 64, 1)
        )

        self.denoising_branch = nn.Sequential(
            Residual_Block(64, 64, 1),
            Residual_Block(64, 64, 2),
            Residual_Block(64, 64, 4),
            Residual_Block(64, 64, 8),
            Residual_Block(64, 64, 1),
        )

        # self.se = SELayer(192)

        self.conv_block = nn.Sequential(
            # nn.Conv2d(192, 64, 1),
            # nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.lumi = kornia.color.rgb_to_grayscale

        self.illumi_loss = LoosenMSE()
        self.lime_loss = LIMELoss2()
        self.denoise_loss = nn.MSELoss()
        # self.tv_loss = TVLoss()
        self.eps = 1e-6

    def forward(self, image_in, image_gt, training=True):
        ill = self.illumi_branch(image_in)
        out1 = image_in / (ill + self.eps)
        r_gt = image_gt * ill
        restor_loss1 = self.illumi_loss(out1, image_gt)
        restor_loss1 += self.illumi_loss(image_in, r_gt)

        ill_loss = self.lime_loss(ill, self.lumi(image_in))

        res_input = self.intro(out1)

        # res1 = self.residual1(res_input)
        # res2 = self.residual2(res1)
        # res3 = self.residual3(res2)
        res = self.denoising_branch(res_input)
        # se_cat = self.se(torch.cat([res1, res2, res3], dim=1))
        # image_out = self.conv_block(se_cat)
        # image_out = self.conv_block(torch.cat([res1, res2, res3], dim=1))
        image_out = self.conv_block(res)

        # restor_loss2 = self.illumi_loss(image_out, image_gt)
        l1_loss = self.denoise_loss(image_out, image_gt)
        # tv_loss = self.tv_loss(image_out)
        psnr = PSNR(image_out, image_gt)
        ssim = SSIM1(image_out, image_gt)

        return ill, out1, image_out, restor_loss1, ill_loss, l1_loss, psnr, ssim


class FixJTNet(nn.Module):
    def __init__(self):
        super(FixJTNet, self).__init__()
        self.fix_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            Residual_Block(64, 64, 1),
            Residual_Block(64, 64, 2),
            Residual_Block(64, 64, 4),
            Residual_Block(64, 64, 1),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.fix_branch(x)
        y += x
        return y

#
# class EnhancementNet2(nn.Module):
#     def __init__(self):
#         super(EnhancementNet2, self).__init__()
#         self.f_num = 32
#
#         self.pre = nn.Sequential(
#             nn.Conv2d(3, self.f_num, 3, 1, 1),
#             nn.InstanceNorm2d(self.f_num),
#             nn.ReLU()
#         )
#
#         self.ill_stage = nn.Sequential(
#             Residual_Block(self.f_num * 2, self.f_num * 2, 1)
#         )
#
#         self.denoising_stage = BaseNet(3, 3)
#
#         self.fix_stage = FixJTNet()
#
#     def forward(self, x, gt):
#         x_pre = self.pre(x)
#         # ill =


if __name__ == '__main__':
    for key in FuseNet(4, 2).state_dict().keys():
        print(key)
