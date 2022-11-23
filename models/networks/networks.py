import kornia
import torch

from models.metrics import PSNR, SSIM1
from models.losses import *
from models.networks.modules import *
from models.networks.unet_2d_illumi import UNetIllumi


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

        self.illumi_branch = UNetIllumi()

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
        self.eps = 1e-3

    def load_ill_weight(self, weight_pth):
        state_dict = torch.load(weight_pth)
        ret = self.illumi_branch.load_state_dict(state_dict)
        print(ret)
        self.illumi_branch.requires_grad_(False)
        self.illumi_branch.eval()

    def forward(self, image_in, image_gt, training=True):

        ill = self.illumi_branch(image_in)**0.8
        out1 = image_in / (ill + self.eps)
        # r_gt = image_gt * ill
        # restor_loss1 = self.illumi_loss(out1, image_gt)
        # restor_loss1 += self.illumi_loss(image_in, r_gt)

        # ill_loss = self.lime_loss(ill, self.lumi(image_in))

        res_input = self.intro(out1)

        res = self.denoising_branch(res_input)
        image_out = self.conv_block(res)

        l1_loss = self.denoise_loss(image_out, image_gt)
        psnr = PSNR(image_out, image_gt)
        ssim = SSIM1(image_out, image_gt)

        # return ill, out1, image_out, restor_loss1, ill_loss, l1_loss, psnr, ssim
        return ill, out1, image_out, l1_loss, psnr, ssim


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

