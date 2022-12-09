import os

import torch
from torch import nn
import torch.nn.functional as F

from models import BaseNet, UNetIllumi
from models.networks.modules import *
from models.networks.slice import batch_bilateral_slice
from tools import saver


def DownSamplingShuffle(x, scale=2):
    _, _, h, w = x.shape
    assert scale == 2
    x1 = x[:, :, 0:h:2, 0:w:2]
    x2 = x[:, :, 0:h:2, 1:w:2]
    x3 = x[:, :, 1:h:2, 0:w:2]
    x4 = x[:, :, 1:h:2, 1:w:2]

    return torch.cat((x1, x2, x3, x4), 1)


class LoosenMSE(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(F.mse_loss(x, y, reduction='none') - self.eps)


class slice_operation(nn.Module):
    def __init__(self):
        super(slice_operation, self).__init__()

    def forward(self, grid, guidemap, align_corners=True):
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        if torch.cuda.is_available():
            hg = hg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1
            wg = wg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1
        else:
            hg = hg.type(torch.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1
            wg = wg.type(torch.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1
        guidemap = guidemap * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1)
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)
        coeff = F.grid_sample(grid, guidemap_guide, padding_mode='border', align_corners=align_corners)
        return coeff.squeeze(2)


class pyramid_shuffle(nn.Module):
    def __init__(self, pyr_num=3, block_size=2):
        super(pyramid_shuffle, self).__init__()
        self.pyr_num = pyr_num
        self.block_size = block_size

    def forward(self, image):
        x = image
        pyramids = []
        pyramids.append(x)

        for i in range(self.pyr_num):
            p_down = DownSamplingShuffle(x, self.block_size)
            pyramids.append(p_down)
            x = p_down

        return pyramids


# class residual_block(nn.Module):
#     def __init__(self, channels=64, stride=1, norm=nn.InstanceNorm2d):
#         super(residual_block, self).__init__()
#         self.inbo_block = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=channels,
#                 out_channels=channels * 5,
#                 kernel_size=1,
#                 stride=stride,
#                 padding=0
#             ),
#             norm(channels * 5, affine=True),
#             nn.LeakyReLU(),
#             nn.Conv2d(
#                 in_channels=channels * 5,
#                 out_channels=channels * 5,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#                 groups=channels * 5
#             ),
#             norm(channels * 5),
#             nn.LeakyReLU(),
#             nn.Conv2d(
#                 in_channels=channels * 5,
#                 out_channels=channels,
#                 kernel_size=1,
#                 stride=stride,
#                 padding=0
#             ),
#             norm(channels),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         y = x
#         outputs = y + self.inbo_block(x)
#         return outputs
class residual_block(nn.Module):
    def __init__(self, channels=64, stride=1, norm=nn.InstanceNorm2d):
        super(residual_block, self).__init__()
        self.inbo_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * 2,
                kernel_size=1,
                stride=stride,
                padding=0
            ),
            norm(channels * 2, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels * 2,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels * 2
            ),
            norm(channels * 2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels,
                kernel_size=1,
                stride=stride,
                padding=0
            ),
            norm(channels),
            nn.LeakyReLU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=stride,
                padding=1
            ),
            norm(channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        y = x
        outputs = y + self.inbo_block(x)
        # outputs = y + self.block(x)
        return outputs


class kernel_est2(nn.Module):
    def __init__(self, in_channels=64, channel=64, n_in=65, n_out=1, gz=1, stage=3, norm=nn.InstanceNorm2d):
        super(kernel_est2, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.gz = gz
        self.stage = stage

        self.pre = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channel,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU()
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
            out_channels=self.gz * self.n_in * self.n_out,
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
            torch.split(out, self.gz, dim=1),
            dim=4
        )
        return out


class slice(nn.Module):
    def __init__(self):
        super(slice, self).__init__()

    def forward(self, grid, guide):
        return batch_bilateral_slice(grid, guide)


class illumiNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, norm=True):
        super(illumiNet, self).__init__()
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


class preNet2(nn.Module):
    def __init__(self, in_channel=1, channel=64, block_num=3, scale=2):
        super(preNet2, self).__init__()
        self.scale = scale
        self.pyrs_shuffle = pyramid_shuffle(block_num, self.scale)
        self.conv_pyramid = []
        self.conv_pyramid1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=channel,
                kernel_size=3,
                padding=1
            ),
            residual_block(channel)
        )
        self.conv_pyramid.append(self.conv_pyramid1)
        self.conv_pyramid2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel * 4,
                out_channels=channel,
                kernel_size=3,
                padding=1
            ),
            residual_block(channel)
        )
        self.conv_pyramid3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel * 16,
                out_channels=channel * 2,
                kernel_size=3,
                padding=1
            ),
            residual_block(channel * 2)
        )
        self.conv_pyramid4 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel * 64,
                out_channels=channel * 4,
                kernel_size=3,
                padding=1
            ),
            residual_block(channel * 4)
        )

        self.conv_pyramid.append(self.conv_pyramid2)
        self.conv_pyramid.append(self.conv_pyramid3)
        self.conv_pyramid.append(self.conv_pyramid4)

    def forward(self, image_in):
        pyrs = self.pyrs_shuffle(image_in)
        outs = []
        for i, image in enumerate(pyrs):
            outs.append(self.conv_pyramid[i](image))
        return outs, pyrs


class denoNet2(nn.Module):
    def __init__(self, channel=64, gz=1, stage=3, group=64, norm=nn.InstanceNorm2d):
        super(denoNet2, self).__init__()
        self.channel = channel
        self.stage = stage
        self.gz = gz
        self.n_in = 1
        self.n_out = 2
        self.b_num = 1
        self.group = group
        self.channel_i = [self.channel, self.channel, self.channel * 2, self.channel * 4]
        self.group_i = [self.group, self.group, self.group * 2, self.group * 4]

        self.pre_net_ill = preNet2(1, channel, stage, 2)
        self.pre_net_image = preNet2(3, channel, stage, 2)

        self.esti_block0 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            kernel_est2(self.channel_i[0], self.channel, self.n_in, self.n_out, self.gz * self.group_i[0],
                        self.stage)
        )
        self.esti_block1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            kernel_est2(self.channel_i[1], self.channel, self.n_in, self.n_out, self.gz * self.group_i[1],
                        self.stage)
        )
        self.esti_block2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            kernel_est2(self.channel_i[2], self.channel, self.n_in, self.n_out, self.gz * self.group_i[2],
                        self.stage)
        )
        self.esti_block3 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            kernel_est2(self.channel_i[3], self.channel, self.n_in, self.n_out, self.gz * self.group_i[3],
                        self.stage)
        )

        self.esti_blocks = [self.esti_block0, self.esti_block1, self.esti_block2, self.esti_block3]

        self.act0 = nn.Sequential(
            norm(self.channel_i[0]),
            nn.LeakyReLU()
        )
        self.act1 = nn.Sequential(
            norm(self.channel_i[1]),
            nn.LeakyReLU()
        )
        self.act2 = nn.Sequential(
            norm(self.channel_i[2]),
            nn.LeakyReLU()
        )
        self.act3 = nn.Sequential(
            norm(self.channel_i[3]),
            nn.LeakyReLU()
        )

        self.acts = [self.act0, self.act1, self.act2, self.act3]

        # self.ratios = [nn.Parameter(torch.tensor(0.), requires_grad=True).cuda() for i in range(4)]
        self.ratio0 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.ratio1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.ratio2 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.ratio3 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.ratios = [self.ratio0, self.ratio1, self.ratio2, self.ratio3]

        self.slice = slice()

        self.conv_pres0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_i[0] + self.channel_i[1],
                out_channels=self.channel_i[0],
                kernel_size=3,
                padding=1
            ),
            norm(self.channel_i[0]),
            residual_block(self.channel_i[0]),
            residual_block(self.channel_i[0])
        )
        self.conv_pres1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_i[1] + self.channel_i[2],
                out_channels=self.channel_i[1],
                kernel_size=3,
                padding=1
            ),
            norm(self.channel_i[1]),
            residual_block(self.channel_i[1]),
            residual_block(self.channel_i[1])
        )
        self.conv_pres2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_i[2] + self.channel_i[3],
                out_channels=self.channel_i[2],
                kernel_size=3,
                padding=1
            ),
            norm(self.channel_i[2]),
            residual_block(self.channel_i[2]),
            residual_block(self.channel_i[2])
        )
        self.conv_pres3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channel_i[3],
                out_channels=self.channel_i[3],
                kernel_size=3,
                padding=1
            ),
            norm(self.channel_i[3]),
            residual_block(self.channel_i[3]),
            residual_block(self.channel_i[3])
        )
        self.conv_pres = [self.conv_pres0, self.conv_pres1, self.conv_pres2, self.conv_pres3]

        self.conv_after0 = nn.Sequential(
            residual_block(self.channel_i[0]),
            residual_block(self.channel_i[0])
        )
        self.conv_after1 = nn.Sequential(
            residual_block(self.channel_i[1]),
            residual_block(self.channel_i[1])
        )
        self.conv_after2 = nn.Sequential(
            residual_block(self.channel_i[2]),
            residual_block(self.channel_i[2])
        )
        self.conv_after3 = nn.Sequential(
            residual_block(self.channel_i[3]),
            residual_block(self.channel_i[3])
        )

        self.conv_after = [self.conv_after0, self.conv_after1, self.conv_after2, self.conv_after3]

    def matrix_multiplication(self, input, grid):
        _, c, _, _ = grid.shape
        _, c0, _, _ = input.shape
        # grid[b,c,h,w]
        # output = torch.sum(input * grid[:, 0:c0, :, :], dim=1, keepdim=True) + grid[:, c0:c, :, :]
        return torch.sum(input * grid[:, 0:c, :, :], dim=1, keepdim=True)

    def forward(self, image_in, ill):
        # using ill to estimate kernel weights first
        # try, where brighter, add more image_in msg to join ill for estimate kernel weights, later
        ills, _ = self.pre_net_ill(ill)
        out0s, out0_pyrs = self.pre_net_image(image_in)

        images = []  # 相反顺序

        for i in range(self.stage + 1):
            idx = self.stage - i
            if i == 0:
                img = out0s[idx]
            else:
                img = torch.cat(
                    [
                        out0s[idx],
                        F.interpolate(images[i - 1], scale_factor=2, mode='bilinear', align_corners=True)
                    ],
                    dim=1)
            img = self.conv_pres[idx](img)
            # grid_esti = self.esti_blocks[idx](ills[idx])
            grid_esti = self.esti_blocks[idx](out0s[idx])

            image_i = []
            group = self.group_i[idx]
            guide = img.permute(0, 2, 3, 1)
            grid_esti = grid_esti.permute(0, 2, 3, 4, 1)
            for j in range(group):
                guide_i = torch.clamp(guide[:, :, :, j], 0, 1)
                grid_esti_i = grid_esti[:, :, :, :, j:j + 1]
                slice_grid = self.slice(grid_esti_i, guide_i)
                slice_grid = slice_grid.permute(0, 3, 1, 2)
                del guide_i, grid_esti_i
                image_i.append(self.matrix_multiplication(img, slice_grid))

            image_i = torch.cat(image_i, dim=1)
            image_i = self.acts[idx](image_i)
            image_i = torch.clamp(image_i, -1., 1.)
            image_i = img + image_i * self.ratios[idx]
            if idx != 0:
                image_i = self.conv_after[idx](image_i)
            images.append(image_i)

        return images, out0_pyrs
    # def forward(self, image_in, ill):
    #     # using ill to estimate kernel weights first
    #     # try, where brighter, add more image_in msg to join ill for estimate kernel weights, later
    #     # ills, _ = self.pre_net_ill(ill)
    #     out0s, out0_pyrs = self.pre_net_image(image_in)
    #
    #     images = []  # 相反顺序
    #
    #     for i in range(self.stage + 1):
    #         idx = self.stage - i
    #         if i == 0:
    #             img = out0s[idx]
    #         else:
    #             img = torch.cat(
    #                 [
    #                     out0s[idx],
    #                     F.interpolate(images[i - 1], scale_factor=2, mode='bilinear', align_corners=True)
    #                 ],
    #                 dim=1)
    #             # img = torch.cat([out0s[idx], self.upsampling(images[i - 1])], dim=1)
    #         img0 = self.conv_pres[idx](img)
    #         img0 = self.conv_after[idx](img0)
    #         # grid_esti = self.esti_blocks[idx](ills[idx])
    #         #
    #         # image_i = []
    #         # group = self.group_i[idx]
    #         # guide = img.permute(0, 2, 3, 1)
    #         # grid_esti = grid_esti.permute(0, 2, 3, 4, 1)
    #         # for j in range(group):
    #         #     guide_i = torch.clamp(guide[:, :, :, j], 0, 1)
    #         #     grid_esti_i = grid_esti[:, :, :, :, j:j + 1]
    #         #     slice_grid = self.slice(grid_esti_i, guide_i)
    #         #     slice_grid = slice_grid.permute(0, 3, 1, 2)
    #         #     del guide_i, grid_esti_i
    #         #     image_i.append(self.matrix_multiplication(img, slice_grid))
    #         #
    #         # image_i = torch.cat(image_i, dim=1)
    #         # image_i = self.acts[idx](image_i)
    #         # img0 = self.conv_try[idx](img)
    #
    #         image_i = out0s[idx] + img0 * self.ratios[idx]
    #
    #         # image_i = self.conv_after[idx](image_i)
    #         images.append(image_i)
    #         del image_i
    #
    #     return images, out0_pyrs


class maJitNet2(nn.Module):
    def __init__(self, channel=64, out_channels=3):
        super(maJitNet2, self).__init__()

        self.ill_branch = UNetIllumi()
        self.eps = 1e-3
        self.channel = channel

        self.denoise_branch = denoNet2(channel=channel)

        self.ill_loss = LoosenMSE()

        self.conv_block = nn.Sequential(
            residual_block(channels=self.channel),
            nn.Conv2d(
                in_channels=channel,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def load_ill_weight(self, weight_pth):
        state_dict = torch.load(weight_pth)
        ret = self.ill_branch.load_state_dict(state_dict)
        print(ret)
        self.ill_branch.requires_grad_(False)

    def forward(self, x_in):
        self.ill_branch.eval()
        ill = self.ill_branch(x_in) ** 0.8
        out0 = x_in / (ill + self.eps)

        outputs, pyrs = self.denoise_branch(out0, ill)
        noise = self.conv_block(outputs[3])
        image_out = pyrs[0] - noise

        return ill, out0, noise, image_out
