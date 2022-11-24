import torch
from torch import nn
import torch.nn.functional as F

from models import BaseNet, UNetIllumi
from models.networks.modules import *
from models.networks.slice import batch_bilateral_slice


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
    def __init__(self, in_channels=64, channel=64, n_in=65, n_out=1, gz=1, stage=3, norm=nn.InstanceNorm2d):
        super(kernel_est, self).__init__()
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
            # torch.split(out, self.n_in * self.n_out, dim=1),
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


class preNet(nn.Module):
    def __init__(self, in_channel=1, channel=64, block_num=3, scale=2):
        super(preNet, self).__init__()
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
        for i in range(3):
            self.conv_pyramid_i = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel * ((scale * 2) ** (i + 1)),
                    out_channels=channel * (2 ** i),
                    kernel_size=3,
                    padding=1
                ),
                residual_block(channel * (2 ** i))
            ).cuda()
            self.conv_pyramid.append(self.conv_pyramid_i)

    def forward(self, image_in):
        pyrs = self.pyrs_shuffle(image_in)
        outs = []
        for i, image in enumerate(pyrs):
            outs.append(self.conv_pyramid[i](image))
        return outs, pyrs


class denoNet(nn.Module):
    def __init__(self, channel=64, gz=1, stage=3, group=64, norm=nn.InstanceNorm2d):
        super(denoNet, self).__init__()
        self.channel = channel
        self.stage = stage
        self.gz = gz
        self.n_in = 1
        self.n_out = 2
        self.b_num = 1
        self.group = group
        self.channel_i = [self.channel, self.channel, self.channel * 2, self.channel * 4]
        self.group_i = [self.group, self.group, self.group * 2, self.group * 4]

        self.pre_net_ill = preNet(1, channel, stage, 2)
        self.pre_net_image = preNet(3, channel, stage, 2)

        self.esti_blocks = [
            nn.Sequential(
                nn.AvgPool2d(2, 2),
                # kernel_est(self.channel_i[i], self.channel, self.channel_i[i] + self.b_num, self.channel_i[i],
                #            self.gz * self.group_i[i],
                kernel_est(self.channel_i[i], self.channel, self.n_in, self.n_out, self.gz * self.group_i[i],
                           self.stage)
            ).cuda() for i in range(4)
        ]

        self.acts = [
            nn.Sequential(
                norm(self.channel_i[i]),
                nn.PReLU()
            ).cuda() for i in range(4)
        ]
        self.ratios = [nn.Parameter(torch.tensor(0.), requires_grad=True).cuda() for i in range(4)]

        self.slice = slice()

        # self.upsampling = nn.Upsample(2, mode='bilinear')

        self.conv_pres = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channel_i[i] if i == 3 else self.channel_i[i] + self.channel_i[i + 1],
                    out_channels=self.channel_i[i],
                    kernel_size=3,
                    padding=1
                ),
                norm(self.channel_i[i]),
                residual_block(self.channel_i[i]),
                residual_block(self.channel_i[i])
            ).cuda() for i in range(4)
        ]
        self.conv_after = [
            nn.Sequential(
                residual_block(self.channel_i[i]),
                residual_block(self.channel_i[i])
            ).cuda() for i in range(4)
        ]

    def matrix_multiplication(self, input, grid):
        _, c, _, _ = grid.shape
        _, c0, _, _ = input.shape
        # grid[b,c,h,w]
        # output = torch.sum(input * grid[:, 0:c0, :, :], dim=1, keepdim=True) + grid[:, c0:c, :, :]
        output = torch.sum(input * grid[:, 0:c, :, :], dim=1, keepdim=True)
        return output

    def forward(self, image_in, ill):
        # using ill to estimate kernel weights first
        # try, where brighter, add more image_in msg to join ill for estimate kernel weights, later
        ills, ill_pyrs = self.pre_net_ill(ill)
        out0s, out0_pyrs = self.pre_net_image(image_in)

        images = []  # 相反顺序

        for i in range(self.stage + 1):
            idx = self.stage - i
            if i == 0:
                img = out0s[idx]
            else:
                image_before = F.interpolate(images[i - 1], scale_factor=2, mode='bicubic', align_corners=True)
                img = torch.cat([out0s[idx], image_before], dim=1)
                # img = torch.cat([out0s[idx], self.upsampling(images[i - 1])], dim=1)
            # img的in_channel是不是需要修改
            img = self.conv_pres[idx](img)
            grid_esti = self.esti_blocks[idx](ills[idx])

            # grid_high_rev = self.slice(grid_esti)

            image_i = []
            group = self.group_i[idx]
            guide = img.permute(0, 2, 3, 1)
            grid_esti = grid_esti.permute(0, 2, 3, 4, 1)
            for j in range(group):
                guide_i = torch.clamp(guide[:, :, :, j], 0, 1)
                grid_esti_i = grid_esti[:, :, :, :, j:j + 1]
                slice_grid = self.slice(grid_esti_i, guide_i)
                slice_grid = slice_grid.permute(0, 3, 1, 2)

                image_i_j = self.matrix_multiplication(img, slice_grid)
                image_i.append(image_i_j)

            image_i = torch.cat(image_i, dim=1)
            image_i = self.acts[idx](image_i)
            image_i = img + image_i * self.ratios[idx]

            image_i = self.conv_after[idx](image_i)
            images.append(image_i)

        return images, out0_pyrs


class maJitNet(nn.Module):
    def __init__(self, channel=64, out_channels=3):
        super(maJitNet, self).__init__()

        self.ill_branch = UNetIllumi()
        self.eps = 1e-3
        self.channel = channel

        self.denoise_branch = denoNet(channel=channel)

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
        self.ill_branch.eval()

    def forward(self, x_in):
        ill = self.ill_branch(x_in) ** 0.8
        out0 = x_in / (ill + self.eps)

        outputs, pyrs = self.denoise_branch(out0, ill)
        image_out = self.conv_block(outputs[0])
        image_out = pyrs[0] - image_out

        return ill, out0, image_out
