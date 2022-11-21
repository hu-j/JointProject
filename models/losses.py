import torch
import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM
from torch.nn import L1Loss, MSELoss
from torchvision.models import vgg16
import torch.nn.functional as F


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class SSIMLoss(nn.Module):
    def __init__(self, channels):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)

    def forward(self, output, target):
        ssim_loss = 1 - self.ssim(output, target)
        return ssim_loss


class SSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = l1_loss + self.alpha * ssim_loss
        return total_loss


class GradSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(GradSSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.grad_loss_func = GradientLoss()
        self.alpha = 1.4

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        grad_loss = self.grad_loss_func(output, target)
        total_loss = l1_loss + self.alpha * ssim_loss + 0.2 * grad_loss
        return total_loss


class SSIML2Loss(nn.Module):
    def __init__(self, channels):
        super(SSIML2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.

    def forward(self, output, target):
        l2_loss = self.l2_loss_func(output, target)
        ssim_loss = 1 - self.ssim(output, target)
        total_loss = l2_loss + self.alpha * ssim_loss
        return total_loss


class MSSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.0

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        total_loss = l1_loss + self.alpha * ms_ssim_loss
        return total_loss


class MSSSIML2Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        # self.alpha = 0.84
        self.alpha = 1.2

    def forward(self, output, target):
        l2_loss = self.l2_loss_func(output, target)
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        total_loss = l2_loss + self.alpha * ms_ssim_loss
        return total_loss


class PerLoss(torch.nn.Module):
    def __init__(self):
        super(PerLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to('cuda')
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model

        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, data, gt):
        loss = []
        if data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        dehaze_features = self.output_features(data)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


class PerL1Loss(torch.nn.Module):
    def __init__(self):
        super(PerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        # total_loss = l1_loss + 0.04 * per_loss
        total_loss = l1_loss + 0.2 * per_loss
        return total_loss


class MSPerL1Loss(torch.nn.Module):
    def __init__(self, channels):
        super(MSPerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        total_loss = l1_loss + 1.2 * ms_ssim_loss + 0.04 * per_loss
        return total_loss


class MSPerL2Loss(torch.nn.Module):
    def __init__(self):
        super(MSPerL2Loss, self).__init__()
        self.l2_loss_func = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=3)
        self.per_loss_func = PerLoss().to('cuda')

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l2_loss = self.l2_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        total_loss = l2_loss + 0.16 * ms_ssim_loss + 0.2 * per_loss
        return total_loss


class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, data):
        w_variance = torch.sum(torch.pow(data[:, :, :, :-1] - data[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(data[:, :, :-1, :] - data[:, :, 1:, :], 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def safe_div(a, b, eps=1e-2):
    return a / torch.clamp_min(b, eps)


class WTVLoss(torch.nn.Module):
    def __init__(self):
        super(WTVLoss, self).__init__()
        self.eps = 1e-2

    def forward(self, data, aux):
        data_dw = data[:, :, :, :-1] - data[:, :, :, 1:]
        data_dh = data[:, :, :-1, :] - data[:, :, 1:, :]
        aux_dw = torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:])
        aux_dh = torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :])

        w_variance = torch.sum(torch.pow(safe_div(data_dw, aux_dw, self.eps), 2))
        h_variance = torch.sum(torch.pow(safe_div(data_dh, aux_dh, self.eps), 2))

        count_h = self._tensor_size(data[:, :, 1:, :])
        count_w = self._tensor_size(data[:, :, :, 1:])

        tv_loss = h_variance / count_h + w_variance / count_w
        return tv_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class WTVLoss2(torch.nn.Module):
    def __init__(self):
        super(WTVLoss2, self).__init__()
        self.eps = 1e-2
        self.criterion = nn.MSELoss()

    def forward(self, data, aux):
        N, C, H, W = data.shape

        data_dw = F.pad(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]), (1, 0, 0, 0))
        data_dh = F.pad(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]), (0, 0, 1, 0))
        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))

        data_d = data_dw + data_dh
        aux_d = aux_dw + aux_dh

        loss1 = self.criterion(data_d, aux_d)
        # loss2 = torch.norm(data_d / (aux_d + self.eps), p=1) / (C * H * W)
        loss2 = torch.norm(data_d / (aux_d + self.eps)) / (C * H * W)
        return loss1 * 0.5 + loss2 * 4.0


class MSTVPerL1Loss(torch.nn.Module):
    def __init__(self):
        super(MSTVPerL1Loss, self).__init__()
        self.l1_loss_func = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=3)
        self.per_loss_func = PerLoss().to('cuda')
        self.tv_loss_func = TVLoss()

    def forward(self, output, target):
        ms_ssim_loss = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss_func(output, target)
        per_loss = self.per_loss_func(output, target)
        tv_loss = self.tv_loss_func(output)
        total_loss = l1_loss + 1.2 * ms_ssim_loss + 0.04 * per_loss + 1e-7 * tv_loss
        return total_loss


class LIMELoss2(torch.nn.Module):
    def __init__(self):
        super(LIMELoss2, self).__init__()
        self.eps = 1e-6
        self.alpha = 0.03

    def forward(self, data, aux):
        N, C, H, W = data.shape
        maxc, _ = torch.max(aux, dim=1, keepdim=True)
        recons_loss = torch.norm(data - maxc, p='fro') ** 2

        data_dw = F.pad(data[:, :, :, :-1] - data[:, :, :, 1:], (1, 0, 0, 0))
        data_dh = F.pad(data[:, :, :-1, :] - data[:, :, 1:, :], (0, 0, 1, 0))

        aux_dw = F.pad(torch.abs(aux[:, :, :, :-1] - aux[:, :, :, 1:]), (1, 0, 0, 0))
        aux_dh = F.pad(torch.abs(aux[:, :, :-1, :] - aux[:, :, 1:, :]), (0, 0, 1, 0))
        # data_d = data_dw + data_dh
        # aux_d = aux_dw + aux_dh
        data_d = torch.sqrt(data_dw ** 2 + data_dh ** 2 + self.eps)
        aux_d = torch.sqrt(aux_dw ** 2 + aux_dh ** 2 + self.eps)

        wtv_loss = torch.norm(data_d / (aux_d + self.eps), p=1)
        total_loss = recons_loss + self.alpha * wtv_loss
        return total_loss / (N * C * H * W)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class anchorLoss(torch.nn.Module):
    def __init__(self, num_classes=80, n_anchors=1, conf_thres=0.3):
        super(anchorLoss, self).__init__()
        self.num_classes = num_classes
        self.n_anchors = n_anchors
        self.conf_thres = conf_thres
        self.iou_loss = IOUloss()
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

    def reshape_out(self, output):
        batch_size = output.shape[0]
        hsize, wsize = output.shape[-2:]
        n_ch = output.shape[1]
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        return output, self.n_anchors * hsize * wsize

    def forward(self, enh_outputs, gt_outputs, need_mask=False):
        iou_losses = []
        cls_losses = []
        obj_losses = []

        for enh_output_i, gt_output_i in zip(enh_outputs, gt_outputs):
            # batch_size, h * w, 85
            enh_output_i, n_fg = self.reshape_out(enh_output_i)
            gt_output_i, _ = self.reshape_out(gt_output_i)
            #
            class_conf, class_pred = torch.max(gt_output_i[:, :, 5:5 + self.num_classes], 2, keepdim=True)
            conf_mask = (gt_output_i[:, :, 4] * class_conf[:, :, 0] >= self.conf_thres).squeeze()

            if need_mask:
                iou_loss = self.iou_loss(
                    enh_output_i[:, :, :4].view(-1, 4)[conf_mask],
                    gt_output_i[:, :, :4].view(-1, 4)[conf_mask]
                ).sum() / n_fg
                cls_loss = self.bcewithlog_loss(
                    enh_output_i[:, :, 5:].view(-1, self.num_classes)[conf_mask],
                    gt_output_i[:, :, 5:].view(-1, self.num_classes)[conf_mask]
                ).sum() / n_fg
                obj_loss = self.bcewithlog_loss(
                    enh_output_i[:, :, 4].unsqueeze(-1).view(-1, 1)[conf_mask],
                    gt_output_i[:, :, 4].unsqueeze(-1).view(-1, 1)[conf_mask]
                ).sum() / n_fg
            else:
                iou_loss = self.iou_loss(
                    enh_output_i[:, :, :4].view(-1, 4),
                    gt_output_i[:, :, :4].view(-1, 4)
                ).sum() / n_fg
                cls_loss = self.bcewithlog_loss(
                    enh_output_i[:, :, 5:].view(-1, self.num_classes),
                    gt_output_i[:, :, 5:].view(-1, self.num_classes)
                ).sum() / n_fg
                obj_loss = self.bcewithlog_loss(
                    enh_output_i[:, :, 4].unsqueeze(-1).view(-1, 1),
                    gt_output_i[:, :, 4].unsqueeze(-1).view(-1, 1)
                ).sum() / n_fg

            iou_losses.append(iou_loss)
            cls_losses.append(cls_loss)
            obj_losses.append(obj_loss)

        return  iou_losses, cls_losses, obj_losses


if __name__ == "__main__":
    MSTVPerL1Loss()
