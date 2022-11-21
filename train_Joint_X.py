import argparse
import datetime
import json
import os
import traceback
from numpy import random

import cv2
import kornia.color
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchviz import make_dot
from torchvision.transforms import Resize, Pad
from tqdm.autonotebook import tqdm

import models
from datasets import LowLightDataset
from tools.jointEvaluator import JointEvaluator
from tools.save_json import COCO_JSON
from tools.plots import plot_one_box
from tools.general import scale_coords
from tools import saver
from models import CosineLR, SSIM1
from tools import SingleSummaryWriter
from models import PSNR, SSIM
from tools import mutils
from yolox.data import COCO_CLASSES
from yolox.exp.build import get_exp_by_file
from yolox.utils import postprocess
import torchvision.transforms.functional as TF


def get_args():
    parser = argparse.ArgumentParser('Joint enhancement and detection net.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')
    parser.add_argument('-m1', '--model1', type=str, default='IAN',
                        help='Model Name')
    parser.add_argument('-m2', '--model2', type=str, default='ANSN',
                        help='Model Name')
    parser.add_argument('-m3', '--model3', type=str, default='JTN',
                        help='Model Name')

    parser.add_argument('-m1w', '--model1_weight', type=str, default='./checkpoints/IAN_335.pth',
                        help='Model weight of IANet')
    parser.add_argument('-m2w', '--model2_weight', type=str, default='./checkpoints/ANSN_422.pth',
                        help='Model weight of NSNet')
    parser.add_argument('-mdet', '--detect_weight', type=str, default='./YOLOX/yolox_s.pth')

    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--comment', type=str, default='trainTest',
                        help='Project comment')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--no_sche', action='store_true')
    parser.add_argument('--sampling', action='store_true')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.5)

    parser.add_argument('--gamma', '-ga', type=float, default=1.5)
    parser.add_argument('--alpha', '-a', type=float, default=0.2)

    parser.add_argument('--optim', type=str, default='adam', help='select optimizer for training, '
                                                                  'suggest using \'admaw\' until the'
                                                                  ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--val_interval', type=int, default=10, help='Number of epoches between valing phases')
    # parser.add_argument('--data_path', type=str, default='/home/xteam/hj/dataset/LOLDataset',
    #                     help='the root folder of dataset')
    parser.add_argument('--data_path', type=str, default='/home/xteam/hj/dataset/LSRWDataset/',
                        help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='111/')
    parser.add_argument('--saved_path', type=str, default='111/')
    args = parser.parse_args()
    return args


class ModelJTNet(nn.Module):
    def __init__(self, model1, model2, model3, model_det, exp):
        super().__init__()

        self.exp = exp
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size

        self.restor_loss = models.MSELoss()
        # self.ssim_loss = models.SSIMLoss()

        self.model_ianet = model1(in_channels=1, out_channels=1)
        self.model_nsnet = model2(in_channels=2, out_channels=1)
        self.model_jointnet = model3(in_channels=2, out_channels=1)
        self.model_det = model_det

        self.load_weight(self.model_ianet, opt.model1_weight)
        self.load_weight(self.model_nsnet, opt.model2_weight)
        self.load_detetction_weight(self.model_det, opt.detect_weight)

        self.model_ianet.eval()
        self.model_nsnet.eval()
        self.model_det.eval()

        self.eps = 1e-6

    def load_weight(self, model, weight_pth):
        if model is not None:
            state_dict = torch.load(weight_pth)
            print(state_dict.keys())
            ret = model.load_state_dict(state_dict, strict=True)
            print(ret)

    def load_detetction_weight(self, model, weight_pth):
        if model is not None:
            ckpt = torch.load(weight_pth)
            ret = model.load_state_dict(ckpt["model"])
            print(ret)

    def noise_syn_exp(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def resizeletterbox(self, img, new_shape=(640, 640)):
        # Resize and pad image while meeting stride-multiple constraints
        _, _, h0, w0 = img.shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / h0, new_shape[1] / w0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(h0 * r)), int(round(w0 * r))  # 收缩后图片的长宽
        dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding  需要填充的边的像素
        # if auto:  # minimum rectangle
        #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        # elif scaleFill:  # stretch
        #     dw, dh = 0.0, 0.0
        #     new_unpad = (new_shape[1], new_shape[0])
        #     ratio = new_shape[1] / w0, new_shape[0] / h0  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if h0 != new_unpad and w0 != new_unpad:  # resize
            # img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            torch_resize = Resize([new_unpad[0], new_unpad[1]])
            img = torch_resize(img)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # top, bottom = 0, int(round(dh + 0.1))
        # left, right = 0, int(round(dw + 0.1))
        # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        img = Pad([left, top, right, bottom], fill=114 / 255, padding_mode='constant')(img)
        return img, ratio, (dw, dh)

    def forward(self, image_in, image_gt, training=True):
        with torch.no_grad():
            # convert rgb to ycbcr
            texture_in, _, _ = torch.split(kornia.color.rgb_to_ycbcr(image_in), 1, dim=1)
            texture_gt, cb_gt, cr_gt = torch.split(kornia.color.rgb_to_ycbcr(image_gt), 1, dim=1)

            # Illumination prediction
            texture_in_down = F.interpolate(texture_in, scale_factor=0.5, mode='bicubic', align_corners=True)
            texture_illumi = self.model_ianet(texture_in_down)
            texture_illumi = F.interpolate(texture_illumi, scale_factor=2, mode='bicubic', align_corners=True)

            # Illumination adjustment
            texture_illumi = torch.clamp(texture_illumi, 0., 1.)
            texture_ia = texture_in / torch.clamp_min(texture_illumi, self.eps)
            texture_ia = torch.clamp(texture_ia, 0., 1.)

            # Noise suppression and fusion
            attention = self.noise_syn_exp(texture_illumi, strength=opt.alpha)
            texture_res = self.model_nsnet(torch.cat([texture_ia, attention], dim=1))
            texture_ns = texture_ia + texture_res

            # Further preserve the texture under brighter illumination
            texture_ns = texture_illumi * texture_in + (1 - texture_illumi) * texture_ns
            texture_ns = torch.clamp(texture_ns, 0, 1)

            # get gt labels through detection
            image_det_gt, ratio, _ = self.resizeletterbox(image_gt, opt.tsize)
            image_det_gt *= 255
            bgr_index = [2, 1, 0]
            image_det_gt = image_det_gt[:, bgr_index, :, :]  # to bgr
            image_det_gt = image_det_gt.int()
            image_det_gt = image_det_gt.float()
            _, gt_outputs, gt_fpn, _ = self.model_det(image_det_gt)

            # view gt_feature
            # import matplotlib.pyplot as plt
            # for i, feature_map in enumerate(gt_fpn):


            gt_outputs = postprocess(
                gt_outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )  # list: batch-size,  tensor (4, 7)  xyxy
            # transform it to res:   x, y, x, y, _, _, class type
            # ... * ratio
            ratios = torch.full([4], ratio[0])

            targets = torch.zeros((opt.batch_size, 120, 5)).cuda()
            h = w = opt.tsize
            gn = torch.tensor((h, w))[[1, 0, 1, 0]].cuda()  # normalization gain whwh
            for det_i, gt_output in enumerate(gt_outputs):
                if gt_output != None:
                    # targets = torch.zeros((gt_output.shape[0], 5))
                    obj_num = gt_output.shape[0]
                    targets[det_i, :obj_num, 1:5] = gt_output[:, 0:4] / gn
                    # targets[det_i, :obj_num, 0:4] = targets[:, 0:4] * ratios
                    targets[det_i, :obj_num, 0] = gt_output[:, 6]

        # joint model
        fusion_attention = self.model_jointnet(torch.cat([texture_ns, texture_ia], dim=1))
        texture_out = torch.clamp(fusion_attention * texture_ia + (1 - fusion_attention) * texture_ns, 0., 1.)
        # texture_out = self.model_jointnet(torch.cat([texture_ns, texture_ia], dim=1))

        # calculate texture loss: MSELoss, SSIMLoss
        restor_loss = self.restor_loss(texture_out, texture_gt)
        # ssim_loss = self.ssim_loss(texture_out, texture_gt)

        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_out, cb_gt, cr_gt], dim=1)
        )

        image_ns = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_ns, cb_gt, cr_gt], dim=1)
        )

        psnr = PSNR(texture_out, texture_gt)
        ssim = SSIM1(texture_out, texture_gt)

        # start detection
        image_det, _, pad = self.resizeletterbox(image_out, opt.tsize)
        image_det *= 255.
        bgr_index = [2, 1, 0]
        image_det = image_det[:, bgr_index, :, :]  # to bgr

        if training:
            self.model_det.head.use_l1 = True
            _, _, det_outputs, det_fpn = self.model_det(image_det, targets=targets, getLosses=True)
            det_loss = det_outputs["total_loss"]
            iou_loss = det_outputs["iou_loss"]
            l1_loss = det_outputs["l1_loss"]
            conf_loss = det_outputs["conf_loss"]
            cls_loss = det_outputs["cls_loss"]

            feature_loss = self.restor_loss(det_fpn[0], gt_fpn[0])

            with torch.no_grad():
                self.model_det.head.use_l1 = False
                _, in_det_outputs, _, _ = self.model_det(image_det, getLosses=False)
                in_det_outputs = postprocess(
                    in_det_outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True)

            return image_out, image_ns, texture_out, restor_loss, det_loss, \
                   iou_loss, l1_loss, conf_loss, cls_loss, \
                   feature_loss, \
                   psnr, ssim, in_det_outputs, gt_outputs
        else:
            self.model_det.head.use_l1 = False
            _, det_outputs, det_fpn, _ = self.model_det(image_det, getLosses=False)
            det_outputs = postprocess(
                det_outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True)

            feature_loss = self.restor_loss(det_fpn[0], gt_fpn[0])

            return image_out, image_ns, fusion_attention, texture_out, det_outputs, gt_outputs, \
                   restor_loss, feature_loss, \
                   psnr, ssim
            # return image_out, image_ns, texture_out, det_outputs, gt_outputs, \
            #        restor_loss, feature_loss, \
            #        psnr, ssim


def train(opt):
    cuda = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    timestamp = mutils.get_formatted_time()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # mk save dir
    save_path = opt.saved_path + f'/{opt.comment}/{timestamp}'
    log_path = opt.saved_path + f'/{opt.comment}/{timestamp}/tensorboard/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # make dataset
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'num_workers': opt.num_workers}

    training_set = LowLightDataset(os.path.join(opt.data_path, 'train'),
                                   images_split='low',
                                   targets_split='high')
    training_generator = DataLoader(training_set, **training_params)

    # val_set = LowLightDataset(os.path.join(opt.data_path, 'eval15'),
    val_set = LowLightDataset(os.path.join(opt.data_path, 'Eval/Huawei'),
                              images_split='low',
                              targets_split='high')
    val_generator = DataLoader(val_set, **val_params)

    # Load detection model
    exp = get_exp_by_file('YOLOX/exps/default/yolox_s.py')
    exp.test_conf = opt.conf
    exp.nmsthre = opt.nms
    exp.test_size = (opt.tsize, opt.tsize)
    model_det = exp.get_model()

    # Load enhance model
    model1 = getattr(models, opt.model1)
    model2 = getattr(models, opt.model2)
    model3 = getattr(models, opt.model3)
    model = ModelJTNet(model1, model2, model3, model_det, exp)
    # print(model)

    writer = SingleSummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    if opt.num_gpus > 0:
        model = model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.model_jointnet.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.model_jointnet.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = CosineLR(optimizer, opt.lr, opt.num_epochs)
    epoch = 0
    step = 0
    val_step = 0

    model.model_jointnet.train()

    num_iter_per_epoch = len(training_generator)

    train_json_file = os.path.join(opt.data_path, 'train/train_3.json')
    # val_json_file = os.path.join(opt.data_path, 'eval15/eval_3.json')
    val_json_file = os.path.join(opt.data_path, 'Eval/Huawei/eval_3.json')

    # try:
    for epoch in range(opt.num_epochs):
        last_epoch = step // num_iter_per_epoch
        if epoch < last_epoch:
            continue

        epoch_loss = []
        progress_bar = tqdm(training_generator)
        saver.base_url = os.path.join(opt.saved_path, 'results', '%03d' % epoch)

        # prepare for json
        train_data_lists = []
        train_gt_data_lists = []
        train_json_saver = COCO_JSON(tsize=opt.tsize)
        # train_evaluator = JointEvaluator(annotation_file=train_json_file, img_size=opt.tsize, confthre=opt.conf,
        #                                  nmsthre=opt.nms, num_classes=exp.num_classes)

        if not opt.sampling:
            for iter, (image_in, image_gt, name) in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    if opt.num_gpus == 1:
                        image_in = image_in.cuda()
                        image_gt = image_gt.cuda()
                    optimizer.zero_grad()

                    image_out, image_ns, texture_out, restor_loss, det_loss, iou_loss, l1_loss, conf_loss, cls_loss, feature_loss, \
                    psnr, ssim, in_det_outputs, gt_outputs = model(
                        image_in, image_gt, training=True
                    )
                    _, _, h0, w0 = image_out.shape

                    # vis_graph = make_dot(feature_loss, params=dict(model.named_parameters()))
                    # vis_graph.render("feature_net_struct", view=False)


                    # delete image that has no label
                    # for (output, img_id) in zip(gt_outputs, name):
                    #     if output is None:
                    #         os.remove(opt.data_path + f'/train/high/{name[0]}.jpg')
                    #         os.remove(opt.data_path + f'/train/low/{name[0]}.jpg')
                    #         print('Image is deleted ', name[0])

                    # save json
                    train_json_saver.add_coco_format(gt_outputs, h0, w0, name)

                    # evaluate
                    # data_list_elem = train_evaluator.convert_to_coco_format(in_det_outputs, h0, w0, name)
                    # gt_list_elem = train_evaluator.convert_to_coco_format(gt_outputs, h0, w0, name)
                    # train_data_lists.extend(data_list_elem)
                    # train_gt_data_lists.extend(gt_list_elem)

                    # backward
                    # theta = 0.1
                    # restor_loss *= 1 - theta
                    # det_loss *= theta

                    # loss = restor_loss + det_loss
                    loss = restor_loss + feature_loss

                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. restor_loss: {:.5f}, det_loss: {:.5f},  psnr: {:.5f}, ssim: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, restor_loss.item(),
                            det_loss.item(),
                            psnr, ssim)
                    )
                    writer.add_scalar('RestorLoss/train', restor_loss, step)
                    writer.add_scalar('DetectionLoss/train', det_loss, step)
                    writer.add_scalar('IouLoss/train', iou_loss, step)
                    writer.add_scalar('L1Loss/train', l1_loss, step)
                    writer.add_scalar('ConfLoss/train', conf_loss, step)
                    writer.add_scalar('ClsLoss/train', cls_loss, step)
                    writer.add_scalar('FeatureLoss/train', feature_loss, step)
                    writer.add_scalar('PSNR/train', psnr, step)
                    writer.add_scalar('SSIM/train', ssim, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            train_json_saver.save_json_file(train_json_file)


        if not opt.no_sche:
            scheduler.step()

        if epoch % opt.val_interval == 0:
            # evaluate before val_interval
            # train_evaluator.evaluate_prediction(train_data_lists, train_gt_data_lists)
            #
            # val_evaluator = JointEvaluator(annotation_file=val_json_file, img_size=opt.tsize, confthre=opt.conf,
            #                                nmsthre=opt.nms, num_classes=exp.num_classes)

            model.model_jointnet.eval()
            psnrs = []
            ssims = []
            restor_losses = []
            featurelosses = []

            # make json file
            json_saver = COCO_JSON(tsize=opt.tsize)

            data_list = []
            gt_data_list = []

            for iter, (image_in, image_gt, name) in enumerate(val_generator):
                try:
                    with torch.no_grad():
                        if opt.num_gpus == 1:
                            image_in = image_in.cuda()
                            image_gt = image_gt.cuda()

                        image_out, image_ns, fusion_attention, texture_out, det_outputs, gt_outputs, restor_loss, feature_loss, \
                        psnr, ssim = model(
                            image_in, image_gt, training=False
                        )

                        _, _, h0, w0 = image_out.shape

                    # make json file
                    json_saver.add_coco_format(gt_outputs, h0, w0, name)

                    # for (output, img_id) in zip(gt_outputs, name):
                    #     if output is None:
                    #         os.remove(opt.data_path + f'/Eval/Huawei/high/{name[0]}.jpg')
                    #         os.remove(opt.data_path + f'/Eval/Huawei/low/{name[0]}.jpg')
                    #         print('Image is deleted ', name[0])

                    # evaluate
                    # data_list_elem = val_evaluator.convert_to_coco_format(det_outputs, h0, w0, name)
                    # gt_list_elem = val_evaluator.convert_to_coco_format(gt_outputs, h0, w0, name)
                    # data_list.extend(data_list_elem)
                    # gt_data_list.extend(gt_list_elem)

                    # save images
                    saver.save_image(texture_out, name=os.path.splitext(name[0])[0] + '_textureout')
                    saver.save_image(fusion_attention, name=os.path.splitext(name[0])[0] + '_fusionAtt')
                    saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_out')
                    saver.save_image(image_ns, name=os.path.splitext(name[0])[0] + '_ns')
                    saver.save_image(image_gt, name=os.path.splitext(name[0])[0] + '_gt')

                    # radio = min(opt.tsize / h0, opt.tsize / w0)
                    # save image_out and image_gt with detection labels

                    saveLabeledImage(gt_outputs, image_gt, image_save_name=name[0] + '_gt_labeled')
                    saveLabeledImage(det_outputs, image_out, image_save_name=name[0] + '_out_labeled')


                    psnrs.append(psnr)
                    ssims.append(ssim.item())
                    restor_losses.append(restor_loss.item())
                    featurelosses.append(feature_loss.item())

                    val_step += 1
                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            json_saver.save_json_file(val_json_file)

            psnr = np.mean(np.array(psnrs))
            ssim = np.mean(np.array(ssims))
            restor_loss = np.mean(np.array(restor_losses))
            featureloss = np.mean(np.array(featurelosses))

            # evaluate
            # val_evaluator.evaluate_prediction(data_list, gt_data_list)

            print(
                'Val. Epoch: {}/{}. psnr: {:.5f}, ssim: {:.5f}'.format(
                    epoch, opt.num_epochs, psnr, ssim))

            writer.add_scalar('PSNR/val', psnr, step)
            writer.add_scalar('SSIM/val', ssim, step)
            writer.add_scalar('RestorLoss/val', restor_loss, step)
            writer.add_scalar('FeatureLoss/val', featureloss, step)

            save_checkpoint(model, f'{opt.model3}_{"%03d" % epoch}_{psnr}_{ssim}_{step}.pth')

            model.model_jointnet.train()

    # except KeyboardInterrupt:
    #     save_checkpoint(model, f'{opt.model3}_{epoch}_{step}_keyboardInterrupt.pth')
    #     writer.close()
    # writer.close()


def save_checkpoint(model, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.model_jointnet.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model_jointnet.state_dict(), os.path.join(opt.saved_path, name))


def saveLabeledImage(outputs, init_image, image_size=(640, 640), image_save_name='image_labeled'):
    for i, det in enumerate(outputs):
        # init_image_i = init_image[i]
        _, _, h0, w0 = init_image.shape
        if det != None:
            det[:, :4] = scale_coords(image_size, det[:, :4], (h0, w0)).round()

            img = init_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for *xyxy, n1, n2, cls in reversed(det):
                conf = n1 * n2
                label = f'{COCO_CLASSES[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, line_thickness=1)

            cv2.imwrite(os.path.join(saver.base_url, f'{image_save_name}.png'), img)


if __name__ == '__main__':
    opt = get_args()
    train(opt)
