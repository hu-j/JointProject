import argparse
import os
import traceback
import datetime

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Pad
from torchviz import make_dot
from tqdm import tqdm

import models
from datasets import LowLightDataset
from models import *
from models.networks.cprmudnet import CPRMUDNet
from models.networks.networks import EnhanceNet
from tools import mutils, SingleSummaryWriter, saver
from tools.jointEvaluator import JointEvaluator
from tools.general import scale_coords
from tools.plots import plot_one_box
from yolox.data import COCO_CLASSES
from yolox.exp.build import get_exp_by_file
from yolox.utils import postprocess


def get_args():
    parser = argparse.ArgumentParser('Joint enhancement and detection net.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus being used')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=1, help='The number of images per batch among all devices')

    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('-m1', '--model1', type=str, default='maJitNet',
                        help='Model Name')
    parser.add_argument('-w1', '--ill_weight', type=str, default='./weights/UNetIllumi_003_101000.pth')
    parser.add_argument('-mdet', '--detect_weight', type=str, default='./weights/yolox_s.pth')

    parser.add_argument('--comment', type=str, default='trainTest',
                        help='Project comment')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--no_sche', action='store_true')
    parser.add_argument('--sampling', action='store_true')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.5)

    parser.add_argument('--gamma', '-ga', type=float, default=1.5)
    parser.add_argument('--alpha', '-a', type=float, default=0.2)
    parser.add_argument('-lw', '--loss_weights', nargs='+', type=float, default=[1, 1, 1],
                        help='weights of loss items')

    parser.add_argument('--optim', type=str, default='adam', help='select optimizer for training, '
                                                                  'suggest using \'admaw\' until the'
                                                                  ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=10, help='Number of epoches between valing phases')
    parser.add_argument('--data_path', type=str, default='/home/xteam/hj/dataset/LOLDataset',
                        help='the root folder of dataset')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--log_path', type=str, default='aTry/1/')
    parser.add_argument('--saved_path', type=str, default='aTry/1/')
    args = parser.parse_args()
    return args


class ModelJT(nn.Module):
    def __init__(self, enhan_model, det_model=None, exp=None):
        super().__init__()

        self.exp = exp
        if self.exp != None:
            self.num_classes = exp.num_classes
            self.confthre = exp.test_conf
            self.nmsthre = exp.nmsthre
            self.test_size = exp.test_size

            self.load_detetction_weight(self.det_model, opt.detect_weight)
            self.det_model.eval()

        self.weights = opt.loss_weights
        self.det_model = det_model
        self.enhan_model = enhan_model()
        self.restor_loss = models.MSELoss()
        self.feature_matching_loss = nn.L1Loss()
        self.restoration_loss = models.MSPerL1Loss(channels=3)
        self.anchor_loss = anchorLoss()

        self.enhan_model.load_ill_weight(opt.ill_weight)


    def load_detetction_weight(self, model, weight_pth):
        if model is not None:
            ckpt = torch.load(weight_pth)
            ret = model.load_state_dict(ckpt["model"])
            print(ret)

    def load_weight(self, model, weight_pth):
        if model is not None:
            state_dict = torch.load(weight_pth)
            print(state_dict.keys())
            ret = model.load_state_dict(state_dict, strict=True)
            print(ret)

    def normalize_tensor_mm(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def normalize_tensor_sigmoid(tensor):
        return nn.functional.sigmoid(tensor)

    def save_detection_feature(self, tensor, name, f_i):
        import torchvision.utils as vutils
        tensors = [tensor, self.normalize_tensor_mm(tensor), self.normalize_tensor_sigmoid(tensor)]
        titles = ['original', 'min-max', 'sigmoid']
        save_path = os.path.join(opt.saved_path, 'feature')
        os.makedirs(save_path, exist_ok=True)

        # for index, tensor in enumerate(tensors):
        _data = tensors.detach().cpu().squeeze(0).unsqueeze(1)
        num_per_row = 8
        grid = vutils.make_grid(_data, nrow=num_per_row)
        vutils.save_image(grid, f'{save_path}/{name}_{titles[f_i]}.png')

    def resizeletterbox(self, img, new_shape=(640, 640)):
        # resize image for detection net
        _, _, h0, w0 = img.shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / h0, new_shape[1] / w0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(h0 * r)), int(round(w0 * r))  # 收缩后图片的长宽
        dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if h0 != new_unpad and w0 != new_unpad:  # resize
            torch_resize = Resize([new_unpad[0], new_unpad[1]])
            img = torch_resize(img)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = Pad([left, top, right, bottom], fill=114 / 255, padding_mode='constant')(img)
        return img, ratio, (dw, dh)

    def resize_image(self, img, new_shape=640):
        _, _, h0, w0 = img.shape
        new_shapes = (new_shape, new_shape)
        r = min(new_shapes[0] / h0, new_shapes[1] / w0)
        new_unpad = int(round(h0 * r)), int(round(w0 * r))  # 收缩后图片的长宽
        if h0 != new_unpad and w0 != new_unpad:  # resize
            torch_resize = Resize([new_unpad[0], new_unpad[1]])
            img = torch_resize(img)

        return img

    def forward(self, image_in, image_gt, training=True):
        # # gt detection
        # with torch.no_grad():
        #     # get gt labels through detection
        #     image_det_gt, ratio, _ = self.resizeletterbox(image_gt, opt.tsize)
        #     image_det_gt *= 255
        #     bgr_index = [2, 1, 0]
        #     image_det_gt = image_det_gt[:, bgr_index, :, :]  # to bgr
        #     image_det_gt = image_det_gt.int()
        #     image_det_gt = image_det_gt.float()
        #     gt_head_feat, gt_outputs, gt_fpn, gt_dark = self.det_model(image_det_gt)
        #     # gt_outputs0 = gt_outputs
        #     # saver.save_cam(image_det_gt, gt_head_feat, 1, '1111')
        #
        #     gt_outputs = postprocess(
        #         gt_outputs, self.num_classes, self.confthre,
        #         self.nmsthre, class_agnostic=True
        #     )  # list: batch-size,  tensor (4, 7)  xyxy
        #     # transform it to res:   x, y, x, y, _, _, class type
        #     # ... * ratio
        #     ratios = torch.full([4], ratio[0])
        #
        #     targets = torch.zeros((opt.batch_size, 120, 5)).cuda()
        #     h = w = opt.tsize
        #     gn = torch.tensor((h, w))[[1, 0, 1, 0]].cuda()  # normalization gain whwh
        #     for det_i, gt_output in enumerate(gt_outputs):
        #         if gt_output != None:
        #             obj_num = gt_output.shape[0]
        #             targets[det_i, :obj_num, 1:5] = gt_output[:, 0:4] / gn
        #             targets[det_i, :obj_num, 0] = gt_output[:, 6]

        ill, out0, image_out = self.enhan_model(image_in)

        # fix
        # image_out = self.fix_model(image_out)
        ssim = SSIM1(image_out, image_gt)
        psnr = PSNR(image_out, image_gt)

        restor_loss = self.restor_loss(image_out, image_gt)

        return ill, out0, image_out, restor_loss, ssim, psnr

        # # start detection
        # image_det, _, pad = self.resizeletterbox(image_out, opt.tsize)
        # image_det *= 255.
        # bgr_index = [2, 1, 0]
        # image_det = image_det[:, bgr_index, :, :]  # to bgr
        #
        # if training:
        #     self.det_model.head.use_l1 = True
        #     det_head_feat, det_outputs, det_loss_outputs, det_fpn = self.det_model(image_det, targets=targets, getLosses=True)
        #     det_loss = det_loss_outputs["total_loss"]
        #     iou_loss = det_loss_outputs["iou_loss"]
        #     l1_loss = det_loss_outputs["l1_loss"]
        #     conf_loss = det_loss_outputs["conf_loss"]
        #     cls_loss = det_loss_outputs["cls_loss"]
        #
        #     # head_loss = self.getHeadLoss(gt_outputs0, det_outputs)
        #
        #     # feature vision
        #     feature_loss = []
        #     for det_fpn_i, gt_fpn_i in zip(det_fpn, gt_fpn):
        #         feature_loss.append(self.restor_loss(det_fpn_i, gt_fpn_i))
        #
        #     # head_loss = []
        #     # for det_head_feat_i, gt_head_feat_i in zip(det_head_feat, gt_head_feat):
        #     #     head_loss.append(self.restor_loss(det_head_feat_i, gt_head_feat_i))
        #
        #     head_iou_losses, head_cls_losses, head_obj_losses = self.anchor_loss(det_head_feat, gt_head_feat)
        #
        #     with torch.no_grad():
        #         self.det_model.head.use_l1 = False
        #         _, in_det_outputs, _, _ = self.det_model(image_det, getLosses=False)
        #         in_det_outputs = postprocess(
        #             in_det_outputs, self.num_classes, self.confthre,
        #             self.nmsthre, class_agnostic=True)
        #
        #     return image_out, restor_loss, det_loss, \
        #            iou_loss, l1_loss, conf_loss, cls_loss, \
        #            feature_loss, head_iou_losses, head_cls_losses, head_obj_losses, \
        #            psnr, ssim, in_det_outputs, gt_outputs
        # else:
        #     self.det_model.head.use_l1 = False
        #     det_head_feat, det_outputs, det_fpn, det_dark = self.det_model(image_det, getLosses=False)
        #
        #     det_outputs = postprocess(
        #         det_outputs, self.num_classes, self.confthre,
        #         self.nmsthre, class_agnostic=True)
        #
        #     return image_out, restor_loss, det_outputs, gt_outputs, \
        #            restor_loss, \
        #            psnr, ssim


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    timestamp = mutils.get_formatted_time()
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

    val_set = LowLightDataset(os.path.join(opt.data_path, 'eval15'),
    # val_set = LowLightDataset(os.path.join(opt.data_path, 'eval_sp'),
    # val_set = LowLightDataset(os.path.join(opt.data_path, 'Eval/Huawei'),
                              images_split='low',
                              targets_split='high')
    val_generator = DataLoader(val_set, **val_params)

    # Load detection model
    # exp = get_exp_by_file('YOLOX/exps/default/yolox_s.py')
    # exp.test_conf = opt.conf
    # exp.nmsthre = opt.nms
    # exp.test_size = (opt.tsize, opt.tsize)
    # model_det = exp.get_model()

    enhan_model = getattr(models, opt.model1)

    # model = ModelJT(enhan_model, model_det, exp)
    model = ModelJT(enhan_model)
    # print(model)

    writer = SingleSummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    if opt.num_gpus > 0:
        model = model.cuda()
        model.enhan_model.cuda()
        if opt.num_gpus > 1:
            model = nn.DataParallel(model)

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.enhan_model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.enhan_model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = CosineLR(optimizer, opt.lr, opt.num_epochs)
    epoch = 0
    step = 0
    val_step = 0

    model.enhan_model.train()
    # model.det_model.eval()

    num_iter_per_epoch = len(training_generator)

    # train_json_file = os.path.join(opt.data_path, 'train/train_3.json')
    # val_json_file = os.path.join(opt.data_path, 'eval15/eval_3.json')
    # val_json_file = os.path.join(opt.data_path, 'Eval/Huawei/eval_3.json')

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)

            # prepare for json
            # train_data_lists = []
            # train_gt_data_lists = []
            # train_json_saver = COCO_JSON(tsize=opt.tsize)
            # train_evaluator = JointEvaluator(annotation_file=train_json_file, img_size=opt.tsize, confthre=opt.conf,
            #                                  nmsthre=opt.nms, num_classes=exp.num_classes)

            if not opt.sampling:
                for iter, (data, target, name) in enumerate(progress_bar):
                    if iter < step - last_epoch * num_iter_per_epoch:
                        progress_bar.update()
                        continue
                    try:
                        if opt.num_gpus == 1:
                            data, target = data.cuda(), target.cuda()
                        optimizer.zero_grad()

                        ill, out1, image_out, restor_loss, ssim, psnr\
                            = model(data, target)

                        # evaluate
                        # _, _, h0, w0 = image_out.shape
                        # data_list_elem = train_evaluator.convert_to_coco_format(in_det_outputs, h0, w0, name)
                        # gt_list_elem = train_evaluator.convert_to_coco_format(gt_outputs, h0, w0, name)
                        # train_data_lists.extend(data_list_elem)
                        # train_gt_data_lists.extend(gt_list_elem)
                        #
                        # feature_losses = feature_loss[0] + feature_loss[1] + feature_loss[2]
                        # head_iou_losses = head_iou_loss[0] + head_iou_loss[1] + head_iou_loss[2]
                        # head_cls_losses = head_cls_loss[0] + head_cls_loss[1] + head_cls_loss[2]
                        # head_obj_losses = head_obj_loss[0] + head_obj_loss[1] + head_obj_loss[2]

                        # vis_graph = make_dot(head_obj_loss[0], params=dict(model.named_parameters()))
                        # vis_graph.render("head_loss_obj_struct", view=False)

                        # if epoch <= 10:
                        #     loss = restor_loss + feature_losses
                        # else:
                        #     loss = restor_loss + feature_losses + det_loss * 0.1
                        loss = restor_loss

                        loss.backward()
                        optimizer.step()

                        epoch_loss.append(float(loss))

                        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. loss: {:.5f}, psnr: {:.3f}, ssim: {:.3f}'.format(
                                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch,
                                loss.item(),
                                psnr, ssim))
                        writer.add_scalar('RestorLoss/train', loss, step)
                        # writer.add_scalar('DetectionLoss/train', det_loss, step)
                        # writer.add_scalar('IouLoss/train', iou_loss, step)
                        # writer.add_scalar('L1Loss/train', l1_loss, step)
                        # writer.add_scalar('ConfLoss/train', conf_loss, step)
                        # writer.add_scalar('ClsLoss/train', cls_loss, step)
                        # writer.add_scalar('FeatureLoss/train', feature_losses, step)
                        # writer.add_scalar('HeadIoULoss/train', head_iou_losses, step)
                        # writer.add_scalar('HeadClsLoss/train', head_cls_losses, step)
                        # writer.add_scalar('HeadObjLoss/train', head_obj_losses, step)
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

            if not opt.no_sche:
                scheduler.step()

            saver.base_url = os.path.join(opt.saved_path, 'results', '%03d' % epoch)

            if epoch % opt.val_interval == 0:
                # evaluate before val_interval
                # train_evaluator.evaluate_prediction(train_data_lists, train_gt_data_lists)
                #
                # val_evaluator = JointEvaluator(annotation_file=val_json_file, img_size=opt.tsize, confthre=opt.conf,
                #                                nmsthre=opt.nms, num_classes=exp.num_classes)

                model.enhan_model.eval()
                restor_loss_ls = []
                feature_loss_ls = []
                psnrs = []
                ssims = []

                data_list = []
                gt_data_list = []

                for iter, (data, target, name) in enumerate(val_generator):
                    with torch.no_grad():
                        if opt.num_gpus == 1:
                            data = data.cuda()
                            target = target.cuda()

                        ill, out1, image_out, restor_loss, ssim, psnr \
                            = model(data, target)

                        # evaluate
                        # _, _, h0, w0 = image_out.shape
                        # data_list_elem = val_evaluator.convert_to_coco_format(det_outputs, h0, w0, name)
                        # gt_list_elem = val_evaluator.convert_to_coco_format(gt_outputs, h0, w0, name)
                        # data_list.extend(data_list_elem)
                        # gt_data_list.extend(gt_list_elem)

                        saver.save_image(image_out, name=os.path.splitext(name[0])[0] + '_out')
                        saver.save_image(out1, name=os.path.splitext(name[0])[0] + '_out1')
                        saver.save_image(ill, name=os.path.splitext(name[0])[0] + '_i')
                        saver.save_image(target, name=os.path.splitext(name[0])[0] + '_gt')
                        # saveLabeledImage(gt_outputs, target, image_save_name=name[0] + '_gt_labeled')
                        # saveLabeledImage(det_outputs, image_out, image_save_name=name[0] + '_out_labeled')

                        # feature_losses = feature_loss[0] + feature_loss[1] + feature_loss[2]
                        loss = restor_loss
                        restor_loss_ls.append(loss.item())
                        # feature_loss_ls.append(feature_loss.item())
                        psnrs.append(psnr)
                        ssims.append(ssim.item())

                        val_step += 1

                restor_loss = np.mean(np.array(restor_loss_ls))
                # feature_loss = np.mean(np.array(feature_loss_ls))
                psnr = np.mean(np.array(psnrs))
                ssim = np.mean(np.array(ssims))

                # evaluate
                # val_evaluator.evaluate_prediction(data_list, gt_data_list)

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
                        epoch, opt.num_epochs, restor_loss, psnr, ssim))
                writer.add_scalar('RestorLoss/val', loss, step)
                # writer.add_scalar('FeatureLoss/train', feature_loss, step)
                writer.add_scalar('PSNR/val', psnr, step)
                writer.add_scalar('SSIM/val', ssim, step)

                save_checkpoint(model, f'JTN_{"%03d" % epoch}_{psnr}_{ssim}_{step}.pth')

                model.enhan_model.train()

    except KeyboardInterrupt:
        save_checkpoint(model, f'JTN_{epoch}_{step}_keyboardInterrupt.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, nn.DataParallel):
        torch.save(model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.state_dict(), os.path.join(opt.saved_path, name))

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
