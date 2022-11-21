import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import time
from tools import mutils

saved_grad = None
saved_name = None

base_url = './results'
os.makedirs(base_url, exist_ok=True)


def normalize_tensor_mm(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_tensor_sigmoid(tensor):
    return nn.functional.sigmoid(tensor)


def save_image(tensor, name=None, save_path=None, exit_flag=False, timestamp=False, norm=False):
    import torchvision.utils as vutils
    os.makedirs(base_url, exist_ok=True)
    if norm:
        tensor = normalize_tensor_mm(tensor)
    grid = vutils.make_grid(tensor.detach().cpu(), nrow=4)

    if save_path:
        vutils.save_image(grid, save_path)
    else:
        if timestamp:
            vutils.save_image(grid, f'{base_url}/{name}_{mutils.get_timestamp()}.png')
        else:
            vutils.save_image(grid, f'{base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_feature(tensor, name, exit_flag=False, timestamp=False):
    import torchvision.utils as vutils
    # tensors = [tensor, normalize_tensor_mm(tensor), normalize_tensor_sigmoid(tensor)]
    # tensors = [tensor]
    tensors = [normalize_tensor_mm(tensor)]
    # titles = ['original', 'min-max', 'sigmoid']
    titles = ['min-max']
    os.makedirs(base_url, exist_ok=True)
    if timestamp:
        name += '_' + str(time.time()).replace('.', '')

    for index, tensor in enumerate(tensors):
        _data = tensor.detach().cpu().squeeze(0).unsqueeze(1)
        num_per_row = 8
        grid = vutils.make_grid(_data, nrow=num_per_row)
        vutils.save_image(grid, f'{base_url}/{name}_{titles[index]}.png')
    if exit_flag:
        exit(0)


def save_cam(img, feature_maps, class_id, name, all_ids=85, image_size=(640, 640), exit_flag=False, timestamp=False, norm=True):
    '''

    :param img: input batch_size must equal 1
    :param feature_maps: feature_maps is a list with 3 tensors and each tensor shape is [batch_size, h_x, w_x, 85]
    '''
    titles = ['score', 'class', 'class*score']
    img_ori = img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    layer0 = feature_maps[0].reshape([-1, all_ids])
    layer1 = feature_maps[1].reshape([-1, all_ids])
    layer2 = feature_maps[2].reshape([-1, all_ids])
    layers = torch.cat([layer0, layer1, layer2], dim=0)

    if norm:
        score_max_v = 1.
        score_min_v = 0.
        class_max_v = 1.
        class_min_v = 0.
    else:
        score_max_v = layers[:, 4].max()
        score_min_v = layers[:, 4].min()
        class_max_v = layers[:, 5 + class_id].max()
        class_min_v = layers[:, 5 + class_id].min()

    for j in range(3):
        layer_one = feature_maps[j]
        if norm:
            anchors_score_max = layer_one[0, :4, :, :].max(0)[0].sigmoid()
            anchors_class_max = layer_one[0, 5 + class_id, :, :].max(0)[0].sigmoid()
        else:
            anchors_score_max = layer_one[0, :4, :, :].max(0)[0]
            anchors_class_max = layer_one[0, 5 + class_id, :, :].max(0)[0].unsqueeze(0)

        scores = ((anchors_score_max - score_min_v) / (score_max_v - score_min_v))
        classes = ((anchors_class_max - class_min_v) / (class_max_v - class_min_v))

        layer_one_list = []
        layer_one_list.append(scores)
        # layer_one_list.append(classes)
        # layer_one_list.append(scores * classes)
        for i, one in enumerate(layer_one_list):
            layer_one = one.cpu().numpy()
            if norm:
                ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
            else:
                ret = ((layer_one - 0.) / (1. - 0.)) * 255

            ret = ret.astype(np.uint8)
            gray = ret[:, :, None]
            ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            ret = cv2.resize(ret, image_size)
            # img_ori = cv2.resize(img_ori, image_size)


            show = ret * 0.5 + img_ori * 0.5
            show = show.astype(np.uint8)
            # cv2.imwrite(f'{base_url}/{name}_{j}_{titles[i]}.png', show)
            cv2.imwrite(f'/home/xteam/hj/workspace/JointEnhanDet/Bread/1119/111/results/000/{name}_{j}_{titles[i]}.png', show)
