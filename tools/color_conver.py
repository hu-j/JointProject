import kornia
import torch

def rgb_to_yuv(rgb_im, dim=0):
    y, u, v = torch.split(kornia.color.rgb_to_yuv(rgb_im), 1, dim=dim)
    u = (u + 0.436) / 0.872
    v = (v + 0.615) / 1.23

    return y, u, v