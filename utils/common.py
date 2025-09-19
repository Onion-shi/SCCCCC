import random, numpy as np, torch
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def upsample_to(x, size_hw):
    return F.interpolate(x, size_hw, mode="bilinear", align_corners=False)

def normalize_img_ch1(x_uint8):
    x = x_uint8.float()/255.0
    return x

def normalize_img_ch3(x_uint8):
    x = x_uint8.float()/255.0
    return x