import numpy as np
from PIL import Image
import torch
from utils.common import normalize_img_ch1, normalize_img_ch3

def base_resize(img: Image.Image, size=(256,256)):
    return img.resize(size, Image.BILINEAR)

def weak_tensor(img_pil: Image.Image):
    # 基础视图（学生）
    arr = np.array(img_pil).astype(np.uint8)
    if len(arr.shape) == 2:  # 如果是灰度图，转换为RGB
        arr = np.stack([arr, arr, arr], axis=2)
    t = normalize_img_ch3(torch.from_numpy(arr).permute(2, 0, 1))  # (3,H,W)
    return t

def strong_view(img_pil: Image.Image):
    # MCR: 随机缩放 + 亮度扰动（可再加形态学/噪声）
    w,h = img_pil.size
    # 简单缩放
    scale = np.random.uniform(0.9, 1.1)
    nw, nh = int(w*scale), int(h*scale)
    im = img_pil.resize((nw,nh))
    if nw>w or nh>h: im = im.crop((0,0,w,h))
    else:
        canvas = Image.new("RGB",(w,h),(0,0,0)); canvas.paste(im,(0,0)); im = canvas
    # 亮度
    arr = np.array(im).astype(np.float32)
    arr = np.clip(arr*np.random.uniform(0.9,1.1)+np.random.uniform(-5,5),0,255).astype(np.uint8)
    t = normalize_img_ch3(torch.from_numpy(arr).permute(2, 0, 1))
    return t
