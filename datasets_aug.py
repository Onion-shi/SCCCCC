import csv, random, os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from aug.img_aug import base_resize, weak_tensor


class UnlabeledOrigSet(Dataset):
    """
    原始无标签数据集
    CSV: image_path, description
    """
    def __init__(self, csv_path, img_root="", size=(256,256)):
        self.size = size
        self.img_root = img_root
        self.recs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.recs.append(r)

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img_path = r["image_path"]  # 使用 image_path
        img = base_resize(Image.open(img_path).convert("RGB"), self.size)
        x = weak_tensor(img)
        txt = r.get("description", "").strip() or "No text provided."
        key = os.path.splitext(os.path.basename(img_path))[0]  #
        return x, txt, key

class UnlabeledAugPool(Dataset):
    """
    无标签增强视图池
    CSV: image_path, text, src_key
    src_key 对应原始图像的 key (文件名不带扩展名)，用来和 UnlabeledOrigSet 对齐
    """
    def __init__(self, csv_path, size=(256,256)):
        self.size = size
        self.recs = []
        self.by_key = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.recs.append(r)
                k = r["src_key"]
                self.by_key.setdefault(k, []).append(r)

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = base_resize(Image.open(r["image_path"]).convert("RGB"), self.size)
        x = weak_tensor(img)
        txt = r.get("text", "").strip()
        if txt == "": txt = "No text provided."
        return x, txt, r["src_key"]

    def sample_views(self, key, n=1):
        """
        随机采样某个原始图像的 n 个增强视图
        """
        if key not in self.by_key:
            raise ValueError(f"No augmented views found for key={key}")
        recs = random.choices(self.by_key[key], k=n)
        imgs, txts = [], []
        for r in recs:
            img = base_resize(Image.open(r["image_path"]).convert("RGB"), self.size)
            x = weak_tensor(img)
            txt = r.get("text", "").strip()
            if txt == "": txt = "No text provided."
            imgs.append(x)
            txts.append(txt)
        return imgs, txts
