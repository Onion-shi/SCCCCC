import csv, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
from aug.img_aug import base_resize, weak_tensor
from utils.common import normalize_img_ch1

class LabeledSet(Dataset):
    """
    CSV 列：image_path,mask_path,text
    mask 单通道 {0,1} png
    """
    def __init__(self, csv_path, size=(256,256)):
        self.size = size
        self.recs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f): self.recs.append(r)

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = base_resize(Image.open(r["image_path"]).convert("RGB"), self.size)
        msk = Image.open(r["mask_path"]).convert("L").resize(self.size, Image.NEAREST)

        x = weak_tensor(img)  # (3,H,W)
        m = (torch.from_numpy(np.array(msk))).float()
        m = (m > 127).float().unsqueeze(0)  # (1,H,W)

        if "text" in r and r["text"].strip() != "":
            txt = r["text"]
        else:
            txt = "No text provided."

        return x, m, txt
