import os, csv, argparse, numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==== 模型 ====
from models.textmatch import TextmatchNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------- Dataset（与训练保持一致，RGB + ImageNet 归一化） -----------------
class ValSet(Dataset):
    """
    CSV 列：image_path,mask_path,text
    """
    def __init__(self, csv_path, size=(256,256)):
        self.size = size
        self.recs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f): self.recs.append(r)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r["image_path"]).convert("RGB").resize(self.size, Image.BILINEAR)
        x = self.to_tensor(img)           # (3,H,W)
        if x.shape[0] == 1:               # 兜底
            x = x.repeat(3,1,1)
        x = self.normalize(x)

        msk = Image.open(r["mask_path"]).convert("L").resize(self.size, Image.NEAREST)
        m = torch.from_numpy(np.array(msk)).float()  # (H,W)
        m = (m > 127).float().unsqueeze(0)           # (1,H,W)

        txt = r.get("text", "")
        txt = txt if isinstance(txt, str) and txt.strip() != "" else "No text provided."
        return x, m, txt, r["image_path"]

# ----------------- Metrics -----------------
def dice_iou_from_logits(logits, gt, thr=0.5, eps=1e-6):
    """
    logits: (B,1,H,W) raw
    gt:     (B,1,H,W) {0,1}
    """
    prob = torch.sigmoid(logits)
    pred = (prob >= thr).float()

    inter = (pred*gt).sum(dim=(1,2,3))
    union = (pred+gt - pred*gt).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3)) + eps)
    iou  = (inter + eps) / (union + eps)
    return dice.cpu().numpy(), iou.cpu().numpy(), prob

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=False,
                    default=r'D:\2025\scc\Textmatch\data\covid19\Test_Folder\new_test_text.csv',
                    type=str, help="CSV: image_path,mask_path,text")
    ap.add_argument("--ckpt", required=False,default=r'D:\2025\scc\Textmatch\ckpts\textmatch_student.pth', type=str, help="student or teacher .pth")
    ap.add_argument("--txt_model", default=r"D:\2025\scc\Textmatch\models\Bio_ClinicalBERT", type=str)
    ap.add_argument("--img_size", default=256, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_pred_dir", type=str, default=r"D:\2025\scc\Textmatch\data\covid19\save", help="如需保存预测 PNG，给目录")
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = ValSet(args.val_csv, size=(args.img_size, args.img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 模型
    model = TextmatchNet(txt_model=args.txt_model, txt_finetune=False, max_len=64).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    if args.save_pred_dir:
        os.makedirs(args.save_pred_dir, exist_ok=True)

    dices, ious = [], []
    with torch.no_grad():
        for xb, mb, tb, paths in tqdm(dl, desc="Eval", ncols=100):
            xb = xb.to(device)          # (B,3,H,W)
            mb = mb.to(device)          # (B,1,H,W)
            texts = list(tb)            # list[str]

            logits, emb_pgcl = model(xb, texts)  # logits: (B,1,H,W)
            d, i, prob = dice_iou_from_logits(logits, mb, thr=args.thr)
            dices.extend(d.tolist()); ious.extend(i.tolist())

            # 保存预测
            if args.save_pred_dir:
                prob_np = prob.squeeze(1).cpu().numpy()  # (B,H,W)
                for k, p in enumerate(paths):
                    name = os.path.splitext(os.path.basename(p))[0]
                    pred_png = (prob_np[k]*255.0).astype(np.uint8)
                    Image.fromarray(pred_png).save(os.path.join(args.save_pred_dir, f"{name}_prob.png"))

    # 汇总
    dice_mean = float(np.mean(dices)) if dices else 0.0
    iou_mean  = float(np.mean(ious))  if ious  else 0.0
    print(f"[Val] Dice={dice_mean:.4f} IoU={iou_mean:.4f}  (N={len(dices)})")

if __name__ == "__main__":
    main()
