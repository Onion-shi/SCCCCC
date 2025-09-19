import os, argparse, itertools, torch
from torch.utils.data import DataLoader
from configs import TrainCfg
from datasets import LabeledSet
from datasets_aug import UnlabeledOrigSet, UnlabeledAugPool
from models.textmatch import TextmatchNet
from losses.seg_losses import dice_loss, bce_loss
from losses.pgcl import ProtoBankEMA, info_nce_two_proto
from utils.ema import ema_update
from utils.common import set_seed, upsample_to
from tqdm import tqdm

def logits_to_prob(x): return torch.sigmoid(x)

def train_one_epoch(student, teacher, proto_bank, loaders, aug_pool, cfg, device):
    student.train(); teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    labeled_loader, unl_loader = loaders
    iters = min(len(labeled_loader), len(unl_loader))
    u_iter = iter(unl_loader)
    logs = {"sup":0., "reg":0., "pl":0., "cl":0., "total":0.}

    pbar = tqdm(enumerate(itertools.islice(labeled_loader, iters)),
                total=iters, desc="Train", ncols=100)

    for i, (imgs_l, masks_l, texts_l) in pbar:
        # ---------------- Labeled ----------------
        imgs_l, masks_l = imgs_l.to(device), masks_l.to(device)
        logits_l, emb_l = student(imgs_l, list(texts_l))
        logits_l = upsample_to(logits_l, masks_l.shape[-2:])
        L_sup = bce_loss(logits_l, masks_l) + dice_loss(logits_l, masks_l)
        proto_bank.update_from_labeled(emb_l.detach(), masks_l.detach())

        # ---------------- Unlabeled ----------------
        try:
            imgs_u, texts_u, keys = next(u_iter)
        except StopIteration:
            u_iter = iter(unl_loader)
            imgs_u, texts_u, keys = next(u_iter)
        imgs_u = imgs_u.to(device)

        logits_u_s, emb_u = student(imgs_u, list(texts_u))

        lt_list = []
        for _ in range(cfg.views_n - 1):
            imgs_aug_batch, texts_aug_batch = [], []
            for k in keys:
                imgs_k, texts_k = aug_pool.sample_views(k, 1)  # (1,3,H,W), [text]
                imgs_aug_batch.append(imgs_k[0])
                texts_aug_batch.append(texts_k[0])
            imgs_aug_batch = torch.stack(imgs_aug_batch, 0).to(device)

            with torch.no_grad():
                logit_t, emb_t = teacher(imgs_aug_batch, texts_aug_batch)
                lt_list.append(upsample_to(logit_t, logits_u_s.shape[-2:]))

        # ---------------- Losses ----------------
        L_reg = sum(torch.nn.functional.mse_loss(logits_u_s, lt) for lt in lt_list) / max(1, len(lt_list))

        with torch.no_grad():
            pl = torch.stack([logits_to_prob(lt) for lt in lt_list], 0).mean(0)  # (B,1,H,W)

        L_pl = dice_loss(logits_u_s, pl)

        y_prob = upsample_to(pl, emb_u.shape[-2:])  # (B,h,w)

        L_cl = info_nce_two_proto(
            emb_u, y_prob,
            proto_bank.pf.detach(), proto_bank.pb.detach(),
            tau=cfg.tau
        )

        # ---------------- Update ----------------
        L_total = L_sup + cfg.lambda_reg*L_reg + cfg.lambda_pl*L_pl + cfg.lambda_cl*L_cl
        opt.zero_grad(); L_total.backward(); opt.step()
        ema_update(student, teacher, m=cfg.ema_m)

        logs["sup"]+=L_sup.item(); logs["reg"]+=L_reg.item()
        logs["pl"]+=L_pl.item();   logs["cl"]+=L_cl.item()
        logs["total"]+=L_total.item()

        pbar.set_postfix({
            "sup": f"{L_sup.item():.3f}",
            "reg": f"{L_reg.item():.3f}",
            "pl":  f"{L_pl.item():.3f}",
            "cl":  f"{L_cl.item():.3f}",
            "tot": f"{L_total.item():.3f}"
        })

    for k in logs: logs[k]/=iters
    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_csv",
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\train_labeled_5.csv',
                        required=False, type=str,
                        help="CSV with labeled images (image_path,text,mask_path)")
    parser.add_argument("--unlabeled_csv",
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\train_unlabeled_5.csv',
                        required=False, type=str,
                        help="CSV with original unlabeled images (image_name,text)")
    parser.add_argument("--aug_pool_csv",
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\aug_pool.csv',
                        required=False, type=str,
                        help="CSV with augmented images (image_path,text,src_key)")
    parser.add_argument("--img_root",
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\img',
                        required=False, type=str,
                        help="Directory where original images are stored")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = TrainCfg()
    cfg.device = args.device
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # datasets
    L = LabeledSet(args.labeled_csv, cfg.img_size)
    U = UnlabeledOrigSet(args.unlabeled_csv, img_root=args.img_root, size=cfg.img_size)
    aug_pool = UnlabeledAugPool(args.aug_pool_csv, size=cfg.img_size)

    dl_L = DataLoader(L, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dl_U = DataLoader(U, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # models
    student = TextmatchNet(txt_model=cfg.txt_model, txt_finetune=False, max_len=cfg.text_max_len).to(device)
    teacher = TextmatchNet(txt_model=cfg.txt_model, txt_finetune=False, max_len=cfg.text_max_len).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad=False

    proto_bank = ProtoBankEMA(dim=128, m=0.9).to(device)

    for ep in range(1, cfg.epochs+1):
        logs = train_one_epoch(student, teacher, proto_bank, (dl_L, dl_U), aug_pool, cfg, device)
        print(f"[Epoch {ep:03d}] sup={logs['sup']:.4f} reg={logs['reg']:.4f} pl={logs['pl']:.4f} cl={logs['cl']:.4f} total={logs['total']:.4f}")

    os.makedirs("ckpts", exist_ok=True)
    torch.save(student.state_dict(), "ckpts/textmatch_student.pth")
    torch.save(teacher.state_dict(), "ckpts/textmatch_teacher.pth")
    print("Saved ckpts/.")
