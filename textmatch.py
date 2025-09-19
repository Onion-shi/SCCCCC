import torch.nn as nn, torch.nn.functional as F
from .enc_image import ConvNeXtTinyEncoder
from .enc_text import BioClinicalBERTEncoder
from .bpd import BPD

# 统一超参
TXT_DIM   = 512
ALIGN_DIM = 512
HEADS     = 8
PROJ_DIM  = 128  # 对应 ProtoBankEMA(dim=128)

class SegHead(nn.Module):
    def __init__(self, in_ch=ALIGN_DIM, mid=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, 1, 1),
        )
    def forward(self, x): return self.body(x)

class PixelProjector(nn.Module):
    """把给定特征投影到 PROJ_DIM，并可上采到目标尺寸。"""
    def __init__(self, in_ch=ALIGN_DIM, out_ch=PROJ_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, feat, out_hw):
        z = self.proj(feat)                          # B × out_ch × h × w
        return F.interpolate(z, out_hw, mode='bilinear', align_corners=False)

class TextmatchNet(nn.Module):
    """
    Backbone: ConvNeXt-Tiny（features_only）
      f1: H/4  (96), f2: H/8  (192), f3: H/16 (384), f4: H/32 (768)

    3×BPD（自深到浅）：
      BPD1: img=f4,      low=reduce3(f3) -> H/16
      BPD2: img=out(H/16), low=reduce2(f2) -> H/8
      BPD3: img=out(H/8),  low=reduce1(f1) -> H/4

    PGCL 对齐论文 Fig.1(d)：
      原型/对比的像素嵌入来自  “Stage3 的投影特征”  —— 即 reduce3(f3) 再投影到 128 维，
      而不是最终解码特征。这样语义更集中、更稳定。
    """
    def __init__(self, txt_model='emilyalsentzer/Bio_ClinicalBERT',
                 txt_finetune=False, max_len=64):
        super().__init__()
        # 编码器
        self.img_enc = ConvNeXtTinyEncoder(out_indices=(0,1,2,3))
        self.txt_enc = BioClinicalBERTEncoder(model_name=txt_model,
                                              out_dim=TXT_DIM,
                                              fine_tune=txt_finetune,
                                              max_len=max_len)
        c1,c2,c3,c4 = self.img_enc.feat_channels  # [96,192,384,768]

        # 对齐到 ALIGN_DIM 作为 skip
        self.reduce1 = nn.Conv2d(c1, ALIGN_DIM, 1)   # stage1 skip (H/4)
        self.reduce2 = nn.Conv2d(c2, ALIGN_DIM, 1)   # stage2 skip (H/8)
        self.reduce3 = nn.Conv2d(c3, ALIGN_DIM, 1)   # stage3 skip (H/16)

        # —— 关键新增：Stage3 的“PGCL投影头” —— #
        # 论文中的 projection head（用在 f^I / stage3 上），输出 128 维像素嵌入供 PGCL/原型使用
        self.pgcl_proj = PixelProjector(in_ch=ALIGN_DIM, out_ch=PROJ_DIM)

        # 三个 BPD（内部已包含 I↔T 双向 cross-attention、上采 & concat）
        self.bpd1 = BPD(c_img_in=c4,        c_txt_in=TXT_DIM,   c_align=ALIGN_DIM, heads=HEADS)
        self.bpd2 = BPD(c_img_in=ALIGN_DIM, c_txt_in=ALIGN_DIM, c_align=ALIGN_DIM, heads=HEADS)
        self.bpd3 = BPD(c_img_in=ALIGN_DIM, c_txt_in=ALIGN_DIM, c_align=ALIGN_DIM, heads=HEADS)

        # 分割头
        self.seg = SegHead(in_ch=ALIGN_DIM)

    def forward(self, imgs, texts):
        # 1) Backbone 多尺度特征
        f1, f2, f3, f4 = self.img_enc(imgs)                 # [H/4, H/8, H/16, H/32]
        t_tok = self.txt_enc(texts, device=imgs.device)     # B × Lt × 512

        # 2) —— PGCL 的像素嵌入来自 Stage3（对齐 Fig.1(d)）——
        #    先把 f3 映射到 ALIGN_DIM，再通过 projection head 投影到 128 维；
        #    最后上采到输入分辨率（与你的训练代码一致，方便 upsample_to 对齐）。
        s3_aligned = self.reduce3(f3)                       # B × ALIGN_DIM × H/16 × W/16
        emb_pgcl   = self.pgcl_proj(s3_aligned, out_hw=imgs.shape[-2:])  # B × 128 × H × W

        # 3) 三级 BPD 解码到 H/4
        low3 = s3_aligned                                   # H/16
        out3, t_tok = self.bpd1(f4, t_tok, low3)            # -> H/16

        low2 = self.reduce2(f2)                             # H/8
        out2, t_tok = self.bpd2(out3, t_tok, low2)          # -> H/8

        low1 = self.reduce1(f1)                             # H/4
        out1, t_tok = self.bpd3(out2, t_tok, low1)          # -> H/4

        # 4) Segmentation 头，输出上采回原图大小
        logits = self.seg(out1)                             # H/4
        logits = F.interpolate(logits, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

        # 返回：分割 logits（监督/伪监督用） + emb_pgcl（PGCL/原型用）
        return logits, emb_pgcl
