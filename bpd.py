import torch, torch.nn as nn, torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, drop=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)

    def forward(self, q_tok, kv_tok):
        x = self.ln1(q_tok); y = self.ln1(kv_tok)
        out,_ = self.mha(x,y,y,need_weights=False)
        z = q_tok + out
        z = z + self.ffn(self.ln2(z))
        return z

class BPD(nn.Module):
    """
    双向提示 & 跳连融合，对齐论文式(1)-(5)的流程  :contentReference[oaicite:6]{index=6}
    """
    def __init__(self, c_img_in, c_txt_in, c_align=512, heads=8):
        super().__init__()
        self.img_proj = nn.Conv2d(c_img_in, c_align, 1)
        self.txt_proj = nn.Linear(c_txt_in, c_align)
        self.ca_img = CrossAttentionBlock(c_align, heads)
        self.ca_txt = CrossAttentionBlock(c_align, heads)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_align*2, c_align, 3, padding=1), nn.GELU(),
            nn.Conv2d(c_align, c_align, 3, padding=1), nn.GELU()
        )

    def forward(self, img_feat, txt_feat, low_skip):
        B,_,Hi,Wi = img_feat.shape
        Hs,Ws = low_skip.shape[-2:]
        fi = self.img_proj(img_feat)                    # (B,Ca,Hi,Wi)
        ft = self.txt_proj(txt_feat)                    # (B,L,Ca)
        ti = fi.flatten(2).transpose(1,2)               # (B,Hi*Wi,Ca)
        tt = ft                                         # (B,L,Ca)
        ti_p = self.ca_img(ti, tt)                      # I<-T
        tt_p = self.ca_txt(tt, ti)                      # T<-I
        fi_p = ti_p.transpose(1,2).view(B,-1,Hi,Wi)
        fi_up = F.interpolate(fi_p, size=(Hs,Ws), mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([fi_up, low_skip], dim=1))
        return fused, tt_p
