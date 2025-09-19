# losses/pgcl.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoBankEMA(nn.Module):
    """
    前景/背景两类原型，用有标注 batch 的像素均值初始化，并用 EMA 更新，对齐 Eq.(9)。
    """
    def __init__(self, dim: int, m: float = 0.9):
        super().__init__()
        self.m = m
        self.register_buffer("pf", torch.zeros(dim))
        self.register_buffer("pb", torch.zeros(dim))
        self.register_buffer("init_f", torch.tensor(0, dtype=torch.long))
        self.register_buffer("init_b", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_from_labeled(self, feat_hw: torch.Tensor, mask_hw: torch.Tensor):
        # feat_hw: (B,C,H,W), mask_hw: (B,1,H,W)
        B, C, H, W = feat_hw.shape
        x = feat_hw.permute(0,2,3,1).reshape(-1, C)  # (N,C)
        y = (mask_hw > 0.5).float().view(-1)         # (N,)

        if y.sum() > 0:
            mf = x[y > 0.5].mean(dim=0)
            if self.init_f.item() == 0:
                self.pf.copy_(mf); self.init_f.fill_(1)
            else:
                self.pf.mul_(self.m).add_(mf, alpha=1 - self.m)

        if (1 - y).sum() > 0:
            mb = x[y <= 0.5].mean(dim=0)
            if self.init_b.item() == 0:
                self.pb.copy_(mb); self.init_b.fill_(1)
            else:
                self.pb.mul_(self.m).add_(mb, alpha=1 - self.m)

def info_nce_two_proto(
    feat_hw: torch.Tensor,     # (B,C,H,W) 学生像素特征（已上采样到输入分辨率）
    y_prob_hw: torch.Tensor,   # (B,1,H,W) 伪标签概率 \hat{y} ∈ [0,1]
    pf: torch.Tensor,          # (C,)
    pb: torch.Tensor,          # (C,)
    tau: float = 0.9,
):
    """
    软权重 PGCL，严格按 Eq.(10)-(11)：
    L_cl = - mean( \hat{y} * log p_fg + (1-\hat{y}) * log p_bg )
    其中 p_fg、p_bg 为对两原型的二元 softmax 概率，使用 cosine sim / tau。
    """
    B, C, H, W = feat_hw.shape
    x = feat_hw.permute(0,2,3,1).reshape(-1, C)       # (N,C)
    y_prob = y_prob_hw.view(-1).clamp(0.0, 1.0)       # (N,)

    # 归一化后做余弦相似
    x = F.normalize(x, dim=1)
    pf = F.normalize(pf, dim=0)
    pb = F.normalize(pb, dim=0)

    sim_f = (x @ pf) / tau                            # (N,)
    sim_b = (x @ pb) / tau                            # (N,)

    den = torch.logsumexp(torch.stack([sim_f, sim_b], dim=1), dim=1)  # (N,)
    log_p_fg = sim_f - den
    log_p_bg = sim_b - den

    loss = -(y_prob * log_p_fg + (1.0 - y_prob) * log_p_bg).mean()
    return loss
