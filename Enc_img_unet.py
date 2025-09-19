# src/models/enc_image_unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.body(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class UNetEncoder(nn.Module):
    """
    简单 U-Net 编码器，返回4级特征: [f1, f2, f3, f4]
    通道默认 [64, 128, 256, 512]，输入灰度(1通道)。
    特征空间分辨率依次为 1/2, 1/4, 1/8, 1/16（如果你想严格对齐 1/4 开始，可在 dataloader 里先下采样一次，或再加一层 down）
    """
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.inc  = DoubleConv(in_ch, c1)   # H, W
        self.down1 = Down(c1, c2)           # H/2
        self.down2 = Down(c2, c3)           # H/4
        self.down3 = Down(c3, c4)           # H/8

        # 为了与原先 “1/4 -> 1/8 -> 1/16 -> 1/32” 的4级层级更接近，
        # 你也可以再加一层 Down 得到 H/16 的 f4；这里保持三次下采样（到 H/8），
        # 若你想再下采一次：
        # self.down4 = Down(c4, c4)  # -> H/16
        # 并把 forward / feat_channels 按需调整。

        self._feat_channels = [c1, c2, c3, c4]

    def feature_info(self):  # 兼容 timm 风格
        class _FI:
            def __init__(self, chs): self._chs = chs
            def channels(self): return self._chs
        return _FI(self._feat_channels)

    def forward(self, x):
        f1 = self.inc(x)      # H,   C=64
        f2 = self.down1(f1)   # H/2, C=128
        f3 = self.down2(f2)   # H/4, C=256
        f4 = self.down3(f3)   # H/8, C=512
        return [f1, f2, f3, f4]

    @property
    def feat_channels(self):
        return self._feat_channels
