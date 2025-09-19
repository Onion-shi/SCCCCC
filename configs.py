from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainCfg:
    seed: int = 0
    device: str = "cuda"
    img_size: Tuple[int,int] = (256,256)
    batch_size: int = 8
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.0
    ema_m: float = 0.99
    views_n: int = 4       # MCR: 1 学生 + (views_n-1) 个教师视图  :contentReference[oaicite:1]{index=1}
    lambda_reg: float = 0.1
    lambda_pl: float = 0.1 # MosMed 可设 0.5 以加强无标注监督  :contentReference[oaicite:2]{index=2}
    lambda_cl: float = 0.1
    tau: float = 0.9
    text_max_len: int = 64
    txt_model: str = "emilyalsentzer/Bio_ClinicalBERT"
