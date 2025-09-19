# src/models/enc_text.py
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

# ====== 硬编码：本地 Bio_ClinicalBERT 目录 ======
BIOBERT_DIR = r"D:\2025\scc\Textmatch\models\Bio_ClinicalBERT"
MAX_LEN = 64
OUT_DIM = 512
FINE_TUNE = False
# ============================================

class BioClinicalBERTEncoder(nn.Module):
    """
    严格离线：只从 BIOBERT_DIR 读取（local_files_only=True）
    需包含: config.json / pytorch_model.bin / vocab.txt / tokenizer_config.json / special_tokens_map.json
    """
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", out_dim=512, fine_tune=False, max_len=64):
        super().__init__()
        # 使用传入的参数或默认值
        self.max_len = max_len
        self.out_dim = out_dim
        
        # 检查本地模型目录是否存在
        if not os.path.isdir(BIOBERT_DIR):
            raise FileNotFoundError(f"BIOBERT_DIR not found: {BIOBERT_DIR}")
        
        self.tok = AutoTokenizer.from_pretrained(BIOBERT_DIR, local_files_only=True)
        self.bert = AutoModel.from_pretrained(BIOBERT_DIR, local_files_only=True)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim)
        
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, texts, device):
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=self.max_len, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.bert(**enc).last_hidden_state  # (B,L,H)
        return self.proj(out)                     # (B,L,OUT_DIM)
