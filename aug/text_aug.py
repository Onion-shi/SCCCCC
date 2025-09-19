# aug/text_aug.py
"""
Text augmentation using (local) BERT Masked-LM + simple token ops.

Usage:
    from aug.text_aug import TextAugmenter, rand_text_view

    # one-time init (in train startup)
    augmenter = TextAugmenter(model_dir="path/to/local/Bio_ClinicalBERT",
                              device="cuda", mask_prob=0.15, replace_prob=0.8)

    # per-sentence call (fast-ish)
    aug_sent = augmenter.augment("Bilateral pulmonary infection in left lung.")
    # or use helper function rand_text_view(sentence) if you set global augmenter
"""

from typing import List, Optional
import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re

# ---------- Defaults ----------
_DEFAULT_MODEL_DIR = r"C:\Users\22094\Desktop\semi-supervised\Textmatch\models\Bio_ClinicalBERT"  # can be local path
# ---------- end defaults ----------

# Simple whitespace tokenizer fallback (used only if transformers tokenizer fails)
def _simple_tokenize(text: str):
    # keep punctuation as separate tokens roughly
    toks = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return toks

class TextAugmenter:
    def __init__(self,
                 model_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 mask_prob: float = 0.15,
                 replace_prob: float = 0.8,
                 top_k: int = 5,
                 max_len: int = 64,
                 do_token_dropout: bool = True,
                 token_dropout_prob: float = 0.05,
                 do_token_shuffle: bool = False,
                 shuffle_max_span: int = 3,
                 seed: Optional[int] = None):
        """
        Args:
            model_dir: local path or HF name for the MLM model (must include tokenizer files and pytorch_model.bin).
            device: "cuda" or "cpu"
            mask_prob: fraction of tokens to consider masking per sentence
            replace_prob: when masked, prob of replacing token with MLM-predicted token (else keep or delete)
            top_k: number of top MLM candidates to sample from
            max_len: max token length (tokens will be truncated)
            do_token_dropout: randomly drop tokens with token_dropout_prob
            token_dropout_prob: prob to drop each token
            do_token_shuffle: whether to shuffle tokens inside small spans (to add noise)
            shuffle_max_span: max span size to shuffle
            seed: random seed for reproducibility
        """
        self.model_dir = model_dir or _DEFAULT_MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.top_k = max(1, int(top_k))
        self.max_len = max_len
        self.do_token_dropout = do_token_dropout
        self.token_dropout_prob = token_dropout_prob
        self.do_token_shuffle = do_token_shuffle
        self.shuffle_max_span = shuffle_max_span
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Load tokenizer + model once
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            self.mlm = AutoModelForMaskedLM.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
            self.mlm.eval()
            # Some tokenizers don't have pad_token defined; ensure it exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        except Exception as e:
            # fallback: disable MLM functionality, but still provide light augmentations
            print(f"[TextAugmenter] Warning: could not load MLM from {self.model_dir}: {e}")
            print("[TextAugmenter] Falling back to simple token ops only.")
            self.tokenizer = None
            self.mlm = None

    def _mlm_predict(self, input_ids: torch.Tensor, mask_positions: List[int], top_k: int):
        """
        Given tokenized input_ids (1,L), predict tokens at mask_positions.
        Returns dict pos -> list of token_ids (top_k candidates).
        """
        if self.mlm is None:
            return {p: [] for p in mask_positions}
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.mlm(input_ids.unsqueeze(0))  # batch dim
            logits = outputs.logits[0]  # (L, V)
            res = {}
            for p in mask_positions:
                # softmax + topk
                scores = logits[p]
                top = torch.topk(scores, k=min(top_k, scores.shape[-1])).indices.tolist()
                res[p] = top
        return res

    def _mask_and_replace(self, text: str) -> str:
        """Core MLM-based mask & replace augmentation."""
        if self.tokenizer is None:
            # fallback: simple dropout/shuffle
            toks = _simple_tokenize(text)
            toks = toks[:self.max_len]
            toks = self._apply_dropout_and_shuffle(toks)
            return " ".join(toks)

        # tokenize into tokens that map back to words via tokenizer
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        input_ids = encoded["input_ids"][0].clone()  # shape (L,)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # find candidate mask positions (exclude special tokens)
        cand_positions = []
        for i, tok in enumerate(tokens):
            if tok in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            # do not mask subword continuation tokens? We'll allow BERT subword masking
            cand_positions.append(i)

        if len(cand_positions) == 0:
            return text

        # sample positions ~ mask_prob
        n_to_mask = max(1, int(len(cand_positions) * self.mask_prob))
        mask_pos = sorted(random.sample(cand_positions, n_to_mask))

        # create masked input for MLM
        masked_ids = input_ids.clone()
        for p in mask_pos:
            masked_ids[p] = self.tokenizer.mask_token_id

        # predict candidates
        candidates = self._mlm_predict(masked_ids, mask_pos, self.top_k)

        # build new token list by replacing masked positions with sampled candidates
        new_ids = input_ids.tolist()
        for p in mask_pos:
            if random.random() <= self.replace_prob and candidates.get(p):
                cand_ids = candidates[p]
                # sample from top_k
                chosen = random.choice(cand_ids)
                new_ids[p] = int(chosen)
            else:
                # either keep original token id or delete token (we keep original to be conservative)
                new_ids[p] = int(input_ids[p].item())

        # decode preserving special tokens
        try:
            new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except:
            # fallback convert tokens:
            new_text = " ".join(self.tokenizer.convert_ids_to_tokens(new_ids))
        # apply optional small token ops
        toks_simple = _simple_tokenize(new_text)[:self.max_len]
        toks_simple = self._apply_dropout_and_shuffle(toks_simple)
        return " ".join(toks_simple)

    def _apply_dropout_and_shuffle(self, toks: List[str]) -> List[str]:
        out = []
        # token dropout
        for t in toks:
            if self.do_token_dropout and random.random() < self.token_dropout_prob:
                continue
            out.append(t)
        # small shuffle
        if self.do_token_shuffle and len(out) > 2:
            span = min(self.shuffle_max_span, len(out))
            # randomly pick a start and permute a small window
            s = random.randint(0, max(0, len(out) - span))
            window = out[s:s+span]
            random.shuffle(window)
            out = out[:s] + window + out[s+span:]
        return out

    def augment(self, text: str) -> str:
        """
        The main entry. Applies mask-replace augmentation; if MLM not available,
        fallback to token ops.
        """
        text = text if isinstance(text, str) else str(text)
        text = text.strip()
        if len(text) == 0:
            return "No text provided."

        # small guard: if too short, return same
        if len(text.split()) <= 2:
            return text

        return self._mask_and_replace(text)

# ----------------- convenience global functions -----------------
_global_augmenter = None

def init_augmenter(model_dir=None, device=None, **kwargs):
    global _global_augmenter
    _global_augmenter = TextAugmenter(model_dir=model_dir, device=device, **kwargs)
    return _global_augmenter

def rand_text_view(text: str):
    """
    Simple wrapper used in train.py: rand_text_view(t)
    Must call init_augmenter(...) once at startup (e.g. in train.py)
    """
    global _global_augmenter
    if _global_augmenter is None:
        # lazy init with default local model (will try local files only)
        _global_augmenter = TextAugmenter(model_dir=_DEFAULT_MODEL_DIR)
    return _global_augmenter.augment(text)
