import torch, torch.nn.functional as F

def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2*(p*target).sum() + eps
    den = p.sum() + target.sum() + eps
    return 1 - num/den

def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)
