import torch, torch.nn as nn

@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m=0.999):
    # Mean-Teacher: EMA 更新教师参数  :contentReference[oaicite:3]{index=3}
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=(1. - m))
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)
