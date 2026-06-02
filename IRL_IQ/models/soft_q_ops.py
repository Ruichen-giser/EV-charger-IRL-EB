"""掩膜 soft Q 辅助（BC 熵统计 / IQ-Learn V(s)）。"""
import torch


def masked_soft_value(q: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    neg_inf = torch.finfo(q.dtype).min
    qm = q.masked_fill(~mask, neg_inf)
    temp = max(float(alpha), 1e-8)
    v = temp * torch.logsumexp(qm / temp, dim=1, keepdim=True)
    # 无合法动作时 logsumexp 为 -inf，避免 IQ 目标里出现 NaN
    no_valid = ~mask.any(dim=-1, keepdim=True)
    return torch.where(no_valid, torch.zeros_like(v), v)


def soft_policy_entropy(q: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    neg_inf = torch.finfo(q.dtype).min
    qm = q.masked_fill(~mask, neg_inf)
    temp = max(float(alpha), 1e-8)
    logp = torch.log_softmax(qm / temp, dim=-1)
    p = logp.exp() * mask.float()
    z = p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    p = p / z
    safe_logp = torch.where(p > 1e-12, logp, torch.zeros_like(logp))
    ent = -(p * safe_logp).sum(dim=-1)
    no_valid = ~mask.any(dim=-1)
    ent = torch.where(no_valid, torch.zeros_like(ent), ent)
    return ent.mean()
