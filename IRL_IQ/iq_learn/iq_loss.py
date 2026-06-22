"""
IQ-Learn 损失（NeurIPS 2021），参考 https://github.com/Div-Infinity/IQ-Learn 的 iq.py。

- offline（value_expert）：Q 项、V 项、χ² 均在专家转移上
- online（value + regularize）：Q 项在专家上；V 项在专家+策略上；(V-γV').mean()；χ² 可选覆盖全 batch
"""
from __future__ import annotations

import torch

IQ_LOSS_MODES = ("online", "offline")


def iq_learn_loss(
    *,
    q_sa: torch.Tensor,
    v_s: torch.Tensor,
    next_v: torch.Tensor,
    done: torch.Tensor,
    is_expert: torch.Tensor | None,
    gamma: float,
    alpha_reg: float = 0.5,
    use_chi: bool = True,
    loss_mode: str = "online",
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    q_sa, v_s, next_v, done: 整 batch（专家 + 策略），shape (B, 1)
    is_expert: bool mask shape (B,) 或 (B, 1)；online 模式必需
    loss_mode: "online" | "offline"
    """
    mode = str(loss_mode).strip().lower()
    if mode not in IQ_LOSS_MODES:
        raise ValueError(f"loss_mode 须为 {IQ_LOSS_MODES}，当前: {loss_mode!r}")

    y = (1.0 - done) * float(gamma) * next_v

    if mode == "online":
        if is_expert is None:
            raise ValueError("online 模式需要 is_expert 标记以计算 softq 专家项")
        m = is_expert.view(-1).bool()
        if not bool(m.any()):
            raise ValueError("online 模式 batch 中无专家样本，无法计算 softq_loss")

        reward_expert = (q_sa - y)[m]
        softq_loss = -reward_expert.mean()
        value_loss = (v_s - y).mean()

        loss = softq_loss + value_loss
        metrics = {
            "softq_loss": float(softq_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }

        if use_chi:
            reward_all = q_sa - y
            chi2_loss = (reward_all.pow(2).mean()) / (4.0 * float(alpha_reg))
            loss = loss + chi2_loss
            metrics["chi2_loss"] = float(chi2_loss.detach())
            metrics["chi2_scope"] = "regularize_all"
    else:
        if is_expert is not None:
            m = is_expert.view(-1).bool()
            if bool(m.any()):
                q_sa = q_sa[m]
                v_s = v_s[m]
                next_v = next_v[m]
                done = done[m]
                y = y[m]

        reward = q_sa - y
        softq_loss = -reward.mean()
        value_loss = (v_s - y).mean()

        loss = softq_loss + value_loss
        metrics = {
            "softq_loss": float(softq_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }

        if use_chi:
            chi2_loss = (reward.pow(2).mean()) / (4.0 * float(alpha_reg))
            loss = loss + chi2_loss
            metrics["chi2_loss"] = float(chi2_loss.detach())
            metrics["chi2_scope"] = "expert_only"

    metrics["total_loss"] = float(loss.detach())
    metrics["iq_loss_mode"] = mode
    return loss, metrics
