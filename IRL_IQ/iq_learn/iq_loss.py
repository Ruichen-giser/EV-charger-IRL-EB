"""
IQ-Learn 损失（NeurIPS 2021），参考 https://github.com/Div-Infinity/IQ-Learn 的 iq.py。

离线专家 + value_expert + χ² 正则（与 CartPole offline 配置一致）。
"""
import torch


def iq_learn_loss(
    *,
    q_sa: torch.Tensor,
    v_s: torch.Tensor,
    next_v: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    alpha_reg: float = 0.5,
    use_chi: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    对专家转移 (s,a,s') 计算 IQ-Learn 目标。
    调用方应仅传入 is_expert=True 的样本（见 DiscreteSoftQAgent.train_step）。

    q_sa: Q(s,a)           shape (B,1)
    v_s:  V(s)=α log Σ exp(Q/α)  shape (B,1)
    next_v: V(s')          shape (B,1)
    done:  episode 结束    shape (B,1)
    """
    y = (1.0 - done) * float(gamma) * next_v
    reward = q_sa - y

    # -E_expert[Q(s,a) - γV(s')]
    softq_loss = -reward.mean()

    # value_expert: E_expert[V(s) - γV(s')]
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

    metrics["total_loss"] = float(loss.detach())
    return loss, metrics
