"""离散 Soft Q：SimpleGridCNN（与 IRL_BC 相同架构与超参）。"""
import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cnn_config import CNN_DROPOUT, CNN_N_CONV_LAYERS
from models.simple_grid_cnn import SimpleGridCNN
from models.soft_q_ops import masked_soft_value, soft_policy_entropy


class SimpleGridCNNQ(nn.Module):
    """SimpleGridCNN + 掩膜 soft V(s)，接口与旧 GridSoftQCNN 一致。"""

    def __init__(
        self,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        action_dim: int,
        *,
        n_conv_layers: int = CNN_N_CONV_LAYERS,
        dropout: float = CNN_DROPOUT,
    ) -> None:
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.net = SimpleGridCNN(
            in_channels=int(in_channels),
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            action_dim=int(action_dim),
            n_conv_layers=int(n_conv_layers),
            dropout=float(dropout),
        )

    def forward(self, obs: torch.Tensor, county_ids: torch.Tensor | None = None) -> torch.Tensor:
        del county_ids
        return self.net(obs)

    def q_values(self, obs: torch.Tensor, actions: torch.Tensor, county_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(obs, county_ids).gather(1, actions.long())

    def soft_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_soft_value(self.forward(obs, county_ids), mask, alpha)


class DiscreteSoftQAgent:
    """IQ-Learn Soft Q agent（Q + target，SimpleGridCNN）。"""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.01,
        alpha_reg: float = 0.5,
        use_chi: bool = True,
        target_update_interval: int = 2,
        grid_h: int = 0,
        grid_w: int = 0,
        in_channels: int = 0,
        n_conv_layers: int = CNN_N_CONV_LAYERS,
        dropout: float = CNN_DROPOUT,
        **_: Any,
    ) -> None:
        del obs_dim
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.alpha_reg = float(alpha_reg)
        self.use_chi = bool(use_chi)
        self.target_update_interval = int(target_update_interval)
        self._step = 0
        self.network_type = "SimpleGridCNN"
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.in_channels = int(in_channels)
        self.n_actions = int(n_actions)

        self.q_net: nn.Module = SimpleGridCNNQ(
            in_channels=self.in_channels,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            action_dim=self.n_actions,
            n_conv_layers=int(n_conv_layers),
            dropout=float(dropout),
        ).to(self.device)

        self.target_net = copy.deepcopy(self.q_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.q_net.train()

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        done = batch["done"]
        mask = batch["mask"]
        next_mask = batch["next_mask"]
        is_expert = batch.get("is_expert")

        q_all = self.q_net(obs)
        q_sa = q_all.gather(1, actions.long())
        v_s = self.q_net.soft_value(obs, mask, self.alpha)

        with torch.no_grad():
            next_v = self.target_net.soft_value(next_obs, next_mask, self.alpha)
            entropy = float(soft_policy_entropy(q_all, mask, self.alpha).detach())
            if is_expert is not None:
                m_all = is_expert.view(-1).bool()
                q_valid = q_all[m_all][mask[m_all]] if bool(m_all.any()) else q_all.new_zeros(0)
            else:
                q_valid = q_all[mask]

        # IQ-Learn：专家对 (s,a) 为主；策略 buffer 的 (s,a) 也参与同一目标，使 closed-loop
        # 访问的状态获得梯度（仅训专家时 policy rollout 指标常会整条曲线不变）。
        if is_expert is not None:
            m = is_expert.view(-1).bool()
            if not bool(m.any()):
                return {
                    "loss": 0.0,
                    "skipped_step": 1.0,
                    "Q_mean": 0.0,
                    "Q_std": 0.0,
                    "Q_max": 0.0,
                    "policy_entropy": entropy,
                }

        from iq_learn.iq_loss import iq_learn_loss

        loss, metrics = iq_learn_loss(
            q_sa=q_sa,
            v_s=v_s,
            next_v=next_v,
            done=done,
            gamma=self.gamma,
            alpha_reg=self.alpha_reg,
            use_chi=self.use_chi,
        )

        self.optimizer.zero_grad()
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            self.optimizer.step()
        else:
            metrics["skipped_step"] = 1.0

        self._step += 1
        if self._step % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        metrics.update(
            {
                "loss": float(loss.detach()),
                "Q_mean": float(q_valid.mean().detach()) if q_valid.numel() else 0.0,
                "Q_std": float(q_valid.std().detach()) if q_valid.numel() > 1 else 0.0,
                "Q_max": float(q_valid.max().detach()) if q_valid.numel() else 0.0,
                "policy_entropy": entropy,
            }
        )
        return metrics

    @torch.no_grad()
    def sample_action_soft(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        rng: np.random.Generator,
    ) -> int:
        """仅在合法动作上做数值稳定的 softmax 采样。"""
        self.q_net.eval()
        q = self.q_net(obs).detach().reshape(-1).cpu().numpy()
        valid = mask.reshape(-1).cpu().numpy().astype(bool)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            raise RuntimeError(
                "sample_action_soft: 无合法动作（action mask 全 False）。"
                "请确认环境在 legacy 模式下已全部建站后 terminated=True，或 rollout 已 reset。"
            )

        qv = np.asarray(q[valid_idx], dtype=np.float64)
        if not np.isfinite(qv).all():
            return int(rng.choice(valid_idx))

        temp = max(float(self.alpha), 1e-8)
        qv = qv - float(np.max(qv))
        logw = np.clip(qv / temp, -50.0, 50.0)
        w = np.exp(logw)
        s = float(w.sum())
        if s <= 0.0 or not np.isfinite(s):
            return int(rng.choice(valid_idx))

        p = w / s
        if not np.isfinite(p).all():
            return int(rng.choice(valid_idx))
        return int(rng.choice(valid_idx, p=p))

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor, mask: torch.Tensor) -> int:
        """在合法动作上取 Q 最大；并列最大时取索引最小者（稳定、可复现）。"""
        self.q_net.eval()
        q = self.q_net(obs).reshape(-1)
        valid = mask.reshape(-1)
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_idx.numel() == 0:
            raise RuntimeError("predict_action: 无合法动作（action mask 全 False）")
        qv = q[valid_idx]
        if not torch.isfinite(qv).all():
            return int(valid_idx[int(torch.randint(valid_idx.numel(), (1,)).item())].item())
        best = float(qv.max().item())
        tied = (qv >= best - 1e-6).nonzero(as_tuple=False).reshape(-1)
        # 并列最大 Q 时取 action id 最小者（与 valid_idx 升序下 argmax(qv) 一致，且可复现）
        pick = int(tied[torch.argmin(valid_idx[tied]).item()].item())
        return int(valid_idx[pick].item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "network": "SimpleGridCNN",
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "alpha": self.alpha,
                "gamma": self.gamma,
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "in_channels": self.in_channels,
                "n_actions": self.n_actions,
            },
            path,
        )

    def load(self, path: str) -> None:
        blob: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(blob["q_net"])
        self.target_net.load_state_dict(blob.get("target_net", blob["q_net"]))
