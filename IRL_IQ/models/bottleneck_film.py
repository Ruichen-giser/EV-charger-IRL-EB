"""Bottleneck FiLM：state / meta 双分支生成通道级 γ、β，调制 encoder 输出。"""
from __future__ import annotations

import torch
import torch.nn as nn


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


class FiLMBranch(nn.Module):
    """单路条件向量 → (γ, β)，输出层零初始化使初始调制近似恒等。"""

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        *,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.to_gamma = nn.Linear(int(hidden_dim), int(out_channels))
        self.to_beta = nn.Linear(int(hidden_dim), int(out_channels))
        _zero_init_linear(self.to_gamma)
        _zero_init_linear(self.to_beta)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(e)
        return self.to_gamma(h), self.to_beta(h)


class BottleneckFiLM(nn.Module):
    """
    state 与 meta 各走一支 FiLM，再相加：
      γ = γ_state(e_state) + γ_meta(e_meta)
      β = β_state(e_state) + β_meta(e_meta)
      feat' = feat ⊙ (1 + γ) + β
    """

    def __init__(
        self,
        embed_dim: int,
        feat_channels: int,
        *,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feat_channels = int(feat_channels)
        self.state_branch = FiLMBranch(
            self.embed_dim,
            self.feat_channels,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.meta_branch = FiLMBranch(
            self.embed_dim,
            self.feat_channels,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )

    def forward(
        self,
        e_state: torch.Tensor,
        e_meta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma_s, beta_s = self.state_branch(e_state)
        gamma_m, beta_m = self.meta_branch(e_meta)
        gamma = gamma_s + gamma_m
        beta = beta_s + beta_m
        return gamma, beta, gamma_s, beta_m

    @staticmethod
    def modulate(
        feat: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        b, c, _, _ = feat.shape
        g = gamma.view(b, c, 1, 1)
        b_ = beta.view(b, c, 1, 1)
        return feat * (1.0 + g) + b_
