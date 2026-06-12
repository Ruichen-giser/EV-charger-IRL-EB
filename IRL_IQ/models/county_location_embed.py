"""County 条件 embedding：state + meta MLP + 可选 residual（e = e_base + α·e_resid）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class CountyLocationEmbed(nn.Module):
    """
    state_id → StateEmbed(51×d)
    county_meta(3) → MLP(3→32→d)  # 受教育程度、人口、家庭收入
    e_base = LayerNorm(Dropout(e_state + e_meta))
    e = e_base + α * ResidualEmbed(county_id)   （默认 α=0）
    """

    def __init__(
        self,
        *,
        n_states: int = 51,
        embed_dim: int = 16,
        meta_dim: int = 3,
        meta_hidden: int = 32,
        n_residual: int = 1149,
        residual_alpha: float = 0.0,
        embed_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_states = int(n_states)
        self.meta_dim = int(meta_dim)
        self.n_residual = int(n_residual)

        self.state_embed = nn.Embedding(self.n_states, self.embed_dim)
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_dim, int(meta_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(meta_hidden), self.embed_dim),
        )
        self.residual_embed = nn.Embedding(self.n_residual, self.embed_dim)
        self.embed_dropout = nn.Dropout(float(embed_dropout))
        self.fuse_norm = nn.LayerNorm(self.embed_dim)

        self.register_buffer(
            "residual_alpha",
            torch.tensor(float(residual_alpha), dtype=torch.float32),
        )

    def forward(
        self,
        state_ids: torch.Tensor,
        county_meta: torch.Tensor,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            state_ids: (B,) long
            county_meta: (B, meta_dim) float
            county_ids: (B,) long，residual 查表（可选）
        Returns:
            (B, embed_dim)
        """
        e_state = self.state_embed(state_ids.long().view(-1))
        e_meta = self.meta_mlp(county_meta.float())
        e_base = self.fuse_norm(self.embed_dropout(e_state + e_meta))

        alpha = float(self.residual_alpha.item())
        if county_ids is not None and alpha != 0.0:
            e_resid = self.residual_embed(county_ids.long().view(-1))
            return e_base + alpha * e_resid

        if county_ids is not None:
            # 保留计算图结构；α=0 时不加 residual
            e_resid = self.residual_embed(county_ids.long().view(-1))
            return e_base + self.residual_alpha * e_resid

        return e_base
