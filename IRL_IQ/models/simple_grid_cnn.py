"""SimpleGridCNN：3×Conv3×3 + 1×1 spatial Q head；县别条件经 Bottleneck FiLM 或输入 concat 注入。"""
from __future__ import annotations

import torch
import torch.nn as nn

from cnn_config import (
    COUNTY_EMBED_DIM,
    COUNTY_META_DIM,
    EMBED_DROPOUT,
    EMBED_MODE,
    FILM_HIDDEN,
    META_MLP_HIDDEN,
    N_MAX_COUNTY_RESIDUAL,
    N_US_STATES,
    RESIDUAL_ALPHA,
)
from models.bottleneck_film import BottleneckFiLM
from models.county_location_embed import CountyLocationEmbed


def _encoder_conv_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    idxs: list[int] = []
    for k, v in state_dict.items():
        if k.startswith("encoder.") and k.endswith(".weight") and v.ndim == 4:
            idxs.append(int(k.split(".")[1]))
    return sorted(idxs)


def infer_simple_grid_cnn_arch(state_dict: dict[str, torch.Tensor]) -> tuple[int, float]:
    conv_idxs = _encoder_conv_indices(state_dict)
    n = len(conv_idxs)
    gap = conv_idxs[1] - conv_idxs[0] if n >= 2 else 3
    dropout = 0.1 if gap >= 4 else 0.0
    return max(n, 2), dropout


def _infer_embed_mode(state_dict: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("film.") for k in state_dict):
        return "bottleneck_film"
    return "concat"


class SimpleGridCNN(nn.Module):
    """
    输入 (B, C_obs, H, W)。
    embed_mode=concat：CountyLocationEmbed 平面拼接到输入。
    embed_mode=bottleneck_film：encoder 只吃 obs，末端用 state/meta FiLM 调制特征。
    输出 (B, H*W) Q 值。
    """

    def __init__(
        self,
        in_channels: int = 8,
        grid_h: int = 47,
        grid_w: int = 26,
        action_dim: int = 1222,
        *,
        n_conv_layers: int = 3,
        dropout: float = 0.1,
        use_location_embed: bool = False,
        embed_mode: str = EMBED_MODE,
        n_states: int = N_US_STATES,
        embed_dim: int = COUNTY_EMBED_DIM,
        meta_dim: int = COUNTY_META_DIM,
        meta_hidden: int = META_MLP_HIDDEN,
        n_residual: int = N_MAX_COUNTY_RESIDUAL,
        residual_alpha: float = RESIDUAL_ALPHA,
        embed_dropout: float = EMBED_DROPOUT,
        film_hidden: int = FILM_HIDDEN,
    ) -> None:
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.action_dim = int(action_dim)
        self.n_conv_layers = int(n_conv_layers)
        self.dropout = float(dropout)
        self.base_in_channels = int(in_channels)
        self.use_location_embed = bool(use_location_embed)
        self.embed_dim = int(embed_dim)
        self.embed_mode = str(embed_mode).strip().lower()
        if self.embed_mode not in {"concat", "bottleneck_film"}:
            raise ValueError(f"未知 embed_mode={embed_mode!r}")

        if self.use_location_embed:
            self.location_embed = CountyLocationEmbed(
                n_states=int(n_states),
                embed_dim=self.embed_dim,
                meta_dim=int(meta_dim),
                meta_hidden=int(meta_hidden),
                n_residual=int(n_residual),
                residual_alpha=float(residual_alpha),
                embed_dropout=float(embed_dropout),
            )
        else:
            self.location_embed = None

        conv_in = self.base_in_channels
        if self.use_location_embed and self.embed_mode == "concat":
            conv_in = self.base_in_channels + self.embed_dim

        widths = [32, 64, 64][: self.n_conv_layers]
        layers: list[nn.Module] = []
        c_in = conv_in
        for c_out in widths:
            layers.extend(
                [
                    nn.Conv2d(c_in, int(c_out), kernel_size=3, padding=1),
                    nn.BatchNorm2d(int(c_out)),
                    nn.ReLU(inplace=True),
                ]
            )
            if self.dropout > 0:
                layers.append(nn.Dropout2d(self.dropout))
            c_in = int(c_out)

        self.encoder = nn.Sequential(*layers)
        self.feat_channels = int(c_in)
        self.head = nn.Conv2d(self.feat_channels, 1, kernel_size=1)

        if self.use_location_embed and self.embed_mode == "bottleneck_film":
            self.film = BottleneckFiLM(
                self.embed_dim,
                self.feat_channels,
                hidden_dim=int(film_hidden),
                dropout=float(embed_dropout),
            )
        else:
            self.film = None

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        action_dim: int,
        use_location_embed: bool = True,
        embed_mode: str | None = None,
        **kwargs,
    ) -> "SimpleGridCNN":
        n_layers, dropout = infer_simple_grid_cnn_arch(state_dict)
        mode = str(embed_mode or _infer_embed_mode(state_dict))
        return cls(
            in_channels,
            grid_h,
            grid_w,
            action_dim,
            n_conv_layers=n_layers,
            dropout=dropout,
            use_location_embed=use_location_embed,
            embed_mode=mode,
            **kwargs,
        )

    def _concat_location_planes(
        self,
        obs: torch.Tensor,
        state_ids: torch.Tensor | None,
        county_meta: torch.Tensor | None,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.location_embed is None:
            return obs
        if state_ids is None or county_meta is None:
            raise ValueError("location embedding 已启用，forward 需要 state_ids 与 county_meta")
        b, _, h, w = obs.shape
        e = self.location_embed(state_ids, county_meta, county_ids)
        planes = e.view(b, self.embed_dim, 1, 1).expand(b, self.embed_dim, h, w)
        return torch.cat([obs, planes], dim=1)

    def _apply_bottleneck_film(
        self,
        feat: torch.Tensor,
        state_ids: torch.Tensor,
        county_meta: torch.Tensor,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.location_embed is None or self.film is None:
            return feat
        e_state, e_meta, _, _ = self.location_embed.forward_components(
            state_ids, county_meta, county_ids
        )
        gamma, beta, _, _ = self.film(e_state, e_meta)
        return self.film.modulate(feat, gamma, beta)

    def forward(
        self,
        obs: torch.Tensor,
        state_ids: torch.Tensor | None = None,
        county_meta: torch.Tensor | None = None,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_location_embed and self.embed_mode == "concat":
            x = self._concat_location_planes(obs, state_ids, county_meta, county_ids)
        else:
            x = obs

        b = x.shape[0]
        feat = self.encoder(x)

        if self.use_location_embed and self.embed_mode == "bottleneck_film":
            if state_ids is None or county_meta is None:
                raise ValueError("bottleneck_film 需要 state_ids 与 county_meta")
            feat = self._apply_bottleneck_film(feat, state_ids, county_meta, county_ids)

        logits = self.head(feat).squeeze(1)
        return logits.reshape(b, -1)
