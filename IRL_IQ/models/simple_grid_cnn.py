"""SimpleGridCNN：3×Conv3×3 + 1×1 spatial Q head；可选 county embedding（输入层 concat）。"""
from __future__ import annotations

import torch
import torch.nn as nn


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


class SimpleGridCNN(nn.Module):
    """
    输入 (B, C_obs, H, W)；若启用 embedding，内部 concat (B, C_obs+D, H, W) 再进 encoder。
    输出 (B, H*W) Q 值（1×1 conv spatial head，无 GAP）。
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
        n_counties: int = 0,
        county_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.action_dim = int(action_dim)
        self.n_conv_layers = int(n_conv_layers)
        self.dropout = float(dropout)
        self.base_in_channels = int(in_channels)
        self.n_counties = int(n_counties)
        self.county_embed_dim = int(county_embed_dim)

        if self.n_counties > 0 and self.county_embed_dim > 0:
            self.county_embed = nn.Embedding(self.n_counties, self.county_embed_dim)
            conv_in = self.base_in_channels + self.county_embed_dim
        else:
            self.county_embed = None
            conv_in = self.base_in_channels

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
        self.head = nn.Conv2d(c_in, 1, kernel_size=1)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        action_dim: int,
        n_counties: int = 0,
        county_embed_dim: int = 0,
    ) -> "SimpleGridCNN":
        n_layers, dropout = infer_simple_grid_cnn_arch(state_dict)
        return cls(
            in_channels,
            grid_h,
            grid_w,
            action_dim,
            n_conv_layers=n_layers,
            dropout=dropout,
            n_counties=n_counties,
            county_embed_dim=county_embed_dim,
        )

    def _concat_county_planes(
        self,
        obs: torch.Tensor,
        county_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.county_embed is None:
            return obs
        if county_ids is None:
            raise ValueError("county embedding 已启用，forward 需要 county_ids")
        b, _, h, w = obs.shape
        emb = self.county_embed(county_ids.long().view(-1))
        planes = emb.view(b, self.county_embed_dim, 1, 1).expand(b, self.county_embed_dim, h, w)
        return torch.cat([obs, planes], dim=1)

    def forward(self, obs: torch.Tensor, county_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self._concat_county_planes(obs, county_ids)
        b = x.shape[0]
        feat = self.encoder(x)
        logits = self.head(feat).squeeze(1)
        return logits.reshape(b, -1)
