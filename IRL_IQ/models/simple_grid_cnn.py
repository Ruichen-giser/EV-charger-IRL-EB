"""SimpleGridCNN：3×Conv3×3 + Dropout + 1×1 head（无残差）。"""
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
    def __init__(
        self,
        in_channels: int = 8,
        grid_h: int = 47,
        grid_w: int = 26,
        action_dim: int = 1222,
        *,
        n_conv_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.action_dim = int(action_dim)
        self.n_conv_layers = int(n_conv_layers)
        self.dropout = float(dropout)

        widths = [32, 64, 64][: self.n_conv_layers]
        layers: list[nn.Module] = []
        c_in = int(in_channels)
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
    ) -> "SimpleGridCNN":
        n_layers, dropout = infer_simple_grid_cnn_arch(state_dict)
        return cls(
            in_channels,
            grid_h,
            grid_w,
            action_dim,
            n_conv_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feat = self.encoder(x)
        logits = self.head(feat).squeeze(1)
        return logits.reshape(b, -1)
