"""离散 Soft Q：SimpleGridCNN + CountyLocationEmbed + 掩膜 soft V(s)。"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cnn_config import (
    CNN_DROPOUT,
    CNN_N_CONV_LAYERS,
    COUNTY_EMBED_DIM,
    COUNTY_META_DIM,
    EMBED_DROPOUT,
    META_MLP_HIDDEN,
    N_MAX_COUNTY_RESIDUAL,
    N_US_STATES,
    RESIDUAL_ALPHA,
)
from models.simple_grid_cnn import SimpleGridCNN
from iq_learn.iq_loss import IQ_LOSS_MODES, iq_learn_loss
from models.soft_q_ops import masked_soft_value, soft_policy_entropy


class SimpleGridCNNQ(nn.Module):
    """SimpleGridCNN wrapper，统一 forward / soft_value 接口。"""

    def __init__(
        self,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        action_dim: int,
        *,
        n_conv_layers: int = CNN_N_CONV_LAYERS,
        dropout: float = CNN_DROPOUT,
        use_location_embed: bool = True,
        embed_dim: int = COUNTY_EMBED_DIM,
        meta_dim: int = COUNTY_META_DIM,
        meta_hidden: int = META_MLP_HIDDEN,
        n_states: int = N_US_STATES,
        n_residual: int = N_MAX_COUNTY_RESIDUAL,
        residual_alpha: float = RESIDUAL_ALPHA,
        embed_dropout: float = EMBED_DROPOUT,
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
            use_location_embed=bool(use_location_embed),
            embed_dim=int(embed_dim),
            meta_dim=int(meta_dim),
            meta_hidden=int(meta_hidden),
            n_states=int(n_states),
            n_residual=int(n_residual),
            residual_alpha=float(residual_alpha),
            embed_dropout=float(embed_dropout),
        )

    def forward(
        self,
        obs: torch.Tensor,
        state_ids: torch.Tensor | None = None,
        county_meta: torch.Tensor | None = None,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(obs, state_ids, county_meta, county_ids)

    def q_values(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state_ids: torch.Tensor | None = None,
        county_meta: torch.Tensor | None = None,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(obs, state_ids, county_meta, county_ids).gather(1, actions.long())

    def soft_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        state_ids: torch.Tensor | None = None,
        county_meta: torch.Tensor | None = None,
        county_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_soft_value(
            self.forward(obs, state_ids, county_meta, county_ids), mask, alpha
        )


class DiscreteSoftQAgent:
    """IQ-Learn Soft Q agent（共享 Q 网络，多县联合训练）。"""

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
        iq_loss_mode: str = "online",
        target_update_interval: int = 2,
        grid_h: int = 0,
        grid_w: int = 0,
        in_channels: int = 0,
        n_counties: int = 0,
        embed_dim: int = COUNTY_EMBED_DIM,
        meta_dim: int = COUNTY_META_DIM,
        meta_hidden: int = META_MLP_HIDDEN,
        n_states: int = N_US_STATES,
        n_residual: int = N_MAX_COUNTY_RESIDUAL,
        residual_alpha: float = RESIDUAL_ALPHA,
        embed_dropout: float = EMBED_DROPOUT,
        n_conv_layers: int = CNN_N_CONV_LAYERS,
        dropout: float = CNN_DROPOUT,
        county_names: list[str] | None = None,
        state_names: list[str] | None = None,
        location_labels: list[str] | None = None,
        **_: Any,
    ) -> None:
        del obs_dim
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.alpha_reg = float(alpha_reg)
        self.use_chi = bool(use_chi)
        mode = str(iq_loss_mode).strip().lower()
        if mode not in IQ_LOSS_MODES:
            raise ValueError(f"iq_loss_mode 须为 {IQ_LOSS_MODES}，当前: {iq_loss_mode!r}")
        self.iq_loss_mode = mode
        self.target_update_interval = int(target_update_interval)
        self._step = 0
        self.network_type = "SimpleGridCNN+CountyLocationEmbed"
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.in_channels = int(in_channels)
        self.n_actions = int(n_actions)
        self.n_counties = int(n_counties)
        self.embed_dim = int(embed_dim)
        self.meta_dim = int(meta_dim)
        self.n_states = int(n_states)
        self.n_residual = int(n_residual)
        self.residual_alpha = float(residual_alpha)
        self.county_names = list(county_names or [])
        self.state_names = list(state_names or [])
        self.location_labels = list(location_labels or self.county_names)

        self.q_net: nn.Module = SimpleGridCNNQ(
            in_channels=self.in_channels,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            action_dim=self.n_actions,
            n_conv_layers=int(n_conv_layers),
            dropout=float(dropout),
            use_location_embed=True,
            embed_dim=self.embed_dim,
            meta_dim=self.meta_dim,
            meta_hidden=int(meta_hidden),
            n_states=self.n_states,
            n_residual=self.n_residual,
            residual_alpha=self.residual_alpha,
            embed_dropout=float(embed_dropout),
        ).to(self.device)

        self.target_net = copy.deepcopy(self.q_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))
        self.use_amp = self.device.type == "cuda"
        self._scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.q_net.train()

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        done = batch["done"]
        mask = batch["mask"]
        next_mask = batch["next_mask"]
        is_expert = batch.get("is_expert")
        state_ids = batch.get("state_ids")
        county_meta = batch.get("county_meta")
        county_ids = batch.get("county_ids")

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            q_all = self.q_net(obs, state_ids, county_meta, county_ids)
            q_sa = q_all.gather(1, actions.long())
            v_s = masked_soft_value(q_all, mask, self.alpha)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                next_v = self.target_net.soft_value(
                    next_obs, next_mask, self.alpha, state_ids, county_meta, county_ids
                )
            entropy = float(soft_policy_entropy(q_all.float(), mask, self.alpha).detach().cpu())
            q_stats = q_all.detach().float()
            if is_expert is not None:
                m_all = is_expert.view(-1).bool()
                q_valid = q_stats[m_all][mask[m_all]] if bool(m_all.any()) else q_stats.new_zeros(0)
            else:
                q_valid = q_stats[mask]

        if self.iq_loss_mode == "online":
            m = is_expert.view(-1).bool() if is_expert is not None else None
            if m is None or not bool(m.any()):
                return {
                    "loss": 0.0,
                    "skipped_step": 1.0,
                    "Q_mean": 0.0,
                    "Q_std": 0.0,
                    "Q_max": 0.0,
                    "policy_entropy": entropy,
                }

        try:
            loss, metrics = iq_learn_loss(
                q_sa=q_sa,
                v_s=v_s,
                next_v=next_v,
                done=done,
                is_expert=is_expert,
                gamma=self.gamma,
                alpha_reg=self.alpha_reg,
                use_chi=self.use_chi,
                loss_mode=self.iq_loss_mode,
            )
        except ValueError:
            return {
                "loss": 0.0,
                "skipped_step": 1.0,
                "Q_mean": 0.0,
                "Q_std": 0.0,
                "Q_max": 0.0,
                "policy_entropy": entropy,
            }

        self.optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            if self.use_amp:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
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
                "Q_mean": float(q_valid.mean().cpu()) if q_valid.numel() else 0.0,
                "Q_std": float(q_valid.std().cpu()) if q_valid.numel() > 1 else 0.0,
                "Q_max": float(q_valid.max().cpu()) if q_valid.numel() else 0.0,
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
        state_ids: torch.Tensor | None = None,
        county_meta: torch.Tensor | None = None,
        county_ids: torch.Tensor | None = None,
    ) -> int:
        self.q_net.eval()
        q = self.q_net(obs, state_ids, county_meta, county_ids).detach().reshape(-1).cpu().numpy()
        valid = mask.reshape(-1).cpu().numpy().astype(bool)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            raise RuntimeError("sample_action_soft: 无合法动作")

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
    def predict_action(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        state_ids: torch.Tensor | None = None,
        county_meta: torch.Tensor | None = None,
        county_ids: torch.Tensor | None = None,
    ) -> int:
        self.q_net.eval()
        q = self.q_net(obs, state_ids, county_meta, county_ids)
        neg_inf = torch.finfo(q.dtype).min
        return int(q.masked_fill(~mask, neg_inf).argmax(dim=1).item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "network": self.network_type,
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "alpha": self.alpha,
                "gamma": self.gamma,
                "iq_loss_mode": self.iq_loss_mode,
                "use_chi": self.use_chi,
                "alpha_reg": self.alpha_reg,
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
                "in_channels": self.in_channels,
                "n_actions": self.n_actions,
                "n_counties": self.n_counties,
                "embed_dim": self.embed_dim,
                "meta_dim": self.meta_dim,
                "n_states": self.n_states,
                "n_residual": self.n_residual,
                "residual_alpha": self.residual_alpha,
                "county_names": self.county_names,
                "state_names": self.state_names,
                "location_labels": self.location_labels,
            },
            path,
        )

    def load(self, path: str) -> None:
        blob: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(blob["q_net"])
        self.target_net.load_state_dict(blob.get("target_net", blob["q_net"]))
        if "iq_loss_mode" in blob:
            self.iq_loss_mode = str(blob["iq_loss_mode"])
        if "county_names" in blob:
            self.county_names = list(blob["county_names"])
        if "state_names" in blob:
            self.state_names = list(blob["state_names"])
        if "location_labels" in blob:
            self.location_labels = list(blob["location_labels"])
        if "n_states" in blob:
            self.n_states = int(blob["n_states"])
        if "embed_dim" in blob:
            self.embed_dim = int(blob["embed_dim"])
        if "residual_alpha" in blob:
            self.residual_alpha = float(blob["residual_alpha"])
            loc = getattr(self.q_net.net, "location_embed", None)
            if loc is not None and hasattr(loc, "residual_alpha"):
                loc.residual_alpha.fill_(self.residual_alpha)
