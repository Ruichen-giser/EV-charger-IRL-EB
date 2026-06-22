"""SimpleGridCNN Q 网络分组学习率：encoder / head / location_embed / film。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

GROUP_ENCODER = "encoder"
GROUP_HEAD = "head"
GROUP_EMBED = "location_embed"
GROUP_FILM = "film"


@dataclass(frozen=True)
class LRScheduleConfig:
    base_lr: float
    lr_head_mult: float = 1.5
    lr_embed_mult: float = 5.0
    lr_film_mult: float = 5.0
    film_warmup_steps: int = 50_000
    lr_decay_step: int = 0
    lr_decay_mult: float = 0.1

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _param_group_name(param_name: str) -> str:
    if param_name.startswith("net.encoder."):
        return GROUP_ENCODER
    if param_name.startswith("net.head."):
        return GROUP_HEAD
    if param_name.startswith("net.location_embed."):
        return GROUP_EMBED
    if param_name.startswith("net.film."):
        return GROUP_FILM
    return GROUP_ENCODER


def build_q_net_param_groups(q_net: nn.Module, cfg: LRScheduleConfig) -> list[dict[str, Any]]:
    buckets: dict[str, list[nn.Parameter]] = {
        GROUP_ENCODER: [],
        GROUP_HEAD: [],
        GROUP_EMBED: [],
        GROUP_FILM: [],
    }
    for name, param in q_net.named_parameters():
        if not param.requires_grad:
            continue
        buckets[_param_group_name(name)].append(param)

    group_specs: list[tuple[str, float]] = [
        (GROUP_ENCODER, 1.0),
        (GROUP_HEAD, float(cfg.lr_head_mult)),
        (GROUP_EMBED, float(cfg.lr_embed_mult)),
        (GROUP_FILM, float(cfg.lr_film_mult)),
    ]
    param_groups: list[dict[str, Any]] = []
    for group_name, mult in group_specs:
        params = buckets[group_name]
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "lr": float(cfg.base_lr) * mult,
                "name": group_name,
            }
        )
    if not param_groups:
        param_groups.append({"params": list(q_net.parameters()), "lr": float(cfg.base_lr), "name": GROUP_ENCODER})
    return param_groups


def learning_rates_for_step(cfg: LRScheduleConfig, step: int) -> dict[str, float]:
    step_i = max(1, int(step))
    decay = (
        float(cfg.lr_decay_mult)
        if int(cfg.lr_decay_step) > 0 and step_i >= int(cfg.lr_decay_step)
        else 1.0
    )
    if int(cfg.film_warmup_steps) > 0:
        warmup = min(1.0, step_i / float(cfg.film_warmup_steps))
    else:
        warmup = 1.0

    base = float(cfg.base_lr) * decay
    return {
        GROUP_ENCODER: base,
        GROUP_HEAD: base * float(cfg.lr_head_mult),
        GROUP_EMBED: base * float(cfg.lr_embed_mult) * warmup,
        GROUP_FILM: base * float(cfg.lr_film_mult) * warmup,
    }


def apply_learning_rates(optimizer: torch.optim.Optimizer, lrs: dict[str, float]) -> None:
    for param_group in optimizer.param_groups:
        name = param_group.get("name")
        if name in lrs:
            param_group["lr"] = float(lrs[name])


def current_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    out: dict[str, float] = {}
    for param_group in optimizer.param_groups:
        name = param_group.get("name")
        if name is not None:
            out[str(name)] = float(param_group["lr"])
    return out


@torch.no_grad()
def measure_film_modulation(
    q_net: nn.Module,
    *,
    state_ids: torch.Tensor,
    county_meta: torch.Tensor,
    county_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    net = getattr(q_net, "net", q_net)
    film = getattr(net, "film", None)
    location_embed = getattr(net, "location_embed", None)
    if film is None or location_embed is None:
        return {}

    e_state, e_meta, _, _ = location_embed.forward_components(state_ids, county_meta, county_ids)
    gamma, beta, _, _ = film(e_state, e_meta)
    return {
        "film_gamma_abs_mean": float(gamma.abs().mean().detach().cpu()),
        "film_beta_abs_mean": float(beta.abs().mean().detach().cpu()),
    }
