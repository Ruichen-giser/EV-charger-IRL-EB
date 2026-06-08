"""多县专家 (s,a,s') 收集与合并（pool）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from envs import (
    ChargingDeploymentEnv,
    MultiChannelGridObservationWrapper,
    action_mask_fn,
    unwrap_charging_env,
)
import mdp_config
from iq_learn.grid_align import (
    JointGridCanvas,
    build_canvas,
    local_action_to_canvas,
    pad_mask_flat,
    pad_obs_hwc,
)
from obs_channels import DEFAULT_OBS_CHANNELS, ObsChannelConfig


def first_visit_expert_actions(expert_actions: np.ndarray, grid_w: int) -> np.ndarray:
    w = int(grid_w)
    seen: set[tuple[int, int]] = set()
    out: list[int] = []
    for a in np.asarray(expert_actions, dtype=np.int64).reshape(-1):
        gx, gy = int(a) % w, int(a) // w
        if (gx, gy) in seen:
            continue
        seen.add((gx, gy))
        out.append(int(a))
    return np.asarray(out, dtype=np.int64)


def expert_action_sequence(expert_actions: np.ndarray, grid_w: int) -> np.ndarray:
    raw = np.asarray(expert_actions, dtype=np.int64).reshape(-1)
    if mdp_config.ONE_STATION_PER_CELL:
        return first_visit_expert_actions(raw, grid_w)
    return raw


@dataclass
class CountyLayout:
    county_id: int
    county_name: str
    grid_npz: str
    H: int
    W: int
    cell_km: float
    n_obs_channels: int
    flat_grid_w: int | None = None  # 联合训练画布宽；None 表示用本地 W

    @property
    def n_actions(self) -> int:
        return int(self.H) * int(self.W)

    def _flat_w(self) -> int:
        return int(self.flat_grid_w if self.flat_grid_w is not None else self.W)

    def action_to_xy(self, action: int, *, grid_w: int | None = None) -> tuple[int, int]:
        w = int(grid_w if grid_w is not None else self._flat_w())
        a = int(action)
        return a % w, a // w

    def grid_center_distance_km(self, action_a: int, action_b: int, *, grid_w: int | None = None) -> float:
        w = int(grid_w if grid_w is not None else self._flat_w())
        gx1, gy1 = self.action_to_xy(action_a, grid_w=w)
        gx2, gy2 = self.action_to_xy(action_b, grid_w=w)
        x1 = (gx1 + 0.5) * self.cell_km
        y1 = (gy1 + 0.5) * self.cell_km
        x2 = (gx2 + 0.5) * self.cell_km
        y2 = (gy2 + 0.5) * self.cell_km
        return float(np.hypot(x1 - x2, y1 - y2))


@dataclass
class CountyExpertBatch:
    """单县本地网格上的专家转移（未 padding）。"""

    obs: np.ndarray
    next_obs: np.ndarray
    actions_local: np.ndarray
    done: np.ndarray
    mask_local: np.ndarray
    next_mask_local: np.ndarray
    layout: CountyLayout


@dataclass
class MergedExpertDataset:
    """多县 pool 后、画布对齐的专家数据集。"""

    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    done: np.ndarray
    mask: np.ndarray
    next_mask: np.ndarray
    county_ids: np.ndarray
    canvas: JointGridCanvas
    counties: list[CountyLayout]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        batch: dict[str, torch.Tensor] = {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            "next_obs": torch.as_tensor(next_obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.long, device=device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
            "mask": torch.as_tensor(self.mask[idx], dtype=torch.bool, device=device),
            "next_mask": torch.as_tensor(self.next_mask[idx], dtype=torch.bool, device=device),
            "county_ids": torch.as_tensor(self.county_ids[idx], dtype=torch.long, device=device),
            "is_expert": torch.ones(len(idx), dtype=torch.bool, device=device),
        }
        return batch


def collect_county_expert_batch(
    grid_npz: str,
    county_id: int,
    *,
    channel_cfg: ObsChannelConfig | None = None,
) -> CountyExpertBatch:
    ch = channel_cfg or DEFAULT_OBS_CHANNELS
    env = MultiChannelGridObservationWrapper(ChargingDeploymentEnv(grid_npz), ch)
    base = unwrap_charging_env(env)
    if base.expert_actions is None or len(base.expert_actions) == 0:
        env.close()
        raise ValueError(f"npz 中无 expert_actions: {grid_npz}")

    obs_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    done_list: list[float] = []
    mask_list: list[np.ndarray] = []
    next_mask_list: list[np.ndarray] = []

    expert_seq = expert_action_sequence(base.expert_actions, base.W)
    obs, _ = env.reset()
    for a in expert_seq:
        mask = action_mask_fn(env)
        a_int = int(a)
        if not bool(mask[a_int]):
            obs, _, terminated, truncated, _ = env.step(a_int)
            if terminated or truncated:
                break
            continue

        obs_list.append(np.asarray(obs, dtype=np.float32))
        next_obs, _, terminated, truncated, info = env.step(a_int)
        if info.get("invalid_action"):
            obs = next_obs
            continue
        next_obs_list.append(np.asarray(next_obs, dtype=np.float32))
        act_list.append(a_int)
        done_list.append(float(terminated or truncated))
        mask_list.append(mask.astype(bool))
        next_mask_list.append(action_mask_fn(env))
        obs = next_obs
        if terminated or truncated:
            break

    env.close()
    if not obs_list:
        raise ValueError(f"无有效专家转移: {grid_npz}")

    layout = CountyLayout(
        county_id=int(county_id),
        county_name=base.county_name,
        grid_npz=str(grid_npz),
        H=int(base.H),
        W=int(base.W),
        cell_km=float(base.cell_km),
        n_obs_channels=int(ch.n_channels),
    )
    return CountyExpertBatch(
        obs=np.stack(obs_list),
        next_obs=np.stack(next_obs_list),
        actions_local=np.asarray(act_list, dtype=np.int64),
        done=np.asarray(done_list, dtype=np.float32).reshape(-1, 1),
        mask_local=np.stack(mask_list),
        next_mask_local=np.stack(next_mask_list),
        layout=layout,
    )


def merge_county_expert_batches(
    batches: list[CountyExpertBatch],
    *,
    canvas: JointGridCanvas | None = None,
) -> MergedExpertDataset:
    """将各县专家转移 pool 到统一画布。"""
    if not batches:
        raise ValueError("merge_county_expert_batches: 空列表")

    counties = [b.layout for b in batches]
    if canvas is None:
        canvas = build_canvas([c.H for c in counties], [c.W for c in counties])

    obs_parts: list[np.ndarray] = []
    next_parts: list[np.ndarray] = []
    act_parts: list[int] = []
    done_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    next_mask_parts: list[np.ndarray] = []
    id_parts: list[int] = []

    for batch in batches:
        layout = batch.layout
        n = int(batch.obs.shape[0])
        for i in range(n):
            obs_parts.append(pad_obs_hwc(batch.obs[i], canvas))
            next_parts.append(pad_obs_hwc(batch.next_obs[i], canvas))
            act_parts.append(
                local_action_to_canvas(int(batch.actions_local[i]), layout.W, canvas.max_w)
            )
            done_parts.append(batch.done[i])
            mask_parts.append(pad_mask_flat(batch.mask_local[i], layout.H, layout.W, canvas))
            next_mask_parts.append(pad_mask_flat(batch.next_mask_local[i], layout.H, layout.W, canvas))
            id_parts.append(int(layout.county_id))

    merged = MergedExpertDataset(
        obs=np.stack(obs_parts),
        next_obs=np.stack(next_parts),
        actions=np.asarray(act_parts, dtype=np.int64).reshape(-1, 1),
        done=np.stack(done_parts),
        mask=np.stack(mask_parts),
        next_mask=np.stack(next_mask_parts),
        county_ids=np.asarray(id_parts, dtype=np.int64),
        canvas=canvas,
        counties=counties,
        meta={
            "n_counties": len(counties),
            "n_transitions": len(obs_parts),
            "max_h": canvas.max_h,
            "max_w": canvas.max_w,
            "n_actions": canvas.n_actions,
            "counties": [
                {
                    "county_id": c.county_id,
                    "county_name": c.county_name,
                    "H": c.H,
                    "W": c.W,
                    "grid_npz": c.grid_npz,
                }
                for c in counties
            ],
        },
    )
    return merged


def build_merged_expert_dataset(
    grid_npz_paths: list[str],
    *,
    channel_cfg: ObsChannelConfig | None = None,
) -> MergedExpertDataset:
    """从多个县 npz 收集并 pool 专家数据。"""
    batches: list[CountyExpertBatch] = []
    for cid, path in enumerate(grid_npz_paths):
        batches.append(collect_county_expert_batch(path, cid, channel_cfg=channel_cfg))
    merged = merge_county_expert_batches(batches)
    for layout in merged.counties:
        layout.flat_grid_w = merged.canvas.max_w
    return merged
