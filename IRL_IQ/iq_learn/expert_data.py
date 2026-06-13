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
from county_meta import compute_county_meta, state_name_to_id
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
    county_id: int  # location_id：唯一 (state, county) 对
    state_id: int
    state_name: str
    county_name: str
    grid_npz: str
    H: int
    W: int
    cell_km: float
    n_obs_channels: int
    county_meta: np.ndarray | None = None  # (3,) float32 socioeconomic
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


def _spatial_batch_from_numpy(
    obs: np.ndarray,
    next_obs: np.ndarray,
    actions: np.ndarray,
    done: np.ndarray,
    mask: np.ndarray,
    next_mask: np.ndarray,
    county_ids: np.ndarray,
    state_ids: np.ndarray,
    county_meta: np.ndarray,
    *,
    device: torch.device,
    is_expert: bool = True,
) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "next_obs": torch.as_tensor(next_obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "done": torch.as_tensor(done, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "next_mask": torch.as_tensor(next_mask, dtype=torch.bool, device=device),
        "county_ids": torch.as_tensor(county_ids, dtype=torch.long, device=device),
        "state_ids": torch.as_tensor(state_ids, dtype=torch.long, device=device),
        "county_meta": torch.as_tensor(county_meta, dtype=torch.float32, device=device),
        "is_expert": torch.full((int(obs.shape[0]),), bool(is_expert), dtype=torch.bool, device=device),
    }


def concat_training_batches(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: torch.cat([a[key], b[key]], dim=0) for key in a}


@dataclass
class MergedExpertDataset:
    """多县专家数据集：保留各县本地网格，采样时再 pad 到联合画布（省内存）。"""

    batches: list[CountyExpertBatch]
    global_index: np.ndarray
    actions: np.ndarray
    done: np.ndarray
    county_ids: np.ndarray
    state_ids: np.ndarray
    county_meta: np.ndarray
    canvas: JointGridCanvas
    counties: list[CountyLayout]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.global_index.shape[0])

    def _pad_indices(self, idx: np.ndarray) -> tuple[np.ndarray, ...]:
        bs = int(idx.shape[0])
        h = int(self.canvas.max_h)
        w = int(self.canvas.max_w)
        na = int(self.canvas.n_actions)
        n_ch = int(self.batches[0].obs.shape[-1])
        obs = np.empty((bs, h, w, n_ch), dtype=np.float32)
        next_obs = np.empty((bs, h, w, n_ch), dtype=np.float32)
        mask = np.empty((bs, na), dtype=bool)
        next_mask = np.empty((bs, na), dtype=bool)
        for j, gi in enumerate(idx):
            b_idx, local_i = (int(self.global_index[gi, 0]), int(self.global_index[gi, 1]))
            batch = self.batches[b_idx]
            layout = batch.layout
            obs[j] = pad_obs_hwc(batch.obs[local_i], self.canvas)
            next_obs[j] = pad_obs_hwc(batch.next_obs[local_i], self.canvas)
            mask[j] = pad_mask_flat(batch.mask_local[local_i], layout.H, layout.W, self.canvas)
            next_mask[j] = pad_mask_flat(batch.next_mask_local[local_i], layout.H, layout.W, self.canvas)
        return (
            obs,
            next_obs,
            self.actions[idx],
            self.done[idx],
            mask,
            next_mask,
            self.county_ids[idx],
            self.state_ids[idx],
            self.county_meta[idx],
        )

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs, next_obs, actions, done, mask, next_mask, county_ids, state_ids, county_meta = self._pad_indices(idx)
        return _spatial_batch_from_numpy(
            obs,
            next_obs,
            actions,
            done,
            mask,
            next_mask,
            county_ids,
            state_ids,
            county_meta,
            device=device,
            is_expert=True,
        )


def collect_county_expert_batch(
    grid_npz: str,
    county_id: int,
    *,
    state_id: int | None = None,
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

    if state_id is not None:
        sid = int(state_id)
    elif base.state_name:
        sid = state_name_to_id(base.state_name)
    else:
        raise ValueError(f"无法确定 state_id: npz 缺少 state_name ({grid_npz})")
    if not base.state_name or not base.county_name:
        raise ValueError(f"无法确定 socioeconomic county_meta: npz 缺少 state/county ({grid_npz})")
    meta_vec = compute_county_meta(base.state_name, base.county_name)
    layout = CountyLayout(
        county_id=int(county_id),
        state_id=sid,
        state_name=base.state_name,
        county_name=base.county_name,
        grid_npz=str(grid_npz),
        H=int(base.H),
        W=int(base.W),
        cell_km=float(base.cell_km),
        n_obs_channels=int(ch.n_channels),
        county_meta=meta_vec,
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

    index_rows: list[tuple[int, int]] = []
    act_parts: list[int] = []
    done_parts: list[np.ndarray] = []
    id_parts: list[int] = []
    state_id_parts: list[int] = []
    meta_parts: list[np.ndarray] = []

    for b_idx, batch in enumerate(batches):
        layout = batch.layout
        meta_vec = (
            layout.county_meta
            if layout.county_meta is not None
            else compute_county_meta(layout.state_name, layout.county_name)
        )
        n = int(batch.obs.shape[0])
        for i in range(n):
            index_rows.append((b_idx, i))
            act_parts.append(
                local_action_to_canvas(int(batch.actions_local[i]), layout.W, canvas.max_w)
            )
            done_parts.append(batch.done[i])
            id_parts.append(int(layout.county_id))
            state_id_parts.append(int(layout.state_id))
            meta_parts.append(np.asarray(meta_vec, dtype=np.float32))

    merged = MergedExpertDataset(
        batches=batches,
        global_index=np.asarray(index_rows, dtype=np.int32),
        actions=np.asarray(act_parts, dtype=np.int64).reshape(-1, 1),
        done=np.stack(done_parts),
        county_ids=np.asarray(id_parts, dtype=np.int64),
        state_ids=np.asarray(state_id_parts, dtype=np.int64),
        county_meta=np.stack(meta_parts).astype(np.float32),
        canvas=canvas,
        counties=counties,
        meta={
            "n_counties": len(counties),
            "n_transitions": len(index_rows),
            "storage_mode": "lazy_pad_on_sample",
            "max_h": canvas.max_h,
            "max_w": canvas.max_w,
            "n_actions": canvas.n_actions,
            "counties": [
                {
                    "county_id": c.county_id,
                    "state_id": c.state_id,
                    "state_name": c.state_name,
                    "county_name": c.county_name,
                    "H": c.H,
                    "W": c.W,
                    "grid_npz": c.grid_npz,
                    "county_meta": (
                        c.county_meta.tolist()
                        if c.county_meta is not None
                        else compute_county_meta(c.state_name, c.county_name).tolist()
                    ),
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
    state_ids: list[int] | None = None,
    county_ids: list[int] | None = None,
) -> MergedExpertDataset:
    """从多个 state-county npz 收集并 pool 专家数据。"""
    batches: list[CountyExpertBatch] = []
    for i, path in enumerate(grid_npz_paths):
        cid = int(county_ids[i]) if county_ids is not None else int(i)
        sid = int(state_ids[i]) if state_ids is not None else None
        batches.append(
            collect_county_expert_batch(path, cid, state_id=sid, channel_cfg=channel_cfg)
        )
    merged = merge_county_expert_batches(batches)
    for layout in merged.counties:
        layout.flat_grid_w = merged.canvas.max_w
    return merged
