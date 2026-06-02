"""单县专家 (s,a,s') 收集，供 IQ-Learn / 评估使用。"""
from dataclasses import dataclass

import numpy as np
import torch

from envs import (
    ChargingDeploymentEnv,
    MultiChannelGridObservationWrapper,
    action_mask_fn,
    unwrap_charging_env,
)
import mdp_config
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
    county_name: str
    grid_npz: str
    H: int
    W: int
    cell_km: float
    n_actions: int
    n_obs_channels: int

    def action_to_xy(self, action: int, *, grid_w: int | None = None) -> tuple[int, int]:
        w = int(grid_w if grid_w is not None else self.W)
        a = int(action)
        return a % w, a // w

    def grid_center_distance_km(self, action_a: int, action_b: int, *, grid_w: int | None = None) -> float:
        w = int(grid_w if grid_w is not None else self.W)
        gx1, gy1 = self.action_to_xy(action_a, grid_w=w)
        gx2, gy2 = self.action_to_xy(action_b, grid_w=w)
        x1 = (gx1 + 0.5) * self.cell_km
        y1 = (gy1 + 0.5) * self.cell_km
        x2 = (gx2 + 0.5) * self.cell_km
        y2 = (gy2 + 0.5) * self.cell_km
        return float(np.hypot(x1 - x2, y1 - y2))


@dataclass
class ExpertTransitionBatch:
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    done: np.ndarray
    mask: np.ndarray
    next_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = rng.integers(0, n, size=min(int(batch_size), n))
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            "next_obs": torch.as_tensor(next_obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.long, device=device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
            "mask": torch.as_tensor(self.mask[idx], dtype=torch.bool, device=device),
            "next_mask": torch.as_tensor(self.next_mask[idx], dtype=torch.bool, device=device),
        }


def collect_expert_transitions(
    grid_npz: str,
    *,
    channel_cfg: ObsChannelConfig | None = None,
) -> tuple[ExpertTransitionBatch, CountyLayout]:
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
        act_list.append(int(a))
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
        county_name=base.county_name,
        grid_npz=str(grid_npz),
        H=int(base.H),
        W=int(base.W),
        cell_km=float(base.cell_km),
        n_actions=int(base.H * base.W),
        n_obs_channels=int(ch.n_channels),
    )
    batch = ExpertTransitionBatch(
        obs=np.stack(obs_list),
        next_obs=np.stack(next_obs_list),
        actions=np.asarray(act_list, dtype=np.int64).reshape(-1, 1),
        done=np.asarray(done_list, dtype=np.float32).reshape(-1, 1),
        mask=np.stack(mask_list),
        next_mask=np.stack(next_mask_list),
    )
    return batch, layout
