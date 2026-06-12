"""分层 Replay Buffer：专家 / 策略各占固定比例；支持 county_id。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    next_obs: np.ndarray
    action: int
    done: float
    mask: np.ndarray
    next_mask: np.ndarray
    county_id: int = 0
    state_id: int = 0
    county_meta: np.ndarray | None = None
    is_expert: bool = False


@dataclass
class StratifiedReplayBuffer:
    """is_expert=True 为专家转移，False 为策略 rollout。"""

    capacity_expert: int = 50_000
    capacity_policy: int = 50_000
    expert_fraction: float = 0.5
    spatial: bool = True
    _expert: list[Transition] = field(default_factory=list)
    _policy: list[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._expert) + len(self._policy)

    def add_expert_from_merged(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        done: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
        county_ids: np.ndarray,
        state_ids: np.ndarray | None = None,
        county_meta: np.ndarray | None = None,
    ) -> None:
        n = int(obs.shape[0])
        for i in range(n):
            t = Transition(
                obs=obs[i].copy(),
                next_obs=next_obs[i].copy(),
                action=int(actions[i, 0] if actions.ndim > 1 else actions[i]),
                done=float(done[i, 0] if done.ndim > 1 else done[i]),
                mask=mask[i].copy(),
                next_mask=next_mask[i].copy(),
                county_id=int(county_ids[i]),
                state_id=int(state_ids[i]) if state_ids is not None else 0,
                county_meta=(
                    county_meta[i].copy()
                    if county_meta is not None
                    else None
                ),
                is_expert=True,
            )
            self._expert.append(t)
        if len(self._expert) > self.capacity_expert:
            self._expert = self._expert[-self.capacity_expert :]

    def add_policy(self, t: Transition) -> None:
        t.is_expert = False
        self._policy.append(t)
        if len(self._policy) > self.capacity_policy:
            self._policy = self._policy[-self.capacity_policy :]

    def sample(self, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        bs = int(batch_size)
        n_exp = max(1, int(round(bs * float(self.expert_fraction))))
        n_exp = min(n_exp, len(self._expert)) if self._expert else 0
        n_pol = bs - n_exp
        if self._policy and n_pol > len(self._policy):
            n_pol = len(self._policy)
            n_exp = bs - n_pol
        if not self._expert and self._policy:
            n_pol, n_exp = bs, 0
        if self._expert and not self._policy:
            n_exp, n_pol = bs, 0

        picks: list[Transition] = []
        if n_exp > 0:
            idx = rng.integers(0, len(self._expert), size=n_exp)
            picks.extend(self._expert[int(i)] for i in idx)
        if n_pol > 0:
            idx = rng.integers(0, len(self._policy), size=n_pol)
            picks.extend(self._policy[int(i)] for i in idx)
        if not picks:
            raise RuntimeError("Replay buffer 为空，无法采样")

        obs = np.stack([p.obs for p in picks])
        next_obs = np.stack([p.next_obs for p in picks])
        actions = np.asarray([p.action for p in picks], dtype=np.int64).reshape(-1, 1)
        done = np.asarray([p.done for p in picks], dtype=np.float32).reshape(-1, 1)
        mask = np.stack([p.mask for p in picks])
        next_mask = np.stack([p.next_mask for p in picks])
        is_expert = np.asarray([p.is_expert for p in picks], dtype=bool)
        county_ids = np.asarray([p.county_id for p in picks], dtype=np.int64)
        state_ids = np.asarray([p.state_id for p in picks], dtype=np.int64)
        county_meta = np.stack(
            [
                p.county_meta if p.county_meta is not None else np.zeros(5, dtype=np.float32)
                for p in picks
            ]
        )

        if self.spatial:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        return {
            "obs": obs_t,
            "next_obs": next_t,
            "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
            "done": torch.as_tensor(done, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
            "next_mask": torch.as_tensor(next_mask, dtype=torch.bool, device=device),
            "is_expert": torch.as_tensor(is_expert, dtype=torch.bool, device=device),
            "county_ids": torch.as_tensor(county_ids, dtype=torch.long, device=device),
            "state_ids": torch.as_tensor(state_ids, dtype=torch.long, device=device),
            "county_meta": torch.as_tensor(county_meta, dtype=torch.float32, device=device),
        }
