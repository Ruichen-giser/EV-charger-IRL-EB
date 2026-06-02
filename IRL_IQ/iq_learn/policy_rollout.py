"""策略 rollout：向 Replay Buffer 写入 is_expert=False 的转移。"""
from typing import TYPE_CHECKING

import numpy as np
import torch

from envs import ChargingDeploymentEnv, MultiChannelGridObservationWrapper, action_mask_fn, unwrap_charging_env
from iq_learn.replay_buffer import StratifiedReplayBuffer, Transition
from obs_channels import DEFAULT_OBS_CHANNELS, ObsChannelConfig

if TYPE_CHECKING:
    from iq_learn.discrete_soft_q import DiscreteSoftQAgent


class PolicyRolloutCollector:
    """在单县环境上用当前 Q 策略收集一步或整局转移。"""

    def __init__(
        self,
        grid_npz: str,
        *,
        spatial: bool = True,
        channel_cfg: ObsChannelConfig | None = None,
    ) -> None:
        self.grid_npz = str(grid_npz)
        self.spatial = bool(spatial)
        self._env = MultiChannelGridObservationWrapper(
            ChargingDeploymentEnv(self.grid_npz),
            channel_cfg or DEFAULT_OBS_CHANNELS,
        )
        self._obs: np.ndarray | None = None

    @property
    def n_actions(self) -> int:
        base = unwrap_charging_env(self._env)
        return int(base.H * base.W)

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return self._obs

    def close(self) -> None:
        self._env.close()

    def _obs_to_tensor(self, obs: np.ndarray, device: torch.device) -> torch.Tensor:
        if self.spatial:
            return torch.as_tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        return torch.as_tensor(obs.reshape(-1), dtype=torch.float32, device=device).unsqueeze(0)

    @torch.no_grad()
    def collect_one_step(
        self,
        agent: "DiscreteSoftQAgent",
        buffer: StratifiedReplayBuffer,
        rng: np.random.Generator,
        *,
        stochastic: bool = True,
    ) -> bool:
        if self._obs is None:
            self.reset()

        obs = self._obs
        mask = action_mask_fn(self._env)
        if not bool(np.any(mask)):
            self.reset()
            return False

        obs_t = self._obs_to_tensor(obs, agent.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=agent.device).unsqueeze(0)

        if stochastic:
            action = agent.sample_action_soft(obs_t, mask_t, rng)
        else:
            action = agent.predict_action(obs_t, mask_t)

        next_obs, _, term, trunc, info = self._env.step(int(action))
        if info.get("invalid_action"):
            self._obs = np.asarray(next_obs, dtype=np.float32)
            return False

        next_mask = action_mask_fn(self._env)
        buffer.add_policy(
            Transition(
                obs=obs.copy(),
                next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                action=int(action),
                done=float(term or trunc),
                mask=mask.copy(),
                next_mask=next_mask.copy(),
                is_expert=False,
            )
        )
        self._obs = np.asarray(next_obs, dtype=np.float32)
        if term or trunc:
            self.reset()
        return True


def warm_start_policy_buffer(
    agent: "DiscreteSoftQAgent",
    buffer: StratifiedReplayBuffer,
    grid_npz: str,
    n_transitions: int,
    rng: np.random.Generator,
    *,
    channel_cfg: ObsChannelConfig | None = None,
) -> int:
    """预填充策略 buffer。"""
    col = PolicyRolloutCollector(grid_npz, spatial=buffer.spatial, channel_cfg=channel_cfg)
    col.reset()
    added = 0
    for _ in range(int(n_transitions)):
        if col.collect_one_step(agent, buffer, rng, stochastic=True):
            added += 1
    col.close()
    return added
