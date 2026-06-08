"""多县策略 rollout：随机选县、画布对齐后写入 Replay Buffer。"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from envs import ChargingDeploymentEnv, MultiChannelGridObservationWrapper, action_mask_fn, unwrap_charging_env
from iq_learn.expert_data import CountyLayout
from iq_learn.grid_align import (
    JointGridCanvas,
    canvas_action_to_local,
    local_action_to_canvas,
    pad_mask_flat,
    pad_obs_hwc,
)
from iq_learn.replay_buffer import StratifiedReplayBuffer, Transition
from obs_channels import DEFAULT_OBS_CHANNELS, ObsChannelConfig

if TYPE_CHECKING:
    from iq_learn.discrete_soft_q import DiscreteSoftQAgent


class CountyRolloutSession:
    """单县环境 session（本地网格）。"""

    def __init__(
        self,
        layout: CountyLayout,
        *,
        channel_cfg: ObsChannelConfig | None = None,
    ) -> None:
        self.layout = layout
        self._env = MultiChannelGridObservationWrapper(
            ChargingDeploymentEnv(layout.grid_npz),
            channel_cfg or DEFAULT_OBS_CHANNELS,
        )
        self._obs: np.ndarray | None = None

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        self._obs = np.asarray(obs, dtype=np.float32)
        return self._obs

    def close(self) -> None:
        self._env.close()

    @property
    def obs(self) -> np.ndarray:
        if self._obs is None:
            raise RuntimeError("CountyRolloutSession 尚未 reset")
        return self._obs

    def local_mask(self) -> np.ndarray:
        return action_mask_fn(self._env)

    def step_local(self, local_action: int):
        return self._env.step(int(local_action))


class JointPolicyRolloutPool:
    """8 县轮换 / 随机 rollout，agent 侧使用画布对齐 obs/mask。"""

    def __init__(
        self,
        counties: list[CountyLayout],
        canvas: JointGridCanvas,
        *,
        channel_cfg: ObsChannelConfig | None = None,
    ) -> None:
        self.canvas = canvas
        self.channel_cfg = channel_cfg or DEFAULT_OBS_CHANNELS
        self.sessions = [CountyRolloutSession(c, channel_cfg=self.channel_cfg) for c in counties]
        for s in self.sessions:
            s.reset()

    def close(self) -> None:
        for s in self.sessions:
            s.close()

    def _pick_session(self, rng: np.random.Generator) -> CountyRolloutSession:
        return self.sessions[int(rng.integers(0, len(self.sessions)))]

    def _to_agent_inputs(
        self,
        session: CountyRolloutSession,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = session.layout
        obs_p = pad_obs_hwc(session.obs, self.canvas)
        mask_p = pad_mask_flat(session.local_mask(), layout.H, layout.W, self.canvas)
        obs_t = torch.as_tensor(obs_p, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        mask_t = torch.as_tensor(mask_p, dtype=torch.bool, device=device).unsqueeze(0)
        county_t = torch.tensor([layout.county_id], dtype=torch.long, device=device)
        return obs_t, mask_t, county_t

    @torch.no_grad()
    def collect_one_step(
        self,
        agent: DiscreteSoftQAgent,
        buffer: StratifiedReplayBuffer,
        rng: np.random.Generator,
        *,
        stochastic: bool = True,
    ) -> bool:
        session = self._pick_session(rng)
        layout = session.layout
        mask_local = session.local_mask()
        if not bool(np.any(mask_local)):
            session.reset()
            return False

        obs_before = session.obs.copy()
        mask_before_local = mask_local.copy()
        obs_t, mask_t, county_t = self._to_agent_inputs(session, agent.device)

        if stochastic:
            canvas_action = agent.sample_action_soft(obs_t, mask_t, rng, county_t)
        else:
            canvas_action = agent.predict_action(obs_t, mask_t, county_t)

        local_action = canvas_action_to_local(
            canvas_action, layout.H, layout.W, self.canvas.max_w
        )
        if local_action is None:
            session.reset()
            return False

        next_obs, _, term, trunc, info = session.step_local(local_action)
        if info.get("invalid_action"):
            session._obs = np.asarray(next_obs, dtype=np.float32)
            return False

        next_mask_local = session.local_mask()
        buffer.add_policy(
            Transition(
                obs=pad_obs_hwc(obs_before, self.canvas),
                next_obs=pad_obs_hwc(np.asarray(next_obs, dtype=np.float32), self.canvas),
                action=int(
                    local_action_to_canvas(int(local_action), layout.W, self.canvas.max_w)
                ),
                done=float(term or trunc),
                mask=pad_mask_flat(mask_before_local, layout.H, layout.W, self.canvas),
                next_mask=pad_mask_flat(next_mask_local, layout.H, layout.W, self.canvas),
                county_id=int(layout.county_id),
                is_expert=False,
            )
        )
        session._obs = np.asarray(next_obs, dtype=np.float32)
        if term or trunc:
            session.reset()
        return True


def warm_start_joint_policy_buffer(
    agent: DiscreteSoftQAgent,
    buffer: StratifiedReplayBuffer,
    pool: JointPolicyRolloutPool,
    n_transitions: int,
    rng: np.random.Generator,
) -> int:
    added = 0
    for _ in range(int(n_transitions)):
        if pool.collect_one_step(agent, buffer, rng, stochastic=True):
            added += 1
    return added
