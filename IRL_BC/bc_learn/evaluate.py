"""BC 评估：expert roll-in（teacher forcing）与 policy rollout（closed-loop）。"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from bc_learn.metrics import (
    empty_eval_dict,
    masked_reciprocal_rank,
    masked_topk_accuracy,
    merge_eval_dicts,
    summarize_policy_deployment,
    summarize_step_metrics,
)
from envs import (
    ChargingDeploymentEnv,
    MultiChannelGridObservationWrapper,
    action_mask_fn,
    unwrap_charging_env,
)
from bc_learn.grid_cnn_bc import GridCNNBCAgent
from expert_data import CountyLayout, expert_action_sequence
from obs_channels import ObsChannelConfig
import mdp_config


def _obs_batch(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)


def _set_agent_eval(agent: GridCNNBCAgent) -> None:
    """评估前切 eval，避免 train 模式下 BatchNorm/Dropout 影响 score 与 argmax。"""
    agent.q_net.eval()


def _station_was_placed(info: dict[str, Any]) -> bool:
    """该步是否成功新建了一个充电站。"""
    if info.get("invalid_action"):
        return False
    if mdp_config.ONE_STATION_PER_CELL and info.get("repeat_station"):
        return False
    return True


def _expert_reference_sequence(layout: CountyLayout) -> list[int]:
    """经 MDP 规则处理后的专家建站序列（legacy: first_visit 去重），不跑 teacher forcing。"""
    blob = np.load(layout.grid_npz, allow_pickle=False)
    if "expert_actions" not in blob:
        return []
    seq = expert_action_sequence(blob["expert_actions"], layout.W)
    return [int(a) for a in seq]


def _score_step(
    *,
    scores: torch.Tensor,
    mask: np.ndarray,
    mask_t: torch.Tensor,
    expert_action: int,
    pred: int,
    layout: CountyLayout,
    device: torch.device,
) -> tuple[int, int, float, float]:
    """返回 (match_inc, top10_inc, mrr, dist_km)。"""
    a = int(expert_action)
    if not bool(mask[a]):
        return 0, 0, 0.0, 0.0
    m = 1 if int(pred) == a else 0
    t10 = 1 if masked_topk_accuracy(scores, torch.tensor([a], device=device), mask_t, k=10) >= 1.0 else 0
    rr = masked_reciprocal_rank(scores, a, mask_t)
    dist = layout.grid_center_distance_km(a, int(pred))
    return m, t10, rr, dist


def _evaluate_expert_rollin(
    layout: CountyLayout,
    channel_cfg: ObsChannelConfig,
    *,
    device: torch.device,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    predict_fn: Callable[[torch.Tensor, torch.Tensor], int],
) -> dict[str, Any]:
    """Expert roll-in（teacher forcing）：环境始终执行专家动作，逐步对比策略预测。"""
    env = MultiChannelGridObservationWrapper(ChargingDeploymentEnv(layout.grid_npz), channel_cfg)
    base = unwrap_charging_env(env)
    if base.expert_actions is None:
        env.close()
        out = empty_eval_dict(eval_mode="expert_rollin")
        return out

    matches = top10_hits = scored_steps = 0
    mrr_sum = 0.0
    distances: list[float] = []
    expert_actions: list[int] = []
    pred_actions: list[int] = []

    obs, _ = env.reset()
    for a_expert in expert_action_sequence(base.expert_actions, base.W):
        mask = action_mask_fn(env)
        a_int = int(a_expert)
        obs_t = _obs_batch(obs, device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        scores = score_fn(obs_t)
        pred = int(predict_fn(obs_t, mask_t))

        if bool(mask[a_int]):
            expert_actions.append(a_int)
            pred_actions.append(pred)
            scored_steps += 1
            dm, dt, rr, dist = _score_step(
                scores=scores,
                mask=mask,
                mask_t=mask_t,
                expert_action=a_int,
                pred=pred,
                layout=layout,
                device=device,
            )
            matches += dm
            top10_hits += dt
            mrr_sum += rr
            distances.append(dist)

        obs, _, term, trunc, info = env.step(a_int)
        if info.get("invalid_action"):
            continue
        if term or trunc:
            break

    env.close()
    return summarize_step_metrics(
        layout,
        expert_actions,
        pred_actions,
        matches=matches,
        top10_hits=top10_hits,
        mrr_sum=mrr_sum,
        scored_steps=scored_steps,
        distances=distances,
        eval_mode="expert_rollin",
    )


def _evaluate_policy_rollout(
    layout: CountyLayout,
    channel_cfg: ObsChannelConfig,
    *,
    device: torch.device,
    predict_fn: Callable[[torch.Tensor, torch.Tensor], int],
    expert_ref: list[int] | None = None,
) -> dict[str, Any]:
    """Policy rollout：closed-loop 跑完，与 MDP 规则处理后的专家建站序列对比。"""
    env = MultiChannelGridObservationWrapper(ChargingDeploymentEnv(layout.grid_npz), channel_cfg)
    base = unwrap_charging_env(env)
    if base.expert_actions is None:
        env.close()
        out = empty_eval_dict(eval_mode="policy_rollout")
        return out

    ref_seq = expert_ref if expert_ref is not None else _expert_reference_sequence(layout)
    policy_built: list[int] = []

    obs, _ = env.reset()
    while True:
        mask = action_mask_fn(env)
        if not bool(np.any(mask)):
            break

        obs_t = _obs_batch(obs, device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        pred = int(predict_fn(obs_t, mask_t))

        obs, _, term, trunc, info = env.step(pred)
        if _station_was_placed(info):
            policy_built.append(pred)
        if info.get("invalid_action"):
            continue
        if term or trunc:
            break

    env.close()
    return summarize_policy_deployment(
        layout,
        ref_seq,
        policy_built,
        eval_mode="policy_rollout",
    )


def evaluate_bc_on_expert(
    agent: GridCNNBCAgent,
    layout: CountyLayout,
    channel_cfg: ObsChannelConfig,
    *,
    topk: int = 10,
) -> dict[str, Any]:
    del topk
    _set_agent_eval(agent)

    def score_fn(obs_t: torch.Tensor) -> torch.Tensor:
        return agent.logits(obs_t)

    def predict_fn(obs_t: torch.Tensor, mask_t: torch.Tensor) -> int:
        return agent.predict_action(obs_t, mask_t)

    return _evaluate_expert_rollin(
        layout,
        channel_cfg,
        device=agent.device,
        score_fn=score_fn,
        predict_fn=predict_fn,
    )


def evaluate_bc_policy_rollout(
    agent: GridCNNBCAgent,
    layout: CountyLayout,
    channel_cfg: ObsChannelConfig,
) -> dict[str, Any]:
    _set_agent_eval(agent)

    def predict_fn(obs_t: torch.Tensor, mask_t: torch.Tensor) -> int:
        return agent.predict_action(obs_t, mask_t)

    return _evaluate_policy_rollout(
        layout,
        channel_cfg,
        device=agent.device,
        predict_fn=predict_fn,
    )


def evaluate_bc_all(
    agent: GridCNNBCAgent,
    layout: CountyLayout,
    channel_cfg: ObsChannelConfig,
) -> dict[str, Any]:
    """合并两类评估：expert_rollin_* 与 policy_rollout_*。"""
    _set_agent_eval(agent)
    expert_ref = _expert_reference_sequence(layout)

    def score_fn(obs_t: torch.Tensor) -> torch.Tensor:
        return agent.logits(obs_t)

    def predict_fn(obs_t: torch.Tensor, mask_t: torch.Tensor) -> int:
        return agent.predict_action(obs_t, mask_t)

    expert = _evaluate_expert_rollin(
        layout,
        channel_cfg,
        device=agent.device,
        score_fn=score_fn,
        predict_fn=predict_fn,
    )
    policy = _evaluate_policy_rollout(
        layout,
        channel_cfg,
        device=agent.device,
        predict_fn=predict_fn,
        expert_ref=expert_ref,
    )
    agent.q_net.train()
    return merge_eval_dicts(expert, policy)
