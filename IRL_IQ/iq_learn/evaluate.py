"""多县 IQ-Learn 评估：画布对齐 + county embedding。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from envs import (
    ChargingDeploymentEnv,
    MultiChannelGridObservationWrapper,
    action_mask_fn,
    unwrap_charging_env,
)
from iq_learn.discrete_soft_q import DiscreteSoftQAgent
from iq_learn.expert_data import CountyLayout, expert_action_sequence
from iq_learn.grid_align import (
    JointGridCanvas,
    canvas_action_to_local,
    local_action_to_canvas,
    pad_mask_flat,
    pad_obs_hwc,
)
from iq_learn.metrics import (
    empty_eval_dict,
    masked_reciprocal_rank,
    masked_topk_accuracy,
    merge_eval_dicts,
    summarize_policy_deployment,
    summarize_step_metrics,
)
from obs_channels import ObsChannelConfig
import mdp_config


def _set_agent_eval(agent: DiscreteSoftQAgent) -> None:
    agent.q_net.eval()
    agent.target_net.eval()


def _restore_agent_train(agent: DiscreteSoftQAgent) -> None:
    agent.q_net.train()
    agent.target_net.train()


def _station_was_placed(info: dict[str, Any]) -> bool:
    if info.get("invalid_action"):
        return False
    if mdp_config.ONE_STATION_PER_CELL and info.get("repeat_station"):
        return False
    return True


def _agent_tensors(
    obs_hwc: np.ndarray,
    mask_local: np.ndarray,
    layout: CountyLayout,
    canvas: JointGridCanvas,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_p = pad_obs_hwc(obs_hwc, canvas)
    mask_p = pad_mask_flat(mask_local, layout.H, layout.W, canvas)
    obs_t = torch.as_tensor(obs_p, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.as_tensor(mask_p, dtype=torch.bool, device=device).unsqueeze(0)
    county_t = torch.tensor([layout.county_id], dtype=torch.long, device=device)
    state_t = torch.tensor([layout.state_id], dtype=torch.long, device=device)
    if layout.county_meta is not None:
        meta_t = torch.as_tensor(layout.county_meta, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        from county_meta import compute_county_meta_from_npz

        meta_t = torch.as_tensor(
            compute_county_meta_from_npz(layout.grid_npz),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    return obs_t, mask_t, county_t, state_t, meta_t


def _expert_deployed_sequence(
    layout: CountyLayout,
    channel_cfg: ObsChannelConfig,
) -> list[int]:
    env = MultiChannelGridObservationWrapper(ChargingDeploymentEnv(layout.grid_npz), channel_cfg)
    base = unwrap_charging_env(env)
    if base.expert_actions is None:
        env.close()
        return []

    deployed: list[int] = []
    obs, _ = env.reset()
    del obs
    for a_expert in expert_action_sequence(base.expert_actions, base.W):
        mask = action_mask_fn(env)
        a_int = int(a_expert)
        obs, _, term, trunc, info = env.step(a_int)
        if bool(mask[a_int]) and _station_was_placed(info):
            deployed.append(a_int)
        if info.get("invalid_action"):
            continue
        if term or trunc:
            break

    env.close()
    return deployed


def evaluate_joint_expert_rollin(
    agent: DiscreteSoftQAgent,
    layout: CountyLayout,
    canvas: JointGridCanvas,
    channel_cfg: ObsChannelConfig,
) -> dict[str, Any]:
    env = MultiChannelGridObservationWrapper(ChargingDeploymentEnv(layout.grid_npz), channel_cfg)
    base = unwrap_charging_env(env)
    if base.expert_actions is None:
        env.close()
        return empty_eval_dict(eval_mode="expert_rollin")

    matches = top10_hits = scored_steps = 0
    mrr_sum = 0.0
    distances: list[float] = []
    expert_actions: list[int] = []
    pred_actions: list[int] = []

    obs, _ = env.reset()
    for a_expert_local in expert_action_sequence(base.expert_actions, base.W):
        mask_local = action_mask_fn(env)
        a_local = int(a_expert_local)
        a_canvas = local_action_to_canvas(a_local, layout.W, canvas.max_w)

        obs_t, mask_t, county_t, state_t, meta_t = _agent_tensors(
            obs, mask_local, layout, canvas, agent.device
        )
        scores = agent.q_net(obs_t, state_t, meta_t, county_t)
        pred_canvas = int(agent.predict_action(obs_t, mask_t, state_t, meta_t, county_t))

        if bool(mask_local[a_local]):
            expert_actions.append(a_canvas)
            pred_actions.append(pred_canvas)
            scored_steps += 1
            m = 1 if pred_canvas == a_canvas else 0
            matches += m
            t10 = (
                1
                if masked_topk_accuracy(scores, torch.tensor([a_canvas], device=agent.device), mask_t, k=10) >= 1.0
                else 0
            )
            top10_hits += t10
            mrr_sum += masked_reciprocal_rank(scores, a_canvas, mask_t)
            distances.append(layout.grid_center_distance_km(a_canvas, pred_canvas))

        obs, _, term, trunc, info = env.step(a_local)
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


def evaluate_joint_policy_rollout(
    agent: DiscreteSoftQAgent,
    layout: CountyLayout,
    canvas: JointGridCanvas,
    channel_cfg: ObsChannelConfig,
    *,
    expert_ref: list[int] | None = None,
) -> dict[str, Any]:
    env = MultiChannelGridObservationWrapper(ChargingDeploymentEnv(layout.grid_npz), channel_cfg)
    base = unwrap_charging_env(env)
    if base.expert_actions is None:
        env.close()
        return empty_eval_dict(eval_mode="policy_rollout")

    ref_local = expert_ref if expert_ref is not None else _expert_deployed_sequence(layout, channel_cfg)
    ref_canvas = [
        local_action_to_canvas(int(a), layout.W, canvas.max_w) for a in ref_local
    ]
    policy_built: list[int] = []

    obs, _ = env.reset()
    while True:
        mask_local = action_mask_fn(env)
        if not bool(np.any(mask_local)):
            break

        obs_t, mask_t, county_t, state_t, meta_t = _agent_tensors(
            obs, mask_local, layout, canvas, agent.device
        )
        pred_canvas = int(agent.predict_action(obs_t, mask_t, state_t, meta_t, county_t))
        local_action = canvas_action_to_local(pred_canvas, layout.H, layout.W, canvas.max_w)
        if local_action is None:
            break

        obs, _, term, trunc, info = env.step(int(local_action))
        if _station_was_placed(info):
            policy_built.append(
                local_action_to_canvas(int(local_action), layout.W, canvas.max_w)
            )
        if info.get("invalid_action"):
            continue
        if term or trunc:
            break

    env.close()
    return summarize_policy_deployment(
        layout,
        ref_canvas,
        policy_built,
        eval_mode="policy_rollout",
    )


def evaluate_joint_county(
    agent: DiscreteSoftQAgent,
    layout: CountyLayout,
    canvas: JointGridCanvas,
    channel_cfg: ObsChannelConfig,
) -> dict[str, Any]:
    _set_agent_eval(agent)
    expert_ref = _expert_deployed_sequence(layout, channel_cfg)
    expert = evaluate_joint_expert_rollin(agent, layout, canvas, channel_cfg)
    policy = evaluate_joint_policy_rollout(
        agent, layout, canvas, channel_cfg, expert_ref=expert_ref
    )
    return merge_eval_dicts(expert, policy)


def evaluate_joint_all(
    agent: DiscreteSoftQAgent,
    counties: list[CountyLayout],
    canvas: JointGridCanvas,
    channel_cfg: ObsChannelConfig,
    *,
    max_counties: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """逐县评估并返回 per_county 列表与跨县平均核心指标。"""
    targets = list(counties)
    if max_counties is not None and int(max_counties) > 0 and len(targets) > int(max_counties):
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(targets), size=int(max_counties), replace=False)
        targets = [targets[int(i)] for i in idx]

    per: list[dict[str, Any]] = []
    for layout in targets:
        ev = evaluate_joint_county(agent, layout, canvas, channel_cfg)
        per.append(
            {
                "state_name": layout.state_name,
                "county_name": layout.county_name,
                "location_key": f"{layout.state_name}/{layout.county_name}",
                **ev,
            }
        )

    match_rates = [float(r["expert_rollin_expert_greedy_match_rate"]) for r in per]
    dists = [float(r["expert_rollin_mean_distance_km"]) for r in per]
    summary = {
        "mean_expert_match_rate": float(np.mean(match_rates)) if match_rates else 0.0,
        "mean_distance_km": float(np.mean(dists)) if dists else 0.0,
    }
    _restore_agent_train(agent)
    return per, summary
