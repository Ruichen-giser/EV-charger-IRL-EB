"""BC 评估指标：Top-K、Hausdorff、Jaccard、LCSS、MRR。

两类评估模式（键前缀）：
  expert_rollin_*   — teacher forcing，环境走专家轨迹，逐步对比预测（含 MRR/top10）
  policy_rollout_*  — closed-loop 策略建站序列 vs 经 MDP 规则处理后的专家建站序列（legacy: first_visit 去重）
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from expert_data import CountyLayout

# expert roll-in：逐步排序/距离（不含集合 Jaccard、Hausdorff）
EXPERT_ROLLIN_CORE_KEYS: tuple[str, ...] = (
    "expert_greedy_match_rate",
    "top10_accuracy",
    "mean_reciprocal_rank",
    "mean_distance_km",
)

# policy rollout：建站序列对比（含 site P/R/F1、Jaccard、Hausdorff、Chamfer）
POLICY_ROLLOUT_CORE_KEYS: tuple[str, ...] = (
    "site_precision",
    "site_recall",
    "site_f1",
    "jaccard_similarity",
    "grid_hausdorff_km",
    "mean_distance_km",
)

# 向后兼容别名
EVAL_CORE_KEYS: tuple[str, ...] = POLICY_ROLLOUT_CORE_KEYS

# policy rollout 额外：LCSS 在三种空间阈值 ε (km) 下的归一化得分
POLICY_ROLLOUT_LCSS_EPS_KM: tuple[float, ...] = (0.0, 2.0, 2.829)


def lcss_eps_metric_key(epsilon_km: float) -> str:
    """ε (km) → metrics 键名，例如 2.829 → lcss_eps2_829_km。"""
    eps = float(epsilon_km)
    if abs(eps) < 1e-9:
        return "lcss_eps0_km"
    if abs(eps - 2.0) < 1e-9:
        return "lcss_eps2_km"
    if abs(eps - 2.829) < 1e-9:
        return "lcss_eps2_829_km"
    tag = f"{eps:.3f}".rstrip("0").rstrip(".").replace(".", "_")
    return f"lcss_eps{tag}_km"


POLICY_ROLLOUT_LCSS_KEYS: tuple[str, ...] = tuple(
    lcss_eps_metric_key(e) for e in POLICY_ROLLOUT_LCSS_EPS_KM
)


def prefix_eval_dict(ev: dict[str, Any], prefix: str) -> dict[str, Any]:
    """为评估结果键加前缀，eval_mode → {prefix}eval_mode。"""
    out: dict[str, Any] = {}
    for k, v in ev.items():
        if k == "eval_mode":
            out[f"{prefix}eval_mode"] = v
        else:
            out[f"{prefix}{k}"] = v
    return out


def merge_eval_dicts(expert: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """合并两类评估结果，键前缀分别为 expert_rollin_ / policy_rollout_。"""
    return {
        **prefix_eval_dict(expert, "expert_rollin_"),
        **prefix_eval_dict(policy, "policy_rollout_"),
    }


def eval_metrics_for_log(ev: dict[str, Any]) -> dict[str, float]:
    """从 merge 后的 ev 提取环境评估指标（供 metrics_log）。"""
    row: dict[str, float] = {}
    for k in EXPERT_ROLLIN_CORE_KEYS:
        row[f"expert_rollin_{k}"] = float(ev.get(f"expert_rollin_{k}", 0.0))
    for k in POLICY_ROLLOUT_CORE_KEYS:
        row[f"policy_rollout_{k}"] = float(ev.get(f"policy_rollout_{k}", 0.0))
    for k in POLICY_ROLLOUT_LCSS_KEYS:
        row[f"policy_rollout_{k}"] = float(ev.get(f"policy_rollout_{k}", 0.0))
    return row


def action_centers_km(layout: CountyLayout, actions: list[int] | np.ndarray) -> np.ndarray:
    pts = []
    for a in actions:
        gx, gy = layout.action_to_xy(int(a))
        pts.append(((gx + 0.5) * layout.cell_km, (gy + 0.5) * layout.cell_km))
    return np.asarray(pts, dtype=np.float64)


def directed_hausdorff_km(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    if q.size == 0:
        return float("inf")
    diff = p[:, None, :] - q[None, :, :]
    return float(np.max(np.min(np.linalg.norm(diff, axis=2), axis=1)))


def symmetric_hausdorff_km(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 and q.size == 0:
        return 0.0
    if p.size == 0 or q.size == 0:
        return float("inf")
    return max(directed_hausdorff_km(p, q), directed_hausdorff_km(q, p))


def grid_hausdorff_km(
    layout: CountyLayout,
    expert_actions: list[int] | np.ndarray,
    pred_actions: list[int] | np.ndarray,
) -> float:
    return symmetric_hausdorff_km(
        action_centers_km(layout, expert_actions),
        action_centers_km(layout, pred_actions),
    )


def jaccard_similarity(
    actions_a: list[int] | np.ndarray,
    actions_b: list[int] | np.ndarray,
) -> float:
    """最终建站集合的交并比 |A∩B|/|A∪B|（按格点 action id）。"""
    sa = {int(a) for a in actions_a}
    sb = {int(a) for a in actions_b}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


@torch.no_grad()
def masked_topk_accuracy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int = 10,
) -> float:
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, neg_inf)
    actions = actions.view(-1).long()
    k_eff = min(int(k), int(mask.sum(dim=1).max().item()))
    k_eff = max(k_eff, 1)
    _, topk = masked.topk(k_eff, dim=1)
    hit = (topk == actions.unsqueeze(1)).any(dim=1)
    return float(hit.float().mean().cpu())


@torch.no_grad()
def masked_reciprocal_rank(
    scores: torch.Tensor,
    expert_action: int,
    mask: torch.Tensor,
) -> float:
    """专家动作在 masked 降序排名中的倒数排名 1/rank；未出现在合法集则为 0。"""
    flat = scores.reshape(-1)
    m = mask.reshape(-1).bool()
    valid_idx = torch.nonzero(m, as_tuple=False).reshape(-1)
    if valid_idx.numel() == 0:
        return 0.0
    a = int(expert_action)
    if not bool(m[a]):
        return 0.0
    qv = flat[valid_idx]
    order = torch.argsort(qv, descending=True)
    ranked = valid_idx[order]
    pos = (ranked == a).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return 0.0
    rank = int(pos[0].item()) + 1
    return 1.0 / float(rank)


def empty_eval_dict(*, eval_mode: str = "none") -> dict[str, float | int | str]:
    if eval_mode == "policy_rollout":
        return {
            "eval_mode": eval_mode,
            "site_precision": 0.0,
            "site_recall": 0.0,
            "site_f1": 0.0,
            "jaccard_similarity": 0.0,
            "grid_hausdorff_km": 0.0,
            "n_steps": 0,
            "n_expert_stations": 0,
            "mean_distance_km": 0.0,
        }
    return {
        "eval_mode": eval_mode,
        "expert_greedy_match_rate": 0.0,
        "top10_accuracy": 0.0,
        "mean_reciprocal_rank": 0.0,
        "n_steps": 0,
        "mean_distance_km": 0.0,
    }


def summarize_step_metrics(
    layout: CountyLayout,
    expert_actions: list[int],
    pred_actions: list[int],
    *,
    matches: int,
    top10_hits: int,
    mrr_sum: float,
    scored_steps: int,
    distances: list[float],
    eval_mode: str,
) -> dict[str, float | int | str]:
    n = scored_steps
    return {
        "eval_mode": eval_mode,
        "expert_greedy_match_rate": matches / max(n, 1),
        "top10_accuracy": top10_hits / max(n, 1),
        "mean_reciprocal_rank": mrr_sum / max(n, 1),
        "n_steps": n,
        "mean_distance_km": float(np.mean(distances)) if distances else 0.0,
    }


def symmetric_chamfer_km(
    layout: CountyLayout,
    actions_a: list[int] | np.ndarray,
    actions_b: list[int] | np.ndarray,
) -> float:
    """两组建站格点之间的对称 Chamfer 距离（km，越小越接近）。"""
    pa = action_centers_km(layout, actions_a)
    pb = action_centers_km(layout, actions_b)
    if pa.size == 0 and pb.size == 0:
        return 0.0
    if pa.size == 0 or pb.size == 0:
        return float("inf")

    def _one_way(p: np.ndarray, q: np.ndarray) -> float:
        diff = p[:, None, :] - q[None, :, :]
        return float(np.mean(np.min(np.linalg.norm(diff, axis=2), axis=1)))

    return (_one_way(pa, pb) + _one_way(pb, pa)) / 2.0


def lcss_length_km(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    epsilon_km: float,
) -> int:
    """最长公共子序列长度：按建站顺序，匹配点空间距离 ≤ ε (km)。"""
    n = int(pts_a.shape[0])
    m = int(pts_b.shape[0])
    if n == 0 or m == 0:
        return 0
    eps = float(epsilon_km)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if float(np.linalg.norm(pts_a[i - 1] - pts_b[j - 1])) <= eps:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def lcss_score_km(
    layout: CountyLayout,
    traj_a: list[int] | np.ndarray,
    traj_b: list[int] | np.ndarray,
    epsilon_km: float,
) -> float:
    """归一化 LCSS = LCSS 长度 / min(len(a), len(b))，取值 [0, 1]。"""
    na = len(traj_a)
    nb = len(traj_b)
    denom = min(na, nb)
    if denom == 0:
        return 1.0 if na == nb == 0 else 0.0
    pa = action_centers_km(layout, traj_a)
    pb = action_centers_km(layout, traj_b)
    return lcss_length_km(pa, pb, epsilon_km) / float(denom)


def lcss_metrics_km(
    layout: CountyLayout,
    expert_ref: list[int],
    policy_built: list[int],
    *,
    epsilons_km: tuple[float, ...] = POLICY_ROLLOUT_LCSS_EPS_KM,
) -> dict[str, float | int]:
    """在若干 ε 阈值下计算 LCSS 归一化得分及原始匹配长度。"""
    pa = action_centers_km(layout, expert_ref)
    pb = action_centers_km(layout, policy_built)
    out: dict[str, float | int] = {}
    for eps in epsilons_km:
        key = lcss_eps_metric_key(eps)
        length = lcss_length_km(pa, pb, eps)
        out[key] = lcss_score_km(layout, expert_ref, policy_built, eps)
        out[f"{key}_len"] = length
    return out


def summarize_policy_deployment(
    layout: CountyLayout,
    expert_ref: list[int],
    policy_built: list[int],
    *,
    eval_mode: str,
) -> dict[str, float | int | str]:
    """Closed-loop 建站完成后，与 MDP 规则处理后的专家建站序列直接对比。"""
    se = {int(a) for a in expert_ref}
    sp = {int(a) for a in policy_built}
    inter = se & sp
    n_pol = len(policy_built)
    n_exp = len(expert_ref)

    site_precision = len(inter) / max(n_pol, 1)
    site_recall = len(inter) / max(n_exp, 1)
    site_f1 = (
        (2.0 * site_precision * site_recall / (site_precision + site_recall))
        if (site_precision + site_recall) > 0
        else 0.0
    )

    result: dict[str, float | int | str] = {
        "eval_mode": eval_mode,
        "site_precision": site_precision,
        "site_recall": site_recall,
        "site_f1": site_f1,
        "jaccard_similarity": jaccard_similarity(expert_ref, policy_built),
        "grid_hausdorff_km": grid_hausdorff_km(layout, expert_ref, policy_built),
        "n_steps": n_pol,
        "n_expert_stations": n_exp,
        "mean_distance_km": symmetric_chamfer_km(layout, expert_ref, policy_built),
    }
    result.update(lcss_metrics_km(layout, expert_ref, policy_built))
    return result


def format_expert_rollin_log(ev: dict[str, Any], *, prefix: str = "expert_rollin_") -> str:
    """格式化 expert roll-in 日志。"""
    p = prefix
    return (
        f"match={float(ev.get(f'{p}expert_greedy_match_rate', 0)):.3f}, "
        f"top10={float(ev.get(f'{p}top10_accuracy', 0)):.3f}, "
        f"mrr={float(ev.get(f'{p}mean_reciprocal_rank', 0)):.3f}, "
        f"dist_km={float(ev.get(f'{p}mean_distance_km', 0)):.2f}"
    )


def format_policy_rollout_log(ev: dict[str, Any], *, prefix: str = "policy_rollout_") -> str:
    """格式化 policy rollout 日志（含 LCSS@ε）。"""
    p = prefix
    return (
        f"prec={float(ev.get(f'{p}site_precision', 0)):.3f}, "
        f"recall={float(ev.get(f'{p}site_recall', 0)):.3f}, "
        f"f1={float(ev.get(f'{p}site_f1', 0)):.3f}, "
        f"jaccard={float(ev.get(f'{p}jaccard_similarity', 0)):.3f}, "
        f"hausdorff_km={float(ev.get(f'{p}grid_hausdorff_km', 0)):.2f}, "
        f"chamfer_km={float(ev.get(f'{p}mean_distance_km', 0)):.2f}, "
        f"lcss@0={float(ev.get(f'{p}lcss_eps0_km', 0)):.3f}, "
        f"lcss@2={float(ev.get(f'{p}lcss_eps2_km', 0)):.3f}, "
        f"lcss@2.829={float(ev.get(f'{p}lcss_eps2_829_km', 0)):.3f}"
    )


def format_eval_log(ev: dict[str, Any]) -> str:
    """合并 expert roll-in 与 policy rollout 评估日志（训练过程打印用）。"""
    return (
        f"expert_rollin: {format_expert_rollin_log(ev)} | "
        f"policy_rollout: {format_policy_rollout_log(ev)}"
    )
