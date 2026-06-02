"""MDP 模式：legacy（旧版）与 repeat（允许同格重复建站）。"""

from __future__ import annotations

import numpy as np

ONE_STATION_PER_CELL: bool = True

VALID_MDP_MODES = ("legacy", "repeat")


def apply_mdp_mode(mode: str) -> None:
    global ONE_STATION_PER_CELL
    key = str(mode).strip().lower()
    if key == "legacy":
        ONE_STATION_PER_CELL = True
    elif key == "repeat":
        ONE_STATION_PER_CELL = False
    else:
        raise ValueError(f"未知 --mdp-mode={mode!r}，可选: {', '.join(VALID_MDP_MODES)}")


def current_mdp_mode() -> str:
    return "legacy" if ONE_STATION_PER_CELL else "repeat"


def expert_trajectory_length(expert_actions: np.ndarray, grid_w: int) -> int:
    """专家决策步数（与 expert_action_sequence 长度一致，供 env max_steps 使用）。"""
    raw = np.asarray(expert_actions, dtype=np.int64).reshape(-1)
    if not raw.size:
        return 1
    if ONE_STATION_PER_CELL:
        seen: set[tuple[int, int]] = set()
        n = 0
        w = int(grid_w)
        for a in raw:
            gx, gy = int(a) % w, int(a) // w
            if (gx, gy) in seen:
                continue
            seen.add((gx, gy))
            n += 1
        return max(n, 1)
    return max(int(raw.size), 1)
