"""MDP 模式：legacy（旧版）与 repeat（允许同格重复建站）。"""

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
