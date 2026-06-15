"""本包路径配置（独立运行，不依赖其它 IRL 子包）。"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_GRID_NPZ_DIR = OUTPUTS_DIR / "prepared_data" / "grid_tensors"
DEFAULT_MDP_GE5_XLSX = REPO_ROOT / "data" / "us_mdp_ge5_state_county_list.xlsx"
DEFAULT_JOINT_USA_MAP_GEOJSON = REPO_ROOT / "data" / "joint_usa_map.geojson"
DEFAULT_IQ_OUTPUT_BASELINE = OUTPUTS_DIR / "iq_output" / "joint_mdp_ge5_baseline"
DEFAULT_IQ_OUTPUT_S1 = OUTPUTS_DIR / "iq_output" / "joint_mdp_ge5_s1_residual"
# 默认实验为 S1（与 DEFAULT_EXPERIMENT 一致）
DEFAULT_IQ_OUTPUT_DIR = DEFAULT_IQ_OUTPUT_S1


def add_package_to_path() -> Path:
    """将本包加入 sys.path，支持直接：python main.py"""
    root = str(PACKAGE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PACKAGE_DIR


def ensure_grid_npz_dir(grid_npz_dir: Path) -> None:
    """确认网格 npz 目录存在。"""
    if not grid_npz_dir.is_dir():
        raise FileNotFoundError(
            f"网格数据目录不存在: {grid_npz_dir}\n"
            "请先运行 IRL_data/main.py，或在目录中放置 <州>__<县>_grid_features.npz"
        )


def default_output_dir(pair, base: Path = DEFAULT_IQ_OUTPUT_DIR) -> Path:
    """按 state-county 生成输出子目录。"""
    from state_county import StateCountyPair, default_output_stem

    if not isinstance(pair, StateCountyPair):
        pair = StateCountyPair("California", str(pair).replace("_", " "))
    return base.parent / default_output_stem(pair)


def policy_checkpoint_filename(
    method: str = "IQ",
    *,
    train_end: date | None = None,
    seed: int | None = None,
) -> str:
    """训练权重文件名，例如 IRL_IQ_20260527_seed0.pt"""
    tag = str(method).strip().upper()
    if tag not in ("BC", "IQ"):
        raise ValueError(f"method 须为 BC 或 IQ，当前: {method!r}")
    end = train_end or date.today()
    base = f"IRL_{tag}_{end.strftime('%Y%m%d')}"
    if seed is not None:
        return f"{base}_seed{int(seed)}.pt"
    return f"{base}.pt"


def policy_checkpoint_path(
    output_dir: Path | str,
    method: str = "IQ",
    *,
    train_end: date | None = None,
    seed: int | None = None,
) -> Path:
    return Path(output_dir) / policy_checkpoint_filename(
        method, train_end=train_end, seed=seed
    )
