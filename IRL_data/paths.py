"""IRL_data 路径配置：相对仓库根目录自动推导，无需写死绝对路径。"""
from __future__ import annotations

from pathlib import Path

IRL_DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = IRL_DATA_DIR.parent
DATA_ROOT = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_PREPARED_DATA_DIR = OUTPUTS_DIR / "prepared_data"
DEFAULT_PREPARED_DATA_PKL = DEFAULT_PREPARED_DATA_DIR / "prepared_irl_dataset.pkl"
DEFAULT_CACHE_DIR = DEFAULT_PREPARED_DATA_DIR / "cache"
DEFAULT_GRID_NPZ_DIR = DEFAULT_PREPARED_DATA_DIR / "grid_tensors"
DEFAULT_MDP_GE5_COUNTY_LIST_XLSX = DATA_ROOT / "us_mdp_ge5_state_county_list.xlsx"


def validate_project_layout() -> None:
    """检查 data/ 目录是否存在。"""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"data directory not found: {DATA_ROOT}")
