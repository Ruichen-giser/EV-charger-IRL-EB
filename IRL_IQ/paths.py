"""本包路径配置（独立运行，不依赖其它 IRL 子包）。"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_GRID_NPZ_DIR = OUTPUTS_DIR / "prepared_data" / "grid_tensors"
DEFAULT_IQ_OUTPUT_DIR = OUTPUTS_DIR / "iq_output" / "joint_8counties"


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
            "请先运行 IRL_data/main.py，或在目录中放置 <县名>_grid_features.npz"
        )


def default_output_dir(county: str, base: Path = DEFAULT_IQ_OUTPUT_DIR) -> Path:
    """按县名生成输出子目录。"""
    return base.parent / county.lower().replace(" ", "_")


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
