"""训练后导出 state / meta / fused embedding（JSON + NPZ + CSV）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from iq_learn.embedding_export import export_location_embeddings  # noqa: E402

_DEFAULT_CKPT = _PKG_ROOT.parent / "outputs" / "iq_output" / "joint_mdp_ge5" / "iq_learn_shared.pt"
_DEFAULT_OUT = _PKG_ROOT.parent / "outputs" / "iq_output" / "joint_mdp_ge5" / "embedding_export"


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 CountyLocationEmbed 分析结果")
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT))
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")

    paths = export_location_embeddings(ckpt, args.output_dir)
    print("[export_location_embeddings] 完成")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
