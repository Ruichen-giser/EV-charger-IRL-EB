"""从 checkpoint 导出 state / meta / fused location embedding，供后续分析。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from county_meta import (
    COUNTY_META_FEATURE_NAMES,
    US_STATE_NAMES,
    compute_county_meta,
    state_name_to_id,
)
from models.county_location_embed import CountyLocationEmbed


def _parse_location_label(label: str) -> tuple[str, str]:
    for sep in ("/", "|", ":"):
        if sep in label:
            state, county = label.split(sep, 1)
            return state.strip(), county.strip()
    raise ValueError(f"无法解析 location label: {label!r}")


def _load_location_embed_from_checkpoint(blob: dict[str, Any]) -> CountyLocationEmbed:
    q_sd = blob["q_net"]
    prefix = "net.location_embed."
    loc_sd = {k[len(prefix) :]: v for k, v in q_sd.items() if k.startswith(prefix)}
    if not loc_sd:
        raise KeyError("checkpoint 缺少 net.location_embed.* 权重")

    embed = CountyLocationEmbed(
        n_states=int(blob.get("n_states", len(US_STATE_NAMES))),
        embed_dim=int(blob.get("embed_dim", 16)),
        meta_dim=int(blob.get("meta_dim", 3)),
        meta_hidden=32,
        n_residual=int(blob.get("n_residual", 1149)),
        residual_alpha=float(blob.get("residual_alpha", 0.0)),
    )
    embed.load_state_dict(loc_sd, strict=True)
    embed.eval()
    return embed


def extract_location_embeddings(
    ckpt_path: str | Path,
) -> dict[str, Any]:
    """
    解析 checkpoint，返回可序列化的 embedding 分析包。

    包含：
      - state_table: 51 州 state_embed 向量
      - counties: 每个训练县的 meta 输入、e_state、e_meta、e_residual、e_fused
    """
    path = Path(ckpt_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    embed = _load_location_embed_from_checkpoint(blob)

    labels = list(blob.get("location_labels") or [])
    if not labels:
        raise ValueError(f"checkpoint 缺少 location_labels: {path}")

    embed_dim = int(blob.get("embed_dim", embed.embed_dim))
    residual_alpha = float(blob.get("residual_alpha", float(embed.residual_alpha.item())))

    with torch.no_grad():
        state_vecs = embed.state_embed.weight.cpu().numpy().astype(np.float64)
        state_table: list[dict[str, Any]] = []
        for sid, name in enumerate(US_STATE_NAMES[: embed.n_states]):
            state_table.append(
                {
                    "state_id": int(sid),
                    "state_name": str(name),
                    "e_state": state_vecs[int(sid)].tolist(),
                }
            )

        counties: list[dict[str, Any]] = []
        e_state_mat: list[np.ndarray] = []
        e_meta_mat: list[np.ndarray] = []
        e_resid_mat: list[np.ndarray] = []
        e_fused_mat: list[np.ndarray] = []
        meta_raw_mat: list[np.ndarray] = []

        for i, label in enumerate(labels):
            state, county = _parse_location_label(label)
            sid = state_name_to_id(state)
            meta_raw = compute_county_meta(state, county)
            state_t = torch.tensor([sid], dtype=torch.long)
            meta_t = torch.as_tensor(meta_raw, dtype=torch.float32).unsqueeze(0)
            county_t = torch.tensor([i], dtype=torch.long)

            e_state = embed.state_embed(state_t).squeeze(0).cpu().numpy()
            e_meta = embed.meta_mlp(meta_t).squeeze(0).cpu().numpy()
            e_resid = embed.residual_embed(county_t).squeeze(0).cpu().numpy()
            e_fused = embed(state_t, meta_t, county_t).squeeze(0).cpu().numpy()

            counties.append(
                {
                    "county_id": int(i),
                    "state_name": state,
                    "county_name": county,
                    "location_key": label,
                    "county_meta": meta_raw.astype(np.float64).tolist(),
                    "county_meta_feature_names": list(COUNTY_META_FEATURE_NAMES),
                    "e_state": e_state.astype(np.float64).tolist(),
                    "e_meta": e_meta.astype(np.float64).tolist(),
                    "e_residual": e_resid.astype(np.float64).tolist(),
                    "e_fused": e_fused.astype(np.float64).tolist(),
                }
            )
            e_state_mat.append(e_state)
            e_meta_mat.append(e_meta)
            e_resid_mat.append(e_resid)
            e_fused_mat.append(e_fused)
            meta_raw_mat.append(meta_raw.astype(np.float64))

    return {
        "checkpoint": str(path.resolve()),
        "embed_dim": embed_dim,
        "meta_dim": int(blob.get("meta_dim", embed.meta_dim)),
        "residual_alpha": residual_alpha,
        "n_states": int(embed.n_states),
        "n_counties": len(counties),
        "state_table": state_table,
        "counties": counties,
        "arrays": {
            "state_names": list(US_STATE_NAMES[: embed.n_states]),
            "location_labels": labels,
            "county_meta_feature_names": list(COUNTY_META_FEATURE_NAMES),
            "state_embed": state_vecs,
            "county_meta": np.stack(meta_raw_mat, axis=0),
            "e_state": np.stack(e_state_mat, axis=0),
            "e_meta": np.stack(e_meta_mat, axis=0),
            "e_residual": np.stack(e_resid_mat, axis=0),
            "e_fused": np.stack(e_fused_mat, axis=0),
        },
    }


def export_location_embeddings(
    ckpt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """写出 JSON + NPZ，返回输出文件路径。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = extract_location_embeddings(ckpt_path)
    arrays = payload.pop("arrays")

    json_path = out / "location_embeddings.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    npz_path = out / "location_embeddings.npz"
    np.savez_compressed(
        npz_path,
        state_names=np.asarray(arrays["state_names"], dtype=object),
        location_labels=np.asarray(arrays["location_labels"], dtype=object),
        county_meta_feature_names=np.asarray(arrays["county_meta_feature_names"], dtype=object),
        state_embed=arrays["state_embed"].astype(np.float32),
        county_meta=arrays["county_meta"].astype(np.float32),
        e_state=arrays["e_state"].astype(np.float32),
        e_meta=arrays["e_meta"].astype(np.float32),
        e_residual=arrays["e_residual"].astype(np.float32),
        e_fused=arrays["e_fused"].astype(np.float32),
        embed_dim=np.int32(payload["embed_dim"]),
        residual_alpha=np.float32(payload["residual_alpha"]),
    )

    state_csv = out / "state_embeddings.csv"
    with state_csv.open("w", encoding="utf-8") as f:
        f.write("state_id,state_name," + ",".join(f"e{i}" for i in range(payload["embed_dim"])) + "\n")
        for row in payload["state_table"]:
            vec = ",".join(f"{x:.8f}" for x in row["e_state"])
            f.write(f"{row['state_id']},{row['state_name']},{vec}\n")

    county_csv = out / "county_embeddings.csv"
    with county_csv.open("w", encoding="utf-8") as f:
        meta_cols = ",".join(COUNTY_META_FEATURE_NAMES)
        e_cols = ",".join(f"e_fused_{i}" for i in range(payload["embed_dim"]))
        f.write(
            "county_id,state_name,county_name,location_key,"
            f"{meta_cols},"
            f"{e_cols}\n"
        )
        for row in payload["counties"]:
            meta = ",".join(f"{x:.8f}" for x in row["county_meta"])
            ef = ",".join(f"{x:.8f}" for x in row["e_fused"])
            f.write(
                f"{row['county_id']},{row['state_name']},{row['county_name']},"
                f"{row['location_key']},{meta},{ef}\n"
            )

    return {
        "json": str(json_path),
        "npz": str(npz_path),
        "state_csv": str(state_csv),
        "county_csv": str(county_csv),
    }
