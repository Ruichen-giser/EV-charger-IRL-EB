"""多通道网格状态（可配置通道子集）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.charging_deployment_env import ChargingDeploymentEnv

STATIC_FEATURE_NAMES = (
    "population_pct",
    "gdp_pct",
    "poi_pct",
    "highway_dist_pct",
    "fuel_station_count_pct",
)

OBS_CHANNEL_NAMES = (
    "dist_norm",
    *STATIC_FEATURE_NAMES,
    "station_mask",
    "valid_mask",
)

N_STATIC_CHANNELS = len(STATIC_FEATURE_NAMES)
N_OBS_CHANNELS = len(OBS_CHANNEL_NAMES)

# Siskiyou 单县训练：仅距离场 + 人口
DIST_POPULATION_CHANNELS = ("dist_norm", "population_pct")

COORD_CHANNEL_NAMES = ("grid_x_norm", "grid_y_norm")

# BC 默认：8 特征 + 归一化坐标（与 Siskiyou 最新配置一致）
ALL_OBS_CHANNELS = OBS_CHANNEL_NAMES + COORD_CHANNEL_NAMES


@dataclass(frozen=True)
class ObsChannelConfig:
    names: tuple[str, ...] = DIST_POPULATION_CHANNELS

    @property
    def n_channels(self) -> int:
        return len(self.names)


DEFAULT_OBS_CHANNELS = ObsChannelConfig()


def grid_coord_planes(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
    ys = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
    grid_x = np.broadcast_to(xs, (h, w)).copy()
    grid_y = np.broadcast_to(ys[:, None], (h, w)).copy()
    return grid_x, grid_y


def build_multichannel_obs(
    env: "ChargingDeploymentEnv",
    channel_cfg: ObsChannelConfig | None = None,
) -> np.ndarray:
    """构建 (H, W, C)。"""
    cfg = channel_cfg or DEFAULT_OBS_CHANNELS
    dist = env.dist_matrix_before_step() / float(env._max_dist)
    static = env.grid_features.astype(np.float32, copy=False)
    if env.stations is None:
        stations = np.zeros((env.H, env.W), dtype=np.float32)
    else:
        stations = env.stations.astype(np.float32, copy=False)
    valid = env.valid_mask.astype(np.float32, copy=False)
    grid_x, grid_y = grid_coord_planes(int(env.H), int(env.W))

    lookup: dict[str, np.ndarray] = {
        "dist_norm": dist,
        "population_pct": static[:, :, 0],
        "gdp_pct": static[:, :, 1],
        "poi_pct": static[:, :, 2],
        "highway_dist_pct": static[:, :, 3],
        "fuel_station_count_pct": static[:, :, 4],
        "station_mask": stations,
        "valid_mask": valid,
        "grid_x_norm": grid_x,
        "grid_y_norm": grid_y,
    }
    planes = [lookup[name][..., None] for name in cfg.names]
    return np.concatenate(planes, axis=-1).astype(np.float32, copy=False)
