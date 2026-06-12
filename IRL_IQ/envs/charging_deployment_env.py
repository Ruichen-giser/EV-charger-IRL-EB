"""
Gymnasium 环境：2 km 网格上顺序部署充电站（IRL/RL 共用）。

MDP：
  - 状态 s：每格到最近已建站点的欧氏距离 d(x,y)，形状 (H, W)
  - 动作 a：在某一格建站，扁平下标 = grid_y * W + grid_x
  - 转移：建站后全场距离取 min 更新
  - 环境奖励：默认 0（IRL 由 IQ-Learn 从专家轨迹学 soft Q）
"""
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import mdp_config


def _km_distances_from_station(gy: int, gx: int, H: int, W: int, cell_km: float) -> np.ndarray:
    """新站落在 (gy,gx) 时，全场各格中心到该站的距离（km）。"""
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    dy = (yy - float(gy)) * float(cell_km)
    dx = (xx - float(gx)) * float(cell_km)
    return np.sqrt(dx * dx + dy * dy, dtype=np.float32)


class ChargingDeploymentEnv(gym.Env):
    """
    从网格 .npz 加载：
      grid_features (H,W,5)、valid_mask、expert_actions、grid_cell_km
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_features_path: str | Path,
        *,
        max_steps: int | None = None,
        grid_cell_km: float | None = None,
        invalid_action_penalty: float = -1.0,
        repeat_station_penalty: float = -0.5,
        expert_actions: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        path = Path(grid_features_path)
        if path.suffix != ".npz":
            raise ValueError("仅支持 .npz 网格特征文件")

        blob = np.load(path, allow_pickle=False)
        self.grid_features = blob["grid_features"].astype(np.float32)
        self.valid_mask = blob["valid_mask"].astype(bool)
        if expert_actions is None and "expert_actions" in blob:
            expert_actions = blob["expert_actions"].astype(np.int64)
        self.cell_km = float(blob["grid_cell_km"]) if grid_cell_km is None else float(grid_cell_km)
        self.state_name = str(blob["state_name"][0]) if "state_name" in blob else ""
        if "county_name" in blob:
            self.county_name = str(blob["county_name"][0])
        else:
            stem = path.stem.replace("_grid_cropped", "").replace("_grid_features", "")
            self.county_name = stem.split("__")[-1].replace("_", " ") if "__" in stem else stem

        if self.grid_features.ndim != 3 or self.grid_features.shape[2] != 5:
            raise ValueError(f"grid_features 须为 (H,W,5)，当前 {self.grid_features.shape}")

        self.H, self.W = int(self.grid_features.shape[0]), int(self.grid_features.shape[1])
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.repeat_station_penalty = float(repeat_station_penalty)
        self.expert_actions = (
            np.asarray(expert_actions, dtype=np.int64).reshape(-1) if expert_actions is not None else None
        )

        self.action_space = spaces.Discrete(self.H * self.W)
        self._max_dist = max(
            float(np.sqrt((self.H * self.cell_km) ** 2 + (self.W * self.cell_km) ** 2)),
            self.cell_km,
        )
        self.observation_space = spaces.Box(0.0, self._max_dist, (self.H, self.W), dtype=np.float32)

        self.stations: np.ndarray | None = None
        self.current_dist_matrix: np.ndarray | None = None
        self.current_step = 0
        n_valid = int(self.valid_mask.sum()) or self.H * self.W
        if max_steps is not None:
            self.max_steps = int(max_steps)
        elif self.expert_actions is not None and len(self.expert_actions) > 0:
            # 与 expert roll-in / policy rollout 一致：步数 = 专家轨迹长度，不与 n_valid 取 max。
            self.max_steps = mdp_config.expert_trajectory_length(self.expert_actions, self.W)
        else:
            self.max_steps = max(n_valid, 1)

    def _action_to_xy(self, action: int) -> tuple[int, int]:
        a = int(action)
        return a % self.W, a // self.W  # gx, gy

    def _is_valid_action(self, gx: int, gy: int) -> bool:
        if gx < 0 or gy < 0 or gx >= self.W or gy >= self.H:
            return False
        return bool(self.valid_mask[gy, gx])

    def _update_dist_matrix(self, gx: int, gy: int) -> None:
        new_d = _km_distances_from_station(gy, gx, self.H, self.W, self.cell_km)
        self.current_dist_matrix = new_d if self.current_dist_matrix is None else np.minimum(
            self.current_dist_matrix, new_d
        )

    def _get_obs(self) -> np.ndarray:
        if self.current_dist_matrix is None:
            return np.full((self.H, self.W), self._max_dist, dtype=np.float32)
        return self.current_dist_matrix.astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.stations = np.zeros((self.H, self.W), dtype=bool)
        self.current_dist_matrix = None
        self.current_step = 0
        return self._get_obs(), {"state_name": self.state_name, "county_name": self.county_name}

    def step(self, action: int):
        gx, gy = self._action_to_xy(action)
        if self.stations is None:
            raise RuntimeError("请先 reset()")

        reward = 0.0
        invalid = not self._is_valid_action(gx, gy)
        repeat = bool(self.stations[gy, gx]) if not invalid else False
        if invalid:
            reward = self.invalid_action_penalty
        elif mdp_config.ONE_STATION_PER_CELL and repeat:
            reward = self.repeat_station_penalty
        else:
            self.stations[gy, gx] = True
            self._update_dist_matrix(gx, gy)

        self.current_step += 1
        # legacy：全部有效格已建站后无合法动作，应结束 episode（避免 rollout 空 mask 采样）
        terminated = self.current_step >= self.max_steps or not bool(self.action_mask_flat().any())
        info = {
            "action_flat": int(action),
            "grid_x": gx,
            "grid_y": gy,
            "invalid_action": invalid,
            "repeat_station": repeat,
        }
        return self._get_obs(), float(reward), terminated, False, info

    def action_mask_flat(self) -> np.ndarray:
        """有效格点为 True；ONE_STATION_PER_CELL 时再排除已建站格。"""
        m = self.valid_mask.copy()
        if mdp_config.ONE_STATION_PER_CELL and self.stations is not None:
            m &= ~self.stations
        return m.reshape(-1)

    def dist_matrix_before_step(self) -> np.ndarray:
        """执行动作前的距离场（分析用）。"""
        if self.current_dist_matrix is None:
            return np.full((self.H, self.W), self._max_dist, dtype=np.float32)
        return self.current_dist_matrix.astype(np.float32, copy=False)


def unwrap_charging_env(env: gym.Env) -> ChargingDeploymentEnv:
    """从 Wrapper 链中取出底层 ChargingDeploymentEnv。"""
    e: gym.Env = env
    for _ in range(8):
        if isinstance(e, ChargingDeploymentEnv):
            return e
        if hasattr(e, "env"):
            e = e.env  # type: ignore[assignment]
        else:
            break
    return e.unwrapped  # type: ignore[return-value]


class MultiChannelGridObservationWrapper(gym.ObservationWrapper):
    """观测 (H, W, C)：距离场 + 静态特征 + 建站掩膜等。"""

    def __init__(self, env: gym.Env, channel_cfg=None):
        from obs_channels import DEFAULT_OBS_CHANNELS, build_multichannel_obs

        super().__init__(env)
        self._build_obs = build_multichannel_obs
        self.channel_cfg = channel_cfg or DEFAULT_OBS_CHANNELS
        base = unwrap_charging_env(env)
        c = int(self.channel_cfg.n_channels)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(int(base.H), int(base.W), c), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return self._build_obs(unwrap_charging_env(self.env), self.channel_cfg)


def action_mask_fn(env: gym.Env) -> np.ndarray:
    """合法动作掩膜（形状 H*W，True=可选）。"""
    return unwrap_charging_env(env).action_mask_flat()
