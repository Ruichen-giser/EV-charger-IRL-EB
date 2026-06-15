"""IQ 联合训练配置（默认 MDP≥5 全量 1149 县 + CountyLocationEmbed）。"""
from county_meta import COUNTY_META_DIM
from obs_channels import ALL_OBS_CHANNELS
from paths import DEFAULT_MDP_GE5_XLSX
from state_county import StateCountyPair

# 开发/调试用小集合（--dev-8-counties 时启用）
DEV_STATE_COUNTIES_8: tuple[StateCountyPair, ...] = (
    StateCountyPair("California", "Siskiyou"),
    StateCountyPair("California", "Modoc"),
    StateCountyPair("California", "Shasta"),
    StateCountyPair("California", "Lassen"),
    StateCountyPair("California", "Los Angeles"),
    StateCountyPair("California", "Sacramento"),
    StateCountyPair("California", "San Diego"),
    StateCountyPair("California", "Kern"),
)

# 兼容旧 API
TRAINING_STATE_COUNTIES = DEV_STATE_COUNTIES_8
TRAINING_COUNTIES: tuple[str, ...] = tuple(
    p.county_name.replace(" ", "_") for p in DEV_STATE_COUNTIES_8
)

CNN_N_CONV_LAYERS = 3
CNN_DROPOUT = 0.0

# CountyLocationEmbed（state + meta MLP + residual）
COUNTY_EMBED_DIM = 16
META_MLP_HIDDEN = 32
N_US_STATES = 51
N_MAX_COUNTY_RESIDUAL = 1149
# baseline：仅 state + meta；S1 通过 --experiment s1 启用 residual_alpha=1.0
RESIDUAL_ALPHA = 0.0
EMBED_DROPOUT = 0.1

# 大规模联合训练默认参数
DEFAULT_MDP_GE5_COUNTY_LIST_XLSX = DEFAULT_MDP_GE5_XLSX
DEFAULT_IQ_LOSS_MODE = "offline"
DEFAULT_TRAIN_STEPS = 1_000_000
DEFAULT_EVAL_EVERY = 200
DEFAULT_LR = 2e-5
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_MAX_COUNTIES = 64
DEFAULT_EVAL_FINAL_MAX_COUNTIES = 64
DEFAULT_TARGET_UPDATE_INTERVAL = 50
DEFAULT_ROLLOUT_MAX_SESSIONS = 32

# 兼容旧字段名
STATE_EMBED_DIM = COUNTY_EMBED_DIM

DEFAULT_OBS_CHANNEL_NAMES = ALL_OBS_CHANNELS
