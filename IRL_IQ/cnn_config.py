"""BC / IQ 共用配置（8 县联合 + county embedding）。"""
from obs_channels import ALL_OBS_CHANNELS

# 联合训练默认 8 县（与 IRL_data/main.py county_names 一致）
TRAINING_COUNTIES: tuple[str, ...] = (
    "Siskiyou",
    "Modoc",
    "Shasta",
    "Lassen",
    "Los_Angeles",
    "Sacramento",
    "San_Diego",
    "Kern",
)

CNN_N_CONV_LAYERS = 3
CNN_DROPOUT = 0.0
COUNTY_EMBED_DIM = 16

DEFAULT_OBS_CHANNEL_NAMES = ALL_OBS_CHANNELS
