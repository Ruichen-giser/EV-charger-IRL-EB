"""BC / IQ 共用的 CNN 与默认 state-county 配置。"""
from obs_channels import ALL_OBS_CHANNELS
from state_county import StateCountyPair

DEFAULT_STATE_COUNTY = StateCountyPair("California", "Los Angeles")
DEFAULT_COUNTY = "Los_Angeles"  # 兼容旧 CLI

# SimpleGridCNN：3 层 Conv3×3，通道 32→64→64
CNN_N_CONV_LAYERS = 3
CNN_DROPOUT = 0.0

DEFAULT_OBS_CHANNEL_NAMES = ALL_OBS_CHANNELS
