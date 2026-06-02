"""BC / IQ 共用的 CNN 与默认县配置。"""
from obs_channels import ALL_OBS_CHANNELS

DEFAULT_COUNTY = "Los_Angeles"

# SimpleGridCNN：3 层 Conv3×3，通道 32→64→64
CNN_N_CONV_LAYERS = 3
CNN_DROPOUT = 0.0

DEFAULT_OBS_CHANNEL_NAMES = ALL_OBS_CHANNELS
