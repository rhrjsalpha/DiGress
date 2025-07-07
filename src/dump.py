import torch
from omegaconf import OmegaConf
from hydra import initialize, compose

# Hydra config 초기화
initialize(config_path="../configs", version_base="1.1")
cfg = compose(config_name="config")

# 출력
print("cfg.general.gpus:", cfg.general.gpus)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

if torch.cuda.is_available():
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))