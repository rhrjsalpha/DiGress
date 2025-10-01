## src/dist_utils.py (수정판)
import os, sys, torch, platform
from typing import Union, Sequence
from pytorch_lightning.strategies import DDPStrategy
from datetime import timedelta

DevicesType = Union[int, Sequence[int], None]

def _world_size(devices: DevicesType) -> int:
    if devices is None:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    if isinstance(devices, (list, tuple)):
        return len(devices)
    try:
        return int(devices)
    except Exception:
        return 0

def _is_wsl() -> bool:
    try:
        return (
            "microsoft" in platform.release().lower()
            or "wsl" in platform.release().lower()
            or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
            or os.environ.get("WSL_DISTRO_NAME")
        )
    except Exception:
        return False

def choose_ddp_strategy(devices: DevicesType, find_unused: bool = False):
    world = _world_size(devices)
    if world <= 1:
        return "auto", None  # 싱글

    # Windows → gloo 고정(루프백은 main.py에서 이미 세팅)
    if sys.platform.startswith("win"):
        return DDPStrategy(process_group_backend="gloo",
                           find_unused_parameters=bool(find_unused),
                           timeout=timedelta(hours=0.1)), "gloo"

    # Linux/WSL: 기본 nccl, 필요 시 gloo로 강제 가능
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # WSL 안정화 옵션
    if _is_wsl():
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
        )
        # 문제가 계속되면 환경변수로 gloo 강제 (선택)
        if os.environ.get("PL_FORCE_GLOO", "0") == "1":
            backend = "gloo"

    return DDPStrategy(process_group_backend=backend,
                       find_unused_parameters=bool(find_unused),
                       timeout=timedelta(hours=3),), backend