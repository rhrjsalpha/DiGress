# utils/dist_utils.py
import os, sys, socket, torch
from pytorch_lightning.strategies import DDPStrategy

def _pick_master_port(default="29500"):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return str(s.getsockname()[1])
    except Exception:
        return default

def choose_ddp_strategy(devices: int, find_unused: bool = False):
    """
    devices>1이면 OS/빌드/가용성에 따라 DDP backend를 고르고,
    MASTER_ADDR/PORT도 안전하게 세팅합니다.

    Returns:
        (strategy, backend)  # strategy: DDPStrategy | None, backend: str | None
    """
    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if devices <= 1:
        return None, None  # DDP 불필요

    # 로컬 기본값
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _pick_master_port())

    is_windows = sys.platform.startswith("win")
    is_linux   = sys.platform.startswith("linux")
    has_cuda   = torch.cuda.is_available()

    # NCCL 빌드 유무 확인
    has_nccl = False
    try:
        import torch.distributed as dist
        has_nccl = getattr(dist, "is_nccl_available", lambda: False)()
    except Exception:
        has_nccl = False

    if is_windows:
        return DDPStrategy(process_group_backend="gloo",
                           find_unused_parameters=find_unused), "gloo"

    if is_linux and has_cuda and has_nccl:
        return DDPStrategy(process_group_backend="nccl",
                           find_unused_parameters=find_unused), "nccl"

    return DDPStrategy(process_group_backend="gloo",
                       find_unused_parameters=find_unused), "gloo"