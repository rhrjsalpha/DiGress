# src/dist_utils.py
import os, sys, torch, tempfile
from typing import Union, Sequence
from pytorch_lightning.strategies import DDPStrategy

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

def _rdzv_file() -> str:
    """
    모든 랭크가 동일하게 볼 수 있는 고정 경로.
    Hydra가 작업 디렉터리를 바꿔도 ranks는 동일 cwd를 상속하므로
    현재 작업 디렉터리에 두는 것이 가장 안전함.
    """
    base = os.getcwd() if os.getcwd() else tempfile.gettempdir()
    path = os.path.join(base, ".pl_ddp_store")
    # Windows file://에서 역슬래시 이슈 회피
    return "file:///" + path.replace("\\", "/")

def choose_ddp_strategy(devices: DevicesType, find_unused: bool = False):
    """
    devices: int | list[int] | None  → Lightning Trainer(devices=...)와 동일 의미
    returns: (strategy, backend_str)
    """
    world = _world_size(devices)
    if world <= 1:
        return "auto", None  # 싱글 GPU/CPU는 자동

    is_windows = sys.platform.startswith("win")
    is_linux   = sys.platform.startswith("linux")
    has_cuda   = torch.cuda.is_available()

    # 공통: rendezvous를 file://로 고정 → MASTER_ADDR/PORT 의존 제거
    init = _rdzv_file()

    if is_windows:
        # Windows는 Gloo만 사용 가능. 호스트명 경로를 피하기 위해 루프백 NIC + uv 강제.
        os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")  # 환경에서 덮어쓸 수 있음
        return DDPStrategy(
            process_group_backend="gloo",
            init_method=init,
            find_unused_parameters=bool(find_unused),
        ), "gloo"

    # Linux: NCCL 가능 시 NCCL, 아니면 gloo
    has_nccl = False
    try:
        import torch.distributed as dist
        has_nccl = getattr(dist, "is_nccl_available", lambda: False)()
    except Exception:
        pass

    backend = "nccl" if (is_linux and has_cuda and has_nccl) else "gloo"
    return DDPStrategy(
        process_group_backend=backend,
        init_method=init,
        find_unused_parameters=bool(find_unused),
    ), backend
