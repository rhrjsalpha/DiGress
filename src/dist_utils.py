# utils/dist_utils.py
import os, sys, socket, torch
from typing import Union, Sequence
from pytorch_lightning.strategies import DDPStrategy

DevicesType = Union[int, Sequence[int], None]

def _as_world_size(devices: DevicesType) -> int:
    if devices is None: return 0
    if isinstance(devices, (list, tuple)): return len(devices)
    try: return int(devices)
    except Exception: return 0

def _pick_master_port(default="29500"):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return str(s.getsockname()[1])
    except Exception:
        return default

def choose_ddp_strategy(devices: DevicesType, find_unused: bool = False):
    world_size = _as_world_size(devices)

    # 싱글이면 Lightning에게 맡김
    if world_size <= 1:
        return "auto", None

    # 로컬에서 항상 loopback 사용
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _pick_master_port())

    is_windows = sys.platform.startswith("win")
    is_linux   = sys.platform.startswith("linux")
    has_cuda   = torch.cuda.is_available()

    # --- Windows 핵심 픽스 ---
    # Windows 빌드의 gloo tcp 디바이스는 제한적 → uv 전송 강제
    if is_windows:
        # 127.0.0.1 + 가용포트 강제
        addr = "127.0.0.1"
        port = os.environ.get("MASTER_PORT") or _pick_master_port()
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        # Gloo 설정(Windows 권장)
        os.environ["GLOO_DEVICE_TRANSPORT"] = "uv"
        # NIC 이름: PowerShell의 Get-NetAdapter로 확인한 이름(가능하면 영문, 예: Ethernet)
        os.environ["GLOO_SOCKET_IFNAME"] = os.environ.get("GLOO_SOCKET_IFNAME", "Ethernet")

        # ★ 요기! env:// 대신 tcp:// 로 명시하여 호스트명 사용을 완전히 회피
        init = f"tcp://127.0.0.1:{port}"
        backend = "gloo"
        return DDPStrategy(
            process_group_backend=backend,
            find_unused_parameters=bool(find_unused),
            init_method=init,
        ), backend

    # Linux는 NCCL 가능하면 NCCL, 아니면 gloo
    has_nccl = False
    try:
        import torch.distributed as dist
        has_nccl = getattr(dist, "is_nccl_available", lambda: False)()
    except Exception:
        pass

    backend = "nccl" if (is_linux and has_cuda and has_nccl) else "gloo"
    return DDPStrategy(process_group_backend=backend,
                       find_unused_parameters=bool(find_unused),
                       init_method="env://"), backend
