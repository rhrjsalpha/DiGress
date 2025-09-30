# src/main_spec.py
# -*- coding: utf-8 -*-
"""
스펙트럼(+ solvent_phase, pH 등) 조건으로 분자 생성 Diffusion 학습 엔트리
- CSVSpecDataset 그대로 사용
- edge_attr(6ch: 4타입+conj+ring) → DiGress 규약 5채널(one-hot: [no,single,double,trip,arom])로 변환
- DatasetInfos를 동봉하여 compute_input_output_dims까지 한 번에 연결
"""

# ─────────────────────────────────────────────────────────────────────────────
# Windows에서 DDP/Gloo 안전옵션 (torch import 전에)
import os, csv
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")
# ─────────────────────────────────────────────────────────────────────────────

import pathlib
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
try:
    import torch.cuda
    torch.cuda.memory_summary = lambda *a, **k: ""  # 아무 것도 반환하지 않게
except Exception:
    pass

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import CSVLogger

# 프로젝트 모듈 (기존 main과 동일한 경로 규칙)
from src import utils
from src.dist_utils import choose_ddp_strategy

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures

from metrics.molecular_metrics import TrainMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from metrics.molecular_metrics import SamplingMolecularMetrics
from analysis.visualization import MolecularVisualization
import re
from pathlib import Path
from typing import Optional
import shutil
import sys, re, csv, os

# 당신이 올린 데이터셋
from datasets.csv_spectrum_dataset import CSVSpecDataset
from src.datasets.csvspec_module import CSVSpecDataModule, CSVSpecInfos
warnings.filterwarnings("ignore", category=PossibleUserWarning)
from pytorch_lightning.callbacks import Callback

QUIET = os.environ.get("QUIET", "0") not in ("0", "", "false", "False", "no", "NO")
if QUIET:
    os.environ.setdefault("PL_DISABLE_TQDM", "0")  # tqdm 막대 끔
    os.environ.setdefault("NCCL_DEBUG", "ERROR")   # NCCL 로그 최소화
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore")

def qprint(*a, **k):
    if not QUIET:
        print(*a, **k)

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PAT = re.compile(
    rf"Epoch\s+(\d+)\s*:.*?Test\s*NLL[: ]\s*({_FLOAT}).*?"
    rf"Atom[\s-]*type\s*KL[: ]\s*({_FLOAT}).*?"
    rf"Edge[\s-]*type\s*KL[: ]\s*({_FLOAT})",
    re.IGNORECASE,
)
# relaxed 먼저 정의 그대로 두고,
_PAT_REL_VALID = re.compile(rf"Relaxed\s+validity\s+over\s+(\d+)\s+molecules:\s+({_FLOAT})%", re.IGNORECASE)

# 일반 validity는 라인 시작(^)에만 매칭되게 보수적으로
_PAT_VALID     = re.compile(rf"^\s*Validity\s+over\s+(\d+)\s+molecules:\s+({_FLOAT})%", re.IGNORECASE)

_PAT_CONN      = re.compile(rf"Number of connected components of\s+(\d+)\s+molecules:\s*min:\s*({_FLOAT})\s*mean:\s*({_FLOAT})\s*max:\s*({_FLOAT})", re.IGNORECASE)
_PAT_UNIQ      = re.compile(rf"Uniqueness\s+over\s+(\d+)\s+valid\s+molecules:\s+({_FLOAT})%", re.IGNORECASE)
_PAT_NOV       = re.compile(rf"Novelty\s+over\s+(\d+)\s+unique\s+valid\s+molecules:\s+({_FLOAT})%", re.IGNORECASE)
_PAT_SPLIT     = re.compile(r"Starting custom metrics", re.IGNORECASE)


class _StdoutTee:
    """stdout을 파일에 동시에 복사하는 간단한 tee."""
    def __init__(self, real_stream, capture_path: Path):
        self._real = real_stream
        self._file = open(capture_path, "w", encoding="utf-8")

    def write(self, s: str):
        self._real.write(s)
        self._file.write(s)

    def flush(self):
        self._real.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    @property
    def real(self):
        return self._real


class TestMetricsCSVCallback(Callback):
    """테스트 중 stdout을 캡처해서 NLL/Atom-KL/Edge-KL을 pl_logs/version_0/test_metrics.csv로 저장"""
    def __init__(self, filename: str = "test_metrics.csv"):
        super().__init__()
        self.filename = filename
        self._initialized = False
        self._log_dir: Optional[Path] = None
        self._csv_path: Optional[Path] = None
        self._header_written = False
        self._tee: Optional[_StdoutTee] = None

    def _init_dir(self, trainer):
        if self._initialized:
            return
        # CSVLogger 경로 찾기
        loggers = getattr(trainer, "loggers", None) or ([trainer.logger] if getattr(trainer, "logger", None) else [])
        if not isinstance(loggers, (list, tuple)):
            loggers = [loggers]
        for lg in loggers:
            if isinstance(lg, CSVLogger):
                self._log_dir = Path(lg.log_dir)  # .../pl_logs/version_0
                break
        if self._log_dir is None:
            self._log_dir = Path(os.getcwd()) / "pl_logs" / "version_0"  # 폴백
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._log_dir / self.filename
        self._header_written = self._csv_path.exists()
        self._initialized = True

    # ---- Lightning hooks ----
    def on_test_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._init_dir(trainer)
        # stdout 캡처 시작
        cap_path = self._log_dir / "test_stdout_capture.log"
        self._tee = _StdoutTee(sys.stdout, cap_path)
        sys.stdout = self._tee  # tee로 교체

    def on_test_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # stdout 복구
        if self._tee is not None:
            sys.stdout = self._tee.real
            self._tee.close()
            self._tee = None
        # 캡처 파일 파싱
        cap_path = self._log_dir / "test_stdout_capture.log"
        row = self._parse_from_capture(cap_path)
        if row:
            with open(self._csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch","test_nll","test_atom_kl","test_edge_kl"])
                if not self._header_written:
                    w.writeheader(); self._header_written = True
                w.writerow(row)
        self._finalize_parse_and_write()

    # ---- helper ----
    def _parse_from_capture(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        last = None
        with open(path, "r", encoding="utf-8", errors="ignore") as g:
            for line in g:
                m = _PAT.search(line)
                if m:
                    last = {
                        "epoch": int(m.group(1)),
                        "test_nll": float(m.group(2)),
                        "test_atom_kl": float(m.group(3)),
                        "test_edge_kl": float(m.group(4)),
                    }
        return last

    def _finalize_parse_and_write(self):
        """캡처 파일을 파싱해 test_metrics.csv와 sampling_metrics_pre_custom.csv를 기록(있으면 건너뜀)."""
        self._init_dir(getattr(self, "_trainer_ref", None) or type("T", (), {"loggers": [], "logger": None}))  # 안전 폴백
        cap_path = self._log_dir / "test_stdout_capture.log"

        # (A) Epoch/Test NLL·KL (이미 썼다면 생략)
        row = self._parse_from_capture(cap_path)
        if row and self._csv_path is not None:
            need_header = not self._csv_path.exists()
            with open(self._csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "test_nll", "test_atom_kl", "test_edge_kl"])
                if need_header: w.writeheader()
                w.writerow(row)

        # (B) pre-custom sampling metrics
        blocks = _parse_sampling_blocks_from_capture(cap_path)
        if blocks:
            out_csv2 = self._log_dir / "sampling_metrics_pre_custom.csv"
            need_header2 = not out_csv2.exists()
            with open(out_csv2, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_SAMPLING_HEADERS)
                if need_header2: w.writeheader()
                for b in blocks:
                    # 누락 키는 빈칸으로
                    row2 = {k: b.get(k, "") for k in _SAMPLING_HEADERS}
                    w.writerow(row2)
            print(f"[OK] saved sampling metrics(pre-custom) → {out_csv2}")

    # Lightning이 콜백에 trainer를 안 넘겨주는 훅에서 사용하기 위함
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        self._trainer_ref = trainer

    def on_exception(self, trainer, pl_module, _):
        # 테스트 중간에 예외가 나도 rank0에서 캡처 파일까지는 남았으니 여기서 저장
        if trainer.is_global_zero:
            self._finalize_parse_and_write()

class TrainCaptureCSVCallback(Callback):
    def __init__(self,
                 metrics_filename="train_metrics.csv",
                 sampling_filename="train_sampling_metrics_pre_custom.csv"):
        super().__init__()
        self.metrics_filename = metrics_filename
        self.sampling_filename = sampling_filename
        self._log_dir: Optional[Path] = None
        self._tee: Optional[_StdoutTee] = None
        self._initialized = False

    def _init_dir(self, trainer):
        if self._initialized:
            return
        # CSVLogger 경로 얻기 (TestMetricsCSVCallback과 동일)
        loggers = getattr(trainer, "loggers", None) or ([trainer.logger] if getattr(trainer, "logger", None) else [])
        if not isinstance(loggers, (list, tuple)):
            loggers = [loggers]
        for lg in loggers:
            if isinstance(lg, CSVLogger):
                self._log_dir = Path(lg.log_dir)  # .../pl_logs/version_0
                break
        if self._log_dir is None:
            # 폴백: 현재 작업 디렉토리 기준으로 생성
            self._log_dir = Path(os.getcwd()) / "pl_logs" / "version_0"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._init_dir(trainer)
        cap = self._log_dir / "train_stdout_capture.log"
        self._tee = _StdoutTee(sys.stdout, cap)
        sys.stdout = self._tee

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # stdout 복구
        if self._tee is not None:
            sys.stdout = self._tee.real
            self._tee.close()
            self._tee = None

        cap = self._log_dir / "train_stdout_capture.log"

        # (1) on_train_end에서 출력한 "Final Train NLL ... KL ..." 파싱 → CSV
        _PAT_TRAIN = re.compile(
            rf"Final\s+Train\s+NLL\s+({_FLOAT}).*?Atom.*?KL\s+({_FLOAT}).*?Edge.*?KL[: ]\s*({_FLOAT})",
            re.IGNORECASE
        )
        last = None
        try:
            with open(cap, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = _PAT_TRAIN.search(line)
                    if m:
                        last = {
                            "train_nll": float(m.group(1)),
                            "train_atom_kl": float(m.group(2)),
                            "train_edge_kl": float(m.group(3)),
                        }
        except FileNotFoundError:
            last = None

        if last:
            out = self._log_dir / self.metrics_filename
            write_header = not out.exists()
            with open(out, "a", newline="") as g:
                w = csv.DictWriter(g, fieldnames=["train_nll","train_atom_kl","train_edge_kl"])
                if write_header: w.writeheader()
                w.writerow(last)

        # (2) Validity/Uniq/Novelty 블록 → train 전용 CSV
        blocks = _parse_sampling_blocks_from_capture(cap)
        if blocks:
            out2 = self._log_dir / self.sampling_filename
            need_header = not out2.exists()
            with open(out2, "a", newline="") as g:
                w = csv.DictWriter(g, fieldnames=_SAMPLING_HEADERS)
                if need_header: w.writeheader()
                for b in blocks:
                    row = {k: b.get(k, "") for k in _SAMPLING_HEADERS}
                    w.writerow(row)
            print(f"[OK] saved train sampling metrics(pre-custom) → {out2}")

# CSV 헤더(항상 이 순서로 기록)
_SAMPLING_HEADERS = [
    "seq",
    "validity_n","validity_pct",
    "conn_n","conn_min","conn_mean","conn_max",
    "relaxed_validity_n","relaxed_validity_pct",
    "uniq_n_valid","uniq_pct",
    "nov_n_unique_valid","nov_pct",
]

_PAT_STABILITY = re.compile(r"Stability metrics:.*?\[([^\]]+)\]", re.IGNORECASE)

def _parse_sampling_blocks_from_capture(path: Path) -> list[dict]:
    """
    stdout에서 pre-custom 샘플링 메트릭(Validity/Conn/Relaxed/Uniq/Nov)만 튼튼하게 파싱.
    - \r → \n, ANSI 제거
    - 'Generated graphs Saved. Computing sampling metrics...' 를 기준으로 청크 분리
    - 각 청크에서 값 추출. 누락되면 'Stability metrics: [...]' 라인으로 폴백 보완
    """
    if not path.exists():
        return []

    # 1) 정규화
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)   # ANSI color 제거
    text = text.replace("\r", "\n")              # tqdm 한 줄 갱신을 줄바꿈으로

    # 2) 샘플링 청크 분리
    chunks = re.split(r"Generated graphs Saved\. Computing sampling metrics\.\s*", text)
    chunks = chunks[1:]  # 첫 블록은 마커 이전이므로 제거

    blocks: list[dict] = []
    for ch in chunks:
        # pre-custom 부분만 남김 (custom 시작 전까지)
        ch_precustom = re.split(r"Starting custom metrics", ch, maxsplit=1)[0]

        cur: dict = {}

        # (a) 직접 매칭
        m = _PAT_VALID.search(ch_precustom)
        if m:
            cur["validity_n"]   = int(m.group(1))
            cur["validity_pct"] = float(m.group(2))

        m = _PAT_CONN.search(ch_precustom)
        if m:
            cur["conn_n"]   = int(m.group(1))
            cur["conn_min"] = float(m.group(2))
            cur["conn_mean"]= float(m.group(3))
            cur["conn_max"] = float(m.group(4))

        m = _PAT_REL_VALID.search(ch_precustom)
        if m:
            cur["relaxed_validity_n"]   = int(m.group(1))
            cur["relaxed_validity_pct"] = float(m.group(2))

        m = _PAT_UNIQ.search(ch_precustom)
        if m:
            cur["uniq_n_valid"] = int(m.group(1))
            cur["uniq_pct"]     = float(m.group(2))

        m = _PAT_NOV.search(ch_precustom)
        if m:
            cur["nov_n_unique_valid"] = int(m.group(1))
            cur["nov_pct"]            = float(m.group(2))

        # (b) 폴백: Validity 라인이 특수한 이유로 인식이 안되면 Stability metrics에서 보완
        if ("validity_pct" not in cur) or ("validity_n" not in cur):
            ms = _PAT_STABILITY.search(ch_precustom)
            # Stability metrics: [valid, relaxed, uniq, nov]  (0~1 범위)
            if ms:
                try:
                    vals = [float(x.strip()) for x in ms.group(1).split(",")]
                    if len(vals) >= 1:
                        cur["validity_pct"] = round(vals[0] * 100, 2)
                    # n은 conn_n과 동일하므로 있으면 재사용, 없으면 uniq/relaxed n 중 하나로 보정
                    if "validity_n" not in cur:
                        if "conn_n" in cur:
                            cur["validity_n"] = cur["conn_n"]
                        elif "relaxed_validity_n" in cur:
                            cur["validity_n"] = cur["relaxed_validity_n"]
                except Exception:
                    pass

        if cur:
            blocks.append(cur)

    # 3) 일련번호(seq)
    for i, b in enumerate(blocks, 1):
        b["seq"] = i
    return blocks

# ─────────────────────────────────────────────────────────────────────────────
# 유틸: 장치 해석
def resolve_devices(gpus_cfg):
    n = torch.cuda.device_count()
    if n == 0:
        return 0
    if isinstance(gpus_cfg, int) and not isinstance(gpus_cfg, bool):
        k = int(gpus_cfg)
        if k <= 0: return 0
        if k == 1: return 1
        return list(range(min(k, n)))
    if isinstance(gpus_cfg, (list, tuple)):
        return [int(i) for i in gpus_cfg if 0 <= int(i) < n]
    if gpus_cfg is True or gpus_cfg == -1 or str(gpus_cfg).lower() == "auto" or gpus_cfg is None:
        return list(range(n))
    return 1


def build_trainer(cfg, callbacks, name: str):
    use_gpu = bool(cfg.general.gpus) and torch.cuda.is_available()
    dev_spec = resolve_devices(cfg.general.gpus) if use_gpu else 0
    world = len(dev_spec) if isinstance(dev_spec, (list, tuple)) else int(dev_spec)

    if not use_gpu or world <= 1:
        devices = 1
        strategy = "auto"
        backend = None
    else:
        devices = dev_spec
        strategy, backend = choose_ddp_strategy(devices, find_unused=True)

    print(f"[Dist] devices={devices}, strategy={strategy if isinstance(strategy,str) else 'DDPStrategy'} "
          f"backend={backend or 'single'} "
          f"GLOO_DEVICE_TRANSPORT={os.environ.get('GLOO_DEVICE_TRANSPORT')} "
          f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}")

    csv_logger = CSVLogger(save_dir=os.getcwd(), name="pl_logs")
    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=devices,
        strategy=strategy,
        precision=getattr(cfg.trainer, "precision", "32-true") if hasattr(cfg, "trainer") else "32-true",
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        gradient_clip_val=cfg.train.clip_grad,
        fast_dev_run=(name == 'debug'),
        enable_progress_bar=True,
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=[csv_logger],
    )
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# (A) edge_attr 6채널 → DiGress 5채널(one-hot) 변환
def to_digress_edge5(data):
    """
    csv_spectrum_dataset.py의 edge_attr:
      [:,0:4] = bond type one-hot (single,double,triple,aromatic)
      [:,4]   = conjugated (0/1)
      [:,5]   = in_ring (0/1)
    → DiGress 규약: [no-bond, single, double, triple, aromatic] (5채널)
      실제 edge에는 no-bond가 0, 그 외는 one-hot(+1 쉬프트)로 인코딩
    """
    if getattr(data, "edge_attr", None) is None or data.edge_attr.numel() == 0:
        data.edge_attr = torch.zeros((0, 5), dtype=torch.float32, device=data.x.device)
        return data
    types4 = data.edge_attr[:, :4]                     # (m, 4)
    t_idx = types4.argmax(dim=-1)                      # 0..3
    data.edge_attr = F.one_hot(t_idx + 1, num_classes=5).to(torch.float32)
    return data

# ─────────────────────────────────────────────────────────────────────────────
def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    # test_only 경로는 그대로 두되 이름만 조정
    cfg.general.name = cfg.general.name + '_resume'
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, None

def get_resume_adaptive(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    # resume 경로 절대화(있으면)
    if cfg.general.resume:
        current_path = os.path.dirname(os.path.realpath(__file__))
        root_dir = current_path.split('outputs')[0]
        cfg.general.resume = os.path.join(root_dir, cfg.general.resume)

    # 여기서는 모델 로드 금지(로드하면 shape mismatch로 터짐)
    cfg.general.name = cfg.general.name + '_resume'
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, None

def _find_hydra_run_dir_by_name(outputs_root: Path, run_name: str) -> Optional[Path]:
    """
    Hydra 기본 구조: outputs/YYYY-MM-DD/HH-MM-SS-<run_name>/
    같은 이름(run_name)을 가진 가장 최신 런 디렉토리를 돌려준다.
    """
    if not outputs_root.exists():
        return None
    # "*-<run_name>" 패턴으로 재귀 탐색
    cands = [p for p in outputs_root.rglob(f"*-{run_name}") if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def _copy_to_pl_logs(outputs_root: Path, run_name: str, src: Path, dst_name: Optional[str] = None) -> Optional[Path]:
    """
    src 파일을 해당 run_name의 pl_logs/version_0/ 아래로 복사.
    """
    run_dir = _find_hydra_run_dir_by_name(outputs_root, run_name)
    if not run_dir:
        print(f"[WARN] hydra run dir not found for name={run_name} under {outputs_root}")
        return None
    pl_dir = run_dir / "pl_logs" / "version_0"
    pl_dir.mkdir(parents=True, exist_ok=True)
    dst = pl_dir / (dst_name or src.name)
    try:
        shutil.copy2(src, dst)
        print(f"[OK] copied {src.name} → {dst}")
        return dst
    except Exception as e:
        print(f"[WARN] failed to copy {src} to {dst}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # ── (1) DataModule/Infos: csvspec만 처리 ────────────────────────────────
    assert cfg.dataset.name in ['csvspec', 'csv_spec', 'csv_spectrum'], \
        f"dataset.name should be 'csvspec' (got {cfg.dataset.name})"

    dm = CSVSpecDataModule(
        cfg,
        train_csv=cfg.dataset.train_csv,
        val_csv=cfg.dataset.val_csv,
        test_csv=cfg.dataset.test_csv,
        smiles_col=getattr(cfg.dataset, "smiles_col", None),
        inchi_col=getattr(cfg.dataset, "inchi_col", "InChI"),
        spectrum_start=getattr(cfg.dataset, "spectrum_start", 200),
        spectrum_end=getattr(cfg.dataset, "spectrum_end", 800),
        global_cols=getattr(cfg.dataset, "global_cols", []),
        spectrum_fill_eps=getattr(cfg.dataset, "spectrum_fill_eps", 1e-8),
        fixed_vocabs=getattr(cfg.dataset, "fixed_vocabs", {}),
        boolean_cols=getattr(cfg.dataset, "boolean_cols", []),
        add_h=getattr(cfg.dataset, "add_h", False),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    infos = CSVSpecInfos(dm, remove_h=getattr(cfg.dataset, "remove_h", False))

    # extra/domain features
    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    # (선택) 모델 차원 계산 훅
    infos.compute_input_output_dims(datamodule=dm, extra_features=extra_features, domain_features=domain_features)

    # src/main_spec.py, dm/infos 만든 직후
    print(f"[DIM] train y_dim={dm.train_dataset.y_dim}, "
          f"val y_dim={getattr(dm.val_dataset, 'y_dim', None)}, "
          f"test y_dim={getattr(dm.test_dataset, 'y_dim', None)}")
    print(
        f"[DIM] spectrum range={getattr(cfg.dataset, 'spectrum_start', 200)}..{getattr(cfg.dataset, 'spectrum_end', 800)}, "
        f"global_cols={getattr(cfg.dataset, 'global_cols', [])}")

    # metrics / viz
    train_metrics = (TrainMolecularMetricsDiscrete(infos)
                     if cfg.model.type == 'discrete' else
                     TrainMolecularMetrics(infos))
    visualization_tools = MolecularVisualization(getattr(cfg.dataset, "remove_h", False), dataset_infos=infos)

    # train SMILES 수집해서 novelty 등의 기준 세트로 사용
    def _collect_train_smiles(dm):
        ds = getattr(dm, "train_dataset", None)
        if ds is None:
            return None
        base = getattr(ds, "dataset", ds)  # Subset 대비
        smi = []
        for i in range(len(base)):
            g = base[i]
            s = getattr(g, "smiles", None)
            if isinstance(s, (list, tuple)):
                smi.extend([si for si in s if isinstance(si, str) and si])
            elif isinstance(s, str) and s:
                smi.append(s)
        return smi or None

    train_smiles = _collect_train_smiles(dm)
    print(f"[INFO] collected {0 if train_smiles is None else len(train_smiles)} train SMILES for novelty")
    sampling_metrics = SamplingMolecularMetrics(infos, train_smiles)

    model_kwargs = {
        'dataset_infos': infos,
        'train_metrics': train_metrics,
        'sampling_metrics': sampling_metrics,
        'visualization_tools': visualization_tools,
        'extra_features': extra_features,
        'domain_features': domain_features,
        'y_loss_mode': getattr(cfg.train, "y_loss_mode", "none"),
    }

    # ── (2) resume/test-only 처리 ───────────────────────────────────────────
    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, model_kwargs)
        # (테스트 전용에서 모든 ckpt를 훑어볼 때만 필요)
        # if getattr(cfg.general, "evaluate_all_checkpoints", False):
        #     os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        # ★중요: 학습 재시작에서는 절대로 chdir 하지 마세요.

    # ── (3) 폴더 생성 및 모델 빌드 ──────────────────────────────────────────
    utils.create_folders(cfg)
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs) if cfg.model.type == 'discrete' \
            else LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    ckpt_for_fit = None

    def _load_weights_flex(model, ckpt_path):
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # (구버전 Torch는 weights_only 인자를 모를 수 있음)
            ckpt = torch.load(ckpt_path, map_location="cpu")
        # 옵티마이저/루프가 있으면 완전 재개 가능
        has_optimizer = (
                isinstance(ckpt, dict)
                and "optimizer_states" in ckpt
                and ckpt["optimizer_states"] is not None
                and (not isinstance(ckpt["optimizer_states"], (list, tuple)) or len(ckpt["optimizer_states"]) > 0)
        )

        if has_optimizer:
            # 완전 재개는 Lightning에 맡김(ckpt_path로 fit에 넘김)
            return ckpt_path, None
        else:
            qprint(f"[resume] no optimizer state in ckpt → weights-only transfer load: {ckpt_path}")

        rank = int(os.environ.get("LOCAL_RANK", "0"))
        qprint(f"[resume] rank{rank} → ckpt_for_fit={ckpt_for_fit}")

        # weights-only → 가중치만 부분 로드
        sd = ckpt.get("state_dict", ckpt)

        # 1) 데이터셋 따라 달라지는 통계 버퍼/로그는 무조건 스킵
        DROP_PREFIXES = ("sampling_metrics.", "train_metrics.")
        sd = {k: v for k, v in sd.items() if not k.startswith(DROP_PREFIXES)}

        # 2) 1차 시도
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            qprint("[resume] weights-only loaded with strict=False",
                   "\n  missing:", missing, "\n  unexpected:", unexpected)
        except RuntimeError as e:
            # 3) y 차원 바뀐 경우 헤드(mlp_in_y)만 제외하고 재시도
            qprint("[resume] retrying without model.mlp_in_y.* due to:", e)
            sd2 = {k: v for k, v in sd.items() if not k.startswith("model.mlp_in_y.")}
            missing, unexpected = model.load_state_dict(sd2, strict=False)
            qprint("[resume] reloaded without mlp_in_y.*",
                   "\n  missing:", missing, "\n  unexpected:", unexpected)

        # weights만 올렸으므로 fit/test에 ckpt_path는 넘기지 않음
        return None, None

    # resume 또는 test_only 경로가 있으면 불러오기
    ckpt_input = cfg.general.resume or cfg.general.test_only
    if ckpt_input:
        ckpt_for_fit, _ = _load_weights_flex(model, ckpt_input)

    callbacks = []
    if cfg.train.save_model:
        periodic_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="ep{epoch:03d}",
            monitor=None,
            every_n_epochs=5,
            save_top_k=-1,
            save_last=False,
            save_weights_only=True,
            save_on_train_epoch_end=True,  # 유지
        )

        # 검증이 없을 수도 있으니 best_ckpt는 옵션 취급(있어도 무방)
        best_ckpt = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}-{val_epoch_NLL:.3f}",
            monitor="val/epoch_NLL",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_weights_only=True,
            # 검증이 없으면 어차피 동작 안 함
        )

        # ★ 중요: last.ckpt를 'train 에폭 종료 시'에도 저장하도록
        last_cb = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="last",
            save_top_k=1,
            monitor=None,  # 모니터 없이
            every_n_epochs=1,
            save_weights_only=True,
            save_on_train_epoch_end=True,  # ★ 추가
        )

        callbacks += [best_ckpt, periodic_ckpt, last_cb]
        callbacks.append(TestMetricsCSVCallback(filename="test_metrics.csv"))
        callbacks.append(TrainCaptureCSVCallback())

    if cfg.train.ema_decay > 0:
        callbacks.append(utils.EMA(decay=cfg.train.ema_decay))

    if cfg.general.name == 'debug':
        print("[WARNING] run name is 'debug' → fast_dev_run enabled")

    trainer = build_trainer(cfg, callbacks, cfg.general.name)

    # ── (4) fit/test ────────────────────────────────────────────────────────
    if not cfg.general.test_only:
        # 재개 가능하면 ckpt_for_fit=경로, 아니면 None(트랜스퍼)
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_for_fit)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=dm)  # ← ckpt_path 전달 금지
    else:
        # test-only: 위에서 이미 가중치 주입했으므로 ckpt_path 없이 테스트
        trainer.test(model, datamodule=dm)  # ← ckpt_path 전달 금지

        # (옵션) 여러 ckpt를 순회 평가하고 싶다면, 각 ckpt를 수동 주입 후 test 호출
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            for file in os.listdir(directory):
                if file.endswith('.ckpt'):
                    ckpt_path = os.path.join(directory, file)
                    print("Loading checkpoint (weights-only test)", ckpt_path)
                    # 동일 모델에 가중치만 다시 주입
                    _ckpt_fit, _ = _load_weights_flex(model, ckpt_path)
                    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
