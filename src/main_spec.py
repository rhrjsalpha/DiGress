# src/main_spec.py
# -*- coding: utf-8 -*-
"""
스펙트럼(+ solvent_phase, pH 등) 조건으로 분자 생성 Diffusion 학습 엔트리
- CSVSpecDataset(당신이 올린 csv_spectrum_dataset.py) 그대로 사용
- edge_attr(6ch: 4타입+conj+ring) → DiGress 규약 5채널(one-hot: [no,single,double,trip,arom])로 변환
- DatasetInfos를 동봉하여 compute_input_output_dims까지 한 번에 연결
"""

# ─────────────────────────────────────────────────────────────────────────────
# Windows에서 DDP/Gloo 안전옵션 (torch import 전에)
import os
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

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

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

# 당신이 올린 데이터셋
from datasets.csv_spectrum_dataset import CSVSpecDataset
from src.datasets.csvspec_module import CSVSpecDataModule, CSVSpecInfos
warnings.filterwarnings("ignore", category=PossibleUserWarning)


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

    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=devices,
        strategy=strategy,
        precision=getattr(cfg.trainer, "precision", "32-true") if hasattr(cfg, "trainer") else "32-true",
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        gradient_clip_val=cfg.train.clip_grad,
        fast_dev_run=(name == 'debug'),
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=[],
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
# (D) resume helpers (기존 main과 동일)
def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'
    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


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

    # metrics / viz
    train_metrics = (TrainMolecularMetricsDiscrete(infos)
                     if cfg.model.type == 'discrete' else
                     TrainMolecularMetrics(infos))
    sampling_metrics = SamplingMolecularMetrics(infos, train_smiles=None)
    visualization_tools = MolecularVisualization(getattr(cfg.dataset, "remove_h", False), dataset_infos=infos)

    model_kwargs = {
        'dataset_infos': infos,
        'train_metrics': train_metrics,
        'sampling_metrics': sampling_metrics,
        'visualization_tools': visualization_tools,
        'extra_features': extra_features,
        'domain_features': domain_features,
    }

    # ── (2) resume/test-only 처리 ───────────────────────────────────────────
    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    # ── (3) 폴더 생성 및 모델 빌드 ──────────────────────────────────────────
    utils.create_folders(cfg)
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs) if cfg.model.type == 'discrete' \
            else LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        ckpt_cb = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='{epoch}',
            monitor='val/epoch_NLL',
            save_top_k=5,
            mode='min',
            every_n_epochs=1
        )
        last_cb = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='last',
            every_n_epochs=1
        )
        callbacks += [last_cb, ckpt_cb]

    if cfg.train.ema_decay > 0:
        callbacks.append(utils.EMA(decay=cfg.train.ema_decay))

    if cfg.general.name == 'debug':
        print("[WARNING] run name is 'debug' → fast_dev_run enabled")

    trainer = build_trainer(cfg, callbacks, cfg.general.name)

    # ── (4) fit/test ────────────────────────────────────────────────────────
    if not cfg.general.test_only:
        trainer.fit(model, datamodule=dm, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            for file in os.listdir(directory):
                if file.endswith('.ckpt'):
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
