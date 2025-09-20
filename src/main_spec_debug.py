# main_spec_debug.py
# -*- coding: utf-8 -*-
"""
디버그 전용 엔트리:
- DataLoader: train/test 모두 shuffle=False(원본 순서), batch_size=1
- 학습 1 epoch만 수행
- GPU=1(가능하면), DDP 비활성화
- 에러는 기록하고(./_bad_batches/bad_rank*.csv), 해당 배치는 0-loss로 스킵
- val 사용 안 함
"""

import os
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

import warnings
from typing import List

import torch
from torch_geometric.loader import DataLoader as GeoLoader

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import CSVLogger

from src import utils

# ▼ 디버그 모듈(에러 스킵) 사용
from diffusion_model_discrete_debug import DiscreteDenoisingDiffusionDebug as DiscreteDenoisingDiffusion
# 만약 파일명이 diffusion_model_descrete_debug.py 라면:
# from diffusion_model_descrete_debug import DiscreteDenoisingDiffusionDebug as DiscreteDenoisingDiffusion

from diffusion_model import LiftedDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization
from src.datasets.csvspec_module import CSVSpecDataModule, CSVSpecInfos

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def _force_debug_overrides(cfg: DictConfig):
    """GPU=1, batch_size=1, n_epochs=1 강제 + val 비활성."""
    cfg.general.gpus = 1
    cfg.train.batch_size = 1
    cfg.train.n_epochs = 1
    cfg.train.save_model = False  # 디버그에선 저장 끔
    cfg.general.samples_to_generate = 4
    cfg.general.final_model_samples_to_generate = 0
    cfg.general.chains_to_save = 0
    cfg.general.sample_every_val = 999_999
    cfg.general.check_val_every_n_epochs = 999_999
    cfg.dataset.val_csv = None  # ★ val 없음


def _patch_ordered_loaders(dm: CSVSpecDataModule):
    """train/test DataLoader를 원본 순서/배치1로 강제."""
    def _mk(ds):
        return GeoLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if getattr(dm, "train_dataset", None) is not None:
        dm.train_dataloader = (lambda self=dm: _mk(self.train_dataset)).__get__(dm, type(dm))
    if getattr(dm, "test_dataset", None) is not None:
        dm.test_dataloader = (lambda self=dm: _mk(self.test_dataset)).__get__(dm, type(dm))
    if hasattr(dm, "batch_size"):
        dm.batch_size = 1


def _build_trainer(cfg: DictConfig) -> Trainer:
    """단일 장치, DDP 비활성."""
    use_gpu = torch.cuda.is_available() and int(cfg.general.gpus) == 1
    return Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1,
        strategy="auto",
        deterministic=True,
        precision=getattr(cfg.trainer, "precision", "32-true") if hasattr(cfg, "trainer") else "32-true",
        max_epochs=cfg.train.n_epochs,   # 1 epoch
        logger=[CSVLogger(save_dir=os.getcwd(), name="pl_logs_debug")],
        log_every_n_steps=1,
        enable_progress_bar=True,
    )


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # 1) 오버라이드 적용
    _force_debug_overrides(cfg)

    # 2) DataModule & Infos
    assert cfg.dataset.name in ['csvspec', 'csv_spec', 'csv_spectrum'], \
        f"dataset.name should be 'csvspec' (got {cfg.dataset.name})"

    dm = CSVSpecDataModule(
        cfg,
        train_csv=cfg.dataset.train_csv,
        val_csv=None,  # ★ 명시적으로 val 비활성
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
        batch_size=1,
        num_workers=0,
    )
    infos = CSVSpecInfos(dm, remove_h=getattr(cfg.dataset, "remove_h", False))

    # 3) features/metrics/viz
    if cfg.model.type == 'discrete' and getattr(cfg.model, "extra_features", None) is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=infos)
        train_metrics = TrainMolecularMetricsDiscrete(infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        train_metrics = TrainMolecularMetrics(infos)

    # 4) 입력/출력 차원 계산
    infos.compute_input_output_dims(datamodule=dm, extra_features=extra_features, domain_features=domain_features)
    print(f"[DIM] train y_dim={dm.train_dataset.y_dim}, test y_dim={getattr(dm.test_dataset, 'y_dim', None)}")
    print(f"[DIM] spectrum range={getattr(cfg.dataset,'spectrum_start',200)}.."
          f"{getattr(cfg.dataset,'spectrum_end',800)}, "
          f"global_cols={getattr(cfg.dataset,'global_cols',[])}")

    visualization = MolecularVisualization(getattr(cfg.dataset, "remove_h", False), dataset_infos=infos)
    sampling_metrics = SamplingMolecularMetrics(infos, train_smiles=None)

    # 5) 폴더 생성 및 모델 빌드
    utils.create_folders(cfg)
    model = (DiscreteDenoisingDiffusion if cfg.model.type == 'discrete' else LiftedDenoisingDiffusion)(
        cfg=cfg,
        dataset_infos=infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization,
        extra_features=extra_features,
        domain_features=domain_features,
        y_loss_mode=getattr(cfg.train, "y_loss_mode", "none"),
    )

    # 6) DataLoader 순서/배치1 강제
    _patch_ordered_loaders(dm)

    # 7) 트레이너 & 실행
    trainer = _build_trainer(cfg)
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # val 없음 → 필요 시 test 수행
    if getattr(cfg.dataset, "test_csv", None):
        trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()


