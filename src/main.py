# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수는 torch import "전에" 최소한만: 루프백 + uv 권장(Windows)
import os
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
# 루프백 NIC 이름은 시스템마다 다를 수 있으나 보통 아래 이름.
# 다른 이름이면 실행 전 외부에서 GLOO_SOCKET_IFNAME 환경변수로 덮어써도 됨.
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")
# ─────────────────────────────────────────────────────────────────────────────

import pathlib
import warnings

import torch
torch.cuda.empty_cache()

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

from src.dist_utils import choose_ddp_strategy
from pytorch_lightning.callbacks import Callback

# ─────────────────────────────────────────────────────────────────────────────
# 유틸: 장치 해석 + GPU 모니터(선택)
def resolve_devices(gpus_cfg):
    n = torch.cuda.device_count()
    if n == 0:
        return 0

    if isinstance(gpus_cfg, int) and not isinstance(gpus_cfg, bool):
        k = int(gpus_cfg)
        if k <= 0:
            return 0
        if k == 1:
            return 1
        return list(range(min(k, n)))

    if isinstance(gpus_cfg, (list, tuple)):
        return [int(i) for i in gpus_cfg if 0 <= int(i) < n]

    if gpus_cfg is True or gpus_cfg == -1 or str(gpus_cfg).lower() == "auto" or gpus_cfg is None:
        return list(range(n))
    return 1

class GpuUsageMonitor(Callback):
    def __init__(self, interval=50): self.interval = interval; self.nvml=None; self.handle=None
    def setup(self, trainer, pl_module, stage=None):
        try:
            import pynvml
            pynvml.nvmlInit(); self.nvml=pynvml
            idx = pl_module.device.index if pl_module.device.type=="cuda" else torch.cuda.current_device()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception as e:
            print(f"[GPU-MON] NVML unavailable → util% skip: {e}")
    def on_train_start(self, trainer, pl_module):
        dev = pl_module.device
        try:
            pdev = next(pl_module.parameters()).device
        except StopIteration:
            pdev = dev
        print(f"[GPU-MON] training device={dev} first_param={pdev}")
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.interval != 0: return
        if not torch.cuda.is_available() or pl_module.device.type != "cuda":
            print(f"[GPU-MON] step={trainer.global_step} CPU"); return
        idx = pl_module.device.index if pl_module.device.index is not None else torch.cuda.current_device()
        mem = torch.cuda.memory_allocated(idx) / 1e9
        mem_max = torch.cuda.max_memory_allocated(idx) / 1e9
        msg = f"[GPU-MON] step={trainer.global_step} gpu{idx} mem={mem:.2f}GB (max {mem_max:.2f}GB)"
        if self.nvml and self.handle:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            meminfo = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            msg += f" util={util.gpu}% nvml_mem={meminfo.used/1e9:.2f}/{meminfo.total/1e9:.2f}GB"
        print(msg)
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(cfg, callbacks, name: str):
    use_gpu = bool(cfg.general.gpus) and torch.cuda.is_available()
    print(use_gpu)
    # 1) 요청 장치 해석
    requested = cfg.general.gpus
    print(requested)
    dev_spec = resolve_devices(requested) if use_gpu else 0  # ex) [0,1] or [0] or 1
    print(dev_spec)
    # 2) world 계산
    world = len(dev_spec) if isinstance(dev_spec, (list, tuple)) else int(dev_spec)
    print(world)
    # 3)
    if not use_gpu or world <= 1:
        devices = 1 if use_gpu else 1          # GPU 1장 or CPU 1개
        strategy = "auto"                      # 절대 DDP 안 켜짐
        backend = None
    else:
        devices = dev_spec                     # ex) [0,1,2,3]
        strategy, backend = choose_ddp_strategy(devices, find_unused=True)

    print(f"[Dist] devices={devices}, strategy={strategy if isinstance(strategy,str) else 'DDPStrategy'} "
          f"backend={backend or 'single'} "
          f"GLOO_DEVICE_TRANSPORT={os.environ.get('GLOO_DEVICE_TRANSPORT')} "
          f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}")

    trainer = Trainer(
        accelerator='gpu' if use_gpu else 'cpu',
        devices=devices,                       # ← 단일이면 정수 1
        strategy=strategy,                     # ← 단일이면 'auto'
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

    try:
        init_m = getattr(trainer.strategy, "_init_method", None)
        if init_m: print(f"[Dist] init_method={init_m}")
    except Exception:
        pass

    return trainer

# ─────────────────────────────────────────────────────────────────────────────

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
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None
        elif dataset_config["name"] == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                                extra_features=extra_features,
                                                domain_features=domain_features)

        train_metrics = (TrainMolecularMetricsDiscrete(dataset_infos)
                         if cfg.model.type == 'discrete' else
                         TrainMolecularMetrics(dataset_infos))

        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError(f"Unknown dataset {cfg['dataset']}")

    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = [GpuUsageMonitor(interval=1)]
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='{epoch}',
            monitor='val/epoch_NLL',
            save_top_k=5,
            mode='min',
            every_n_epochs=1
        )
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='last',
            every_n_epochs=1
        )
        callbacks += [last_ckpt_save, checkpoint_callback]

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run name is 'debug' → fast_dev_run enabled")

    trainer = build_trainer(cfg, callbacks, name)

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
