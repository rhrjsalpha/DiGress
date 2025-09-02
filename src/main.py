import os
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
import torch, os

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29511"            # 충돌 시 숫자만 바꿔주세요
os.environ["GLOO_DEVICE_TRANSPORT"] = "uv"
os.environ["GLOO_SOCKET_IFNAME"] = "Ethernet"  # 1)에서 바꾼 이름; 안 바꿨다면 "이더넷"

def resolve_devices(gpus_cfg):
    import torch, os, platform
    n = torch.cuda.device_count()
    if n == 0:
        return 0  # CPU
    if gpus_cfg in (-1, "auto", None):
        return list(range(n))              # 모든 GPU
    if isinstance(gpus_cfg, int):
        return list(range(min(gpus_cfg, n)))  # 앞에서 n개
    if isinstance(gpus_cfg, (list, tuple)):
        # 유효 ID만 남기기 (CUDA_VISIBLE_DEVICES 쓰면 재매핑된 인덱스 기준)
        return [int(i) for i in gpus_cfg if int(i) < n]
    return 1

class GpuUsageMonitor(Callback):
    def __init__(self, interval=50):
        self.interval = interval
        self.nvml = None
        self.handle = None

    def setup(self, trainer, pl_module, stage=None):
        # NVML(=nvidia-smi API) 초기화
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            # 현재 랭크가 쓰는 로컬 GPU 인덱스
            idx = pl_module.device.index if pl_module.device.type == "cuda" else torch.cuda.current_device()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception as e:
            print(f"[GPU-MON] NVML unavailable → util%는 생략합니다: {e}")

    def on_train_start(self, trainer, pl_module):
        dev = pl_module.device
        is_cuda = (dev.type == "cuda")
        print(f"[GPU-MON] training device={dev} is_cuda={is_cuda}")
        # 파라미터가 실제 CUDA에 올라갔는지(DDP면 on_train_start 시점에 CUDA로 이동)
        try:
            pdev = next(pl_module.parameters()).device
            print(f"[GPU-MON] first parameter device={pdev}")
            if pdev.type != "cuda":
                print("[GPU-MON][WARN] 모델 파라미터가 CUDA가 아닙니다.")
        except StopIteration:
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.interval != 0:
            return
        if not torch.cuda.is_available() or pl_module.device.type != "cuda":
            print(f"[GPU-MON] step={trainer.global_step} CPU mode")
            return

        idx = pl_module.device.index if pl_module.device.index is not None else torch.cuda.current_device()
        mem = torch.cuda.memory_allocated(idx) / 1e9
        mem_max = torch.cuda.max_memory_allocated(idx) / 1e9
        print(f"[GPU-MON] step={trainer.global_step} gpu{idx} mem={mem:.2f}GB (max {mem_max:.2f}GB)")

        # NVML 이용해 실시간 Util(%)
        if self.nvml and self.handle:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            meminfo = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            print(f"[GPU-MON] util={util.gpu}% mem_used={meminfo.used/1e9:.2f}GB/{meminfo.total/1e9:.2f}GB")

def build_trainer(cfg, callbacks, name: str):
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    # devices = cfg.general.gpus if use_gpu else 1
    devices = resolve_devices(cfg.general.gpus) if use_gpu else 0  # ← 변경

    # old ckpt 호환을 위해 find_unused_parameters=True 권장
    # (기존에 strategy="ddp_find_unused_parameters_true" 쓰시던 부분을 대체)
    strategy, backend = choose_ddp_strategy(devices, find_unused=True)
    print(len(devices))
    if len(devices) <= 1:
        strategy = "auto" # auto, ddp, ddp_spawn, deepspeed
        backend = None

    print(f"[Dist] devices={devices}, backend={backend or 'single'} "
          f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
          f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
          f"GLOO_DEVICE_TRANSPORT={os.environ.get('GLOO_DEVICE_TRANSPORT')} "
          f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}")

    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy=strategy,                                 # None | DDPStrategy
        accelerator='gpu' if use_gpu else 'cpu',
        devices=devices,
        precision=getattr(cfg.trainer, "precision", "32-true") if hasattr(cfg, "trainer") else "32-true",
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=(name == 'debug'),
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != 'debug' else 1,
        logger=[],
    )
    return trainer

def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
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
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
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

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
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

        elif dataset_config.name == 'moses':
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

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    callbacks.append(GpuUsageMonitor(interval=1))  # 50배치마다 출력
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)


    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    #use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()



    trainer = build_trainer(cfg, callbacks, name)
    #trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
    #                  strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
    #                  accelerator='gpu' if use_gpu else 'cpu',
    #                  devices=cfg.general.gpus if use_gpu else 1,
    #                  max_epochs=cfg.train.n_epochs,
    #                  check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
    #                  fast_dev_run=cfg.general.name == 'debug',
    #                  enable_progress_bar=False,
    #                  callbacks=callbacks,
    #                  log_every_n_steps=50 if name != 'debug' else 1,
    #                  logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
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
