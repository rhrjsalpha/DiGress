# These imports are tricky because they use c++, do not move them
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import pathlib
import warnings

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.datasets import guacamol_dataset, qm9_dataset, moses_dataset
from src.datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, SpectreDatasetInfos
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization, NonMolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    # 이전에 저장된 체크포인트에서 모델과 설정(config)을 복원 #
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    chains_to_save = cfg.general.chains_to_save

    # 지정된 체크포인트로부터 diffusion 모델을 불러옴 (이전 학습 결과)
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.chains_to_save = chains_to_save
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    # 기존 config를 불러오되 일부 항목만 현재 값으로 덮어씀

    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    # 지정된 체크포인트로부터 diffusion 모델을 불러옴 (이전 학습 결과)
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg: # 현재 cfg에 있는 모든 키들을 복원된 new_cfg에 덮어씀
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume' # 실행 이름에 _resume을 붙이고, 재개 경로 저장

    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg) # 복원된 config에 빠진 새로운 키들을 현재 config에서 가져와 추가함 (오류 방지)
    return new_cfg, model


def setup_wandb(cfg):
    # 현재 구성(config)을 기반으로 wandb 프로젝트를 초기화하고 .txt 파일들을 자동 저장하도록 설정
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) # OmegaConf 객체(cfg)를 일반 Python 딕셔너리로 변환
    # resolve=True: ${var} 형태의 변수 참조를 실제 값으로 평가
    # throw_on_missing=True: 빠진 항목이 있으면 오류 발생

    # 항목	                     설명
    # name	                     wandb에서 이 실행(run)의 이름
    # project	                 wandb 프로젝트 이름 (graph_ddm_qm9 등)
    # config	                 현재 Hydra config 전체를 wandb에 기록
    # settings._disable_stats	 시스템 자원 통계 수집 비활성화 (성능 최적화용)
    # reinit=True	             여러 번 wandb.init() 호출 시에도 재초기화 허용
    # mode	"online", "offline" 또는 "disabled" 가능 (cfg.general.wandb)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    ##### 데이터셋 준비 #####
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm-20', 'planar']: # SPECTRE 기반 비분자 그래프
        if dataset_config['name'] == 'sbm':
            datamodule = SBMDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
        elif dataset_config['name'] == 'comm-20':
            datamodule = Comm20DataModule(cfg)
            sampling_metrics = Comm20SamplingMetrics(datamodule.dataloaders)
        else:
            datamodule = PlanarDataModule(cfg)
            sampling_metrics = PlanarSamplingMetrics(datamodule.dataloaders)

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
        print(dataset_infos.output_dims)
        assert False

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']: # 분자 그래프 데이터셋
        if dataset_config["name"] == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            datamodule.prepare_data()
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            datamodule.prepare_data()
            train_smiles = None

        elif dataset_config.name == 'moses':
            datamodule = moses_dataset.MOSESDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            datamodule.prepare_data()
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

    ##### 학습/테스트 재개 처리 #####
    # test_only: 테스트만 수행 (고정 config)
    # resume: 학습 재개 (config 수정 가능)
    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    ##### WandB 설정 #####
    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    ##### 모델 생성 #####
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    ##### 체크포인트 및 EMA 콜백 설정 #####
    callbacks = []

    # "val/epoch_NLL" 값이 가장 낮을 때의 모델을 저장
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(last_ckpt_save)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'test':
        print("[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    ##### Trainer 실행 #####
    print("cfg.general.gpus",cfg.general.name)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      logger=[])
    ##### 학습/테스트 실행 #####
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
