# These imports are tricky because they use C++, do not move them
from rdkit import Chem

import os
import pathlib
import warnings

import torch
# import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.datasets import guacamol_dataset, qm9_dataset, moses_dataset
from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    chains_to_save = cfg.general.chains_to_save

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
    new_cfg = cfg.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


#def setup_wandb(cfg):
#    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
#    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
#              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
#    wandb.init(**kwargs)
#    wandb.save('*.txt')
#    return cfg
def setup_wandb(cfg):
    return cfg

@hydra.main(version_base='1.1', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] == 'qm9':
        datamodule = qm9_dataset.QM9DataModule(cfg)
        dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        datamodule.prepare_data()
        train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                    dataset_infos=dataset_infos, evaluate_dataset=False)
    elif dataset_config["name"] == 'guacamol':
        datamodule = guacamol_dataset.GuacamolDataModule(cfg)
        dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
        datamodule.prepare_data()
        train_smiles = None
    elif dataset_config["name"] == 'moses':
        datamodule = moses_dataset.MOSESDataModule(cfg)
        dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
        datamodule.prepare_data()
        train_smiles = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_config['name']}")

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

    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                         filename='last', every_n_epochs=1)
        callbacks += [checkpoint_callback, last_ckpt_save]

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    if torch.cuda.is_available() and cfg.general.gpus > 0:
        accelerator = 'gpu'
        devices = cfg.general.gpus
        strategy = 'ddp' if cfg.general.gpus > 1 else 'auto'
    else:
        accelerator = 'cpu'
        devices = 1  # CPU에서는 최소 1 이상이어야 함
        strategy = 'auto'

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator=accelerator,
                      devices=devices,
                      limit_train_batches=20 if cfg.general.name == 'test' else None,
                      limit_val_batches=20 if cfg.general.name == 'test' else None,
                      limit_test_batches=20 if cfg.general.name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy=strategy,
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      logger=[])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            files_list = os.listdir(directory)
            for file in files_list:
                if file.endswith('.ckpt') and os.path.join(directory, file) != cfg.general.test_only:
                    print("Loading checkpoint", file)
                    trainer.test(model, datamodule=datamodule, ckpt_path=os.path.join(directory, file))


if __name__ == '__main__':
    main()
