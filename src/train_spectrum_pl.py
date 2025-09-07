# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수는 torch import "전에" 최소한만: 루프백 + uv 권장(Windows)
import os
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")
# ─────────────────────────────────────────────────────────────────────────────

import pathlib
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm

warnings.filterwarnings("ignore", category=PossibleUserWarning)
from pytorch_lightning.callbacks import Callback

class EpochPrinter(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        tr = m.get("train/mse")
        va = m.get("val/mse")
        tr_v = float(tr) if tr is not None else float("nan")
        va_v = float(va) if va is not None else float("nan")
        print(f"[epoch {trainer.current_epoch:03d}] train/mse={tr_v:.6f}  val/mse={va_v:.6f}")

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

class GpuUsageMonitor(pl.callbacks.Callback):
    def __init__(self, interval=50): self.interval = interval; self.nvml=None; self.handle=None
    def setup(self, trainer, pl_module, stage=None):
        try:
            import pynvml
            pynvml.nvmlInit(); self.nvml=pynvml
            idx = pl_module.device.index if pl_module.device and pl_module.device.type=="cuda" else torch.cuda.current_device()
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
        if not torch.cuda.is_available() or (pl_module.device and pl_module.device.type != "cuda"):
            print(f"[GPU-MON] step={trainer.global_step} CPU"); return
        idx = pl_module.device.index if pl_module.device and pl_module.device.index is not None else torch.cuda.current_device()
        mem = torch.cuda.memory_allocated(idx) / 1e9
        mem_max = torch.cuda.max_memory_allocated(idx) / 1e9
        msg = f"[GPU-MON] step={trainer.global_step} gpu{idx} mem={mem:.2f}GB (max {mem_max:.2f}GB)"
        if self.nvml and self.handle:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            meminfo = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            msg += f" util={util.gpu}% nvml_mem={meminfo.used/1e9:.2f}/{meminfo.total/1e9:.2f}GB"
        print(msg)

def build_trainer(cfg, callbacks, name: str):
    use_gpu = bool(cfg.general.gpus) and torch.cuda.is_available()
    requested = cfg.general.gpus
    dev_spec = resolve_devices(requested) if use_gpu else 0
    world = len(dev_spec) if isinstance(dev_spec, (list, tuple)) else int(dev_spec)

    if not use_gpu or world <= 1:
        devices = 1
        strategy = "auto"
    else:
        devices = dev_spec
        # DDP 전략 선택(간단화): 기본값
        strategy = "ddp_find_unused_parameters_true"

    print(f"[Dist] devices={devices}, strategy={strategy} "
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
# ============================================================================

# === Dataset (기존 CSVSpecDataset 사용) ======================================
from src.datasets.csv_spectrum_dataset import CSVSpecDataset

class CSVSpecDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.spec_start = cfg.dataset.spec_start
        self.spec_end   = cfg.dataset.spec_end
        self.global_cols = ["solvent_phase","is_qm","dielectric_constant_avg","pH_label"]
        self.fixed_vocabs = {
            "solvent_phase": cfg.dataset.solvent_vocab,
            "pH_label":      cfg.dataset.ph_vocab,
        }
        self.boolean_cols = ["is_qm"]
        self.train_csv = cfg.dataset.train_csv
        self.val_csv   = cfg.dataset.val_csv
        self.test_csv  = cfg.dataset.test_csv
        self.num_workers = cfg.train.num_workers
        self.batch_size  = cfg.train.batch_size

    def setup(self, stage: Optional[str] = None):
        spec_s = self.spec_start; spec_e = self.spec_end

        self.ds_train = CSVSpecDataset(
            csv_path=self.train_csv, stage="train",
            inchi_col="InChI", smiles_col=None,
            spectrum_start=spec_s, spectrum_end=spec_e,
            global_cols=self.global_cols, stats_path=None,
            spectrum_fill_eps=1e-8, fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols,
        )

        if self.val_csv:
            self.ds_val = CSVSpecDataset(
                csv_path=self.val_csv, stage="val",
                inchi_col="InChI", smiles_col=None,
                spectrum_start=spec_s, spectrum_end=spec_e,
                global_cols=self.global_cols, stats_path=str(Path(self.train_csv).with_suffix("")) + "_stats.json",
                spectrum_fill_eps=1e-8, fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols,
            )
        else:
            # 90/10 split
            n_total = len(self.ds_train)
            n_val = max(1, int(0.1 * n_total))
            n_train = n_total - n_val
            self.ds_train, self.ds_val = torch.utils.data.random_split(
                self.ds_train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
            )

        self.ds_test = CSVSpecDataset(
            csv_path=self.test_csv, stage="test",
            inchi_col="InChI", smiles_col=None,
            spectrum_start=spec_s, spectrum_end=spec_e,
            global_cols=self.global_cols, stats_path=str(Path(self.train_csv).with_suffix("")) + "_stats.json",
            spectrum_fill_eps=1e-8, fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols,
        )

        # spec_len/cond_dim 계산
        self.spec_len = self.spec_end - self.spec_start + 1
        y_dim0 = int(self.ds_train[0].y.numel())
        self.cond_dim = max(0, y_dim0 - self.spec_len)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.ds_val,   batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.ds_test,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# === Model (LightningModule로 래핑) ==========================================
class GraphSpectrumNet(nn.Module):
    def __init__(self, node_in=10, edge_in=6, hidden=256, layers=4, cond_dim=0, out_dim=601, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            nn_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            conv = GINEConv(nn_mlp)
            self.gnn_layers.append(conv); self.norms.append(BatchNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden + cond_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data, cond=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_embed(x)
        e = self.edge_mlp(edge_attr) if edge_attr is not None else None
        for conv, bn in zip(self.gnn_layers, self.norms):
            h = conv(h, edge_index, e)
            h = bn(h); h = torch.relu(h); h = self.dropout(h)
        g = global_mean_pool(h, batch)
        if cond is not None and cond.numel() > 0:
            g = torch.cat([g, cond], dim=-1)
        return self.readout(g)

class SpectrumModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, spec_len: int, cond_dim: int):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.spec_len = spec_len
        self.model = GraphSpectrumNet(
            node_in=10, edge_in=6,
            hidden=cfg.model.hidden, layers=cfg.model.layers, dropout=cfg.model.dropout,
            cond_dim=cond_dim, out_dim=spec_len
        )
        self.loss_fn = nn.MSELoss()

    def _split_y(self, batch_y: torch.Tensor):
        # batch_y: (B, spec_len + cond)
        y_spec = batch_y[:, :self.spec_len]
        cond = batch_y[:, self.spec_len:] if batch_y.size(1) > self.spec_len else None
        return y_spec, cond

    def training_step(self, batch, _):
        y_spec, cond = self._split_y(batch.y)
        y_hat = self.model(batch, cond=cond)
        loss = self.loss_fn(y_hat, y_spec)
        self.log("train/mse", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_spec.size(0))
        return loss

    def validation_step(self, batch, _):
        y_spec, cond = self._split_y(batch.y)
        y_hat = self.model(batch, cond=cond)
        loss = self.loss_fn(y_hat, y_spec)
        self.log("val/mse", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_spec.size(0))

    def test_step(self, batch, _):
        y_spec, cond = self._split_y(batch.y)
        y_hat = self.model(batch, cond=cond)
        loss = self.loss_fn(y_hat, y_spec)
        self.log("test/mse", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_spec.size(0))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=1e-4)
        return opt

# === Hydra 런처 (main.py와 동일 패턴) =========================================
@hydra.main(version_base='1.3', config_path='../configs', config_name='config_spectrum')
def main(cfg: DictConfig):
    # DataModule 준비
    dm = CSVSpecDataModule(cfg)
    dm.setup()

    # 콜백 구성 (GPU 모니터는 설정으로 on/off)
    callbacks = []
    if getattr(cfg.general, "gpu_monitor", False):
        callbacks.append(GpuUsageMonitor(interval=getattr(cfg.general, "gpu_monitor_interval", 50)))
    callbacks.append(EpochPrinter())  # ← 추가
    if cfg.train.save_model:
        callbacks += [
            ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1),
            ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='{epoch}',
                            monitor='val/mse', mode='min', save_top_k=5, every_n_epochs=1),
        ]

    trainer = build_trainer(cfg, callbacks, cfg.general.name)
    model = SpectrumModule(cfg, spec_len=dm.spec_len, cond_dim=dm.cond_dim)

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=dm)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=cfg.general.test_only)

if __name__ == "__main__":
    main()
