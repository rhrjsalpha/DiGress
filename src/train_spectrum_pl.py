# -*- coding: utf-8 -*-
# src/train_spectrum_pl.py
# X,E,y→X′,E′,y′로 중간 표현을 갱신하고, 이런 층을 L번 쌓은 뒤 그래프 수준 벡터 (y^L) 를 꺼내 MLP로 보내서 스펙트럼(601차원)을 회귀
# ── (1) 환경 ─────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

import warnings
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig
import time

torch.set_float32_matmul_precision("high")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DIGRESS_ROOT", str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)


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

class GraphFilesCallback(pl.callbacks.Callback):
    """graphs/ 폴더와 텍스트 로그 생성"""
    def __init__(self):
        self.fp = None

    def on_fit_start(self, trainer, pl_module):
        gdir = Path("graphs")
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "final_smiles.txt").touch(exist_ok=True)         # 형식만 맞춰 보관
        self.fp = open(gdir / "generated_samples1.txt", "a", encoding="utf-8")

    def on_validation_epoch_end(self, trainer, pl_module):
        val_mse = trainer.callback_metrics.get("val_mse")
        if self.fp and val_mse is not None:
            self.fp.write(f"epoch={trainer.current_epoch}, val_mse={float(val_mse):.6f}\n")
            self.fp.flush()

    def on_fit_end(self, trainer, pl_module):
        if self.fp:
            self.fp.close()
            self.fp = None



# ── (2) 데이터 모듈 ───────────────────────────────────────────────────────────
from src.datasets.csv_spectrum_dataset import CSVSpecDataset

class CSVSpecDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.spec_start = cfg.dataset.spec_start
        self.spec_end   = cfg.dataset.spec_end
        self.global_cols = ["solvent_phase","is_qm","dielectric_constant_avg","pH_label"]
        self.fixed_vocabs = {"solvent_phase": cfg.dataset.solvent_vocab,
                             "pH_label": cfg.dataset.ph_vocab}
        self.boolean_cols = ["is_qm"]
        self.train_csv, self.val_csv, self.test_csv = cfg.dataset.train_csv, cfg.dataset.val_csv, cfg.dataset.test_csv
        self.num_workers, self.batch_size = cfg.train.num_workers, cfg.train.batch_size

    def setup(self, stage: Optional[str] = None):
        s, e = self.spec_start, self.spec_end
        self.ds_train = CSVSpecDataset(self.train_csv, "train", None, "InChI", s, e,
                                       self.global_cols, None, spectrum_fill_eps=1e-8,
                                       fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols)
        if self.val_csv:
            self.ds_val = CSVSpecDataset(self.val_csv, "val", None, "InChI", s, e,
                                         self.global_cols, str(Path(self.train_csv).with_suffix("")) + "_stats.json",
                                         spectrum_fill_eps=1e-8, fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols)
        else:
            n_total = len(self.ds_train); n_val = max(1, int(0.1 * n_total))
            self.ds_train, self.ds_val = torch.utils.data.random_split(self.ds_train, [n_total-n_val, n_val],
                                                                       generator=torch.Generator().manual_seed(42))
        self.ds_test = CSVSpecDataset(self.test_csv, "test", None, "InChI", s, e,
                                      self.global_cols, str(Path(self.train_csv).with_suffix("")) + "_stats.json",
                                      spectrum_fill_eps=1e-8, fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols)
        self.spec_len = e - s + 1
        self.cond_dim = int(self.ds_train[0].y.numel()) - self.spec_len

    def train_dataloader(self): return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers)
    def val_dataloader(self):   return DataLoader(self.ds_val,   batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def test_dataloader(self):  return DataLoader(self.ds_test,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

# ── (3) 모델 ─────────────────────────────────────────────────────────────────
class GraphSpectrumNet(nn.Module):
    def __init__(self, node_in=10, edge_in=6, hidden=256, layers=4, cond_dim=0, out_dim=601, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn_layers = nn.ModuleList(); self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.gnn_layers.append(GINEConv(mlp)); self.norms.append(BatchNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden + cond_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, data, cond=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_embed(x); e = self.edge_mlp(edge_attr) if edge_attr is not None else None
        for conv, bn in zip(self.gnn_layers, self.norms):
            h = conv(h, edge_index, e); h = bn(h); h = torch.relu(h); h = self.dropout(h)
        g = global_mean_pool(h, batch)
        if cond is not None and cond.numel() > 0: g = torch.cat([g, cond], dim=-1)
        return self.readout(g)

class SpectrumModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, spec_len: int, cond_dim: int):
        super().__init__(); self.save_hyperparameters(ignore=["cfg"]); self.cfg = cfg; self.spec_len = spec_len
        self.model = GraphSpectrumNet(hidden=cfg.model.hidden, layers=cfg.model.layers, dropout=cfg.model.dropout,
                                      cond_dim=cond_dim, out_dim=spec_len)
        self.loss_fn = nn.MSELoss()

    def _split(self, y):
        # 스펙트럼 부위와, 분자 전역 특징 부위로 나눔
        return y[:, :self.spec_len], (y[:, self.spec_len:] if y.size(1) > self.spec_len else None)

    def training_step(self, batch, _): ys, c = self._split(batch.y); yh = self.model(batch, c); loss = self.loss_fn(yh, ys); self.log("train_mse", loss, on_epoch=True, prog_bar=True, batch_size=ys.size(0)); return loss
    def validation_step(self, batch, _): ys, c = self._split(batch.y); yh = self.model(batch, c); self.log("val_mse", self.loss_fn(yh, ys), on_epoch=True, prog_bar=True, batch_size=ys.size(0))
    def test_step(self, batch, _): ys, c = self._split(batch.y); yh = self.model(batch, c); self.log("test_mse", self.loss_fn(yh, ys), on_epoch=True, prog_bar=True, batch_size=ys.size(0))
    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, weight_decay=1e-4)

# ── (4) 콜백: 에폭 로그, 플롯, 그래프 파일 ─────────────────────────────────────
class EpochPrinter(Callback):
    def __init__(self):
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        tr = float(m.get("train_mse", float("nan")))
        va = float(m.get("val_mse", float("nan")))
        elapsed = (time.perf_counter() - self._t0) if self._t0 else float("nan")
        print(f"[epoch {trainer.current_epoch:03d}] time={elapsed:.2f}s  train/mse={tr:.6f}  val/mse={va:.6f}")
        # (옵션) 로그에도 남기고 싶으면:
        if trainer.logger is not None:
            try:
                trainer.logger.log_metrics({"epoch_time_sec": elapsed}, step=trainer.current_epoch)
            except Exception:
                pass


class ValPlotCallback(pl.callbacks.Callback):
    def __init__(self, every_n_epochs: int = 1, n_samples: int = 8):
        self.every = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every != 0:
            return
        dm = trainer.datamodule
        val_loader = dm.val_dataloader()
        if val_loader is None:
            return

        batch = next(iter(val_loader)).to(pl_module.device)
        spec_len = dm.spec_len
        with torch.no_grad():
            y_true = batch.y[:, :spec_len]
            cond   = batch.y[:, spec_len:] if dm.cond_dim > 0 else None
            y_pred = pl_module.model(batch, cond=cond)

        # 저장 경로: outputs_spectrum/<job>/<ts>/chains/<job>/epochXX/chains/
        outdir = Path("chains") / dm.cfg.general.name / f"epoch{epoch:02d}" / "chains"
        outdir.mkdir(parents=True, exist_ok=True)

        # 헤드리스 환경에서도 저장되도록 Agg 백엔드 사용
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        wl = torch.arange(dm.spec_start, dm.spec_end + 1).cpu().numpy()
        k = min(self.n_samples, y_true.size(0))
        for i in range(k):
            fig = plt.figure(figsize=(6, 3))
            plt.plot(wl, y_true[i].detach().cpu().numpy(), label="true")
            plt.plot(wl, y_pred[i].detach().cpu().numpy(), label="pred")
            plt.xlabel("wavelength"); plt.ylabel("intensity")
            plt.legend(); plt.tight_layout()
            fig.savefig(outdir / f"sample_{i}.png", dpi=140)
            plt.close(fig)

class GraphFilesCallback(Callback):
    def __init__(self): self.fp = None
    def on_fit_start(self, trainer, pl_module):
        gdir = Path("graphs"); gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "final_smiles.txt").touch(exist_ok=True)
        self.fp = open(gdir / "generated_samples1.txt", "a", encoding="utf-8")
    def on_validation_epoch_end(self, trainer, pl_module):
        vm = trainer.callback_metrics.get("val_mse")
        if self.fp and vm is not None: self.fp.write(f"epoch={trainer.current_epoch}, val_mse={float(vm):.6f}\n"); self.fp.flush()
    def on_fit_end(self, trainer, pl_module):
        if self.fp: self.fp.close(); self.fp = None

# ── (5) Trainer 빌더 ─────────────────────────────────────────────────────────
def build_trainer(cfg, extra_callbacks: Optional[List[Callback]] = None):
    job_name = cfg.general.name
    ckpt_dir = Path("checkpoints") / job_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # config는 train.n_epochs를 사용
    epochs = int(getattr(cfg.train, "n_epochs", 200))

    checkpoint_best = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_mse",   # ← LightningModule에서 self.log("val_mse", ...) 로 기록
        mode="min",
        save_top_k=1
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="last",
        every_n_epochs=1
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_best, checkpoint_last, lr_cb]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    loggers = [CSVLogger("logs", name=job_name)]
    try:
        import tensorboard  # 설치돼 있으면 텐서보드도 같이 로깅
        from pytorch_lightning.loggers import TensorBoardLogger
        loggers.append(TensorBoardLogger("tb_logs", name=job_name))
    except Exception:
        print("[WARN] TensorBoard not found; using CSVLogger only.")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    return trainer

# ── (6) Hydra 진입점 ─────────────────────────────────────────────────────────


@hydra.main(version_base="1.3", config_path="../configs", config_name="config_spectrum")
def main(cfg: DictConfig):
    # 결과 디렉토리: outputs_spectrum/<job>/<ts>/
    # configs/config_spectrum.yaml 에서 hydra.run.dir 을 outputs_spectrum로 지정해 두세요.
    dm = CSVSpecDataModule(cfg); dm.setup()
    callbacks = [
        EpochPrinter(),
        ValPlotCallback(every_n_epochs=1, n_samples=8),
        GraphFilesCallback(),
    ]
    if getattr(cfg.general, "gpu_monitor", False):
        from src.dist_utils import choose_ddp_strategy  # 선택적
        callbacks.append(GpuUsageMonitor(interval=getattr(cfg.general, "gpu_monitor_interval", 50)))
    trainer = build_trainer(cfg, extra_callbacks=callbacks)
    model = SpectrumModule(cfg, spec_len=dm.spec_len, cond_dim=dm.cond_dim)
    trainer.fit(model, datamodule=dm); trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()

