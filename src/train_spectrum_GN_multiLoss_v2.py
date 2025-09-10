# -*- coding: utf-8 -*-
# train_spectrum_pl_reworked.py
# - GNN/DiGress 백본으로 연속 nm 그리드 스펙트럼 회귀
# - 멀티로스 + GradNorm + 마일스톤 그림/지표 저장
# - is_cv 플래그 지원 (train/val vs train/test)
# - import 가능한 API(run_from_cfg) + 스크립트 직접 실행( hydra.main )

import os
import csv
import json
import time
import math
import platform
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem import Draw

# ----- custom losses -----
from src.custom_loss.SID_loss import sid_loss
from src.custom_loss.soft_dtw_cuda import SoftDTW
from src.custom_loss.GradNorm import GradNorm
#####
from src.train_spectrum_GN_multiLoss import DualMilestoneEval
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
# ===================== OS / ENV =====================
if platform.system() == "Windows":
    os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME",
        os.environ.get("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1"),
    )
else:
    # 리눅스: 깨끗이
    os.environ.pop("GLOO_DEVICE_TRANSPORT", None)
    os.environ.pop("GLOO_SOCKET_IFNAME", None)
    for k in (
        "NCCL_DEBUG", "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "NCCL_SHM_DISABLE", "NCCL_P2P_DISABLE"
    ):
        os.environ.pop(k, None)

torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
#                           유틸 / 공통
# =====================================================================

def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _rmse_from_mse(mse: float) -> float:
    return float(math.sqrt(max(0.0, mse)))

def _device_of(pl_module: "SpectrumModule"):
    if hasattr(pl_module, "device"):
        return pl_module.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
#                         마일스톤 평가+그림 저장
# =====================================================================

class MilestoneEvalAndFigure(Callback):
    def __init__(
        self,
        milestones: List[int],
        n_samples: int = 8,
        plot_all: bool = False,
        save_train: bool = True,
        save_val: bool = True,
        save_test: bool = False,
        outdir_name: str = "milestones",
        split_names: Tuple[str, str] = ("train", "val"),  # ("train","test")로도 사용
        enable_plots: bool = True
    ):
        self.milestones = set(int(m) for m in milestones)
        self.n_samples = int(n_samples)
        self.plot_all = bool(plot_all)
        self.save_train, self.save_val, self.save_test = save_train, save_val, save_test
        self.outdir_name = outdir_name
        self.split_names = split_names  # ("train","val") or ("train","test")

    @staticmethod
    def _render_mol_ax(ax, data):
        ax.axis("off")
        try:
            mol = None
            if hasattr(data, "smiles") and data.smiles:
                mol = Chem.MolFromSmiles(data.smiles)
            elif hasattr(data, "inchi") and data.inchi:
                mol = Chem.MolFromInchi(data.inchi)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(380, 320))
                ax.imshow(img)
                return
        except Exception:
            pass
        # fallback: 간단 그래프
        try:
            import networkx as nx
            G = nx.Graph()
            ei = data.edge_index.cpu().numpy()
            G.add_nodes_from(range(int(data.num_nodes)))
            G.add_edges_from([(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])])
            pos = nx.spring_layout(G, seed=0, k=0.4)
            nx.draw(G, pos, ax=ax, node_size=150, width=1.0)
        except Exception:
            ax.text(0.5, 0.5, "Mol unavailable", ha="center", va="center")

    @torch.no_grad()
    def _eval_split(self, pl_module, loader, wl, outdir_split, collect_limit=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        device = _device_of(pl_module)
        outdir_split.mkdir(parents=True, exist_ok=True)

        total_values = 0
        total_samples = 0
        sum_abs = 0.0
        sum_sq = 0.0
        sum_sid = 0.0
        sum_sdtw = 0.0

        saved = 0
        need = 10**12 if (self.plot_all or collect_limit is None) else int(collect_limit)

        spec_len = wl.shape[0]
        for batch in loader:
            batch = batch.to(device)
            ys = batch.y[:, :spec_len]
            cond = batch.y[:, spec_len:] if batch.y.size(1) > spec_len else None
            yh = pl_module.model(batch, cond)

            diff = (yh - ys)
            sum_abs += torch.sum(torch.abs(diff)).item()
            sum_sq += torch.sum(diff * diff).item()
            total_values += ys.numel()

            mask = torch.ones_like(ys, dtype=torch.bool)
            sid_b = sid_loss(yh, ys, mask, eps=1e-6, reduction="mean_valid")
            sum_sid += float(sid_b) * ys.size(0)
            sdtw_b = pl_module.softdtw(yh.unsqueeze(-1), ys.unsqueeze(-1))
            sum_sdtw += float(sdtw_b.sum().item())
            total_samples += ys.size(0)

            # figures
            data_list = batch.to_data_list()
            b = ys.size(0)
            for i in range(b):
                if saved >= need:
                    break
                fig = plt.figure(figsize=(11, 4))
                gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[0, 1])
                self._render_mol_ax(ax0, data_list[i])
                ax1.plot(wl, _to_numpy(ys[i]), label="true")
                ax1.plot(wl, _to_numpy(yh[i]), label="pred")
                ax1.set_xlabel("wavelength")
                ax1.set_ylabel("intensity")
                ax1.legend()
                fig.tight_layout()
                fig.savefig(outdir_split / f"sample_{saved:05d}.png", dpi=140)
                plt.close(fig)
                saved += 1

        mae = sum_abs / max(1, total_values)
        mse = sum_sq / max(1, total_values)
        rmse = _rmse_from_mse(mse)
        sid_v = sum_sid / max(1, total_samples)
        sdtw = sum_sdtw / max(1, total_samples)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "sid": sid_v, "softdtw": sdtw}
        (outdir_split / "metrics.json").write_text(json.dumps(metrics, indent=2))
        try:
            import pandas as pd
            pd.DataFrame([metrics]).to_csv(outdir_split / "metrics.csv", index=False)
        except Exception:
            pass
        return metrics

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch not in self.milestones:
            return

        dm = trainer.datamodule
        wl = torch.arange(dm.spec_start, dm.spec_end + 1).cpu().numpy()
        base = Path(os.getcwd()) / self.outdir_name / f"epoch{epoch:03d}"
        print(f"[Milestone] evaluating at epoch {epoch} → {base}")

        # split 이름은 ("train","val") 또는 ("train","test")
        train_name, second_name = self.split_names

        if self.save_train:
            self._eval_split(pl_module, dm.train_dataloader(), wl, base / train_name, collect_limit=self.n_samples)

        if self.save_val and dm.val_dataloader() is not None:
            self._eval_split(pl_module, dm.val_dataloader(), wl, base / "val", collect_limit=self.n_samples)

        if self.save_test and dm.test_dataloader() is not None:
            self._eval_split(pl_module, dm.test_dataloader(), wl, base / "test", collect_limit=self.n_samples)

# =====================================================================
#                         에폭별 CSV 라이터
# =====================================================================

class EpochCSVWriter(Callback):
    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.rows = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        row = {"epoch": trainer.current_epoch}
        for k, v in m.items():
            if k.startswith(("train_", "val_")):
                try:
                    row[k] = float(v)
                except Exception:
                    pass
        self.rows.append(row)

    def on_fit_end(self, trainer, pl_module):
        if not self.rows:
            return
        keys = sorted(set().union(*[r.keys() for r in self.rows]))
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

# =====================================================================
#                              GPU 모니터
# =====================================================================

class GpuUsageMonitor(Callback):
    def __init__(self, interval=50):
        self.interval = interval
        self.nvml = None
        self.handle = None

    def setup(self, trainer, pl_module, stage=None):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            idx = pl_module.device.index if pl_module.device.type == "cuda" else torch.cuda.current_device()
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
        if (batch_idx + 1) % self.interval != 0:
            return
        if not torch.cuda.is_available() or pl_module.device.type != "cuda":
            print(f"[GPU-MON] step={trainer.global_step} CPU")
            return
        idx = pl_module.device.index if pl_module.device.index is not None else torch.cuda.current_device()
        mem = torch.cuda.memory_allocated(idx) / 1e9
        mem_max = torch.cuda.max_memory_allocated(idx) / 1e9
        msg = f"[GPU-MON] step={trainer.global_step} gpu{idx} mem={mem:.2f}GB (max {mem_max:.2f}GB)"
        if self.nvml and self.handle:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            meminfo = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            msg += f" util={util.gpu}% nvml_mem={meminfo.used/1e9:.2f}/{meminfo.total/1e9:.2f}GB"
        print(msg)

# =====================================================================
#                           보조 로그 콜백
# =====================================================================

class GraphFilesCallback(pl.callbacks.Callback):
    def __init__(self):
        self.fp = None

    def on_fit_start(self, trainer, pl_module):
        gdir = Path(PROJECT_ROOT) / "graphs"
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "final_smiles.txt").touch(exist_ok=True)
        self.fp = open(gdir / "generated_samples1.txt", "a", encoding="utf-8")

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or not self.fp:
            return
        has_val = getattr(trainer.datamodule, "val_dataloader", None)
        if has_val is None or trainer.datamodule.val_dataloader() is None:
            tm = trainer.callback_metrics.get("train_total")
            if tm is not None:
                ep = int(trainer.current_epoch)
                self.fp.write(f"epoch={ep}, train_total={float(tm):.6f}\n")
                self.fp.flush()

    def on_fit_end(self, trainer, pl_module):
        if self.fp:
            self.fp.close()
            self.fp = None

# =====================================================================
#                             데이터 모듈
# =====================================================================

from src.datasets.csv_spectrum_dataset import CSVSpecDataset

class CSVSpecDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.spec_start = cfg.dataset.spec_start
        self.spec_end = cfg.dataset.spec_end
        self.global_cols = ["solvent_phase", "is_qm", "dielectric_constant_avg", "pH_label"]
        self.fixed_vocabs = {"solvent_phase": cfg.dataset.solvent_vocab,
                             "pH_label": cfg.dataset.ph_vocab}
        self.boolean_cols = ["is_qm"]
        self.train_csv, self.val_csv, self.test_csv = cfg.dataset.train_csv, cfg.dataset.val_csv, cfg.dataset.test_csv
        self.num_workers, self.batch_size = cfg.train.num_workers, cfg.train.batch_size
        self.smiles_col = getattr(cfg.dataset, "smiles_col", None)
        self.inchi_col = getattr(cfg.dataset, "inchi_col", "InChI")
        self.add_h = bool(getattr(cfg.dataset, "add_h", False))

    def override_splits(self, *, use_val: bool, use_test: bool, val_from_test: bool = False):
        """
        - use_val=True, val_from_test=True  → test_csv를 검증셋으로 사용, test는 비활성화.
        - use_test=True                     → test_csv 사용(검증은 비활성화).
        """
        if use_val and val_from_test:
            self.val_csv = self.test_csv
            self.test_csv = None
        elif use_test:
            self.val_csv = None
        else:
            self.val_csv = None
            self.test_csv = None

    def setup(self, stage: Optional[str] = None):
        s, e = self.spec_start, self.spec_end
        self.ds_train = CSVSpecDataset(
            self.train_csv, "train",
            smiles_col=self.smiles_col, inchi_col=self.inchi_col,
            spectrum_start=s, spectrum_end=e,
            global_cols=self.global_cols, stats_path=None,
            spectrum_fill_eps=1e-8,
            fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols,
            add_h=self.add_h
        )

        self.ds_val = None
        if self.val_csv:
            self.ds_val = CSVSpecDataset(
                self.val_csv, "val",
                smiles_col=self.smiles_col, inchi_col=self.inchi_col,
                spectrum_start=s, spectrum_end=e,
                global_cols=self.global_cols, stats_path=str(Path(self.train_csv).with_suffix("")) + "_stats.json",
                spectrum_fill_eps=1e-8,
                fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols,
                add_h=self.add_h
            )

        self.ds_test = None
        if self.test_csv:
            self.ds_test = CSVSpecDataset(
                self.test_csv, "test",
                smiles_col=self.smiles_col, inchi_col=self.inchi_col,
                spectrum_start=s, spectrum_end=e,
                global_cols=self.global_cols, stats_path=str(Path(self.train_csv).with_suffix("")) + "_stats.json",
                spectrum_fill_eps=1e-8,
                fixed_vocabs=self.fixed_vocabs, boolean_cols=self.boolean_cols,
                add_h=self.add_h
            )

        self.spec_len = e - s + 1
        self.cond_dim = int(self.ds_train[0].y.numel()) - self.spec_len
        self.node_dim = int(self.ds_train[0].x.size(1))
        self.edge_dim = int(self.ds_train[0].edge_attr.size(1)) if self.ds_train[0].edge_attr is not None else 0

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.ds_val is None:
            return None
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.ds_test is None:
            return None
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

# =====================================================================
#                             모델 백본들
# =====================================================================

class GraphSpectrumNet(nn.Module):
    def __init__(self, node_in=10, edge_in=6, hidden=256, layers=4, cond_dim=0, out_dim=601, dropout=0.1):
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.gnn_layers.append(GINEConv(mlp))
            self.norms.append(BatchNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden + cond_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, data, cond=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_embed(x)
        e = self.edge_mlp(edge_attr) if edge_attr is not None else None
        for conv, bn in zip(self.gnn_layers, self.norms):
            h = conv(h, edge_index, e)
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
        g = global_mean_pool(h, batch)
        if cond is not None and cond.numel() > 0:
            g = torch.cat([g, cond], dim=-1)
        output = self.readout(g)
        return output

# ---- DiGress 백본 ----
from src.models.transformer_model import GraphTransformer

def pyg_batch_to_dense(batch, y_in, edge_in_dim):
    device = batch.x.device
    bs = int(batch.batch.max().item()) + 1
    dx = batch.x.size(1)
    de = edge_in_dim if (batch.edge_attr is None) else batch.edge_attr.size(1)
    counts = torch.bincount(batch.batch, minlength=bs).tolist()
    nmax = max(counts)
    X = torch.zeros(bs, nmax, dx, device=device)
    E = torch.zeros(bs, nmax, nmax, de, device=device)
    node_mask = torch.zeros(bs, nmax, dtype=torch.bool, device=device)
    start = 0
    for b, n in enumerate(counts):
        end = start + n
        idx = torch.arange(start, end, device=device)
        X[b, :n] = batch.x[idx]; node_mask[b, :n] = True
        if batch.edge_index.numel() > 0 and batch.edge_attr is not None:
            mask_e = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end) & \
                     (batch.edge_index[1] >= start) & (batch.edge_index[1] < end)
            ei = batch.edge_index[:, mask_e] - start
            if ei.numel() > 0:
                E[b, ei[0], ei[1]] = batch.edge_attr[mask_e]
        start = end
    if y_in is None:
        y_in = torch.zeros(bs, 0, device=device)
    return X, E, y_in, node_mask

class DiGressSpectrumModel(nn.Module):
    def __init__(self, node_in, edge_in, cond_dim, spec_len,
                 hidden=256, n_layers=6, n_head=8, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.edge_in = edge_in
        self.cond_dim = cond_dim

        self.emb_X = nn.Linear(node_in, hidden)
        self.emb_E = nn.Linear(max(1, edge_in), hidden)
        self.emb_y = nn.Linear(max(1, cond_dim), hidden) if cond_dim > 0 else None

        self.gt = GraphTransformer(
            n_layers=n_layers,
            input_dims={'X': hidden, 'E': hidden, 'y': hidden},
            hidden_mlp_dims={'X': hidden, 'E': hidden, 'y': hidden},
            hidden_dims={'dx': hidden, 'de': hidden, 'dy': hidden,
                        'n_head': n_head, 'dim_ffX': hidden * 4,
                        'dim_ffE': max(128, hidden // 2), 'dim_ffy': hidden * 4},
            output_dims={'X': hidden, 'E': hidden, 'y': hidden},
            act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, spec_len)
        )

    def forward(self, batch, cond):
        if cond is not None and cond.numel() > 0:
            y_in = cond if self.emb_y is None else self.emb_y(cond)
        else:
            y_in = torch.zeros(batch.num_graphs, self.hidden, device=batch.x.device)

        X, E, y_in, node_mask = pyg_batch_to_dense(batch, y_in, self.edge_in)
        if E.size(-1) == 0:
            E = E.new_zeros(E.shape[:-1] + (1,))

        X = self.emb_X(X)
        E = self.emb_E(E)

        out = self.gt(X, E, y_in, node_mask)  # out.y: [B, hidden]
        return self.readout(out.y)

# 백본 선택
def build_backbone(cfg: DictConfig, dm: CSVSpecDataModule, cond_dim: int, spec_len: int) -> nn.Module:
    backend = str(getattr(cfg.model, "backend", "gnn")).lower()
    if backend == "digress":
        return DiGressSpectrumModel(
            node_in=dm.node_dim, edge_in=dm.edge_dim, cond_dim=cond_dim, spec_len=spec_len,
            hidden=cfg.model.hidden, n_layers=cfg.model.layers,
            n_head=getattr(cfg.model, "n_head", 8), dropout=cfg.model.dropout
        )
    elif backend == "gnn":
        return GraphSpectrumNet(
            node_in=dm.node_dim or 10, edge_in=dm.edge_dim or 6,
            hidden=cfg.model.hidden, layers=cfg.model.layers,
            cond_dim=cond_dim, out_dim=spec_len, dropout=cfg.model.dropout
        )
    else:
        raise ValueError(f"Unknown model.backend={backend} (use 'gnn' or 'digress')")

# =====================================================================
#                         Lightning 모듈
# =====================================================================

class SpectrumModule(pl.LightningModule):
    """
    cfg.train.losses: ["SID","MAE"] 등
    cfg.train.use_gradnorm: bool
    cfg.train.alpha: GradNorm alpha
    """
    def __init__(self, cfg: DictConfig, spec_len: int, cond_dim: int, backbone: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "backbone"])
        self.cfg = cfg
        self.spec_len = spec_len
        self.model = backbone

        device_cuda = torch.cuda.is_available()
        self.softdtw = SoftDTW(use_cuda=device_cuda,
                               gamma=float(getattr(cfg.train, "softdtw_gamma", 0.2)),
                               bandwidth=None, normalize=True)

        self.loss_names: List[str] = [s.upper() for s in (cfg.train.losses or ["MSE"])]
        self.num_losses = len(self.loss_names)
        self.initial_losses: torch.Tensor | None = None

        self.use_gradnorm: bool = bool(getattr(cfg.train, "use_gradnorm", True))
        self.alpha: float = float(getattr(cfg.train, "alpha", 0.12))
        self.gradnorm = GradNorm(num_losses=self.num_losses, alpha=self.alpha) if self.use_gradnorm else None

        self.lr = float(cfg.train.lr)

    def _split(self, y):
        return y[:, :self.spec_len], (y[:, self.spec_len:] if y.size(1) > self.spec_len else None)

    @staticmethod
    def _build_mask_like(y: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
        if threshold is None:
            return torch.ones_like(y, dtype=torch.bool)
        return (y > threshold)

    def _compute_each_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name in self.loss_names:
            if name == "MSE":
                out[name] = torch.mean((y_pred - y_true) ** 2)
            elif name == "MAE":
                out[name] = torch.mean(torch.abs(y_pred - y_true))
            elif name == "HUBER":
                out[name] = nn.functional.smooth_l1_loss(y_pred, y_true, reduction="mean")
            elif name == "SID":
                out[name] = sid_loss(y_pred, y_true, mask, eps=1e-6, reduction="mean_valid")
            elif name == "SOFTDTW":
                out[name] = self.softdtw(y_pred.unsqueeze(-1), y_true.unsqueeze(-1)).mean()
            else:
                raise ValueError(f"Unknown loss: {name}")
        return out

    def _combine_losses(self, each: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        losses_vec = torch.stack([each[n] for n in self.loss_names])
        if self.initial_losses is None:
            self.initial_losses = losses_vec.detach().clamp_min(1e-12)
        norm_vec = losses_vec / self.initial_losses

        if self.use_gradnorm and mode == "train":
            weights = self.gradnorm.compute_weights(norm_vec, self.model)
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, device=norm_vec.device, dtype=norm_vec.dtype)
            total = torch.sum(weights * norm_vec)
            for i, name in enumerate(self.loss_names):
                self.log(f"{mode}_weight_{name}", weights[i], on_epoch=True, prog_bar=False, batch_size=self.current_batch_size)
        else:
            total = torch.mean(norm_vec)

        for i, name in enumerate(self.loss_names):
            self.log(f"{mode}_loss_{name}", losses_vec[i], on_epoch=True, prog_bar=False, batch_size=self.current_batch_size)
            self.log(f"{mode}_norm_{name}", norm_vec[i], on_epoch=True, prog_bar=False, batch_size=self.current_batch_size)
        return total

    @property
    def current_batch_size(self) -> int:
        try:
            return self._last_batch_size
        except Exception:
            return 1

    def training_step(self, batch, _):
        ys, cond = self._split(batch.y)
        self._last_batch_size = ys.size(0)
        yhat = self.model(batch, cond)
        mask = self._build_mask_like(ys, threshold=None)
        each = self._compute_each_loss(yhat, ys, mask)
        total = self._combine_losses(each, mode="train")
        self.log("train_total", total, on_epoch=True, prog_bar=True, batch_size=ys.size(0))
        self.log("train_mse", ((yhat - ys) ** 2).mean(), on_epoch=True, prog_bar=False, batch_size=ys.size(0))
        return total

    def validation_step(self, batch, _):
        ys, cond = self._split(batch.y)
        self._last_batch_size = ys.size(0)
        yhat = self.model(batch, cond)
        mask = self._build_mask_like(ys, threshold=None)
        each = self._compute_each_loss(yhat, ys, mask)
        total = self._combine_losses(each, mode="val")
        self.log("val_total", total, on_epoch=True, prog_bar=True, batch_size=ys.size(0))
        self.log("val_mse", ((yhat - ys) ** 2).mean(), on_epoch=True, prog_bar=False, batch_size=ys.size(0))

    def test_step(self, batch, _):
        ys, cond = self._split(batch.y)
        self._last_batch_size = ys.size(0)
        yhat = self.model(batch, cond)
        mask = self._build_mask_like(ys, threshold=None)
        each = self._compute_each_loss(yhat, ys, mask)
        total = self._combine_losses(each, mode="test")
        self.log("test_total", total, on_epoch=True, prog_bar=True, batch_size=ys.size(0))
        self.log("test_mse", ((yhat - ys) ** 2).mean(), on_epoch=True, prog_bar=False, batch_size=ys.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

class EpochPrinter(pl.Callback):
    def __init__(self):
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        tr = float(m.get("train_total", float("nan")))
        va = float(m.get("val_total", float("nan")))
        elapsed = (time.perf_counter() - self._t0) if self._t0 else float("nan")
        print(f"[epoch {trainer.current_epoch:03d}] time={elapsed:.2f}s  train/total={tr:.6f}  val/total={va:.6f}")
        if trainer.logger is not None:
            try:
                trainer.logger.log_metrics({"epoch_time_sec": elapsed}, step=trainer.current_epoch)
            except Exception:
                pass

    def on_fit_start(self, trainer, pl_module):
        lr = int(os.environ.get("LOCAL_RANK", -1))
        if torch.cuda.is_available():
            cd = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(cd).total_memory / 1e9
            free = torch.cuda.mem_get_info()[0] / 1e9
            print(f"[RANK{lr}] cuda_device={cd}  free={free:.2f}GB / total={total:.2f}GB")
        else:
            print(f"[RANK{lr}] CPU mode")


class ValPlotCallback(pl.Callback):
    """검증 배치 일부를 플롯. (is_cv=False이면 val_loader가 없으니 자연히 skip)"""
    def __init__(self, every_n_epochs: int = 1, n_samples: int = 8):
        self.every = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
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
            cond = batch.y[:, spec_len:] if dm.cond_dim > 0 else None
            y_pred = pl_module.model(batch, cond=cond)
        outdir = Path(os.getcwd()) / "chains" / dm.cfg.general.name / f"epoch{epoch:02d}" / "chains"
        outdir.mkdir(parents=True, exist_ok=True)
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


# =====================================================================
#                              Trainer
# =====================================================================

def build_trainer(cfg, *, extra_callbacks: Optional[List[pl.Callback]] = None, has_val: bool = True):
    job_name = cfg.general.name
    ckpt_dir = _ensure_dir(Path(PROJECT_ROOT) / "checkpoints" / job_name)
    precision = getattr(cfg.trainer, "precision", getattr(cfg.train, "precision", 32))

    monitor_key = "val_total" if has_val else "train_total"
    checkpoint_best = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best",
                                      monitor=monitor_key, mode="min", save_top_k=1)
    checkpoint_last = ModelCheckpoint(dirpath=str(ckpt_dir), filename="last", every_n_epochs=1)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_best, checkpoint_last, lr_cb]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    is_windows = (platform.system() == "Windows")
    accel = getattr(cfg.train, "accelerator", "gpu" if torch.cuda.is_available() else "cpu")
    yaml_devices = getattr(cfg.train, "devices", 1)
    devices = 1 if is_windows else yaml_devices
    num_nodes = int(getattr(cfg.train, "num_nodes", 1))
    strategy_cfg = str(getattr(cfg.train, "strategy", "auto"))

    if devices != 1 and strategy_cfg in ("ddp", "ddp_find_unused_false"):
        strategy = DDPStrategy(
            process_group_backend="gloo",
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            broadcast_buffers=False,
            bucket_cap_mb=12,
        )
    else:
        strategy = "auto"

    print(f"[DIST] system={'Windows' if is_windows else 'Linux'} "
          f"→ backend={getattr(strategy, 'process_group_backend', strategy)} devices={devices}")

    # ➜ 여기 추가: has_val=False일 때 검증 루프 완전 비활성화
    trainer_kwargs = dict(
        max_epochs=int(getattr(cfg.train, "n_epochs", 200)),
        accelerator=accel,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        enable_progress_bar=True,
        default_root_dir=str(PROJECT_ROOT),
    )
    if not has_val:
        trainer_kwargs.update(
            num_sanity_val_steps=0,  # sanity check에서 val loop 건너뛰기
            limit_val_batches=0,     # 에폭 중 val loop 자체 비활성화
        )

    trainer = pl.Trainer(**trainer_kwargs)
    return trainer



# =====================================================================
#                         평가 유틸리티 함수
# =====================================================================

@torch.no_grad()
def compute_metrics_on_loader(
    pl_module: SpectrumModule, loader, spec_len: int,
    *, device_pref: str = "auto",               # "auto"|"cuda"|"cpu"
    include_softdtw: bool = True,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    if loader is None:
        return {}

    # ---- 평가 디바이스 결정
    if device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = pl_module.device if hasattr(pl_module, "device") else (
            next(pl_module.parameters()).device if any(True for _ in pl_module.parameters()) else torch.device("cpu")
        )

    # ---- 모델을 평가 디바이스로 이동 + eval 모드
    try:
        orig_device = next(pl_module.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")
    was_training = pl_module.training
    pl_module.to(device)
    pl_module.eval()

    # ---- SoftDTW (디바이스 맞춰 생성)
    local_softdtw = None
    if include_softdtw:
        local_softdtw = SoftDTW(
            use_cuda=(device.type == "cuda"),
            gamma=float(getattr(pl_module.cfg.train, "softdtw_gamma", 0.2)),
            bandwidth=None, normalize=True
        )

    try:
        # ---- 누적 계산
        total_values = total_samples = 0
        sum_abs = sum_sq = sum_sid = sum_sdtw = 0.0

        it = loader
        try:
            from tqdm import tqdm
            if show_progress:
                it = tqdm(loader, total=len(loader), desc="[final-metrics]")
        except Exception:
            pass

        for b_idx, batch in enumerate(it):
            if max_batches is not None and b_idx >= max_batches:
                break
            batch = batch.to(device)
            ys = batch.y[:, :spec_len]
            cond = batch.y[:, spec_len:] if batch.y.size(1) > spec_len else None
            yh = pl_module.model(batch, cond)

            diff = (yh - ys)
            sum_abs += torch.sum(torch.abs(diff)).item()
            sum_sq += torch.sum(diff * diff).item()
            total_values += ys.numel()

            mask = torch.ones_like(ys, dtype=torch.bool)
            sid_b = sid_loss(yh, ys, mask, eps=1e-6, reduction="mean_valid")
            sum_sid += float(sid_b) * ys.size(0)

            if local_softdtw is not None:
                sdtw_b = local_softdtw(yh.unsqueeze(-1), ys.unsqueeze(-1))
                sum_sdtw += float(sdtw_b.sum().item())

            total_samples += ys.size(0)

        mae = sum_abs / max(1, total_values)
        mse = sum_sq / max(1, total_values)
        rmse = float(max(mse, 0.0) ** 0.5)
        sid_v = sum_sid / max(1, total_samples)
        sdtw = (sum_sdtw / max(1, total_samples)) if include_softdtw else float("nan")
        return {"mae": mae, "mse": mse, "rmse": rmse, "sid": sid_v, "softdtw": sdtw}
    finally:
        # ---- 원상 복구
        try:
            pl_module.to(orig_device)
            if was_training:
                pl_module.train()
        except Exception:
            pass

def merge_milestone_summaries(milestone_root: Path,
                              split_names: Tuple[str, str] = ("train", "val"),
                              save_path: Optional[Path] = None) -> Optional[Path]:
    """
    milestones/epochXXX/{train|val|test}/metrics.csv 파일들을 스캔해 하나로 합친다.
    컬럼: epoch, split, mae, mse, rmse, sid, softdtw
    """
    if not milestone_root.exists():
        return None
    rows = []
    import re
    import pandas as pd

    for p in sorted(milestone_root.glob("epoch*")):
        m = re.search(r"epoch(\d+)", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        for sp in ("train", split_names[1]):  # train + (val|test)
            csv_path = p / sp / "metrics.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        rec = df.iloc[0].to_dict()
                        rec.update({"epoch": ep, "split": sp})
                        rows.append(rec)
                except Exception:
                    pass

    if not rows:
        return None

    df_all = pd.DataFrame(rows).sort_values(["epoch", "split"])
    if save_path is None:
        save_path = milestone_root / "milestones_summary.csv"
    _ensure_dir(save_path.parent)
    df_all.to_csv(save_path, index=False)
    return save_path


# =====================================================================
#                         실행 엔진 (API + main)
# =====================================================================

def run_from_cfg(cfg: DictConfig,
                 *,
                 is_cv: Optional[bool] = None,
                 fold_tag: Optional[str] = None,
                 final_csv_name: Optional[str] = None) -> Dict[str, str]:
    """
    외부(CV 코드)에서 import하여 호출 가능한 실행 함수.
    - is_cv=True  → (train, val) 모드. test_csv를 검증으로 사용.
    - is_cv=False → (train, test) 모드. test_csv를 테스트로 사용.
    반환: {"final_metrics_csv": ..., "milestones_summary_csv": ..., "ckpt_dir": ...}
    """
    # ---- is_cv 해석
    if is_cv is None:
        is_cv = bool(getattr(cfg.general, "is_cv", False))

    # ---- DataModule 구성(+ 분할 오버라이드)
    dm = CSVSpecDataModule(cfg)
    if is_cv:
        # test_csv → val 로 사용, test 비활성화
        dm.override_splits(use_val=True, use_test=False, val_from_test=True)
        split_names = ("train", "val")
    else:
        # test_csv를 테스트로, val 비활성화
        dm.override_splits(use_val=False, use_test=True)
        split_names = ("train", "test")

    metrics_epochs = getattr(cfg.train, "milestones_metrics",
                             getattr(cfg.train, "milestones", []))  # 없으면 기존 것을 metrics로
    plot_epochs = getattr(cfg.train, "milestones_plots", [])  # 기본은 비어있음

    dm.setup()
    spec_len = dm.spec_len

    # ---- 백본
    backbone = build_backbone(cfg, dm, cond_dim=dm.cond_dim, spec_len=spec_len)
    model = SpectrumModule(cfg, spec_len=spec_len, cond_dim=dm.cond_dim, backbone=backbone)

    # ---- 콜백들
    callbacks: List[pl.Callback] = [
        EpochPrinter(),
        GraphFilesCallback(),
        EpochCSVWriter(Path(os.getcwd()) / "metrics_epoch.csv"),
        DualMilestoneEval(
            metrics_milestones=metrics_epochs,
            plot_milestones=plot_epochs,
            n_samples=(0 if is_cv else int(getattr(cfg.train, "milestone_n_samples", 16))),
            plot_all=(False if is_cv else bool(getattr(cfg.train, "milestone_plot_all", False))),
            save_train=True,
            save_val=is_cv,  # CV: val 저장
            save_test=(not is_cv),  # Final: test 저장
            outdir_name="milestones",
            split_names=split_names,
            global_enable_plots=(not is_cv),  # CV 전체에서 그림 off
        ),
    ]
    if getattr(cfg.general, "gpu_monitor", False):
        callbacks.append(GpuUsageMonitor(interval=getattr(cfg.general, "gpu_monitor_interval", 50)))

    # ---- Trainer
    trainer = build_trainer(cfg, extra_callbacks=callbacks, has_val=is_cv)

    # ---- 학습
    print(f"[MODEL] backend={getattr(cfg.model,'backend','gnn')}  -> {type(backbone).__name__}")
    print("cuda_count=", torch.cuda.device_count(),
          "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("world_size=", trainer.world_size,
          "num_devices=", trainer.num_devices,
          "local_rank=", getattr(trainer, "local_rank", None))

    trainer.fit(model, datamodule=dm)

    # ---- (선택) 테스트 호출: CV면 val만 있으므로 test는 skip, Final이면 test 호출
    if not is_cv:
        trainer.test(model, datamodule=dm, ckpt_path="best")

    # ---- 마일스톤 요약 CSV 병합
    milestones_root = Path(os.getcwd()) / "milestones"
    milestones_summary_csv = merge_milestone_summaries(milestones_root, split_names=split_names)

    # ---- 최종 지표 계산 (Train + Val|Test)
    print("[FINAL] computing metrics on train/val ...")
    t0 = time.perf_counter()

    # CV 모드라면 train/val, Final 모드라면 train/test
    train_metrics = compute_metrics_on_loader(
        model, dm.train_dataloader(), dm.spec_len,
        device_pref="cuda",  # 가능하면 GPU로
        include_softdtw=True,  # 느리면 False로 끄기
        max_batches=None,  # 느리면 20 같은 상한
        show_progress=True,
    )

    if is_cv:
        second_loader = dm.val_dataloader()
    else:
        second_loader = dm.test_dataloader()

    second_metrics = compute_metrics_on_loader(
        model, second_loader, dm.spec_len,
        device_pref="cuda",
        include_softdtw=True,
        max_batches=None,
        show_progress=True,
    )

    mode_label = "CV" if is_cv else "Final"
    second_tag = "val" if is_cv else "test"
    # 컬럼명 규칙: CV_training_sid, CV_val_sid / Final_training_sid, Final_test_sid 등
    final_rows = {}
    for k in ("mae", "mse", "rmse", "sid", "softdtw"):
        if k in train_metrics:
            final_rows[f"{mode_label}_training_{k}"] = train_metrics[k]
        if k in second_metrics:
            final_rows[f"{mode_label}_{second_tag}_{k}"] = second_metrics[k]

    # 메타 정보(선택): fold_tag, job_name, backend
    final_rows.update({
        "job_name": str(getattr(cfg.general, "name", "")),
        "backend": str(getattr(cfg.model, "backend", "")),
        "fold_tag": ("" if fold_tag is None else str(fold_tag)),
        "is_cv": bool(is_cv),
    })

    # ---- 최종 CSV 저장
    if final_csv_name is None:
        # fold 구분이 필요하면 파일명에 태그 부여
        stem = "final_metrics"
        if fold_tag is not None and str(fold_tag) != "":
            stem += f"_{fold_tag}"
        final_csv_name = stem + ".csv"

    final_csv_path = Path(os.getcwd()) / final_csv_name
    with open(final_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows.keys()))
        w.writeheader()
        w.writerow(final_rows)

    # 체크포인트 디렉토리 경로 반환
    ckpt_dir = Path(PROJECT_ROOT) / "checkpoints" / str(getattr(cfg.general, "name", ""))
    out = {
        "final_metrics_csv": str(final_csv_path),
        "milestones_summary_csv": ("" if milestones_summary_csv is None else str(milestones_summary_csv)),
        "ckpt_dir": str(ckpt_dir),
    }
    print("[DONE] final metrics saved →", out["final_metrics_csv"])
    if milestones_summary_csv:
        print("[DONE] milestones summary →", milestones_summary_csv)
    return out


# =====================================================================
#                              Hydra main
# =====================================================================

@hydra.main(version_base="1.3", config_path="../configs", config_name="config_spectrum")
def main(cfg: DictConfig):
    """
    실행 예)
      # CV 모드: test_csv를 val로 사용
      python train_spectrum_pl_reworked.py general.is_cv=true general.name=cv_job1

      # Final 모드: train/test
      python train_spectrum_pl_reworked.py general.is_cv=false general.name=final_job1
    """
    # 외부에서 import 없이도 바로 실행 가능
    is_cv = bool(getattr(cfg.general, "is_cv", False))
    fold_tag = getattr(cfg.general, "fold_tag", None)
    run_from_cfg(cfg, is_cv=is_cv, fold_tag=fold_tag)


if __name__ == "__main__":
    main()