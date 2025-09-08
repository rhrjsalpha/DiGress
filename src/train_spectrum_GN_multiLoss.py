# -*- coding: utf-8 -*-
# X,E,y→X′,E′,y′를 갱신하는 백본(GNN/DiGress)을 선택해 그래프 수준 y를 얻고,
# 이를 MLP로 스펙트럼(연속 nm 그리드)으로 회귀. 멀티로스 + GradNorm + 로깅/그림 저장.

import os
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict

import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader
import csv
from pytorch_lightning.strategies import DDPStrategy
from rdkit import Chem
from rdkit.Chem import Draw

# ----- custom losses -----
from src.custom_loss.SID_loss import sid_loss
from src.custom_loss.soft_dtw_cuda import SoftDTW
from src.custom_loss.GradNorm import GradNorm

torch.set_float32_matmul_precision("high")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DIGRESS_ROOT", str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)
RUN_DIR = Path(os.getcwd())

# ======================== Milestone 평가 + 그림 저장 =========================
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
    ):
        self.milestones = set(int(m) for m in milestones)
        self.n_samples = int(n_samples)
        self.plot_all = bool(plot_all)
        self.save_train, self.save_val, self.save_test = save_train, save_val, save_test
        self.outdir_name = outdir_name

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
        # fallback: 토폴로지
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

        device = pl_module.device
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

            # metrics
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
            data_list = batch.to_data_list()  # 커스텀 속성 보존
            b = ys.size(0)
            for i in range(b):
                if saved >= need:
                    break
                fig = plt.figure(figsize=(11, 4))
                gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[0, 1])
                self._render_mol_ax(ax0, data_list[i])
                ax1.plot(wl, ys[i].detach().cpu().numpy(), label="true")
                ax1.plot(wl, yh[i].detach().cpu().numpy(), label="pred")
                ax1.set_xlabel("wavelength")
                ax1.set_ylabel("intensity")
                ax1.legend()
                fig.tight_layout()
                fig.savefig(outdir_split / f"sample_{saved:05d}.png", dpi=140)
                plt.close(fig)
                saved += 1

        mae = sum_abs / max(1, total_values)
        mse = sum_sq / max(1, total_values)
        sid = sum_sid / max(1, total_samples)
        sdtw = sum_sdtw / max(1, total_samples)

        import json, pandas as pd
        metrics = {"mae": mae, "mse": mse, "sid": sid, "softdtw": sdtw}
        (outdir_split / "metrics.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame([metrics]).to_csv(outdir_split / "metrics.csv", index=False)
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
        if self.save_train:
            self._eval_split(pl_module, dm.train_dataloader(), wl, base / "train", collect_limit=self.n_samples)
        if self.save_val:
            self._eval_split(pl_module, dm.val_dataloader(), wl, base / "val", collect_limit=self.n_samples)
        if self.save_test and hasattr(dm, "test_dataloader"):
            self._eval_split(pl_module, dm.test_dataloader(), wl, base / "test", collect_limit=self.n_samples)


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


# ============================== GPU 모니터 ==============================
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


# ============================== 그래프 로그 파일 ==============================
class GraphFilesCallback(pl.callbacks.Callback):
    def __init__(self):
        self.fp = None

    def on_fit_start(self, trainer, pl_module):
        gdir = Path(PROJECT_ROOT) / "graphs"
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / "final_smiles.txt").touch(exist_ok=True)
        self.fp = open(gdir / "generated_samples1.txt", "a", encoding="utf-8")

    def on_validation_epoch_end(self, trainer, pl_module):
        vm = trainer.callback_metrics.get("val_total")
        if self.fp and vm is not None:
            self.fp.write(f"epoch={trainer.current_epoch}, val_total={float(vm):.6f}\n")
            self.fp.flush()

    def on_fit_end(self, trainer, pl_module):
        if self.fp:
            self.fp.close()
            self.fp = None


# ============================== 데이터 모듈 ==============================
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
        else:
            n_total = len(self.ds_train)
            n_val = max(1, int(0.1 * n_total))
            self.ds_train, self.ds_val = torch.utils.data.random_split(
                self.ds_train, [n_total - n_val, n_val],
                generator=torch.Generator().manual_seed(42)
            )
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
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# ============================== 두 개의 백본 ==============================
# ---- (A) GNN 백본 (기존) ----
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
        return self.readout(g)

# ---- (B) DiGress 백본 ----
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

# ---- DiGress 백본 (입력 임베딩 포함) ----
from src.models.transformer_model import GraphTransformer

class DiGressSpectrumModel(nn.Module):
    def __init__(self, node_in, edge_in, cond_dim, spec_len,
                 hidden=256, n_layers=6, n_head=8, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.edge_in = edge_in
        self.cond_dim = cond_dim

        # 1) 입력을 모두 hidden 차원으로 투영
        self.emb_X = nn.Linear(node_in, hidden)
        self.emb_E = nn.Linear(max(1, edge_in), hidden)               # edge_attr가 없으면 1로 대체
        self.emb_y = nn.Linear(max(1, cond_dim), hidden) if cond_dim > 0 else None

        # 2) GraphTransformer는 모두 hidden 차원으로 받도록 설정
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
        # cond → hidden (없으면 hidden차원 제로-벡터)
        if cond is not None and cond.numel() > 0:
            y_in = cond if self.emb_y is None else self.emb_y(cond)  # [B, hidden]
        else:
            y_in = torch.zeros(batch.num_graphs, self.hidden, device=batch.x.device)

        # Dense로 변환 (y_in은 이미 [B, hidden])
        X, E, y_in, node_mask = pyg_batch_to_dense(batch, y_in, self.edge_in)  # X:[B,N,dx], E:[B,N,N,de]

        # edge_attr가 완전히 없을 때 de=0인 텐서가 올 수 있으니 1로 보정
        if E.size(-1) == 0:
            E = E.new_zeros(E.shape[:-1] + (1,))

        # 입력 임베딩: 마지막 차원에 선형 적용
        X = self.emb_X(X)      # [B, N, hidden]
        E = self.emb_E(E)      # [B, N, N, hidden]
        # y_in은 이미 [B, hidden]

        out = self.gt(X, E, y_in, node_mask)  # out.y: [B, hidden]
        return self.readout(out.y)


# ============================== 백본 선택 팩토리 ==============================
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


# ============================== Lightning 모듈 ==============================
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


# ============================== 에폭 프린터 & 밸리데이션 플로터 ==============================
class EpochPrinter(Callback):
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
            cond = batch.y[:, spec_len:] if dm.cond_dim > 0 else None
            y_pred = pl_module.model(batch, cond=cond)
        outdir = Path(PROJECT_ROOT) / "chains" / dm.cfg.general.name / f"epoch{epoch:02d}" / "chains"
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


# ============================== Trainer ==============================
def build_trainer(cfg, extra_callbacks: Optional[List[Callback]] = None):
    job_name = cfg.general.name
    ckpt_dir = Path(PROJECT_ROOT) / "checkpoints" / job_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(getattr(cfg.train, "n_epochs", 200))

    checkpoint_best = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best",
                                      monitor="val_total", mode="min", save_top_k=1)
    checkpoint_last = ModelCheckpoint(dirpath=str(ckpt_dir), filename="last", every_n_epochs=1)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_best, checkpoint_last, lr_cb]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # --- DDP 전략 선택 (권장: find_unused_parameters=False) ---
    strategy_cfg = getattr(cfg.train, "strategy", "auto")
    if strategy_cfg in ("ddp", "ddp_find_unused_false"):
        strategy = DDPStrategy(
            find_unused_parameters=(strategy_cfg != "ddp_find_unused_false"),
            gradient_as_bucket_view=True,
        )
    else:
        strategy = strategy_cfg  # "auto" 등

    loggers = [CSVLogger(save_dir=str(RUN_DIR), name="pl_logs", version="")]
    try:
        import tensorboard  # noqa
        loggers.append(TensorBoardLogger(str(Path(PROJECT_ROOT) / "tb_logs"), name=job_name))
    except Exception:
        print("[WARN] TensorBoard not found; using CSVLogger only.")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        default_root_dir=str(PROJECT_ROOT),
    )
    return trainer


# ============================== Hydra 엔트리 ==============================
@hydra.main(version_base="1.3", config_path="../configs", config_name="config_spectrum")
def main(cfg: DictConfig):
    """
    설정 예:
      model:
        backend: gnn        # gnn | digress
        hidden: 256
        layers: 6
        n_head: 8
        dropout: 0.1
    """
    RUN_DIR = Path(os.getcwd())
    callbacks = [
        EpochPrinter(),
        ValPlotCallback(every_n_epochs=1, n_samples=8),
        GraphFilesCallback(),
        EpochCSVWriter(RUN_DIR / "metrics_epoch.csv"),
        MilestoneEvalAndFigure(
            milestones=getattr(cfg.train, "milestones", [10, 50, 100]),
            n_samples=getattr(cfg.train, "milestone_n_samples", 16),
            plot_all=bool(getattr(cfg.train, "milestone_plot_all", False)),
            save_train=True, save_val=True,
            save_test=bool(getattr(cfg.train, "milestone_eval_test", False)),
            outdir_name="milestones",
        ),
    ]

    dm = CSVSpecDataModule(cfg); dm.setup()
    if getattr(cfg.general, "gpu_monitor", False):
        callbacks.append(GpuUsageMonitor(interval=getattr(cfg.general, "gpu_monitor_interval", 50)))

    # 백본 선택/생성
    backbone = build_backbone(cfg, dm, cond_dim=dm.cond_dim, spec_len=dm.spec_len)
    print(f"[MODEL] backend={getattr(cfg.model,'backend','gnn')}  -> {type(backbone).__name__}")

    # (디버그) 문자열 확인 — processed 캐시 갱신 필요 시 force_reprocess로 재생성하세요
    try:
        batch = next(iter(dm.train_dataloader()))
        d0 = batch.to_data_list()[0]
        print("smiles:", getattr(d0, "smiles", None))
        print("inchi :", getattr(d0, "inchi", None))
    except Exception:
        pass

    trainer = build_trainer(cfg, extra_callbacks=callbacks)
    model = SpectrumModule(cfg, spec_len=dm.spec_len, cond_dim=dm.cond_dim, backbone=backbone)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
