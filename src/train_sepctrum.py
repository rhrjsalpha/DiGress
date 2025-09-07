# -*- coding: utf-8 -*-
# src/train_spectrum.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm

# 여러분의 경로에 맞게 import 경로 확인
from src.datasets.csv_spectrum_dataset import CSVSpecDataset

# ---------- 모델 ----------
class GraphSpectrumNet(nn.Module):
    def __init__(self, node_in=10, edge_in=6, hidden=256, layers=4,
                 cond_dim=0, out_dim=601, dropout=0.1):
        super().__init__()
        # 노드 임베딩
        self.node_in = node_in
        self.edge_in = edge_in
        self.cond_dim = cond_dim
        self.out_dim = out_dim

        self.node_embed = nn.Linear(node_in, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            conv = GINEConv(nn_mlp)
            self.gnn_layers.append(conv)
            self.norms.append(BatchNorm(hidden))
        self.dropout = nn.Dropout(dropout)

        # 리드아웃: 풀링 후 cond와 concat → MLP → 스펙트럼
        readin = hidden + cond_dim
        self.readout = nn.Sequential(
            nn.Linear(readin, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data, cond=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.node_embed(x)
        e = self.edge_mlp(edge_attr) if edge_attr is not None else None
        for conv, bn in zip(self.gnn_layers, self.norms):
            h = conv(h, edge_index, e)  # GINE는 edge_attr가 내부 mlp에 들어감
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
        g = global_mean_pool(h, batch)  # (B, hidden)

        if cond is not None and cond.numel() > 0:
            g = torch.cat([g, cond], dim=-1)  # (B, hidden+cond_dim)
        y_hat = self.readout(g)               # (B, out_dim)
        return y_hat


# ---------- 학습 루프 ----------
def split_y(batch_y: torch.Tensor, spec_len: int):
    # batch_y: (B, L)
    y_spec = batch_y[:, :spec_len]
    cond = batch_y[:, spec_len:] if batch_y.size(1) > spec_len else None
    return y_spec, cond

def train_one_epoch(model, loader, spec_len, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y_spec, cond = split_y(batch.y, spec_len)
        y_hat = model(batch, cond=cond)
        loss = loss_fn(y_hat, y_spec)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y_spec.size(0)
        n += y_spec.size(0)
    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, spec_len, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y_spec, cond = split_y(batch.y, spec_len)
        y_hat = model(batch, cond=cond)
        loss = loss_fn(y_hat, y_spec)
        total_loss += float(loss.item()) * y_spec.size(0)
        n += y_spec.size(0)
    return total_loss / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="Train spectrum regressor from CSVSpecDataset")
    # CSV 경로들
    p.add_argument("--train-csv", default="../data/csv/EM_stratified_train_clustered_resplit_with_mu_eps.csv")
    p.add_argument("--val-csv",   default="", help="비우면 train에서 split")
    p.add_argument("--test-csv",  default="../data/csv/EM_stratified_test_clustered_resplit_with_mu_eps.csv")
    # 스펙트럼 범위
    p.add_argument("--spec-start", type=int, default=200)
    p.add_argument("--spec-end",   type=int, default=800)
    # 고정 vocab/불리언 (코드 내에서 고정)
    p.add_argument("--solvent-vocab", nargs="*", default=["solid","liquid","gas"])
    p.add_argument("--ph-vocab",      nargs="*", default=["acidic","basic","neutral"])
    p.add_argument("--bool-cols",     nargs="*", default=["is_qm"])
    # 학습 하이퍼
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec_len = args.spec_end - args.spec_start + 1

    # ----- Dataset 준비 -----
    fixed_vocabs = {"solvent_phase": args.solvent-vocab if hasattr(args, "solvent-vocab") else args.solvent_vocab,
                    "pH_label": args.ph_vocab}
    boolean_cols = args.bool_cols
    GLOBAL_COLS = ["solvent_phase", "is_qm", "dielectric_constant_avg", "pH_label"]

    # train ds (stage=train → stats.json 생성)
    ds_train = CSVSpecDataset(
        csv_path=args.train_csv, stage="train",
        inchi_col="InChI", smiles_col=None,
        spectrum_start=args.spec_start, spectrum_end=args.spec_end,
        global_cols=GLOBAL_COLS, stats_path=None,
        spectrum_fill_eps=1e-8, fixed_vocabs=fixed_vocabs, boolean_cols=boolean_cols,
    )

    # val ds
    if args.val_csv:
        ds_val = CSVSpecDataset(
            csv_path=args.val_csv, stage="val",
            inchi_col="InChI", smiles_col=None,
            spectrum_start=args.spec_start, spectrum_end=args.spec_end,
            global_cols=GLOBAL_COLS, stats_path=str(Path(args.train_csv).with_suffix("")) + "_stats.json",
            spectrum_fill_eps=1e-8, fixed_vocabs=fixed_vocabs, boolean_cols=boolean_cols,
        )
    else:
        # train을 90/10으로 나누기
        n_total = len(ds_train)
        n_val = max(1, int(0.1 * n_total))
        n_train = n_total - n_val
        ds_train, ds_val = random_split(ds_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # test ds (선택)
    ds_test = CSVSpecDataset(
        csv_path=args.test_csv, stage="test",
        inchi_col="InChI", smiles_col=None,
        spectrum_start=args.spec_start, spectrum_end=args.spec_end,
        global_cols=GLOBAL_COLS, stats_path=str(Path(args.train_csv).with_suffix("")) + "_stats.json",
        spectrum_fill_eps=1e-8, fixed_vocabs=fixed_vocabs, boolean_cols=boolean_cols,
    )

    # cond_dim 추정: 첫 샘플에서 y 크기 - spec_len
    y_dim0 = int(ds_train[0].y.numel())
    cond_dim = max(0, y_dim0 - spec_len)

    # ----- Dataloader -----
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # ----- Model -----
    model = GraphSpectrumNet(node_in=10, edge_in=6, hidden=args.hidden, layers=args.layers,
                             cond_dim=cond_dim, out_dim=spec_len, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf"); best_state = None
    print(f"Training... (B={args.batch_size}, epochs={args.epochs}, cond_dim={cond_dim}, out={spec_len})")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, spec_len, opt, device)
        va_loss = evaluate(model, val_loader, spec_len, device)
        print(f"[{epoch:03d}] train={tr_loss:.6f}  val={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ----- Test -----
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss = evaluate(model, test_loader, spec_len, device)
    print(f"[TEST] MSE={te_loss:.6f}")

    # ----- 저장 -----
    outdir = Path("./generated_samples"); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / "spectrum_regressor.pt"
    torch.save({"model": model.state_dict(),
                "spec_len": spec_len,
                "cond_dim": cond_dim,
                "config": vars(args)}, ckpt)
    print(f"[Saved] {ckpt.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
