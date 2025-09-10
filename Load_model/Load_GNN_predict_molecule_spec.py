# tools/predict_smiles_from_ckpt.py
from pathlib import Path
import sys
import csv
import torch
from omegaconf import OmegaConf, open_dict
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.datasets.csv_spectrum_dataset import CSVSpecDataset ,
from src.train_spectrum_GN_multiLoss_v2
from src.train_spectrum_GN_multiLoss_v2 import (
    CSVSpecDataModule, SpectrumModule, build_backbone
)
from src.datasets.csv_spectrum_dataset import build_graph  # 그래프 생성 재사용

def load_cfg(cfg_path, train_csv=None, backend="gnn"):
    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        if train_csv: cfg.dataset.train_csv = str(Path(train_csv).resolve())
        cfg.general.is_cv = False
        cfg.model.backend = backend
    return cfg

def load_model_env(ckpt_path, cfg):
    dm = CSVSpecDataModule(cfg)
    # 차원/통계 확보를 위해 train만 있으면 충분
    dm.override_splits(use_val=False, use_test=False)
    dm.setup()
    backbone = build_backbone(cfg, dm, cond_dim=dm.cond_dim, spec_len=dm.spec_len)
    model = SpectrumModule.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        backbone=backbone,
        map_location=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.eval()
    device = model.device
    return model, dm, device

def predict_spectra(model, dm, device, smiles_list):
    preds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            preds.append((smi, None))
            continue
        g = build_graph(mol)                 # x, edge_index, edge_attr 세팅
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        # cond(글로벌 피처)가 있더라도 추론만 할 때는 0으로 채워도 OK
        cond = torch.zeros(1, dm.cond_dim, dtype=torch.float32) if dm.cond_dim > 0 else None

        g = g.to(device)
        if cond is not None:
            cond = cond.to(device)
        with torch.no_grad():
            yhat = model.model(g, cond).squeeze(0).detach().cpu().numpy()  # (spec_len,)
        preds.append((smi, yhat))
    return preds

def main():
    import argparse, numpy as np
    ap = argparse.ArgumentParser(description="Predict spectra from SMILES using a saved GNN checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg",  required=True)
    ap.add_argument("--train_csv", required=True, help="차원/통계 복원용 학습 CSV (필수)")
    ap.add_argument("--smiles", nargs="+", help="예측할 SMILES들 (공백 구분)")
    ap.add_argument("--smiles_csv", default=None, help="SMILES가 들어있는 CSV (컬럼명: smiles)")
    ap.add_argument("--out_csv", default="predicted_spectra.csv")
    args = ap.parse_args()

    # SMILES 수집
    smi_list = []
    if args.smiles:
        smi_list.extend(args.smiles)
    if args.smiles_csv:
        import pandas as pd
        df = pd.read_csv(args.smiles_csv)
        smi_list.extend(df["smiles"].astype(str).tolist())
    smi_list = [s for s in smi_list if s and s.strip()]
    assert len(smi_list) > 0, "예측할 SMILES가 없습니다."

    cfg = load_cfg(args.cfg, train_csv=args.train_csv, backend="gnn")
    model, dm, device = load_model_env(args.ckpt, cfg)
    preds = predict_spectra(model, dm, device, smi_list)

    # 저장
    wl_start, wl_end = cfg.dataset.spec_start, cfg.dataset.spec_end
    header = ["smiles"] + [str(w) for w in range(wl_start, wl_end + 1)]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for smi, arr in preds:
            if arr is None:
                w.writerow([smi] + [""] * (wl_end - wl_start + 1))
            else:
                w.writerow([smi] + list(map(float, arr)))
    print(f"[OK] saved → {Path(args.out_csv).resolve()}")

if __name__ == "__main__":
    main()
