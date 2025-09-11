# -*- coding: utf-8 -*-
"""
예측 전용 스크립트 (모듈 고정판: src.train_spectrum_GN_multiLoss_v2)
- 학습된 ckpt + config_spectrum.yaml + 입력 CSV(SMILES/전역조건) -> pred_200~pred_800 컬럼 추가 저장
- 입력 CSV에 200~800 컬럼이 없어도 자동으로 0.0으로 채워 예측 가능
- (옵션) 앞에서 N개 플롯 저장

사용 예)
python tools/predict_from_ckpt_v2.py \
  --ckpt ./checkpoints/spectrum_regression/best.ckpt \
  --config ./configs/config_spectrum.yaml \
  --csv ./data/new_molecules.csv \
  --out ./data/new_molecules_pred.csv \
  --batch-size 128 --cuda --plot-n 8
"""
from __future__ import annotations
import os, sys, argparse, tempfile, warnings
from pathlib import Path
from typing import List, Optional

import torch
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from datetime import datetime

OmegaConf.register_new_resolver("now", lambda fmt="": datetime.now().strftime(fmt))

# -------- Repo 경로 추가 --------
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]  # tools/ 상위
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# -------- 학습 모듈(고정) 임포트 --------
from src.train_spectrum_GN_multiLoss_v2 import (
    CSVSpecDataModule,
    SpectrumModule,
    build_backbone,
)
from torch_geometric.loader import DataLoader
from types import SimpleNamespace


# ==== 인라인 설정 (CLI 대신 코드에서 직접 설정) ====
USE_INLINE_ARGS = True  # <- True면 아래 INLINE_ARGS를 사용, False면 기존 argparse 사용

INLINE_ARGS = {
    "ckpt":   r"/root/PycharmProjects/DiGress/checkpoints/spectrum_regression/last-v11.ckpt",
    "config": r"/root/PycharmProjects/DiGress/configs/config_spectrum.yaml",
    "csv":    r"/root/PycharmProjects/DiGress/data/csv/Search.csv",
    "out":    r"/root/PycharmProjects/DiGress/Load_model/new_mol_pred.csv",  # None이면 <입력>_pred.csv로 저장
    "batch_size": 128,
    "cuda": True,             # GPU 사용(가능할 때)
    "plot_n": 8,              # 0이면 플롯 저장 안 함
    "plots_dir": "pred_plots",
    "overwrite_pred_cols": True,  # pred_XXX 컬럼이 이미 있으면 덮어쓰기
}

def parse_or_inline_args():
    ap = argparse.ArgumentParser(description="Predict spectra (200~800nm) from a trained ckpt.")
    ap.add_argument("--ckpt", required=not USE_INLINE_ARGS, help="Lightning checkpoint (.ckpt)")
    ap.add_argument("--config", required=not USE_INLINE_ARGS, help="config_spectrum.yaml (학습 시 사용)")
    ap.add_argument("--csv", required=not USE_INLINE_ARGS, help="입력 CSV (SMILES/InChI + 전역조건)")
    ap.add_argument("--out", default=None, help="출력 CSV (기본: <입력>_pred.csv)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--cuda", action="store_true", help="가능하면 CUDA 사용")
    ap.add_argument("--plot-n", type=int, default=0, help="앞에서 N개 플롯 저장 (0=off)")
    ap.add_argument("--plots-dir", default="pred_plots", help="플롯 저장 폴더")
    ap.add_argument("--overwrite-pred-cols", action="store_true",
                    help="이미 존재하는 pred_XXX 컬럼 덮어쓰기")

    if USE_INLINE_ARGS:
        # argparse를 건너뛰고 코드 내 상수로부터 args 생성
        ia = INLINE_ARGS
        return SimpleNamespace(
            ckpt=ia["ckpt"],
            config=ia["config"],
            csv=ia["csv"],
            out=ia.get("out"),
            batch_size=ia.get("batch_size", 128),
            cuda=bool(ia.get("cuda", False)),
            plot_n=int(ia.get("plot_n", 0)),
            plots_dir=ia.get("plots_dir", "pred_plots"),
            overwrite_pred_cols=bool(ia.get("overwrite_pred_cols", False)),
        )
    else:
        return ap.parse_args()

def ensure_wavelength_columns(df: pd.DataFrame, start: int, end: int, fill_value: float = 0.0) -> pd.DataFrame:
    """df에 start~end(정수 문자열) 파장 컬럼이 없으면 생성."""
    df = df.copy()
    for wl in range(start, end + 1):
        col = str(wl)
        if col not in df.columns:
            df[col] = fill_value
    return df


def save_prediction_plots(
    outdir: Path,
    wl_grid: np.ndarray,
    preds: np.ndarray,
    trues: Optional[np.ndarray],
    smiles_list: Optional[List[str]],
    top_n: int = 8,
):
    outdir.mkdir(parents=True, exist_ok=True)
    k = min(top_n, preds.shape[0])
    for i in range(k):
        fig = plt.figure(figsize=(6, 3))
        plt.plot(wl_grid, preds[i], label="pred")
        if trues is not None:
            plt.plot(wl_grid, trues[i], label="true")
            plt.legend()
        if smiles_list and i < len(smiles_list) and isinstance(smiles_list[i], str):
            plt.title(smiles_list[i][:60])
        plt.xlabel("wavelength (nm)")
        plt.ylabel("intensity")
        plt.tight_layout()
        fig.savefig(outdir / f"sample_{i:03d}.png", dpi=140)
        plt.close(fig)


def main():
    args = parse_or_inline_args()

    # -------- config 로드 / 파장 범위 --------
    cfg = OmegaConf.load(args.config)
    spec_start = int(cfg.dataset.spec_start)
    spec_end   = int(cfg.dataset.spec_end)
    wl_grid = np.arange(spec_start, spec_end + 1, dtype=np.int32)
    spec_len = int(spec_end - spec_start + 1)

    # -------- 입력 CSV 확인 --------
    in_path = Path(args.csv)
    df_in = pd.read_csv(in_path)

    required_global = ["solvent_phase", "is_qm", "dielectric_constant_avg", "pH_label"]
    for c in required_global:
        if c not in df_in.columns:
            raise ValueError(f"필수 전역조건 컬럼이 없습니다: '{c}'")

    smiles_col = getattr(cfg.dataset, "smiles_col", "SMILES")
    inchi_col  = getattr(cfg.dataset, "inchi_col",  "InChI")
    if (smiles_col not in df_in.columns) and (inchi_col not in df_in.columns):
        raise ValueError(f"SMILES({smiles_col}) 또는 InChI({inchi_col}) 컬럼이 필요합니다.")

    # 200~800 컬럼 없으면 0.0으로 보정해 임시 CSV 생성
    df_tmp = ensure_wavelength_columns(df_in, spec_start, spec_end, fill_value=0.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tf:
        tmp_csv_path = Path(tf.name)
    df_tmp.to_csv(tmp_csv_path, index=False)

    # -------- DataModule (단일 split) --------
    cfg_pred = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg_pred.dataset.train_csv = str(tmp_csv_path)
    cfg_pred.dataset.val_csv = None
    cfg_pred.dataset.test_csv = None

    dm = CSVSpecDataModule(cfg_pred)
    try:
        dm.override_splits(use_val=False, use_test=False)
    except Exception:
        pass
    dm.setup()

    # -------- 모델 & ckpt 로드 --------
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    backbone = build_backbone(cfg_pred, dm, cond_dim=dm.cond_dim, spec_len=dm.spec_len)
    module = SpectrumModule(cfg_pred, spec_len=dm.spec_len, cond_dim=dm.cond_dim, backbone=backbone)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)
    module.eval().to(device)

    # -------- 예측 --------
    loader = DataLoader(
        dm.ds_train,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(getattr(cfg.train, "num_workers", 0)),
        pin_memory=True,
        persistent_workers=(int(getattr(cfg.train, "num_workers", 0)) > 0),
    )

    preds_all: List[np.ndarray] = []
    trues_all:  List[np.ndarray] = []
    smiles_list: List[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ys   = batch.y[:, :dm.spec_len]  # true (0으로 채워져 있을 수 있음)
            cond = batch.y[:, dm.spec_len:] if dm.cond_dim > 0 else None
            yh   = module.model(batch, cond)  # [B, spec_len]
            preds_all.append(yh.detach().cpu().numpy())
            trues_all.append(ys.detach().cpu().numpy())

            if hasattr(batch, "smiles"):
                try:
                    smiles_list.extend(list(batch.smiles))
                except Exception:
                    smiles_list.extend([""] * yh.size(0))
            else:
                smiles_list.extend([""] * yh.size(0))

    preds = np.vstack(preds_all) if preds_all else np.zeros((0, spec_len), dtype=np.float32)
    trues = np.vstack(trues_all) if trues_all else None

    # -------- 결과 저장 --------
    out_path = Path(args.out) if args.out else in_path.with_suffix("").with_name(in_path.stem + "_pred.csv")
    df_out = df_in.copy()

    pred_cols = [f"pred_{wl}" for wl in wl_grid]
    if not args.overwrite_pred_cols:
        clash = [c for c in pred_cols if c in df_out.columns]
        if clash:
            raise ValueError(f"출력 컬럼과 충돌: {clash}\n--overwrite-pred-cols 로 덮어쓰거나 기존 컬럼명 변경 필요.")

    if preds.shape[0] != len(df_out):
        print(f"[WARN] 예측 행수({preds.shape[0]})와 입력 행수({len(df_out)})가 다릅니다. 인덱스/필터링 여부 확인하세요.")

    for j, wl in enumerate(wl_grid):
        df_out[f"pred_{wl}"] = preds[:, j].astype(np.float32)

    df_out.to_csv(out_path, index=False)
    print(f"[OK] Saved predictions → {out_path}")

    # -------- (옵션) 플롯 --------
    if args.plot_n > 0:
        plots_dir = Path(args.plots_dir)
        has_true = all(str(wl) in df_in.columns for wl in wl_grid)
        save_prediction_plots(
            plots_dir,
            wl_grid.astype(np.int32),
            preds=preds,
            trues=trues if has_true else None,
            smiles_list=smiles_list,
            top_n=args.plot_n,
        )
        print(f"[OK] Saved plots (first {args.plot_n}) → {plots_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
