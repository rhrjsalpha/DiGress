# -*- coding: utf-8 -*-
"""
grid_search_spectrum_v2.py
- src 밖 독립 폴더에서 실행 가능하도록 프로젝트 루트를 자동 탐색하여 sys.path에 추가
- DiGress/GNN 학습 엔진(run_from_cfg) + 내부 CV 루프를 이용해 그리드 서치
- MODE_GRID(글로벌 노드) 미사용, 대신 cond(y)에 들어갈 전역 피처 조합(GLOBAL_FEATURE_SETS)만 바꿔가며 실행
- 각 조합마다:
    (1) K-fold CV (is_cv=True, fold val=검증) → 폴더별 final_metrics_CV*.csv
    (2) Full-train + 외부 test (is_cv=False) → final_metrics_Final.csv
  + 조합별 결과 요약 CSV 생성

실행 예)
python grid_search_spectrum_v2.py \
  --train_csv /path/to/train.csv \
  --test_csv  /path/to/test.csv  \
  --base_cfg  /path/to/config_spectrum.yaml \
  --out_root  ./grid_runs --n_splits 5 --shuffle
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from omegaconf import OmegaConf, DictConfig, open_dict

# ------------------------------------------------------------
# 0) 프로젝트 루트/ src 경로 자동 합류
# ------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    """start에서 위로 올라가며 src/train_spectrum_GN_multiLoss_v2.py 가 있는 곳을 찾는다."""
    cur = start
    while True:
        src_dir = cur / "src"
        if (src_dir / "train_spectrum_GN_multiLoss_v2.py").exists():
            return cur
        if cur.parent == cur:
            # 못 찾으면 start를 반환(사용자가 직접 PYTHONPATH 를 맞춘 상태일 수 있음)
            return start
        cur = cur.parent

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")) if os.environ.get("PROJECT_ROOT") else find_project_root(THIS_FILE.parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_spectrum_GN_multiLoss_v2 import run_from_cfg  # noqa: E402

# ------------------------------------------------------------
# 1) 그리드 정의
#   - GLOBAL_FEATURE_SETS: cond(y)에 넣을 전역 컬럼 조합
#   - LOSS_GRID: 멀티로스 조합
#   - USE_GRADNORM_GRID: GradNorm on/off
#   - BACKEND_GRID: gnn/digress
# ------------------------------------------------------------
GLOBAL_FEATURE_SETS: List[Dict[str, Any]] = [
    {"tag": "GF=None",        "cols": []},
    {"tag": "GF=solv",        "cols": ["solvent_phase"]},
    {"tag": "GF=pH",          "cols": ["pH_label"]},
    {"tag": "GF=solv+pH",     "cols": ["solvent_phase", "pH_label"]},
    {"tag": "GF=dielectric",  "cols": ["dielectric_constant_avg"]},
    {"tag": "GF=all",         "cols": ["solvent_phase", "is_qm", "dielectric_constant_avg", "pH_label"]},
]

LOSS_GRID: List[List[str]] = [
    ["MSE", "MAE", "SID", "SOFTDTW"],
    ["MSE", "SID"],
    ["MAE", "SID"],
]

USE_GRADNORM_GRID: List[bool] = [True, False]

BACKEND_GRID: List[str] = ["digress"]  # 필요 시 ["gnn","digress"]

# ------------------------------------------------------------
# 2) 유틸
# ------------------------------------------------------------
@contextmanager
def pushd(new_dir: Path):
    prev = Path.cwd()
    new_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

def safe_register_now_resolver():
    """base YAML에 ${now:...}가 있어도 터지지 않도록 resolver 등록."""
    try:
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    except Exception:
        pass  # 이미 등록되어 있으면 무시

def load_base_cfg(base_cfg_path: Path) -> DictConfig:
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
    safe_register_now_resolver()
    cfg = OmegaConf.load(str(base_cfg_path))
    with open_dict(cfg):
        cfg.setdefault("dataset", {})
        cfg.dataset.setdefault("train_csv", "")
        cfg.dataset.setdefault("val_csv", "")
        cfg.dataset.setdefault("test_csv", "")
        cfg.setdefault("general", {})
        cfg.general.setdefault("name", "SpectrumJob")
        cfg.general.setdefault("is_cv", False)
        cfg.general.setdefault("fold_tag", "")
        cfg.setdefault("train", {})
        cfg.train.setdefault("n_epochs", 200)
        cfg.setdefault("model", {})
    return cfg

def write_split_csv(df: pd.DataFrame, idx: np.ndarray, out_path: Path):
    df.iloc[idx].to_csv(out_path, index=False)

def run_one_cv_and_final(
    base_cfg: DictConfig,
    train_csv: Path,
    test_csv: Path,
    out_root: Path,
    *,
    job_prefix: str,
    global_cols: List[str],
    losses: List[str],
    use_gn: bool,
    backend: str,
    n_splits: int,
    seed: int,
    shuffle: bool,
) -> Dict[str, Any]:
    """
    한 조합으로 CV + Final 실행 후 주요 결과 경로 반환.
    """
    cfg0 = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    with open_dict(cfg0):
        cfg0.model.backend = backend
        cfg0.train.losses = list(losses)
        cfg0.train.use_gradnorm = bool(use_gn)

        # ★ 전역 피처 조합을 dataset.global_cols 로 주입 (데이터모듈이 이를 읽도록 되어 있어야 함)
        cfg0.dataset.global_cols = list(global_cols)
        # 필요하다면 bool 컬럼도 지정 가능 (기본은 ["is_qm"])
        cfg0.dataset.boolean_cols = list(getattr(cfg0.dataset, "boolean_cols", ["is_qm"]))

        # CV에서는 그림 OFF (지표만 저장)
        cfg0.train.milestones_plots = []
        # 지표 마일스톤(없으면 비활성)
        cfg0.train.milestones_metrics = cfg0.train.get("milestones_metrics", cfg0.train.get("milestones", []))

    df_train = pd.read_csv(train_csv)
    if len(df_train) < n_splits:
        raise ValueError(f"Train rows({len(df_train)}) < n_splits({n_splits})")

    # ---------- (1) CV ----------
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=(seed if shuffle else None))
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    fold_dirs: List[Path] = []

    for fold_idx, (idx_tr, idx_val) in enumerate(kf.split(df_train), start=1):
        tag = f"{job_prefix}_CV{fold_idx}"
        work = out_root / f"CV{fold_idx}"
        fold_dirs.append(work)

        tr_csv = splits_dir / f"CV{fold_idx}_train.csv"
        vl_csv = splits_dir / f"CV{fold_idx}_val.csv"
        write_split_csv(df_train, idx_tr, tr_csv)
        write_split_csv(df_train, idx_val, vl_csv)

        cfg = OmegaConf.create(OmegaConf.to_container(cfg0, resolve=True))
        with open_dict(cfg):
            cfg.dataset.train_csv = str(tr_csv)
            cfg.dataset.val_csv = None
            cfg.dataset.test_csv = str(vl_csv)  # is_cv=True → 내부에서 val로 사용
            cfg.general.name = tag
            cfg.general.is_cv = True
            cfg.general.fold_tag = f"CV{fold_idx}"

        with pushd(work):
            _ = run_from_cfg(cfg, is_cv=True, fold_tag=f"CV{fold_idx}")

    # CV 요약
    cv_summary = out_root / "CV_summary"
    cv_summary.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in fold_dirs:
        f = d / f"final_metrics_{d.name}.csv"  # final_metrics_CV{i}.csv
        if f.exists():
            rows.append(pd.read_csv(f))
    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        (cv_summary / "CV_all_folds_summary.csv").write_text(all_df.to_csv(index=False))
        value_cols = [c for c in all_df.columns if c.startswith("CV_")]
        if value_cols:
            stat_df = pd.DataFrame({
                "metric": value_cols,
                "mean": [all_df[c].mean() for c in value_cols],
                "std":  [all_df[c].std(ddof=1) for c in value_cols],
            })
            (cv_summary / "CV_mean_std.csv").write_text(stat_df.to_csv(index=False))

    # ---------- (2) Final ----------
    final_dir = out_root / "Final"
    cfg_final = OmegaConf.create(OmegaConf.to_container(cfg0, resolve=True))
    with open_dict(cfg_final):
        cfg_final.dataset.train_csv = str(train_csv)
        cfg_final.dataset.val_csv = None
        cfg_final.dataset.test_csv = str(test_csv)
        cfg_final.general.name = f"{job_prefix}_Final"
        cfg_final.general.is_cv = False
        cfg_final.general.fold_tag = "Final"
        # 최종 러닝에서 그림을 원하면 base_cfg 쪽 milestones_plots를 사용 (여기선 건드리지 않음)

    with pushd(final_dir):
        _ = run_from_cfg(cfg_final, is_cv=False, fold_tag="Final")

    return {
        "cv_dir": str(out_root),
        "cv_summary": str(cv_summary / "CV_all_folds_summary.csv"),
        "cv_meanstd": str(cv_summary / "CV_mean_std.csv"),
        "final_metrics": str(final_dir / "final_metrics_Final.csv"),
    }

# ------------------------------------------------------------
# 3) 메인
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=None, help="CV에 사용할 train CSV")
    ap.add_argument("--test_csv",  default=None, help="Final에서 평가할 external test CSV")
    ap.add_argument("--base_cfg",  default=None, help="config_spectrum.yaml (미지정시 PROJECT_ROOT/configs/config_spectrum.yaml)")
    ap.add_argument("--out_root",  default="./grid_runs", help="조합별 결과 상위 폴더")
    ap.add_argument("--n_splits",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--shuffle",   action="store_true")
    args = ap.parse_args()

    # base cfg 경로
    if args.base_cfg:
        base_cfg_path = Path(args.base_cfg).resolve()
    else:
        base_cfg_path = PROJECT_ROOT / "configs" / "config_spectrum.yaml"
    base_cfg = load_base_cfg(base_cfg_path)

    # train/test 경로: CLI > ENV > base_cfg
    train_csv = Path(args.train_csv or os.getenv("TRAIN_CSV") or base_cfg.dataset.train_csv).resolve()
    test_csv  = Path(args.test_csv  or os.getenv("TEST_CSV")  or base_cfg.dataset.test_csv ).resolve()
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"train_csv={train_csv}, test_csv={test_csv} (둘 다 존재해야 함)")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    master_rows: List[Dict[str, Any]] = []
    total = len(GLOBAL_FEATURE_SETS) * len(LOSS_GRID) * len(USE_GRADNORM_GRID) * len(BACKEND_GRID)
    idx = 0

    for be in BACKEND_GRID:
        for gf in GLOBAL_FEATURE_SETS:
            for ls in LOSS_GRID:
                for use_gn in USE_GRADNORM_GRID:
                    idx += 1
                    tag = f"{tstamp}__{be}__{gf['tag']}__{'GN' if use_gn else 'NoGN'}__{'-'.join(ls)}"
                    work = out_root / tag
                    print(f"\n=== [{idx}/{total}] {tag} ===")
                    res = run_one_cv_and_final(
                        base_cfg, train_csv, test_csv, work,
                        job_prefix=tag,
                        global_cols=gf["cols"],
                        losses=ls,
                        use_gn=use_gn,
                        backend=be,
                        n_splits=args.n_splits,
                        seed=args.seed,
                        shuffle=bool(args.shuffle),
                    )
                    master_rows.append({
                        "tag": tag,
                        "backend": be,
                        "global_features": gf["tag"],
                        "losses": "-".join(ls),
                        "use_gradnorm": use_gn,
                        **res
                    })

    # 조합별 요약
    summary = out_root / f"grid_summary_{tstamp}.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
        w.writeheader()
        w.writerows(master_rows)

    print(f"\n✅ GRID DONE")
    print(f"- Project root: {PROJECT_ROOT}")
    print(f"- Summary:      {summary}")

if __name__ == "__main__":
    main()
