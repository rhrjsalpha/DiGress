# -*- coding: utf-8 -*-
# CV_spectrum_GN_multiLoss_v2.py
#
# - 같은 폴더의 train_spectrum_GN_multiLoss_v2.py 에서 run_from_cfg()를 import하여 사용
# - 입력: train_csv, test_csv
# - 절차:
#   (1) train_csv로 K-Fold CV 수행 (is_cv=True, fold_val은 각 Fold의 holdout)
#   (2) Fold별 결과 final_metrics_CV{i}.csv 생성 + 한 곳에 취합/평균/표준편차
#   (3) 전체 train으로 최종 학습 후 test 평가 (is_cv=False) → final_metrics_Final.csv
#
# 실행 예시:
#   python CV_spectrum_GN_multiLoss_v2.py \
#       --train_csv ./data/train.csv \
#       --test_csv  ./data/test.csv \
#       --n_splits 5 \
#       --seed 42 \
#       --job_name MySpectrumJob \
#       --base_cfg ../configs/config_spectrum.yaml
#
# 메모:
# - CV 모드(is_cv=True)에서는 내부 run_from_cfg가 "test_csv를 val로 사용하는" 로직을 갖고 있으므로,
#   여기서는 "fold_val_csv를 cfg.dataset.test_csv로 설정"합니다.
# - 최종(Final) 모드(is_cv=False)에서는 원래 test_csv로 테스트 평가합니다.

from __future__ import annotations
import os
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from omegaconf import OmegaConf, DictConfig, open_dict
from datetime import datetime
# 같은 폴더의 training 엔진
from train_spectrum_GN_multiLoss_v2 import run_from_cfg

@contextmanager
def pushd(new_dir: Path):
    prev = Path.cwd()
    new_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

def dump_cfg_to_yaml(cfg, out_dir: Path, tag: str):
    """현재 cfg를 YAML로 저장하고 콘솔에도 예쁘게 출력"""
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = OmegaConf.to_yaml(cfg, resolve=False)  # Hydra 보간자 충돌 방지
    (out_dir / f"cfg_input_{tag}.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\n[CFG:{tag}] -----------------------")
    print(yaml_text)
    print(f"-----------------------------------\n")

def load_base_cfg(base_cfg_path: Path) -> DictConfig:
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
    cfg = OmegaConf.load(str(base_cfg_path))
    # 필요한 키가 없으면 만들어 둠
    with open_dict(cfg):
        cfg.setdefault("dataset", {})
        cfg.dataset.setdefault("train_csv", "")
        cfg.dataset.setdefault("val_csv", "")
        cfg.dataset.setdefault("test_csv", "")
        cfg.setdefault("general", {})
        cfg.general.setdefault("name", "job")
        cfg.general.setdefault("is_cv", False)
        cfg.general.setdefault("fold_tag", "")
        cfg.setdefault("train", {})
        cfg.train.setdefault("n_epochs", 200)
    return cfg


def write_split_csv(df: pd.DataFrame, idx: np.ndarray, out_path: Path):
    df.iloc[idx].to_csv(out_path, index=False)


def collect_and_save_cv_summary(fold_dirs: List[Path],
                                out_dir: Path,
                                summary_csv_name: str = "CV_all_folds_summary.csv",
                                stats_csv_name: str = "CV_mean_std.csv") -> Dict[str, str]:
    """
    fold_dir/final_metrics_CV{i}.csv 들을 모아서 하나로 합치고,
    주요 컬럼(CV_training_*, CV_val_*)에 대해 평균/표준편차를 계산해 별도 CSV 저장.
    """
    rows = []
    for d in fold_dirs:
        f = d / f"final_metrics_{d.name}.csv"  # d.name = CV{i}
        if f.exists():
            rows.append(pd.read_csv(f))
    if not rows:
        return {"summary_csv": "", "stats_csv": ""}

    all_df = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / summary_csv_name
    all_df.to_csv(summary_path, index=False)

    # 평균/표준편차는 CV_* 로 시작하는 수치 컬럼만 대상으로 계산
    value_cols = [c for c in all_df.columns if c.startswith("CV_")]
    stats = {}
    if value_cols:
        stats["metric"] = value_cols
        stats["mean"] = [all_df[c].mean() for c in value_cols]
        stats["std"] = [all_df[c].std(ddof=1) for c in value_cols]
    stats_df = pd.DataFrame(stats) if stats else pd.DataFrame()

    stats_path = out_dir / stats_csv_name
    stats_df.to_csv(stats_path, index=False)

    return {"summary_csv": str(summary_path), "stats_csv": str(stats_path)}


# ---- Excution Test  ----
USE_INLINE_ARGS = True  # ← True이면 아래 INLINE 값이 기본으로 쓰임(CLI가 있으면 CLI가 우선)
INLINE = {
    "train_csv": "/home/user/Spectral_Data/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
    "test_csv":  "/home/user/Spectral_Data/EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv",
    "n_splits": 5,
    "seed": 42,
    "shuffle": False,              # True/False
    "job_name": "SpectrumJob",
    "base_cfg": None,              # None이면 기본 ../configs/config_spectrum.yaml 사용
    "out_dir": "./_cv_runs",
}

def _none_if_empty(x):
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None

def _coalesce(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None
def main():
    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    parser = argparse.ArgumentParser(description="K-Fold CV + Final Train/Test for Spectrum model")
    # ✅ required 제거하고 기본값은 None으로 둠(실제 기본은 INLINE/ENV/base_cfg 순으로 보충)
    parser.add_argument("--train_csv", type=str, default=None, help="Input training CSV (used for CV folds)")
    parser.add_argument("--test_csv", type=str, default=None, help="External test CSV (Final run)")
    parser.add_argument("--n_splits", type=int, default=None, help="Number of folds for K-Fold CV")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for K-Fold shuffling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting (default: False)")
    parser.add_argument("--job_name", type=str, default=None, help="Base job name")
    parser.add_argument("--base_cfg", type=str, default=None,
                        help="Path to base hydra yaml (default: ../configs/config_spectrum.yaml)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to create fold/final working dirs")
    args = parser.parse_args()

    # --- base cfg 경로 결정
    default_base_cfg = Path(__file__).resolve().parents[1] / "configs" / "config_spectrum.yaml"
    base_cfg_path = Path(_coalesce(args.base_cfg, INLINE.get("base_cfg"), str(default_base_cfg))).resolve()
    base_cfg = load_base_cfg(base_cfg_path)

    # --- 각 인자 최종값 결정(우선순위: CLI -> ENV -> INLINE -> (train/test만) base_cfg)
    train_csv_str = _coalesce(
        args.train_csv,
        os.environ.get("TRAIN_CSV"),
        INLINE.get("train_csv"),
        _none_if_empty(base_cfg.dataset.train_csv),
    )
    test_csv_str = _coalesce(
        args.test_csv,
        os.environ.get("TEST_CSV"),
        INLINE.get("test_csv"),
        _none_if_empty(base_cfg.dataset.test_csv),
    )
    n_splits = int(_coalesce(args.n_splits, os.environ.get("N_SPLITS"), INLINE.get("n_splits"), 5))
    seed = int(_coalesce(args.seed, os.environ.get("SEED"), INLINE.get("seed"), 42))
    # shuffle: CLI에 --shuffle 주면 True, 아니면 INLINE 값/ENV 값 사용
    shuffle = bool(args.shuffle or str(os.environ.get("SHUFFLE", "")).lower() in ("1", "true", "yes") or INLINE.get("shuffle", False))
    job_name = str(_coalesce(args.job_name, os.environ.get("JOB_NAME"), INLINE.get("job_name"), "SpectrumJob"))
    out_dir_str = str(_coalesce(args.out_dir, os.environ.get("OUT_DIR"), INLINE.get("out_dir"), "./_cv_runs"))

    # --- 필수 경로 체크
    if not train_csv_str or not test_csv_str:
        raise SystemExit(
            "train_csv/test_csv가 설정되지 않았습니다. "
            "① 코드 상단 INLINE 경로를 채우거나 ② CLI 인자(--train_csv/--test_csv)나 "
            "③ 환경변수 TRAIN_CSV/TEST_CSV를 설정하세요."
        )

    train_csv = Path(train_csv_str).resolve()
    test_csv = Path(test_csv_str).resolve()
    out_root = Path(out_dir_str).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Read full train
    df_train = pd.read_csv(train_csv)
    n_total = len(df_train)
    if n_total < n_splits:
        raise ValueError(f"Train rows({n_total}) < n_splits({n_splits})")

    # Prepare split writer dir
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # (1) K-Fold CV
    # -----------------------------
    kf_random_state = seed if shuffle else None
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=kf_random_state)
    fold_dirs: List[Path] = []
    for fold_idx, (idx_tr, idx_val) in enumerate(kf.split(df_train), start=1):
        fold_tag = f"CV{fold_idx}"
        fold_dir = out_root / fold_tag
        fold_dirs.append(fold_dir)

        # Save split CSVs
        tr_csv = splits_dir / f"{fold_tag}_train.csv"
        vl_csv = splits_dir / f"{fold_tag}_val.csv"
        write_split_csv(df_train, idx_tr, tr_csv)
        write_split_csv(df_train, idx_val, vl_csv)

        # Compose cfg for this fold (is_cv=True → fold val을 test_csv로 주입)
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        #cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
        with open_dict(cfg):
            cfg.dataset.train_csv = str(tr_csv)
            cfg.dataset.val_csv = None
            cfg.dataset.test_csv = str(vl_csv)
            cfg.general.name = f"{job_name}_{fold_tag}"
            cfg.general.is_cv = True
            cfg.general.fold_tag = fold_tag

        print(f"\n[CV] Running {fold_tag} → train={tr_csv.name}, val={vl_csv.name}")
        with pushd(fold_dir):
            dump_cfg_to_yaml(cfg, Path.cwd(), tag=fold_tag)
            _ = run_from_cfg(cfg, is_cv=True, fold_tag=fold_tag)

    # CV 요약 저장
    cv_summary_dir = out_root / "CV_summary"
    merged = collect_and_save_cv_summary(fold_dirs, cv_summary_dir)

    # -----------------------------
    # (2) Final train on full train, test on external test
    # -----------------------------
    final_dir = out_root / "Final"
    final_tag = "Final"
    cfg_final = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    #cfg_final = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
    with open_dict(cfg_final):
        cfg_final.dataset.train_csv = str(train_csv)
        cfg_final.dataset.val_csv = None
        cfg_final.dataset.test_csv = str(test_csv)
        cfg_final.general.name = f"{job_name}_{final_tag}"
        cfg_final.general.is_cv = False
        cfg_final.general.fold_tag = final_tag

    print(f"\n[Final] Running final training → train={train_csv.name}, test={test_csv.name}")
    with pushd(final_dir):
        dump_cfg_to_yaml(cfg_final, Path.cwd(), tag=final_tag)
        _ = run_from_cfg(cfg_final, is_cv=False, fold_tag=final_tag)

    # -----------------------------
    # (3) Top-level 안내 출력
    # -----------------------------
    print("\n================ DONE ================")
    print(f"CV fold dirs: {[str(d) for d in fold_dirs]}")
    print(f"CV merged summary: {merged.get('summary_csv','')}")
    print(f"CV mean/std:       {merged.get('stats_csv','')}")
    print(f"Final run dir:     {str(final_dir)}")
    print("======================================")

if __name__ == "__main__":
    main()
