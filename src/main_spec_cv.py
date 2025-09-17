# -*- coding: utf-8 -*-
"""
main_spec_cv.py — 인자 없이 바로 실행 가능한 K-fold 교차검증 (val→test 자동 치환)

핵심:
- 코드 안에 프로젝트/데이터/오버라이드 기본값을 모두 정의 (CLI 인자 없어도 동작)
- 교차검증 중 '검증 루프' 비활성화: dataset.val_csv=null
- 각 fold의 val_fold.csv를 test로 사용해 fold 성능을 평가 (cond_y_base 요구 회피)
- 서브프로세스는 [sys.executable, main_spec.py, overrides...] 리스트 호출(shell=False)

원하면 CLI 인자로 언제든 덮어쓸 수 있음.
"""
from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# =========================
# 1) 코드 내부 기본값 정의
# =========================
def _auto_project_root() -> Path:
    """repo 루트를 자동 탐색. 못 찾으면 Windows 예시 경로로 폴백."""
    here = Path(__file__).resolve()
    # src/main_spec.py를 발견하면 그 상위 폴더를 루트로 간주
    for p in [here.parent, *here.parents]:
        if (p / "src" / "main_spec.py").exists():
            return p
    # Windows 사용자 폴백 (원하면 바꿔도 됨)
    fallback = Path(r"C:\Users\kogun\PycharmProjects\DiGress")
    return fallback if (fallback / "src" / "main_spec.py").exists() else here.parent

PROJECT_ROOT_DEFAULT = _auto_project_root()
DATA_DIR_DEFAULT     = PROJECT_ROOT_DEFAULT / "data" / "csv"
print(DATA_DIR_DEFAULT)
# ⚠️ 데이터 파일 기본값 (원하면 파일명만 바꿔도 됩니다)
TRAIN_CSV_DEFAULT = DATA_DIR_DEFAULT / "EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"# "EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
TEST_CSV_DEFAULT  = DATA_DIR_DEFAULT / "EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"#"EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"

OUT_ROOT_DEFAULT     = PROJECT_ROOT_DEFAULT / "cv_runs"
NAME_PREFIX_DEFAULT  = "specCV"
N_SPLITS_DEFAULT     = 5
SEED_DEFAULT         = 100
STRATIFY_BY_DEFAULT  = None   # 예: "solvent_class,pH_bin"

n_epochs = 1

# 하이드라 오버라이드 기본값 (원하면 여기서 수정)
EXTRA_OVERRIDES_DEFAULT = [
    "dataset.name=csvspec",
    "train.save_model=True",
    f"train.n_epochs={n_epochs}",
    "general.gpus=1",
    # "trainer.precision=16-mixed",
]


# ================
# 2) 유틸 함수들
# ================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _combine_stratify_labels(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        raise ValueError("cols is empty for stratification")
    work = []
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"stratify_by column '{c}' not in CSV")
        s = df[c].astype(str).fillna("NA")
        work.append(s)
    combo = pd.Series(["|".join(x) for x in zip(*work)], index=df.index)
    return combo.values

def _write_fold_csvs(df: pd.DataFrame, tr_idx: np.ndarray, va_idx: np.ndarray, out_dir: Path) -> Tuple[Path, Path]:
    _ensure_dir(out_dir)
    tr = out_dir / "train_fold.csv"
    va = out_dir / "val_fold.csv"
    df.iloc[tr_idx].to_csv(tr, index=False)
    df.iloc[va_idx].to_csv(va, index=False)
    return tr, va

def _run_hydra(main_spec: Path, overrides: List[str]) -> int:
    """
    sys.executable과 리스트 인자로 안전 실행(shell=False).
    예: [python, main_spec.py, 'dataset.train_csv=...', ...]
    """
    cmd = [sys.executable, str(main_spec), *overrides]
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, shell=False)
    return proc.returncode

def _collect_metrics(search_dir: Path, pattern_glob: str = "final_metrics*.csv") -> List[Path]:
    return sorted(search_dir.rglob(pattern_glob))

def _aggregate_fold_metrics(fold_dirs: List[Path], out_csv: Path) -> None:
    frames = []
    for d in fold_dirs:
        csvs = _collect_metrics(d)
        if not csvs:
            print(f"[WARN] no final_metrics CSV under {d}")
            continue
        csv_path = csvs[-1]  # 최신 파일 하나 선택
        try:
            df = pd.read_csv(csv_path)
            df.insert(0, "fold_dir", str(d))
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {csv_path}: {e}")
    if not frames:
        print("[WARN] no fold metrics collected; skip aggregation")
        return
    big = pd.concat(frames, ignore_index=True)
    _ensure_dir(out_csv.parent)
    big.to_csv(out_csv, index=False)
    print(f"[OK] aggregated fold metrics → {out_csv}")


# ==================
# 3) 인자 파서 (모두 기본값 제공)
# ==================
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-contained K-fold CV runner (val→test)")
    p.add_argument("--train_csv", default=str(TRAIN_CSV_DEFAULT))
    p.add_argument("--test_csv",  default=str(TEST_CSV_DEFAULT))  # 전체 재학습용(옵션)
    p.add_argument("--out_root",  default=str(OUT_ROOT_DEFAULT))
    p.add_argument("--n_splits",  type=int, default=N_SPLITS_DEFAULT)
    p.add_argument("--seed",      type=int, default=SEED_DEFAULT)
    p.add_argument("--stratify_by", default=STRATIFY_BY_DEFAULT, help='예: "solvent_class,pH_bin"')

    p.add_argument("--project_root", default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--main_spec", default=None)  # 기본 None이면 <project_root>/src/main_spec.py

    p.add_argument("--name_prefix", default=NAME_PREFIX_DEFAULT)
    # 공백으로 나눠 여러 개 전달 가능. default는 코드 상단 리스트(EXTRA_OVERRIDES_DEFAULT)
    p.add_argument("--extra_overrides", default=None)

    p.add_argument("--retrain_full", action="store_true",
                   help="CV 완료 후 train 전체+외부 test_csv로 1회 최종 평가")
    return p


# ==================
# 4) 메인 실행
# ==================
def main():
    args = make_parser().parse_args()

    project_root = Path(args.project_root).resolve()
    out_root     = Path(args.out_root).resolve()
    _ensure_dir(out_root)

    main_spec = Path(args.main_spec).resolve() if args.main_spec else (project_root / "src" / "main_spec.py")
    if not main_spec.exists():
        raise FileNotFoundError(f"main_spec.py not found: {main_spec}")

    train_csv = Path(args.train_csv).resolve()
    test_csv  = Path(args.test_csv).resolve() if args.test_csv else None

    # extra overrides 확정 (리스트 형태 유지)
    if args.extra_overrides:
        extra_over = args.extra_overrides.strip().split()
    else:
        extra_over = list(EXTRA_OVERRIDES_DEFAULT)

    # 데이터 불러와 split
    df = pd.read_csv(train_csv)
    n = len(df)
    print(f"[INFO] TrainPool rows: {n}")
    if args.stratify_by:
        cols = [c.strip() for c in args.stratify_by.split(",") if c.strip()]
        labels = _combine_stratify_labels(df, cols)
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.arange(n), labels))
        print(f"[INFO] StratifiedKFold by {cols}, n_splits={args.n_splits}, seed={args.seed}")
    else:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.arange(n)))
        print(f"[INFO] KFold n_splits={args.n_splits}, seed={args.seed}")

    fold_dirs: List[Path] = []

    # ===== 각 폴드 실행 (val→test 자동 치환) =====
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        fold_dir = out_root / f"fold_{fold_id:02d}"
        _ensure_dir(fold_dir)
        fold_dirs.append(fold_dir)

        f_train, f_val = _write_fold_csvs(df, tr_idx, va_idx, fold_dir)

        overrides = [
            f"dataset.train_csv={str(f_train)}",
            "dataset.val_csv=null",                 # ✅ 검증 비활성화
            f"dataset.test_csv={str(f_val)}",       # ✅ val을 test로 사용
            f"general.name={args.name_prefix}_fold{fold_id}",
            # 추가 오버라이드들
            *extra_over,
        ]

        ret = _run_hydra(main_spec, overrides)
        if ret != 0:
            print(f"[ERROR] fold {fold_id} failed with code {ret}")
            sys.exit(ret)

    # ===== 폴드 결과 집계 =====
    _aggregate_fold_metrics(fold_dirs, out_csv=out_root / "cv_folds_aggregated.csv")

    # ===== (옵션) 전체 재학습 + 외부 test =====
    if args.retrain_full and test_csv is not None:
        final_dir = out_root / "final_train_full"
        _ensure_dir(final_dir)

        overrides = [
            f"dataset.train_csv={str(train_csv)}",
            "dataset.val_csv=null",
            f"dataset.test_csv={str(test_csv)}",
            f"general.name={args.name_prefix}_final",
            *extra_over,
        ]
        ret = _run_hydra(main_spec, overrides)
        if ret != 0:
            print("[ERROR] final train+test failed with code", ret)
            sys.exit(ret)

        # 최종 결과를 out_root로 복사(선택)
        for p in _collect_metrics(final_dir):
            dst = out_root / p.name
            try:
                shutil.copy2(p, dst)
                print(f"[OK] copied {p.name} → {dst}")
            except Exception as e:
                print(f"[WARN] copy failed {p}: {e}")


if __name__ == "__main__":
    main()


