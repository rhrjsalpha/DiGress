#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cv_runner.py — K-fold 교차검증 + (선택) 전체 재학습 & Test 평가 오케스트레이터

핵심 아이디어
- 입력 CSV(TrainPool)를 K-fold로 나누어 각 fold마다 기존 학습 스크립트(main_spec.py 등)를
  "그대로" 호출(subprocess)하여 재사용. (프로젝트 코드 최소 수정)
- fold별로 train/val CSV를 임시로 생성해 전달 → 내부 DataModule은 평소처럼 동작.
- (선택) 외부 test_csv가 있으면 마지막에 TrainPool 전체로 재학습 후 test 평가 수행.
- fold별 결과 CSV를 자동 수집/병합하여 요약본 생성.

사용 예시
python cv_runner.py \
  --train_csv /path/to/train.csv \
  --out_root ./cv_runs --n_splits 5 --seed 42 \
  --train_cmd "python /root/PycharmProjects/DiGress/src/main_spec.py --train_csv {train_csv} --val_csv {val_csv} --out_dir {out_dir} --seed {seed}"

(선택) 외부 test가 있을 때, 전체 재학습 후 test 평가까지:
python cv_runner.py \
  --train_csv /path/to/train.csv --test_csv /path/to/test.csv \
  --out_root ./cv_runs --n_splits 5 --seed 42 --retrain_full \
  --train_cmd "python /root/PycharmProjects/DiGress/src/main_spec.py --train_csv {train_csv} --val_csv {val_csv} --out_dir {out_dir} --seed {seed}" \
  --final_cmd "python /root/PycharmProjects/DiGress/src/main_spec.py --train_csv {train_csv} --test_csv {test_csv} --out_dir {out_dir} --seed {seed}"

참고
- {train_csv}, {val_csv}, {test_csv}, {out_dir}, {seed}, {fold} 플레이스홀더를 자유롭게 쓰세요.
- main_spec.py가 val_csv 미제공 시 검증을 끄도록 이미 구현되어 있으니, fold 학습엔 반드시 {val_csv}를 넣으세요.
- (선택) novelty를 fold의 train 기준으로 계산해야 한다면, 본인의 학습 스크립트가 참조 세트를 받아들이도록 한 뒤
  "--novelty_ref {train_csv}" 같은 인자를 추가해 {train_csv}를 넘기면 됩니다.
"""

from __future__ import annotations
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _combine_stratify_labels(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """여러 열을 결합해 계층화 라벨을 생성 (결측치는 'NA'로 대체)"""
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


def _write_fold_csvs(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, out_dir: Path) -> Tuple[Path, Path]:
    _ensure_dir(out_dir)
    train_path = out_dir / "train_fold.csv"
    val_path = out_dir / "val_fold.csv"
    df.iloc[train_idx].to_csv(train_path, index=False)
    df.iloc[val_idx].to_csv(val_path, index=False)
    return train_path, val_path


def _run_cmd(cmd: str) -> int:
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode


def _collect_metrics(search_dir: Path, pattern: str = "final_metrics*.csv") -> List[Path]:
    return sorted(search_dir.rglob(pattern))


def _aggregate_fold_metrics(fold_dirs: List[Path], out_csv: Path) -> None:
    """fold 디렉터리 안의 final_metrics CSV들을 합쳐 하나의 요약본으로 저장"""
    frames = []
    for d in fold_dirs:
        csvs = _collect_metrics(d)
        if not csvs:
            print(f"[WARN] no final_metrics CSV under {d}")
            continue
        # 가장 최신 파일 하나만 선택 (필요시 정책 변경)
        csv_path = csvs[-1]
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
    # 간단 요약: fold별 행이 이미 단일 요약이라면 평균/표준편차를 추가로 계산해도 됨
    # 여기서는 원본 개별 행만 합쳐 저장
    _ensure_dir(out_csv.parent)
    big.to_csv(out_csv, index=False)
    print(f"[OK] aggregated fold metrics → {out_csv}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K-fold CV runner for DiGress (subprocess-based)")
    p.add_argument("--train_csv", required=True, help="TrainPool CSV (train+val 분할용 전체 CSV)")
    p.add_argument("--test_csv", default=None, help="(선택) 최종 test CSV")
    p.add_argument("--out_root", required=True, help="실행 루트 디렉토리")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # 계층화 분할 옵션 (없으면 일반 KFold)
    p.add_argument("--stratify_by", default=None, help="콤마로 구분된 열 이름들(e.g. solvent_class,pH_bin)")

    # 학습/최종 재학습 커맨드 템플릿
    p.add_argument(
        "--train_cmd",
        required=True,
        help=(
            "Fold 학습 커맨드 템플릿. 사용할 플레이스홀더: "
            "{train_csv} {val_csv} {test_csv} {out_dir} {seed} {fold}"
        ),
    )
    p.add_argument(
        "--final_cmd",
        default=None,
        help=(
            "(선택) TrainPool 전체 재학습+Test 평가 커맨드 템플릿. "
            "플레이스홀더: {train_csv} {test_csv} {out_dir} {seed}"
        ),
    )
    p.add_argument("--retrain_full", action="store_true", help="CV 후 TrainPool 전체로 재학습+Test 평가 수행")
    return p


def main():
    args = make_parser().parse_args()
    out_root = Path(args.out_root).resolve()
    _ensure_dir(out_root)

    train_csv = Path(args.train_csv).resolve()
    test_csv = Path(args.test_csv).resolve() if args.test_csv else None

    df = pd.read_csv(train_csv)
    n = len(df)
    print(f"[INFO] TrainPool size: {n}")

    # 분할기 준비
    if args.stratify_by:
        cols = [c.strip() for c in args.stratify_by.split(",") if c.strip()]
        labels = _combine_stratify_labels(df, cols)
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.arange(n), labels))
        print(f"[INFO] StratifiedKFold by {cols} (n_splits={args.n_splits}, seed={args.seed})")
    else:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.arange(n)))
        print(f"[INFO] KFold (n_splits={args.n_splits}, seed={args.seed})")

    fold_dirs: List[Path] = []

    # ====== 각 Fold 학습/검증 실행 ======
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        fold_dir = out_root / f"fold_{fold_id}"
        _ensure_dir(fold_dir)
        fold_dirs.append(fold_dir)

        # fold용 CSV 생성
        f_train_csv, f_val_csv = _write_fold_csvs(df, tr_idx, va_idx, fold_dir)

        # 커맨드 템플릿 렌더링
        cmd = args.train_cmd.format(
            train_csv=str(f_train_csv),
            val_csv=str(f_val_csv),
            test_csv=(str(test_csv) if test_csv else ""),
            out_dir=str(fold_dir),
            seed=args.seed,
            fold=fold_id,
        )
        ret = _run_cmd(cmd)
        if ret != 0:
            print(f"[ERROR] fold {fold_id} failed with code {ret}")
            sys.exit(ret)

    # ====== Fold 결과 집계 ======
    _aggregate_fold_metrics(fold_dirs, out_csv=out_root / "cv_folds_aggregated.csv")

    # ====== (선택) 전체 재학습 + Test 평가 ======
    if args.retrain_full:
        if not args.final_cmd:
            print("[WARN] --retrain_full requires --final_cmd; skip final stage")
            return
        if test_csv is None:
            print("[WARN] --retrain_full without --test_csv; skip final stage")
            return

        final_dir = out_root / "final_train_full"
        _ensure_dir(final_dir)

        cmd_final = args.final_cmd.format(
            train_csv=str(train_csv),
            test_csv=str(test_csv),
            out_dir=str(final_dir),
            seed=args.seed,
        )
        ret = _run_cmd(cmd_final)
        if ret != 0:
            print(f"[ERROR] final train+test failed with code {ret}")
            sys.exit(ret)

        # 최종 결과 스냅샷(선택): fold 요약본과 함께 복사
        for p in _collect_metrics(final_dir):
            dst = out_root / p.name
            try:
                shutil.copy2(p, dst)
                print(f"[OK] copied {p.name} → {dst}")
            except Exception as e:
                print(f"[WARN] copy failed {p}: {e}")


if __name__ == "__main__":
    main()
