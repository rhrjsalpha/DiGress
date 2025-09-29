# -*- coding: utf-8 -*-
"""
main_spec_boot_resample.py

목표
- 스모크 단계(사전 검증) 없이 바로 EVAL을 수행.
- EVAL(각 폴드)이 실패(ret != 0)하면 중단하지 않고, 새로운 train/test 부트스트랩 분할을
  다시 샘플링하여 해당 폴드를 재시도.
- 목표 개수(target_safe)만큼 성공한 폴드를 확보할 때까지 반복.
- 마지막에 전체 train으로 재학습 + (선택) 외부 test 평가 수행.

출력 구조(예시)
out_root/
  safe_split/                 # 성공한 SAFE 분할만 저장
    safe_001/{train_boot.csv, test_oob.csv, SAFE_BOOTSTRAP_SPLIT_INFO.json}
    ...
  trials/                     # 시도했던 trial(성공/실패 포함) 스냅샷
    trial_0001/{train_boot.csv, test_oob.csv}
    ...
  bootstrap_eval_runs/
    fold_001/...
    fold_002/...
    ...
  final_train_full/ ...
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# 0) 상단 인라인 설정
# ============================================================================
USE_INLINE_SETTINGS = True

# 공통 기본값
DATASET_NAME   = "csvspec"
GENERAL_GPUS   = 3
N_EPOCHS       = 10
FINAL_SAMPLES_for_Val  = 100
FINAL_SAMPLES = 1000
batch_size = 32
samples_to_generate = 512
samples_to_save = 20
chains_to_save = 10
number_chain_steps = 50
# (선택) 외부 test CSV를 쓰지 않으려면 TEST_CSV_DEF = None
TRAIN_CSV_DEF = r"/home/user/Spectral_Data/train_good_rows.csv" # /home/user/Spectral_Data/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv
TEST_CSV_DEF  = r"/home/user/Spectral_Data/test_good_rows.csv" # /home/user/Spectral_Data/EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv

# (A) 공통 설정
COMMON_SETTINGS = dict(
    project_root=None,                 # <repo>/src/main_spec.py 자동 탐색
    train_csv=TRAIN_CSV_DEF,
    test_csv=TEST_CSV_DEF,             # 외부 test (없으면 None)
    out_root=None,                     # 기본: <project_root>/bootstrap_cv_like_notval_resample
    name_prefix="specBSEVAL",

    # 목표 SAFE 개수 및 재사용
    target_safe=5,
    reuse_existing_safe=True,
    pick_order="asc",                  # asc | desc (재사용 정렬)

    # 부트스트랩 샘플링 파라미터
    bootstrap_trials=2000,             # 전체 최대 시도 수(성공/실패 포함)
    bootstrap_size=1.0,
    with_replacement=True,
    seed=10,
    stratify_by="pH_label,type,Solvent",
    stratified_bootstrap=True,
    min_test_size=1,

    # 중복/레벨 체크
    dedupe_by="train",                 # off | train | train+test
    level_check=True,
    level_check_mode="warn",           # off | warn | hard
    level_check_ignore="",

    # 폴드 실패 시 재시도 한도
    max_resample_per_fold=50,          # 각 폴드가 실패하면 최대 이 횟수만큼 새 분할로 재시도

    # EVAL 러닝 설정(기본 오버라이드로 들어감)
    eval_epochs=N_EPOCHS,
    eval_extra_overrides=[
        f"dataset.name={DATASET_NAME}",
        f"general.gpus={GENERAL_GPUS}",
        f"train.n_epochs={N_EPOCHS}",
        f"train.batch_size={batch_size}",
        "train.save_model=True",
        # 시각화로 인한 오류 피하기(필요시 변경 가능)
        f"general.samples_to_generate={samples_to_generate}",
        f"general.samples_to_save={samples_to_save}",
        f"general.chains_to_save={chains_to_save}",
        f"general.number_chain_steps={number_chain_steps}",
        f"general.final_model_samples_to_generate={FINAL_SAMPLES_for_Val}",
    ],

    # 파이널 재학습 설정
    do_full_retrain=True,
    full_extra_overrides=[
        f"dataset.name={DATASET_NAME}",
        f"general.gpus={GENERAL_GPUS}",
        f"train.n_epochs={N_EPOCHS}",
        f"train.batch_size={batch_size}",
        "train.save_model=True",
        f"general.samples_to_generate={samples_to_generate}",
        f"general.samples_to_save={samples_to_save}",
        f"general.chains_to_save={chains_to_save}",
        f"general.number_chain_steps={number_chain_steps}",
        f"general.final_model_samples_to_generate={FINAL_SAMPLES}",
        # 파이널에서 시각화/체인 저장을 켜고 싶으면 위 값 조정
    ],
)


# ============================================================================
# 경로/기본
# ============================================================================
def _auto_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src" / "main_spec.py").exists():
            return p
    # Windows 개발 경로 백업(필요 없으면 무시)
    fallback = Path(r"C:\Users\kogun\PycharmProjects\DiGress")
    return fallback if (fallback / "src" / "main_spec.py").exists() else here.parent


PROJECT_ROOT_DEFAULT = _auto_project_root()
OUT_ROOT_DEFAULT     = PROJECT_ROOT_DEFAULT / "bootstrap_cv_like_notval_resample"


# NEW: cv_runs 루트
def _cv_runs_root(out_root: Path) -> Path:
    p = out_root / "cv_runs"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ============================================================================
# 유틸
# ============================================================================
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _combine_stratify_labels(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"stratify_by column '{c}' not in CSV")
    return pd.Series(["|".join(x) for x in zip(*[df[c].astype(str).fillna("NA") for c in cols])]).values

def _quote_if_needed(v) -> str:
    s = str(v)
    if any(ch in s for ch in [' ', '=', ',', ';', ':', '\\']):
        return f'"{s}"'
    return s

def _split_overrides(str_or_list) -> list[str]:
    if not str_or_list:
        return []
    if isinstance(str_or_list, str):
        return shlex.split(str_or_list)
    return list(str_or_list)

def _run_with_logs(cmd: List[str], log_dir: Optional[Path], enable_log: bool) -> int:
    env = dict(os.environ)
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    if cmd and cmd[0] == sys.executable and "-u" not in cmd[1:3]:
        cmd = [sys.executable, "-u"] + cmd[1:]
    if log_dir:
        (log_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(log_dir / "logs" / "stdout.txt", "w", encoding="utf-8") as so, \
                open(log_dir / "logs" / "stderr.txt", "w", encoding="utf-8") as se:
            return subprocess.run(cmd, shell=False, text=True, stdout=so, stderr=se, env=env).returncode
    else:
        return subprocess.run(cmd, shell=False, env=env).returncode

def _sanity_check_levels(train_df: pd.DataFrame, test_df: pd.DataFrame, ignore_cols: List[str]) -> Dict[str, List[str]]:
    bad: Dict[str, List[str]] = {}
    ig = set(i.strip() for i in ignore_cols if i)
    for col in train_df.columns:
        if col in ig:
            continue
        if train_df[col].dtype == "object" or pd.api.types.is_categorical_dtype(train_df[col]):
            tr = set(map(str, train_df[col].dropna().unique()))
            te = set(map(str, test_df[col].dropna().unique()))
            miss = sorted(list(te - tr))
            if miss:
                bad[col] = miss
    return bad

def _sig_from_indices(idxs: np.ndarray) -> str:
    arr = np.array(sorted(np.unique(idxs)), dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()

def _pair_sig(train_idx: np.ndarray, test_idx: Optional[np.ndarray]) -> str:
    ts = _sig_from_indices(train_idx)
    return ts + "|" + (_sig_from_indices(test_idx) if test_idx is not None else "none")

def _sample_bootstrap_indices(labels: Optional[np.ndarray], n_total: int, size: float,
                              with_repl: bool, rng: np.random.Generator) -> np.ndarray:
    if labels is None:
        n_pick = max(1, int(round(n_total * size)))
        return (rng.integers(0, n_total, size=n_pick, endpoint=False)
                if with_repl else rng.choice(n_total, size=n_pick, replace=False))
    idxs = np.arange(n_total)
    df_tmp = pd.DataFrame({"lab": labels, "i": idxs})
    res = []
    for _, grp in df_tmp.groupby("lab"):
        m = len(grp)
        k = max(1, int(round(m * size)))
        pool = grp["i"].to_numpy()
        if with_repl:
            res.append(np.random.default_rng(rng.integers(0, 2**31-1)).choice(pool, size=k, replace=True))
        else:
            res.append(np.random.default_rng(rng.integers(0, 2**31-1)).choice(pool, size=min(k, m), replace=False))
    return np.concatenate(res)

def _collect_metrics(root: Path, pattern_glob: str = "final_metrics*.csv") -> List[Path]:
    return sorted(root.rglob(pattern_glob))

def _aggregate_metrics(run_dirs: List[Path], out_all: Path, out_summary: Path):
    frames = []
    for d in run_dirs:
        csvs = _collect_metrics(d)
        if not csvs:
            print(f"[WARN] no metrics under {d}")
            continue
        try:
            df = pd.read_csv(csvs[-1])
            df.insert(0, "run_dir", str(d))
            frames.append(df)
        except Exception as e:
            print(f"[WARN] read failed {csvs[-1]}: {e}")
    if not frames:
        print("[WARN] nothing to aggregate")
        return
    big = pd.concat(frames, ignore_index=True)
    _ensure_dir(out_all.parent)
    big.to_csv(out_all, index=False)
    num_cols = big.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        summ = pd.DataFrame({
            "metric": num_cols,
            "mean": [big[c].mean() for c in num_cols],
            "std":  [big[c].std(ddof=1) for c in num_cols],
            "n":    [big[c].notna().sum() for c in num_cols],
        })
        summ.to_csv(out_summary, index=False)
    print(f"[OK] aggregated → {out_all} / {out_summary}")

# ============================================================================
# argparse / 인라인
# ============================================================================
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap EVAL with resampling on failure (no val) → Full retrain+Test")
    p.add_argument("--project_root", default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--train_csv", default=str(TRAIN_CSV_DEF))
    p.add_argument("--test_csv",  default=str(TEST_CSV_DEF))
    p.add_argument("--out_root",  default=str(OUT_ROOT_DEFAULT))
    p.add_argument("--name_prefix", default="specBSEVAL")

    p.add_argument("--target_safe", type=int, default=5)
    p.add_argument("--reuse_existing_safe", action="store_true")
    p.add_argument("--pick_order", default="asc", choices=["asc", "desc"])

    p.add_argument("--bootstrap_trials", type=int, default=2000)
    p.add_argument("--bootstrap_size", type=float, default=1.0)
    p.add_argument("--with_replacement", action="store_true")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--stratify_by", default=None)
    p.add_argument("--stratified_bootstrap", action="store_true")
    p.add_argument("--min_test_size", type=int, default=1)

    p.add_argument("--dedupe_by", default="train", choices=["off", "train", "train+test"])
    p.add_argument("--level_check", action="store_true")
    p.add_argument("--level_check_mode", default="warn", choices=["off", "warn", "hard"])
    p.add_argument("--level_check_ignore", default="")

    p.add_argument("--max_resample_per_fold", type=int, default=50)

    p.add_argument("--eval_epochs", type=int, default=N_EPOCHS)
    p.add_argument("--eval_extra_overrides", default=None)
    p.add_argument("--do_full_retrain", action="store_true")
    p.add_argument("--full_extra_overrides", default=None)

    p.add_argument("--main_spec", default=None)
    p.add_argument("--enable_logs", action="store_true")
    return p

def _resolve_project_root(user_root: Optional[str | Path]) -> Path:
    if user_root:
        p = Path(user_root).resolve()
        if (p / "src" / "main_spec.py").exists():
            return p
        print(f"[WARN] main_spec.py가 {p}/src 아래에 없습니다. 자동 탐색으로 대체.")
    return PROJECT_ROOT_DEFAULT

def _load_args() -> argparse.Namespace:
    parser = make_parser()
    if not USE_INLINE_SETTINGS:
        return parser.parse_args()

    a = parser.parse_args([])
    S = dict(COMMON_SETTINGS)

    setattr(a, "project_root", str(_resolve_project_root(S.get("project_root"))))
    setattr(a, "train_csv", str(S.get("train_csv") or TRAIN_CSV_DEF))
    setattr(a, "test_csv",  str(S.get("test_csv")) if S.get("test_csv") else None)
    setattr(a, "out_root",  str(S.get("out_root") or OUT_ROOT_DEFAULT))
    setattr(a, "name_prefix", str(S.get("name_prefix", "specBSEVAL")))

    # 숫자/토글
    for k, dflt in [
        ("target_safe", 5), ("reuse_existing_safe", True), ("pick_order", "asc"),
        ("bootstrap_trials", 2000), ("bootstrap_size", 1.0), ("with_replacement", True),
        ("seed", 10), ("stratify_by", "pH_label,type,Solvent"), ("stratified_bootstrap", True),
        ("min_test_size", 1), ("dedupe_by", "train"),
        ("level_check", True), ("level_check_mode", "warn"), ("level_check_ignore", ""),
        ("max_resample_per_fold", 50),
        ("eval_epochs", N_EPOCHS),
        ("do_full_retrain", True),
    ]:
        setattr(a, k, S.get(k, dflt))

    # 오버라이드
    eval_list = S.get("eval_extra_overrides", [])
    full_list = S.get("full_extra_overrides", [])
    setattr(a, "eval_extra_overrides", " ".join(eval_list) if isinstance(eval_list, list) else eval_list)
    setattr(a, "full_extra_overrides", " ".join(full_list) if isinstance(full_list, list) else full_list)

    setattr(a, "main_spec", None)
    setattr(a, "enable_logs", False)
    return a

# ============================================================================
# 분할/실행 로직
# ============================================================================
def _list_existing_safe(safe_root: Path, order: str) -> List[Path]:
    cands = [p for p in safe_root.glob("safe_*") if p.is_dir()]
    cands.sort(key=lambda p: p.name, reverse=(order == "desc"))
    return cands

def _ensure_test_oob_file(safe_dir: Path) -> Optional[Path]:
    f_test = safe_dir / "test_oob.csv"
    if f_test.exists():
        return f_test
    f_val = safe_dir / "val_oob.csv"
    if f_val.exists():
        try:
            shutil.copy2(f_val, f_test)
        except Exception:
            pass
        return f_test if f_test.exists() else f_val
    f_legacy = safe_dir / "test_external.csv"
    return f_legacy if f_legacy.exists() else None

def _maybe_reuse_safe(a, safe_root: Path, need_n: int) -> List[Dict]:
    reused = []
    if not a.reuse_existing_safe:
        return reused
    cands = _list_existing_safe(safe_root, a.pick_order)
    for c in cands:
        f_train = c / "train_boot.csv"
        f_test  = _ensure_test_oob_file(c)
        if f_train.exists() and (f_test and f_test.exists()):
            info_file = c / "SAFE_BOOTSTRAP_SPLIT_INFO.json"
            meta = json.load(open(info_file, "r", encoding="utf-8")) if info_file.exists() else {}
            reused.append({
                "safe_dir": str(c),
                "train_csv": str(f_train),
                "test_csv": str(f_test),
                "meta": meta,
            })
            if len(reused) >= need_n:
                break
    if reused:
        print(f"[INFO] Reusing {len(reused)} SAFE splits from {safe_root}")
    return reused

def _sample_one_split(df: pd.DataFrame, labels: Optional[np.ndarray], size: float, with_repl: bool,
                      rng: np.random.Generator, out_dir: Path,
                      strat_cols: List[str], a) -> Tuple[Path, Path, Dict]:
    """train_boot.csv, test_oob.csv를 out_dir에 저장하고 경로/메타 반환"""
    _ensure_dir(out_dir)
    n_total = len(df)
    train_idx = _sample_bootstrap_indices(labels, n_total, size, with_repl, rng)
    uni_train = np.unique(train_idx)
    oob_mask = np.ones(n_total, dtype=bool); oob_mask[uni_train] = False
    test_idx = np.where(oob_mask)[0]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # 레벨 체크
    if a.level_check and a.level_check_mode != "off":
        bad = _sanity_check_levels(train_df, test_df, [s.strip() for s in a.level_check_ignore.split(",") if s.strip()])
        if bad and a.level_check_mode == "hard":
            raise RuntimeError(f"Level-check failed (hard): {bad}")

    f_train = out_dir / "train_boot.csv"
    f_test  = out_dir / "test_oob.csv"
    train_df.to_csv(f_train, index=False)
    test_df.to_csv(f_test,  index=False)

    meta = {
        "paths": {"train_csv": str(f_train), "test_csv": str(f_test)},
        "bootstrap_size": a.bootstrap_size, "with_replacement": a.with_replacement,
        "seed": a.seed, "stratify_by": strat_cols if strat_cols else None
    }
    return f_train, f_test, meta

def _eval_one_split(main_spec: Path, f_train: Path, f_test: Path, name: str,
                    eval_extra: List[str], run_dir: Path,
                    trial_no: int, seed: int, fold_idx_zero_based: int,
                    cv_runs_dir: Path) -> int:
    overrides = [
        f'dataset.train_csv="{str(f_train)}"',
        "dataset.val_csv=null",
        f'dataset.test_csv="{str(f_test)}"',
        f"general.name={name}",
        *eval_extra,
    ]
    cmd = [sys.executable, str(main_spec), *overrides]
    print(f"[EVAL RUN] TRIAL {trial_no:04d} | SEED {seed} | CV fold_{fold_idx_zero_based:02d} →", " ".join(map(str, cmd)))
    ret = _run_with_logs(cmd, run_dir, enable_log=True)

    # 실패하면 cv_runs에 로그/요약 저장
    if ret != 0:
        cv_trial_dir = cv_runs_dir / f"_trial_{trial_no:03d}_seed{seed}" / f"fold_{fold_idx_zero_based:02d}"
        _mirror_logs_to_cv_runs(run_dir, cv_trial_dir, ret, trial_no, seed, fold_idx_zero_based, f_train, f_test)
        print(f"[EVAL FAIL] TRIAL {trial_no:04d} | CV fold_{fold_idx_zero_based:02d} | ret={ret} → logs saved to {cv_trial_dir}")
    return ret

def _promote_to_safe(safe_root: Path, src_train: Path, src_test: Path, meta: Dict) -> Path:
    k = 1
    safe_dir = safe_root / f"safe_{k:03d}"
    while safe_dir.exists():
        k += 1
        safe_dir = safe_root / f"safe_{k:03d}"
    _ensure_dir(safe_dir)
    dst_train = safe_dir / "train_boot.csv"
    dst_test  = safe_dir / "test_oob.csv"
    shutil.copy2(src_train, dst_train)
    shutil.copy2(src_test,  dst_test)
    meta2 = {"idx": k, **meta, "paths": {"train_csv": str(dst_train), "test_csv": str(dst_test), "safe_dir": str(safe_dir)}}
    with open(safe_dir / "SAFE_BOOTSTRAP_SPLIT_INFO.json", "w", encoding="utf-8") as f:
        json.dump(meta2, f, ensure_ascii=False, indent=2)
    return safe_dir

def _mirror_logs_to_cv_runs(eval_fold_dir: Path, cv_trial_dir: Path, ret: int,
                            trial_no: int, seed: int, fold_idx_zero_based: int,
                            f_train: Path, f_test: Path):
    """eval 폴더의 logs 를 cv_runs 쪽으로 복사하고, 요약 error.txt 생성"""
    (cv_trial_dir).mkdir(parents=True, exist_ok=True)
    src_stdout = eval_fold_dir / "logs" / "stdout.txt"
    src_stderr = eval_fold_dir / "logs" / "stderr.txt"
    dst_stdout = cv_trial_dir / "stdout.txt"
    dst_stderr = cv_trial_dir / "stderr.txt"
    try:
        if src_stdout.exists(): shutil.copy2(src_stdout, dst_stdout)
        if src_stderr.exists(): shutil.copy2(src_stderr, dst_stderr)
    except Exception as e:
        print(f"[WARN] log copy failed → {cv_trial_dir}: {e}")
    # 요약 파일
    with open(cv_trial_dir / "error.txt", "w", encoding="utf-8") as f:
        f.write(
            f"[ERROR] TRIAL {trial_no:04d} | SEED {seed} | CV fold_{fold_idx_zero_based:02d} | ret={ret}\n"
            f"train_csv: {f_train}\n"
            f"test_csv : {f_test}\n"
            f"eval_logs: {eval_fold_dir / 'logs'}\n"
        )

def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _pair_sig_from_files(f_train: Path, f_test: Path) -> str:
    return f"{_sha1_file(f_train)}|{_sha1_file(f_test)}"
# ============================================================================
# 메인
# ============================================================================
def main():
    a = _load_args()

    project_root = Path(a.project_root).resolve()
    out_root     = Path(a.out_root).resolve(); _ensure_dir(out_root)
    main_spec    = Path(a.main_spec).resolve() if a.main_spec else (project_root / "src" / "main_spec.py")

    cv_runs_dir = _cv_runs_root(out_root)

    if not main_spec.exists():
        raise FileNotFoundError(f"main_spec.py not found: {main_spec}")

    train_csv = Path(a.train_csv).resolve()
    test_csv  = Path(a.test_csv).resolve() if getattr(a, "test_csv", None) else None

    df_full = pd.read_csv(train_csv)

    # 라벨 결합(층화 옵션)
    strat_cols = [c.strip() for c in (a.stratify_by.split(",") if a.stratify_by else []) if c.strip()]
    labels = _combine_stratify_labels(df_full, strat_cols) if (a.stratified_bootstrap and strat_cols) else None

    # 디렉토리
    safe_root = out_root / "safe_split"; _ensure_dir(safe_root)
    trials_root = out_root / "trials"; _ensure_dir(trials_root)
    eval_root = out_root / "bootstrap_eval_runs"; _ensure_dir(eval_root)

    # 재사용
    successes: List[Dict] = []
    seen: set[str] = set()
    eval_count = 0  # ★ fold 번호(시도 횟수). 성공/실패와 무관하게 1씩 증가

    if a.reuse_existing_safe:
        reused = _maybe_reuse_safe(a, safe_root, a.target_safe)
        for i, s in enumerate(reused, start=1):
            if len(successes) >= a.target_safe:
                break
            f_train = Path(s["train_csv"])
            f_test = Path(s["test_csv"]) if s.get("test_csv") else _ensure_test_oob_file(Path(s["safe_dir"]))
            if not (f_train and f_train.exists() and f_test and f_test.exists()):
                continue

            # 중복 방지
            sig = _pair_sig_from_files(f_train, f_test)
            if a.dedupe_by != "off":
                if sig in seen:
                    ...
                seen.add(sig)

            # ★ fold 번호는 시도 순서대로 증가
            eval_count += 1
            fold_zero = eval_count - 1
            fold_dir = eval_root / f"fold_{eval_count:03d}"
            _ensure_dir(fold_dir)

            ret = _eval_one_split(
                main_spec, f_train, f_test, f"{a.name_prefix}_EVAL_{eval_count:03d}",
                _split_overrides(a.eval_extra_overrides), fold_dir,
                trial_no=0, seed=a.seed, fold_idx_zero_based=fold_zero, cv_runs_dir=cv_runs_dir
            )
            if ret == 0:
                successes.append({"safe_dir": s["safe_dir"], "train_csv": str(f_train), "test_csv": str(f_test)})
            else:
                print(f"[REUSE FAIL] fold_{fold_zero:02d} ret={ret} → drop and continue")

    # 필요 개수만큼 성공 폴드 확보될 때까지 (최대 bootstrap_trials)
    # 필요 SAFE 수를 채울 때까지: 매 시도마다 fold 번호 1 증가
    rng = np.random.default_rng(a.seed)
    trials = 0
    while len(successes) < a.target_safe and trials < a.bootstrap_trials:
        trials += 1
        trial_dir = trials_root / f"trial_{trials:04d}";
        _ensure_dir(trial_dir)

        try:
            f_train, f_test, meta = _sample_one_split(
                df_full, labels, a.bootstrap_size, a.with_replacement,
                rng, trial_dir, strat_cols, a
            )
        except Exception as e:
            print(f"[TRIAL {trials:04d}] sampling failed: {e}")
            shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        # 중복 체크
        sig = _pair_sig(pd.read_csv(f_train).index.values,
                        pd.read_csv(f_test).index.values) if a.dedupe_by != "off" else None
        if sig and sig in seen:
            print(f"[TRIAL {trials:04d}] duplicate split; skip")
            shutil.rmtree(trial_dir, ignore_errors=True)
            continue
        if sig: seen.add(sig)

        # ★ fold 번호는 시도할 때마다 증가
        eval_count += 1
        fold_zero = eval_count - 1
        fold_dir = eval_root / f"fold_{eval_count:03d}";
        _ensure_dir(fold_dir)

        ret = _eval_one_split(
            main_spec, f_train, f_test, f"{a.name_prefix}_EVAL_{eval_count:03d}",
            _split_overrides(a.eval_extra_overrides), fold_dir,
            trial_no=trials, seed=a.seed, fold_idx_zero_based=fold_zero, cv_runs_dir=cv_runs_dir
        )

        if ret == 0:
            safe_dir = _promote_to_safe(safe_root, f_train, f_test, meta)
            successes.append({
                "safe_dir": str(safe_dir),
                "train_csv": str(safe_dir / "train_boot.csv"),
                "test_csv": str(safe_dir / "test_oob.csv")
            })
            print(f"[OK] fold_{fold_zero:02d} succeeded ({len(successes)}/{a.target_safe})")
        else:
            print(f"[FAIL] fold_{fold_zero:02d} (ret={ret}); continue to next fold")

    if len(successes) < a.target_safe:
        print(f"[WARN] need {a.target_safe} successful folds, but only {len(successes)} succeeded.")

    # 집계
    run_dirs = [eval_root / f"fold_{i:03d}" for i in range(1, eval_count + 1)]
    if run_dirs:
        _aggregate_metrics(run_dirs,
                           out_all=eval_root / "bootstrap_eval_metrics_all.csv",
                           out_summary=eval_root / "bootstrap_eval_metrics_summary.csv")
    else:
        print("[WARN] No successful eval runs to aggregate.")

    # 전체 재학습 + 외부 테스트
    if a.do_full_retrain:
        final_dir = out_root / "final_train_full"; _ensure_dir(final_dir)
        overrides = [
            f'dataset.train_csv="{str(train_csv)}"',
            "dataset.val_csv=null",
            (f'dataset.test_csv="{str(test_csv)}"' if test_csv else "dataset.test_csv=null"),
            f"general.name={a.name_prefix}_FINAL_full",
            *_split_overrides(a.full_extra_overrides),
        ]
        cmd = [sys.executable, str(main_spec), *overrides]
        print("[FINAL RUN]", " ".join(map(str, cmd)))
        ret = _run_with_logs(cmd, final_dir, enable_log=a.enable_logs)
        if ret != 0:
            print("[ERROR] Final full retrain failed."); sys.exit(ret)

    # 요약 저장
    summary = {
        "target_safe": a.target_safe,
        "successful_folds": len(successes),
        "total_folds_attempted": eval_count,  # ← 추가
        "used_safes": successes,
        "eval_root": str(eval_root),
        "final_retrain": bool(a.do_full_retrain),
    }
    with open(out_root / "PIPELINE_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE] Resample-on-failure pipeline finished.")
    print(f"Summary: {out_root / 'PIPELINE_SUMMARY.json'}")


if __name__ == "__main__":
    main()
