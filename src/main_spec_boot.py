# -*- coding: utf-8 -*-
"""
main_spec_bootstrap_nfold_then_full_notval.py

목표
- main_spec.py의 'val 경로'는 절대 사용하지 않음.
- 부트스트랩으로 (train_boot, test_oob) 쌍을 스모크 호출로 검증하며 '안전 분할' N개 확보.
- 확보된 N개 분할 각각에 대해 train=train_boot, test=test_oob(또는 external)로 본평가 수행(검증 비활성화).
- 마지막에 전체 train으로 재학습 + 외부 test 평가.

파일 구조(예시)
out_root/
  safe_split/safe_001/{train_boot.csv, test_oob.csv, SAFE_BOOTSTRAP_SPLIT_INFO.json}
  safe_split/safe_002/ ...
  bootstrap_eval_runs/fold_001/ ...
  final_train_full/ ...
"""

from __future__ import annotations

import argparse, json, shutil, subprocess, sys, hashlib
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

# =========================
# 0) PYCHARM QUICK SETTINGS
# =========================
USE_INLINE_SETTINGS = True
QUICK_SETTINGS = dict(
    # 경로
    project_root=None,  # <repo>/src/main_spec.py 자동 탐색
    train_csv=r"C:\Users\kogun\PycharmProjects\DiGress\data\csv\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
    test_csv=r"C:\Users\kogun\PycharmProjects\DiGress\data\csv\EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv",
    out_root=None,  # 기본: <project_root>/bootstrap_cv_like_notval
    name_prefix="specBSEVAL",

    # --- N개 안전 분할 타깃 & 재사용 ---
    target_safe=5,              # 확보할 안전 분할 개수
    reuse_existing_safe=True,   # safe_split/에 이미 있으면 재사용
    pick_order="asc",           # asc(오래된 순) | desc(최신 우선)

    # --- 안전 분할 탐색(부트스트랩 + 스모크; 항상 train/test만 사용) ---
    bootstrap_trials=1000,      # 전체 시도 상한
    bootstrap_size=1.0,         # 학습 샘플 비율
    with_replacement=True,      # 복원추출(True 권장 → OOB 확보)
    seed=100,
    stratify_by="pH_label,type,Solvent",   # None이면 비층화
    stratified_bootstrap=True,
    min_test_size=1,            # OOB 최소 샘플 수

    # 중복 방지
    dedupe_by="train",          # off | train | train+test

    # level-check(문자/범주형 컬럼: train ⊇ test)
    level_check=True,
    level_check_mode="warn",    # off | warn | hard
    level_check_ignore="",

    # 스모크 학습(main_spec 호출) — 항상 val_csv=null, test_csv=OOB(or external)
    smoke_epochs=1,
    log_smoke=True,
    keep_failed_trials=True,
    extra_overrides_smoke=[
        "dataset.name=csvspec",
        "train.save_model=False",
        "general.gpus=1",
        # "trainer.precision=16-mixed",
    ],

    # --- N개 분할 본평가(각 분할을 fold처럼 실행, 항상 train/test만 사용) ---
    # 항상 dataset.val_csv=null, dataset.test_csv=각 분할의 test_oob.csv
    extra_overrides_eval=[
        "dataset.name=csvspec",
        "train.save_model=True",
        "train.n_epochs=100",
        "general.gpus=1",
        "general.final_model_samples_to_generate=100"
    ],

    # --- 전체 재학습 + 외부 테스트 ---
    do_full_retrain=True,
    extra_overrides_full=[
        "dataset.name=csvspec",
        "train.save_model=True",
        "train.n_epochs=100",
        "general.gpus=1",
        "general.final_model_samples_to_generate=100"
    ],
)

# ============ 경로/기본 ============
def _auto_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src" / "main_spec.py").exists():
            return p
    fallback = Path(r"C:\Users\kogun\PycharmProjects\DiGress")
    return fallback if (fallback / "src" / "main_spec.py").exists() else here.parent

PROJECT_ROOT_DEFAULT = _auto_project_root()
DATA_DIR_DEFAULT     = PROJECT_ROOT_DEFAULT / "data" / "csv"
TRAIN_CSV_DEFAULT    = DATA_DIR_DEFAULT / "QM_EM_ABS_stratified_train_resplit_with_mu_eps.csv"
TEST_CSV_DEFAULT     = DATA_DIR_DEFAULT / "QM_EM_ABS_stratified_test_resplit_with_mu_eps.csv"
OUT_ROOT_DEFAULT     = PROJECT_ROOT_DEFAULT / "bootstrap_cv_like_notval"

# ============ 유틸 ============
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _combine_stratify_labels(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    for c in cols:
        if c not in df.columns: raise KeyError(f"stratify_by column '{c}' not in CSV")
    return pd.Series(["|".join(x) for x in zip(*[df[c].astype(str).fillna("NA") for c in cols])]).values

def _run_with_logs(cmd: List[str], log_dir: Optional[Path], enable: bool) -> int:
    if enable and log_dir is not None:
        (log_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(log_dir / "logs" / "stdout.txt","w",encoding="utf-8") as so, \
             open(log_dir / "logs" / "stderr.txt","w",encoding="utf-8") as se:
            return subprocess.run(cmd, shell=False, text=True, stdout=so, stderr=se).returncode
    else:
        return subprocess.run(cmd, shell=False).returncode

def _sanity_check_levels(train_df: pd.DataFrame, test_df: pd.DataFrame, ignore_cols: List[str]) -> Dict[str,List[str]]:
    bad: Dict[str,List[str]] = {}
    ig = set(i.strip() for i in ignore_cols if i)
    for col in train_df.columns:
        if col in ig: continue
        if train_df[col].dtype == "object" or pd.api.types.is_categorical_dtype(train_df[col]):
            tr = set(map(str, train_df[col].dropna().unique()))
            te = set(map(str, test_df[col].dropna().unique()))
            miss = sorted(list(te - tr))
            if miss: bad[col] = miss
    return bad

def _sig_from_indices(idxs: np.ndarray) -> str:
    arr = np.array(sorted(np.unique(idxs)), dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()

def _pair_sig(train_idx: np.ndarray, test_idx: Optional[np.ndarray]) -> str:
    ts = _sig_from_indices(train_idx)
    return ts + "|" + (_sig_from_indices(test_idx) if test_idx is not None else "none")

def _sample_bootstrap_indices(labels: Optional[np.ndarray], n_total:int, size:float, with_repl:bool, rng:np.random.Generator)->np.ndarray:
    if labels is None:
        n_pick = max(1, int(round(n_total * size)))
        return (rng.integers(0, n_total, size=n_pick, endpoint=False)
                if with_repl else rng.choice(n_total, size=n_pick, replace=False))
    idxs = np.arange(n_total)
    df_tmp = pd.DataFrame({"lab": labels, "i": idxs})
    res = []
    for _, grp in df_tmp.groupby("lab"):
        m = len(grp); k = max(1, int(round(m * size)))
        pool = grp["i"].to_numpy()
        res.append((rng.choice(pool, size=k, replace=True)) if with_repl else (rng.choice(pool, size=min(k,m), replace=False)))
    return np.concatenate(res)

def _collect_metrics(root: Path, pattern_glob: str = "final_metrics*.csv") -> List[Path]:
    return sorted(root.rglob(pattern_glob))

def _aggregate_metrics(run_dirs: List[Path], out_all: Path, out_summary: Path):
    frames=[]
    for d in run_dirs:
        csvs = _collect_metrics(d)
        if not csvs:
            print(f"[WARN] no metrics under {d}"); continue
        try:
            df = pd.read_csv(csvs[-1]); df.insert(0,"run_dir",str(d)); frames.append(df)
        except Exception as e:
            print(f"[WARN] read failed {csvs[-1]}: {e}")
    if not frames:
        print("[WARN] nothing to aggregate"); return
    big = pd.concat(frames, ignore_index=True); _ensure_dir(out_all.parent); big.to_csv(out_all, index=False)
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

# ============ argparse / 인라인 ============
def make_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Bootstrap N-fold-like EVAL (no val) → Full retrain+Test")
    p.add_argument("--project_root", default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--train_csv", default=str(TRAIN_CSV_DEFAULT))
    p.add_argument("--test_csv",  default=str(TEST_CSV_DEFAULT))
    p.add_argument("--out_root",  default=str(OUT_ROOT_DEFAULT))
    p.add_argument("--name_prefix", default="specBSEVAL")

    p.add_argument("--target_safe", type=int, default=5)
    p.add_argument("--reuse_existing_safe", action="store_true")
    p.add_argument("--pick_order", default="asc", choices=["asc","desc"])

    p.add_argument("--bootstrap_trials", type=int, default=1000)
    p.add_argument("--bootstrap_size", type=float, default=1.0)
    p.add_argument("--with_replacement", action="store_true")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--stratify_by", default=None)
    p.add_argument("--stratified_bootstrap", action="store_true")
    p.add_argument("--min_test_size", type=int, default=1)

    p.add_argument("--dedupe_by", default="train", choices=["off","train","train+test"])
    p.add_argument("--level_check", action="store_true")
    p.add_argument("--level_check_mode", default="warn", choices=["off","warn","hard"])
    p.add_argument("--level_check_ignore", default="")

    p.add_argument("--smoke_epochs", type=int, default=1)
    p.add_argument("--log_smoke", action="store_true")
    p.add_argument("--keep_failed_trials", action="store_true")
    p.add_argument("--extra_overrides_smoke", default=None)

    p.add_argument("--extra_overrides_eval", default=None)

    p.add_argument("--do_full_retrain", action="store_true")
    p.add_argument("--extra_overrides_full", default=None)

    p.add_argument("--main_spec", default=None)
    return p

def _resolve_project_root(user_root: Optional[str|Path])->Path:
    if user_root:
        p=Path(user_root).resolve()
        if (p/"src"/"main_spec.py").exists(): return p
        print(f"[WARN] main_spec.py가 {p}/src 아래에 없습니다. 자동 탐색으로 대체.")
    return PROJECT_ROOT_DEFAULT

def _load_args()->argparse.Namespace:
    parser=make_parser()
    if not USE_INLINE_SETTINGS: return parser.parse_args()
    a=parser.parse_args([])
    S=QUICK_SETTINGS
    setattr(a,"project_root", str(_resolve_project_root(S.get("project_root"))))
    setattr(a,"train_csv", str(S.get("train_csv") or TRAIN_CSV_DEFAULT))
    setattr(a,"test_csv",  str(S.get("test_csv") or TEST_CSV_DEFAULT))
    setattr(a,"out_root",  str(S.get("out_root") or OUT_ROOT_DEFAULT))
    setattr(a,"name_prefix", S.get("name_prefix","specBSEVAL"))

    for k, dflt in [
        ("target_safe",5), ("reuse_existing_safe",True), ("pick_order","asc"),
        ("bootstrap_trials",1000), ("bootstrap_size",1.0), ("with_replacement",True),
        ("seed",100), ("stratify_by",None), ("stratified_bootstrap",True),
        ("min_test_size",1), ("dedupe_by","train"),
        ("level_check",True), ("level_check_mode","warn"), ("level_check_ignore",""),
        ("smoke_epochs",1), ("log_smoke",True), ("keep_failed_trials",True),
        ("do_full_retrain",True),
    ]:
        setattr(a, k, S.get(k, dflt))
    eo_smoke = S.get("extra_overrides_smoke", ["dataset.name=csvspec","train.save_model=False","general.gpus=1"])
    setattr(a,"extra_overrides_smoke", " ".join(eo_smoke) if isinstance(eo_smoke,list) else eo_smoke)
    eo_eval = S.get("extra_overrides_eval", ["dataset.name=csvspec","train.save_model=True","train.n_epochs=10","general.gpus=1"])
    setattr(a,"extra_overrides_eval", " ".join(eo_eval) if isinstance(eo_eval,list) else eo_eval)
    eo_full = S.get("extra_overrides_full", ["dataset.name=csvspec","train.save_model=True","train.n_epochs=10","general.gpus=1"])
    setattr(a,"extra_overrides_full", " ".join(eo_full) if isinstance(eo_full,list) else eo_full)
    setattr(a,"main_spec", S.get("main_spec"))
    return a

# ============ SAFE 재사용/탐색 ============
def _list_existing_safe(safe_root: Path, order: str)->List[Path]:
    cands = [p for p in safe_root.glob("safe_*") if p.is_dir()]
    cands.sort(key=lambda p: p.name, reverse=(order=="desc"))
    return cands

def _ensure_test_oob_file(safe_dir: Path)->Optional[Path]:
    """과거 스크립트가 val_oob.csv로 저장했을 수도 있어 호환 처리."""
    f_test = safe_dir/"test_oob.csv"
    if f_test.exists(): return f_test
    f_val = safe_dir/"val_oob.csv"
    if f_val.exists():
        try: shutil.copy2(f_val, f_test)
        except Exception: pass
        return f_test if f_test.exists() else f_val
    # 더 과거 형식 대비
    f_legacy = safe_dir/"test_external.csv"
    return f_legacy if f_legacy.exists() else None

def _maybe_reuse_safe(a, safe_root: Path, need_n: int)->List[Dict]:
    reused=[]
    if not a.reuse_existing_safe: return reused
    cands = _list_existing_safe(safe_root, a.pick_order)
    for c in cands:
        f_train = c/"train_boot.csv"
        f_test  = _ensure_test_oob_file(c)
        if f_train.exists() and (f_test and f_test.exists()):
            info_file = c/"SAFE_BOOTSTRAP_SPLIT_INFO.json"
            meta = json.load(open(info_file,"r",encoding="utf-8")) if info_file.exists() else {}
            reused.append({
                "safe_dir": str(c),
                "train_csv": str(f_train),
                "test_csv": str(f_test),
                "meta": meta,
            })
            if len(reused) >= need_n: break
    if reused:
        print(f"[INFO] Reusing {len(reused)} SAFE splits from {safe_root}")
    return reused

def _search_safe_needed(a, df: pd.DataFrame, main_spec: Path, out_root: Path, ext_test_csv: Optional[Path], already:int)->List[Dict]:
    safe_root = out_root/"safe_split"; _ensure_dir(safe_root)
    rng = np.random.default_rng(a.seed)
    n_total = len(df)
    strat_cols = [c.strip() for c in (a.stratify_by.split(",") if a.stratify_by else []) if c.strip()]
    labels = _combine_stratify_labels(df, strat_cols) if (a.stratified_bootstrap and strat_cols) else None

    # dedupe
    seen: set[str] = set()
    found=[]
    trials=0
    while len(found)+already < a.target_safe and trials < a.bootstrap_trials:
        trials += 1
        trial_dir = out_root / f"trial_{trials:04d}"
        _ensure_dir(trial_dir)

        train_idx = _sample_bootstrap_indices(labels, n_total, a.bootstrap_size, a.with_replacement, rng)
        uni_train = np.unique(train_idx)
        oob_mask = np.ones(n_total, dtype=bool); oob_mask[uni_train]=False
        test_idx = np.where(oob_mask)[0]

        # dedupe
        sig=None
        if a.dedupe_by!="off":
            sig = _pair_sig(uni_train, test_idx) if a.dedupe_by=="train+test" else _sig_from_indices(uni_train)
            if sig in seen:
                print(f"[TRIAL {trials:04d}] duplicate split; skip")
                if not a.keep_failed_trials: shutil.rmtree(trial_dir, ignore_errors=True)
                continue

        if test_idx.size < a.min_test_size:
            print(f"[TRIAL {trials:04d}] OOB too small ({test_idx.size}<{a.min_test_size}); skip")
            if not a.keep_failed_trials: shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        # level-check (train ⊇ test)
        if a.level_check and a.level_check_mode!="off":
            bad=_sanity_check_levels(train_df, test_df, [s.strip() for s in a.level_check_ignore.split(",") if s.strip()])
            if bad:
                print(f"[LEVEL-CHECK][trial {trials:04d}] test-only cats: {bad}")
                if a.level_check_mode=="hard":
                    if not a.keep_failed_trials: shutil.rmtree(trial_dir, ignore_errors=True)
                    continue

        # CSV 저장
        f_train = trial_dir/"train_boot.csv"
        f_test  = trial_dir/"test_oob.csv"
        train_df.to_csv(f_train, index=False)
        test_df.to_csv(f_test,  index=False)

        # ✅ 스모크 호출: 항상 val_csv=null, test_csv=OOB(or external)
        extra_over = a.extra_overrides_smoke.strip().split() if a.extra_overrides_smoke else []
        overrides = [
            f"dataset.train_csv={str(f_train)}",
            "dataset.val_csv=null",
            f"dataset.test_csv={str(f_test)}",
            f"general.name={a.name_prefix}_SMOKE_trial{trials:04d}",
            f"train.n_epochs={a.smoke_epochs}",
            *extra_over,
        ]
        ret = _run_with_logs([sys.executable, str(main_spec), *overrides], trial_dir, enable=a.log_smoke)
        if ret != 0:
            print(f"[TRIAL {trials:04d}] smoke FAILED(ret={ret})")
            if not a.keep_failed_trials: shutil.rmtree(trial_dir, ignore_errors=True)
            continue

        # SAFE로 승격 → safe_split/safe_XXX
        k=1; safe_dir = safe_root/f"safe_{k:03d}"
        while safe_dir.exists(): k+=1; safe_dir = safe_root/f"safe_{k:03d}"
        _ensure_dir(safe_dir)
        shutil.copy2(f_train, safe_dir/"train_boot.csv")
        shutil.copy2(f_test,  safe_dir/"test_oob.csv")

        info = {
            "idx": k, "trial": trials,
            "paths":{"train_csv":str(safe_dir/"train_boot.csv"), "test_csv":str(safe_dir/"test_oob.csv"), "safe_dir":str(safe_dir)},
            "bootstrap_size": a.bootstrap_size, "with_replacement": a.with_replacement,
            "seed": a.seed, "stratify_by": strat_cols if strat_cols else None
        }
        with open(safe_dir/"SAFE_BOOTSTRAP_SPLIT_INFO.json","w",encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        seen.add(sig if sig else f"safe{k:03d}")
        print(f"[OK] SAFE split saved: {safe_dir}")
        found.append({"safe_dir": str(safe_dir), "train_csv": str(safe_dir/"train_boot.csv"), "test_csv": str(safe_dir/"test_oob.csv"), "meta": info})

    if len(found)+already < a.target_safe:
        print(f"[WARN] need {a.target_safe} SAFE, but only {len(found)+already} available.")
    return found

# ============ 메인 ============
def main():
    a=_load_args()

    project_root = Path(a.project_root).resolve()
    out_root     = Path(a.out_root).resolve(); _ensure_dir(out_root)
    main_spec    = Path(a.main_spec).resolve() if a.main_spec else (project_root/"src"/"main_spec.py")
    if not main_spec.exists(): raise FileNotFoundError(f"main_spec.py not found: {main_spec}")

    train_csv = Path(a.train_csv).resolve()
    test_csv  = Path(a.test_csv).resolve() if a.test_csv else None

    df_full = pd.read_csv(train_csv)

    # 1) SAFE 분할 확보 (재사용 + 부족분 검색)
    safe_root = out_root/"safe_split"; _ensure_dir(safe_root)
    reused = _maybe_reuse_safe(a, safe_root, a.target_safe)
    need_more = max(0, a.target_safe - len(reused))
    found = _search_safe_needed(a, df_full, main_spec, out_root, test_csv, already=len(reused)) if need_more>0 else []
    safes = (reused + found)[:a.target_safe]

    if not safes:
        print("[FATAL] No SAFE splits available."); sys.exit(2)

    # 2) N개 분할 본평가(검증 비활성화, test만 사용)
    eval_root = out_root/"bootstrap_eval_runs"; _ensure_dir(eval_root)
    run_dirs=[]
    for i, s in enumerate(safes, start=1):
        safe_dir = Path(s["safe_dir"])
        f_train  = Path(s["train_csv"])
        f_test   = Path(s["test_csv"]) if s.get("test_csv") else _ensure_test_oob_file(safe_dir)
        if f_test is None or not f_test.exists():
            print(f"[WARN] Missing test file in {safe_dir}; skip this fold"); continue

        extra_over = a.extra_overrides_eval.strip().split() if a.extra_overrides_eval else []
        overrides = [
            f"dataset.train_csv={str(f_train)}",
            "dataset.val_csv=null",                 # ✅ 항상 비활성화
            f"dataset.test_csv={str(f_test)}",      # ✅ test만 사용
            f"general.name={a.name_prefix}_EVAL_{i:03d}",
            *extra_over,
        ]
        fold_dir = eval_root/f"fold_{i:03d}"; _ensure_dir(fold_dir)
        print("[EVAL RUN]", " ".join(map(str, [sys.executable, str(main_spec), *overrides])))
        # ret = _run_with_logs([sys.executable, str(main_spec), *overrides], fold_dir, enable=True)
        ret = _run_with_logs([sys.executable, "-u", str(main_spec), *overrides], fold_dir, enable=False)
        if ret != 0:
            print(f"[ERROR] fold {i} failed"); sys.exit(ret)
        run_dirs.append(fold_dir)

    # 집계
    if run_dirs:
        _aggregate_metrics(run_dirs, out_all=eval_root/"bootstrap_eval_metrics_all.csv", out_summary=eval_root/"bootstrap_eval_metrics_summary.csv")
    else:
        print("[WARN] No successful eval runs to aggregate.")

    # 3) 전체 재학습 + 외부 테스트
    if a.do_full_retrain:
        final_dir = out_root/"final_train_full"; _ensure_dir(final_dir)
        extra_over = a.extra_overrides_full.strip().split() if a.extra_overrides_full else []
        overrides = [
            f"dataset.train_csv={str(train_csv)}",
            "dataset.val_csv=null",
            f"dataset.test_csv={str(test_csv)}" if test_csv else "dataset.test_csv=null",
            f"general.name={a.name_prefix}_FINAL_full",
            *extra_over,
        ]
        print("[FINAL RUN]", " ".join(map(str, [sys.executable, str(main_spec), *overrides])))
        ret = _run_with_logs([sys.executable, "-u", str(main_spec), *overrides], final_dir, enable=False)
        if ret != 0:
            print("[ERROR] Final full retrain failed."); sys.exit(ret)

    # 요약 저장
    summary = {
        "target_safe": a.target_safe,
        "used_safes": [{"safe_dir": s["safe_dir"], "train_csv": s["train_csv"], "test_csv": s.get("test_csv")} for s in safes],
        "eval_root": str(eval_root),
        "final_retrain": bool(a.do_full_retrain),
    }
    with open(out_root/"PIPELINE_SUMMARY.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE] Bootstrap N-eval (no val) → Full retrain pipeline finished.")
    print(f"Summary: {out_root/'PIPELINE_SUMMARY.json'}")


if __name__ == "__main__":
    main()
