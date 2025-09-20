# export_bad_good_rows_train_test_v2.py
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

# ==== 경로 수정 ====
BAD_DIR   = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\outputs\2025-09-20\12-47-17-graph-tf-model\_bad_batches")
SRC_TRAIN = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\data\csv\ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv")
SRC_TEST  = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\data\csv\ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv")

def _read_log():
    files = sorted(BAD_DIR.glob("bad_rank*.csv"))
    if not files:
        raise SystemExit(f"No bad_rank*.csv in {BAD_DIR}")
    log = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # split 컬럼이 없으면 'test'로 채워서 호환
    if "split" not in log.columns:
        log["split"] = "test"

    # 안전성: batch_idx가 문자열이면 정수화
    if "batch_idx" in log.columns:
        log["batch_idx"] = pd.to_numeric(log["batch_idx"], errors="coerce").astype("Int64")
    else:
        # 아주 옛 포맷이면 대신 'batch' 같은 걸 찾을 수도 있음
        pass
    return log

from typing import Optional, List, Set
def _collect_indices(log: pd.DataFrame, split: str, n_rows: Optional[int] = None) -> List[int]:
    sub = log[log["split"] == split].copy()
    idx: Set[int] = set()

    # 1) indices 우선
    if "indices" in sub.columns:
        for s in sub["indices"].fillna("").astype(str):
            s = s.strip()
            if not s:
                continue
            for tok in s.split():
                try:
                    idx.add(int(tok))
                except Exception:
                    pass

    # 2) 비었으면 batch_idx 사용 (shuffle=False, batch_size=1 전제)
    if not idx and "batch_idx" in sub.columns:
        for b in sub["batch_idx"].dropna().tolist():
            try:
                i = int(b)
                if n_rows is None or (0 <= i < n_rows):
                    idx.add(i)
            except Exception:
                pass

    return sorted(idx)

def _export(src_csv: Path, idx_bad: list[int], out_prefix: str):
    if not src_csv or not src_csv.exists():
        print(f"[SKIP] {out_prefix}: source csv not found -> {src_csv}")
        return
    src = pd.read_csv(src_csv)
    bad_df  = src.iloc[idx_bad] if idx_bad else src.iloc[[]]
    good_df = src[~src.index.isin(idx_bad)] if idx_bad else src

    BAD_DIR.mkdir(parents=True, exist_ok=True)
    bad_path  = BAD_DIR / f"{out_prefix}_bad_rows.csv"
    good_path = BAD_DIR / f"{out_prefix}_good_rows.csv"
    bad_df.to_csv(bad_path, index=False)
    good_df.to_csv(good_path, index=False)
    print(f"[SAVE] {out_prefix}: bad={len(bad_df)}, good={len(good_df)}")
    print(f"       -> {bad_path.name}, {good_path.name}")

def main():
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    log = _read_log()

    # 원본 CSV 길이(선택): batch_idx 범위 체크용
    n_train = pd.read_csv(SRC_TRAIN).shape[0] if SRC_TRAIN.exists() else None
    n_test  = pd.read_csv(SRC_TEST ).shape[0] if SRC_TEST.exists()  else None

    train_bad = _collect_indices(log, "train", n_rows=n_train)
    test_bad  = _collect_indices(log, "test",  n_rows=n_test)

    _export(SRC_TRAIN, train_bad, "train")
    _export(SRC_TEST,  test_bad,  "test")

if __name__ == "__main__":
    main()


