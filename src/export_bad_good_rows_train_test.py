# export_bad_good_rows_by_ident.py
# -*- coding: utf-8 -*-
"""
bad_rank*.csv 의 ident 컬럼(SMILES | InChI … 형태)을 파싱해
원본 CSV에서 동일 InChI/SMILES 를 가진 행을 bad/good 으로 분리 저장.

출력:
  <BAD_DIR>/train_bad_rows.csv,  train_good_rows.csv
  <BAD_DIR>/test_bad_rows.csv,   test_good_rows.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Set, Optional, List

import pandas as pd


# ====== 경로를 환경에 맞게 수정하세요 ======
BAD_DIR   = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\outputs\2025-09-20\14-22-17-graph-tf-model\_bad_batches")
SRC_TRAIN = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\data\csv\ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv")
SRC_TEST  = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\data\csv\ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv")

# 원본 CSV의 컬럼명(필요시 바꾸세요)
INCHI_COL  = "InChI"
SMILES_COL = "SMILES"


def _read_log_concat(bad_dir: Path) -> pd.DataFrame:
    files = sorted(bad_dir.glob("bad_rank*.csv"))
    if not files:
        raise SystemExit(f"No bad_rank*.csv in {bad_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "split" not in df.columns:
        df["split"] = "test"
    if "ident" not in df.columns:
        df["ident"] = ""
    return df


def _parse_ident_cell(cell: str) -> Tuple[Set[str], Set[str]]:
    """'SMILES ... | InChI=...' 섞인 ident 문자열에서 InChI/SMILES 집합을 분리."""
    inchi_set: Set[str] = set()
    smiles_set: Set[str] = set()

    if not isinstance(cell, str):
        return inchi_set, smiles_set

    parts = [p.strip() for p in cell.split("|") if isinstance(p, str) and p.strip()]
    for p in parts:
        if p.startswith("InChI="):
            inchi_set.add(p.strip())
        else:
            # 공백 없는 토큰을 SMILES 후보로 수집(간단 기준)
            if " " not in p:
                smiles_set.add(p.strip())
    return inchi_set, smiles_set


def _collect_bad_idents(log: pd.DataFrame, split: str) -> Tuple[Set[str], Set[str]]:
    sub = log.loc[log["split"] == split]
    inchi_all: Set[str] = set()
    smiles_all: Set[str] = set()
    for cell in sub["ident"].fillna("").astype(str):
        i_set, s_set = _parse_ident_cell(cell)
        inchi_all |= i_set
        smiles_all |= s_set
    return inchi_all, smiles_all


def _export_by_ident(
    src_csv: Path,
    bad_inchi: Set[str],
    bad_smiles: Set[str],
    out_prefix: str,
) -> None:
    if not src_csv.exists():
        print(f"[SKIP] {out_prefix}: source csv not found -> {src_csv}")
        return

    df = pd.read_csv(src_csv)
    cols = set(df.columns)

    # ident가 없으면 아무 것도 분리하지 않음(경고만 출력)
    if not bad_inchi and not bad_smiles:
        print(f"[WARN] {out_prefix}: ident 집합이 비어 있어 아무 것도 분리하지 않습니다.")
        bad_df = df.iloc[[]]
        good_df = df
    else:
        bad_mask = pd.Series(False, index=df.index)
        if INCHI_COL in cols and bad_inchi:
            bad_mask |= df[INCHI_COL].astype(str).str.strip().isin({s.strip() for s in bad_inchi})
        if SMILES_COL in cols and bad_smiles:
            bad_mask |= df[SMILES_COL].astype(str).str.strip().isin({s.strip() for s in bad_smiles})
        bad_df = df.loc[bad_mask]
        good_df = df.loc[~bad_mask]

    BAD_DIR.mkdir(parents=True, exist_ok=True)
    (BAD_DIR / f"{out_prefix}_bad_rows.csv").write_text("", encoding="utf-8")  # touch
    bad_df.to_csv(BAD_DIR / f"{out_prefix}_bad_rows.csv", index=False)
    good_df.to_csv(BAD_DIR / f"{out_prefix}_good_rows.csv", index=False)

    print(f"[SAVE] {out_prefix}: bad={len(bad_df)}, good={len(good_df)}")
    print(f"       -> {(BAD_DIR / f'{out_prefix}_bad_rows.csv').name}, "
          f"{(BAD_DIR / f'{out_prefix}_good_rows.csv').name}")


def main():
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    log = _read_log_concat(BAD_DIR)

    # train
    train_inchi, train_smiles = _collect_bad_idents(log, "train")
    _export_by_ident(SRC_TRAIN, train_inchi, train_smiles, "train")

    # test
    test_inchi, test_smiles = _collect_bad_idents(log, "test")
    _export_by_ident(SRC_TEST, test_inchi, test_smiles, "test")


if __name__ == "__main__":
    main()

