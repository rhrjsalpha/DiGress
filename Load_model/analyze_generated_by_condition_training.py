# analyze_generated_by_condition.py
# -*- coding: utf-8 -*-
"""
generated_molecules_with_conditions.csv 를 읽어서

- cond_id(조건) 별로
    * validity
    * relaxed_validity (여기서는 validity와 동일하게 취급)
    * uniqueness
    * novelty (train 구조 기준)

을 계산하고,
조건별로 "유효 + 유니크 + 노벨" 한 분자들을 따로 모아서 CSV로 저장한다.

여기서는 Gaussian 파라미터(center_nm, sigma_nm, height)를 쓰지 않고,
대신 실제 데이터셋의 lambda_max_nm, spectrum_list 를 조건 정보로 사용한다.
"""

from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs  # (지금은 안 쓰지만 확장용)

# ------------------------------------------------------------------
# 1. 경로 설정 (★ 여기만 바꿔서 사용)
# ------------------------------------------------------------------
# 생성 단계에서 만든 CSV
GENERATED_CSV = Path(
    r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\generated_from_condY_trainingset\generated_molecules_with_conditions_with_conditions_from_train.csv"
)

# train 구조 목록 (novelty 계산용)
# - TRAIN_SMILES_COL 이 InChI 인지 SMILES 인지에 따라 TRAIN_IS_INCHI 를 설정
TRAIN_SMILES_CSV: Optional[Path] = Path(
    r"EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
)
TRAIN_SMILES_COL = "InChI"   # "InChI" 또는 "SMILES"
TRAIN_IS_INCHI   = True      # InChI면 True, SMILES면 False

# 결과 저장 폴더
OUT_DIR = GENERATED_CSV.parent / "metrics_per_condition"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 2. 유틸 함수
# ------------------------------------------------------------------
def canonical_from_strings(str_list: List[str], is_inchi: bool = False) -> List[str]:
    """
    문자열 리스트에서
    - NaN / 빈 문자열 / None 제거
    - is_inchi 에 따라 MolFromInchi 또는 MolFromSmiles 사용
    - canonical SMILES 로 변환해서 리턴
    """
    valids: List[str] = []
    for s in str_list:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s or s.lower() == "none":
            continue

        if is_inchi:
            mol = Chem.MolFromInchi(s)
        else:
            mol = Chem.MolFromSmiles(s)

        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        valids.append(can)
    return valids


def canonical_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    SMILES 전용 canonical 함수 (generated CSV용)
    """
    return canonical_from_strings(smiles_list, is_inchi=False)


def compute_metrics_for_condition(
    smiles_series: pd.Series,
    train_smiles_set: Optional[set] = None,
) -> tuple[Dict[str, float], List[str], List[str], List[str]]:
    """
    한 조건(cond_id)에 해당하는 SMILES Series 에 대해
    validity / uniqueness / novelty 계산.

    반환:
      metrics_dict,
      valid_smiles(canonical),
      unique_smiles,
      novel_smiles
    """
    total = len(smiles_series)

    # 1) valid + canonical
    valid_smiles = canonical_valid_smiles(smiles_series.tolist())
    n_valid = len(valid_smiles)

    validity = n_valid / total if total > 0 else float("nan")

    # 2) unique (valid 안에서만)
    unique_smiles = sorted(set(valid_smiles))
    n_unique = len(unique_smiles)
    uniqueness = n_unique / n_valid if n_valid > 0 else float("nan")

    # 3) novel (unique 안에서만)
    if train_smiles_set is not None:
        novel_smiles = [s for s in unique_smiles if s not in train_smiles_set]
        n_novel = len(novel_smiles)
        novelty = n_novel / n_unique if n_unique > 0 else float("nan")
    else:
        novel_smiles = unique_smiles[:]
        n_novel = len(novel_smiles)
        novelty = 1.0 if n_unique > 0 else float("nan")

    metrics = {
        "total": float(total),
        "n_valid": float(n_valid),
        "validity": float(validity),
        "relaxed_validity": float(validity),
        "n_unique": float(n_unique),
        "uniqueness": float(uniqueness),
        "n_novel": float(n_novel),
        "novelty": float(novelty),
    }

    return metrics, valid_smiles, unique_smiles, novel_smiles


# ------------------------------------------------------------------
# 3. 메인 로직
# ------------------------------------------------------------------
def main():
    # (1) 생성 결과 로드
    df = pd.read_csv(GENERATED_CSV)

    if "SMILES" not in df.columns:
        raise ValueError("입력 CSV 에 'SMILES' 컬럼이 없습니다.")
    if "cond_id" not in df.columns:
        raise ValueError("입력 CSV 에 'cond_id' 컬럼이 없습니다.")

    # (2) train 구조 로딩 (novelty 용)
    train_smiles_set: Optional[set] = None
    if TRAIN_SMILES_CSV is not None and TRAIN_SMILES_CSV.is_file():
        train_df = pd.read_csv(TRAIN_SMILES_CSV)
        if TRAIN_SMILES_COL not in train_df.columns:
            raise ValueError(
                f"train CSV 에 '{TRAIN_SMILES_COL}' 컬럼이 없습니다. "
                f"다른 이름이면 TRAIN_SMILES_COL 을 수정하세요."
            )

        train_str_list = train_df[TRAIN_SMILES_COL].tolist()
        train_smiles = canonical_from_strings(
            train_str_list,
            is_inchi=TRAIN_IS_INCHI,
        )
        train_smiles_set = set(train_smiles)
        print(f"[INFO] Loaded {len(train_smiles_set)} train 구조 for novelty 계산.")
    else:
        print("[WARN] TRAIN_SMILES_CSV 를 찾을 수 없어서 novelty 를 계산하지 않습니다.")
        train_smiles_set = None

    # (3) cond_id 별 그룹핑
    cond_groups = df.groupby("cond_id")

    metrics_rows: List[Dict] = []
    filtered_rows: List[Dict] = []

    for cond_id, g in cond_groups:
        # 공통 메타 정보
        pH_label = g["pH_label"].iloc[0] if "pH_label" in g.columns else None
        dielectric = (
            g["dielectric_constant_avg"].iloc[0]
            if "dielectric_constant_avg" in g.columns
            else None
        )
        type_label = g["type"].iloc[0] if "type" in g.columns else None

        # ★ 새 조건 정보: lambda_max_nm, spectrum_list
        lambda_max_nm = (
            g["lambda_max_nm"].iloc[0] if "lambda_max_nm" in g.columns else None
        )
        spectrum_list = (
            g["spectrum_list"].iloc[0] if "spectrum_list" in g.columns else None
        )

        # DB/ID 대표값 (cond-level 메타로 넣어도 되고, 없어도 됨)
        db_repr = g["DB"].iloc[0] if "DB" in g.columns else None
        id_repr = g["ID"].iloc[0] if "ID" in g.columns else None

        metrics, valid_list, unique_list, novel_list = compute_metrics_for_condition(
            g["SMILES"], train_smiles_set
        )

        # --- cond_id별 metric 저장 ---
        metrics_rows.append(
            {
                "cond_id": cond_id,
                "pH_label": pH_label,
                "dielectric_constant_avg": dielectric,
                "type": type_label,
                "lambda_max_nm": lambda_max_nm,
                "spectrum_list": spectrum_list,
                "DB_repr": db_repr,
                "ID_repr": id_repr,
                **metrics,
            }
        )

        # --- 그룹 내에서 canonical SMILES → 대표 row 메타정보 매핑 ---
        can2meta: Dict[str, Dict] = {}

        for _, row in g.iterrows():
            s_raw = row.get("SMILES", None)
            if not isinstance(s_raw, str):
                continue
            s_raw = s_raw.strip()
            if not s_raw:
                continue
            mol = Chem.MolFromSmiles(s_raw)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            if can in can2meta:
                continue

            can2meta[can] = {
                "DB": row["DB"] if "DB" in g.columns else None,
                "ID": row["ID"] if "ID" in g.columns else None,
                "lambda_max_nm": row["lambda_max_nm"] if "lambda_max_nm" in g.columns else None,
                "spectrum_list": row["spectrum_list"] if "spectrum_list" in g.columns else None,
            }

        # --- 이 조건에서 valid & unique & novel 인 분자만 저장 ---
        for smi in novel_list:
            meta = can2meta.get(smi, {})
            filtered_rows.append(
                {
                    "cond_id": cond_id,
                    "pH_label": pH_label,
                    "dielectric_constant_avg": dielectric,
                    "type": type_label,
                    # per-molecule 메타
                    "DB": meta.get("DB", None),
                    "ID": meta.get("ID", None),
                    "lambda_max_nm": meta.get("lambda_max_nm", lambda_max_nm),
                    "spectrum_list": meta.get("spectrum_list", spectrum_list),
                    "SMILES": smi,   # canonical SMILES
                }
            )

    # (4) 결과 저장
    metrics_df = pd.DataFrame(metrics_rows)
    filtered_df = pd.DataFrame(filtered_rows)

    metrics_path = OUT_DIR / "condition_metrics.csv"
    filtered_path = OUT_DIR / "valid_unique_novel_molecules.csv"

    metrics_df.to_csv(metrics_path, index=False)
    filtered_df.to_csv(filtered_path, index=False)

    print(f"[INFO] 조건별 metrics 를 {metrics_path} 에 저장했습니다.")
    print(f"[INFO] 필터 통과 분자들을 {filtered_path} 에 저장했습니다.")


if __name__ == "__main__":
    main()

