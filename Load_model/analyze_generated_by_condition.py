# analyze_generated_by_condition.py
# -*- coding: utf-8 -*-
"""
generated_molecules_with_conditions.csv 를 읽어서

- cond_id(조건) 별로
    * validity
    * relaxed_validity (여기서는 validity와 동일하게 취급)
    * uniqueness
    * novelty (train SMILES 기준)

을 계산하고,
조건별로 "유효 + 유니크 + 노벨" 한 분자들을 따로 모아서 CSV로 저장한다.

조건 사이에서는 같은 SMILES 가 나와도 상관 없으므로,
모든 계산은 cond_id 그룹 안에서만 수행한다.
"""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs  # <<< 추가

# ------------------------------------------------------------------
# 1. 경로 설정 (★ 여기만 바꿔서 사용)
# ------------------------------------------------------------------
# 생성 단계에서 만든 CSV
GENERATED_CSV = Path(
    r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\generated_from_condY_trainingset\generated_molecules_with_conditions.csv"
)

# train SMILES 목록 (novelty 계산용)
# - 컬럼 이름은 기본으로 "SMILES" 라고 가정.
# - novelty 안 쓸 거면 TRAIN_SMILES_CSV = None 로 두면 됨.
TRAIN_SMILES_CSV: Optional[Path] = Path(
    r"EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
)
TRAIN_SMILES_COL = "InChI"

# 결과 저장 폴더
OUT_DIR = GENERATED_CSV.parent / "metrics_per_condition"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_THRESHOLD = 0.6   # 원하면 0.7, 0.8 등으로 조절
FP_RADIUS = 2
FP_NBITS = 2048

# ------------------------------------------------------------------
# 2. 유틸 함수
# ------------------------------------------------------------------

def canonical_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    문자열 리스트에서
    - NaN / 빈 문자열 / None 제거
    - RDKit MolFromSmiles 로 파싱 가능한 것만 남기고
    - canonical SMILES 로 변환해서 리턴
    """
    valids: List[str] = []
    for s in smiles_list:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s or s.lower() == "none":
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        # canonical 로 통일
        can = Chem.MolToSmiles(mol)
        valids.append(can)
    return valids


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
        # train 셋이 없으면 unique 전부를 novel 로 취급할 수도 있고,
        # novelty 지표는 1.0 으로 놓거나 NaN 으로 둘 수 있음.
        novel_smiles = unique_smiles[:]
        n_novel = len(novel_smiles)
        novelty = 1.0 if n_unique > 0 else float("nan")

    metrics = {
        "total": float(total),
        "n_valid": float(n_valid),
        "validity": float(validity),
        # relaxed_validity 는 여기서는 validity 와 동일하게 둠
        "relaxed_validity": float(validity),
        "n_unique": float(n_unique),
        "uniqueness": float(uniqueness),  # valid 중 unique 비율
        "n_novel": float(n_novel),
        "novelty": float(novelty),        # unique 중 novel 비율
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

    # (2) train SMILES 로딩 (novelty 용)
    train_smiles_set: Optional[set] = None
    if TRAIN_SMILES_CSV is not None and TRAIN_SMILES_CSV.is_file():
        train_df = pd.read_csv(TRAIN_SMILES_CSV)
        if TRAIN_SMILES_COL not in train_df.columns:
            raise ValueError(
                f"train CSV 에 '{TRAIN_SMILES_COL}' 컬럼이 없습니다. "
                f"다른 이름이면 TRAIN_SMILES_COL 을 수정하세요."
            )
        train_smiles = canonical_valid_smiles(train_df[TRAIN_SMILES_COL].tolist())
        train_smiles_set = set(train_smiles)
        print(f"[INFO] Loaded {len(train_smiles_set)} train SMILES for novelty 계산.")
    else:
        print("[WARN] TRAIN_SMILES_CSV 를 찾을 수 없어서 novelty 를 계산하지 않습니다.")
        train_smiles_set = None

    # (3) cond_id 별 그룹핑
    cond_groups = df.groupby("cond_id")

    metrics_rows = []
    filtered_rows = []

    for cond_id, g in cond_groups:
        # 이 조건의 기본 정보 (center_nm, sigma_nm 등은 그룹에서 대표값 한 번만 사용)
        center_nm = g["center_nm"].iloc[0] if "center_nm" in g.columns else None
        sigma_nm = g["sigma_nm"].iloc[0] if "sigma_nm" in g.columns else None
        height = g["height"].iloc[0] if "height" in g.columns else None
        pH_label = g["pH_label"].iloc[0] if "pH_label" in g.columns else None
        dielectric = (
            g["dielectric_constant_avg"].iloc[0]
            if "dielectric_constant_avg" in g.columns
            else None
        )
        is_qm = g["is_qm"].iloc[0] if "is_qm" in g.columns else None
        type_label = g["type"].iloc[0] if "type" in g.columns else None

        metrics, valid_list, unique_list, novel_list = compute_metrics_for_condition(
            g["SMILES"], train_smiles_set
        )

        # --- cond_id별 metric 저장 ---
        metrics_rows.append(
            {
                "cond_id": cond_id,
                "center_nm": center_nm,
                "sigma_nm": sigma_nm,
                "height": height,
                "pH_label": pH_label,
                "dielectric_constant_avg": dielectric,
                "is_qm": is_qm,
                "type": type_label,
                **metrics,
            }
        )

        # --- 이 조건에서 valid & unique & novel 인 분자만 저장 ---
        for smi in novel_list:
            filtered_rows.append(
                {
                    "cond_id": cond_id,
                    "center_nm": center_nm,
                    "sigma_nm": sigma_nm,
                    "height": height,
                    "pH_label": pH_label,
                    "dielectric_constant_avg": dielectric,
                    "is_qm": is_qm,
                    "type": type_label,
                    "SMILES": smi,
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
