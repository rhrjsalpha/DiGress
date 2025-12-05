# fill_conditions_from_training.py

import pandas as pd


def fill_conditions_from_training(
    generated_csv: str,
    train_csv: str,
    out_csv: str = None,
    id_col: str = "ID",
    cols_to_copy = ("dielectric_constant_avg", "Solvent"),
):
    """
    1) generated_csv : DiGress로 생성한 결과
       - cond_row_index, cond_id, rep_idx, SMILES, lambda_max_nm,
         spectrum_list, pH_label, DB, ID, type, ...

    2) train_csv : Graphormer 학습에 사용한 원본 training set
       - ID, pH_label, dielectric_constant_avg, Solvent, type, ...

    3) ID 기준으로 cols_to_copy 에 있는 컬럼들을 generated_csv에 채워넣고
       새 CSV 로 저장.
    """

    # 1. CSV 읽기
    gen_df = pd.read_csv(generated_csv)
    train_df = pd.read_csv(train_csv)

    if id_col not in gen_df.columns:
        raise ValueError(f"generated_csv 에 '{id_col}' 컬럼이 없습니다.")
    if id_col not in train_df.columns:
        raise ValueError(f"train_csv 에 '{id_col}' 컬럼이 없습니다.")

    # 2. train_df에서 필요한 컬럼만 추출
    missing_in_train = [c for c in cols_to_copy if c not in train_df.columns]
    if missing_in_train:
        raise ValueError(
            f"train_csv 에 {missing_in_train} 컬럼이 없습니다. "
            f"(cols_to_copy={cols_to_copy} 확인 필요)"
        )

    # ID, 필요한 컬럼만 남기고 ID 기준 중복 제거
    train_sub = (
        train_df[[id_col, *cols_to_copy]]
        .drop_duplicates(subset=id_col)
        .copy()
    )

    # 3. generated 결과와 ID로 merge (left join)
    merged = gen_df.merge(train_sub, on=id_col, how="left")

    # 4. 출력 경로 결정
    if out_csv is None:
        out_csv = generated_csv.replace(".csv", "_with_conditions_from_train.csv")

    merged.to_csv(out_csv, index=False)
    print(f"[INFO] 조건이 채워진 CSV 저장 완료 → {out_csv}")

    return merged


if __name__ == "__main__":
    GENERATED = r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\generated_from_condY_trainingset/generated_molecules_with_conditions.csv"
    TRAIN_CSV = r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"

    fill_conditions_from_training(
        generated_csv=GENERATED,
        train_csv=TRAIN_CSV,
        cols_to_copy=("dielectric_constant_avg", "Solvent"),  # 필요하면 여기 더 추가 가능
    )

