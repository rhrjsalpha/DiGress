# make_multitask_data.py
import os
import sys
import random
import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

SOLVENTS = ["water", "ethanol", "methanol", "acetonitrile"]
PHS = ["acidic", "neutral", "basic"]

def pick(items):
    return items[np.random.randint(0, len(items))]

def gen_gaussian_spectrum(nm_min, nm_max, n_peaks=(2, 5), noise=0.01, rng=None):
    """가우시안 혼합 스펙트럼 (관측 구간 내부에서 0~1 정규화)."""
    if rng is None:
        rng = np.random.default_rng()
    xs = np.arange(nm_min, nm_max + 1)
    y = np.zeros_like(xs, dtype=float)
    k = rng.integers(n_peaks[0], n_peaks[1] + 1)
    for _ in range(k):
        mu = rng.uniform(nm_min, nm_max)
        sigma = rng.uniform(8, 40)
        amp = rng.uniform(0.3, 1.0)
        y += amp * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    y += rng.normal(0.0, noise, size=y.shape)
    y = np.clip(y, 0, None)
    m = y.max()
    if m > 0:
        y = y / m
    return xs, y

def add_global_columns(df):
    df = df.copy()
    df["Solvent"] = [pick(SOLVENTS) for _ in range(len(df))]
    df["pH"] = [pick(PHS) for _ in range(len(df))]
    return df

def build_block_with_random_window(df_base, full_min, full_max, min_len=150, tag="UV"):
    """
    전체 축 [full_min, full_max]를 컬럼으로 두고,
    각 샘플마다 [start,end] 랜덤 윈도우(길이 >= min_len)를 뽑아 그 구간만 스펙트럼 생성, 나머지는 NaN.

    반환: (spec_df, meta_range_df)
      - spec_df: intensity 표(관측 외 구간 NaN)
      - meta_range_df: 각 행의 관측 시작/끝 nm (ex: UV_start_nm, UV_end_nm)
    """
    rng = np.random.default_rng(RNG_SEED)
    n = len(df_base)
    grid = np.arange(full_min, full_max + 1, dtype=int)
    n_cols = len(grid)

    # 결과 버퍼 (NaN으로 초기화)
    mat = np.full((n, n_cols), np.nan, dtype=float)
    starts = np.empty(n, dtype=int)
    ends = np.empty(n, dtype=int)

    max_len = n_cols  # 최대 가능 길이
    if min_len > max_len:
        raise ValueError(f"min_len({min_len}) > available length({max_len})")

    for i in range(n):
        # 길이 무작위 (최소 min_len ~ 전체 길이)
        spec_len = rng.integers(min_len, max_len + 1)
        start_nm = rng.integers(full_min, full_max - spec_len + 2)  # +2 because high is exclusive
        end_nm = start_nm + spec_len - 1

        xs, y = gen_gaussian_spectrum(start_nm, end_nm, rng=rng)
        # 관측 구간을 전체 축에 매핑
        col_start = start_nm - full_min
        col_end = end_nm - full_min + 1
        mat[i, col_start:col_end] = y

        starts[i] = start_nm
        ends[i] = end_nm

    spec_df = pd.DataFrame(mat, columns=[f"{nm}" for nm in grid], index=df_base.index)
    meta_df = pd.DataFrame({f"{tag}_start_nm": starts, f"{tag}_end_nm": ends}, index=df_base.index)
    return spec_df, meta_df

def single_split_indices(n, train_ratio=0.8, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    return idx[:n_train], idx[n_train:]

def build_qm_nm_block(df_base, nm_min=200, nm_max=800, n_peaks=(2, 5), noise=0.01):
    """
    QM_nm용 스펙트럼 블록.
    - 200~800nm 전체 축을 모두 채움(중간 NaN 없음).
    - 각 행은 가우시안 피크 혼합으로 0~1 정규화된 강도.
    반환: (spec_df, meta_df)
      - spec_df: 200..800 컬럼이 모두 값으로 채워진 DataFrame
      - meta_df: 범위 메타(고정: 200, 800)
    """
    rng = np.random.default_rng(RNG_SEED)
    grid = np.arange(nm_min, nm_max + 1, dtype=int)
    n = len(df_base)

    mat = np.zeros((n, len(grid)), dtype=float)
    for i in range(n):
        xs, y = gen_gaussian_spectrum(nm_min, nm_max, n_peaks=n_peaks, noise=noise, rng=rng)
        # gen_gaussian_spectrum는 [nm_min, nm_max] 전체에서 0~1 스케일로 생성
        # xs는 nm_min..nm_max 등간격이므로 바로 할당
        mat[i, :] = y

    spec_df = pd.DataFrame(mat, columns=[f"{nm}" for nm in grid], index=df_base.index)
    meta_df = pd.DataFrame({"QM_start_nm": nm_min, "QM_end_nm": nm_max}, index=df_base.index)
    return spec_df, meta_df


def add_qm_global_columns(df):
    """QM 데이터의 글로벌 메타를 고정값으로 부여: Solvent='QM', pH='neutral'"""
    df = df.copy()
    df["Solvent"] = "QM"
    df["pH"] = "neutral"
    return df

def main(photochemcad_csv="Photochemcad_only.csv", smiles_col="First(SMILES)"):
    if not os.path.exists(photochemcad_csv):
        print(f"[ERROR] '{photochemcad_csv}' 파일이 없습니다. 경로를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    raw = pd.read_csv(photochemcad_csv)
    if smiles_col not in raw.columns:
        print(f"[ERROR] 입력 파일에 '{smiles_col}' 컬럼이 없습니다.", file=sys.stderr)
        sys.exit(1)

    smiles_pool = (
        raw[smiles_col].dropna().astype(str).drop_duplicates().tolist()
    )
    if len(smiles_pool) < 100:
        print(f"[ERROR] 고유 SMILES가 100개 미만입니다. ({len(smiles_pool)}개)", file=sys.stderr)
        sys.exit(1)

    # 100개 샘플
    smiles_100 = np.random.choice(smiles_pool, size=100, replace=False)
    base = pd.DataFrame({"smiles": smiles_100})
    base = add_global_columns(base)

    # ===== UV-Vis (200–800nm), 개별 랜덤 관측 구간(>=150nm) =====
    uvvis_block, uv_ranges = build_block_with_random_window(
        base, full_min=200, full_max=800, min_len=150, tag="UV"
    )
    uvvis_full = pd.concat([base, uv_ranges, uvvis_block], axis=1)

    # ===== Fluorescence (300–700nm), 개별 랜덤 관측 구간(>=150nm) =====
    fluor_block, fl_ranges = build_block_with_random_window(
        base, full_min=300, full_max=700, min_len=150, tag="FL"
    )
    fluor_full = pd.concat([base, fl_ranges, fluor_block], axis=1)

    # ===== QM_nm (200–800nm, Full Grid, No NaN) =====
    # QM은 글로벌 메타를 고정(Solvent='QM', pH='neutral')
    base_qm = add_qm_global_columns(pd.DataFrame({"smiles": smiles_100}))
    qm_block, qm_ranges = build_qm_nm_block(base_qm, nm_min=200, nm_max=800, n_peaks=(2, 5), noise=0.01)
    qm_full = pd.concat([base_qm, qm_ranges, qm_block], axis=1)

    # ===== 동일 인덱스로 80/20 split (UV/FL/QM 공통) =====
    train_idx, test_idx = single_split_indices(len(base), train_ratio=0.8, seed=RNG_SEED)

    uv_train = uvvis_full.iloc[train_idx].reset_index(drop=True)
    uv_test = uvvis_full.iloc[test_idx].reset_index(drop=True)

    fl_train = fluor_full.iloc[train_idx].reset_index(drop=True)
    fl_test = fluor_full.iloc[test_idx].reset_index(drop=True)

    qm_train = qm_full.iloc[train_idx].reset_index(drop=True)
    qm_test = qm_full.iloc[test_idx].reset_index(drop=True)

    # ===== 저장 =====
    uv_train.to_csv("uvvis_fake_train.csv", index=False)
    uv_test.to_csv("uvvis_fake_test.csv", index=False)

    fl_train.to_csv("fluorescence_fake_train.csv", index=False)
    fl_test.to_csv("fluorescence_fake_test.csv", index=False)

    qm_train.to_csv("qm_nm_fake_train.csv", index=False)
    qm_test.to_csv("qm_nm_fake_test.csv", index=False)

    print("✅ Saved:")
    print(" - uvvis_fake_train.csv")
    print(" - uvvis_fake_test.csv")
    print(" - fluorescence_fake_train.csv")
    print(" - fluorescence_fake_test.csv")
    print(" - qm_nm_fake_train.csv")
    print(" - qm_nm_fake_test.csv")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = "Photochemcad_only.csv"
    if len(sys.argv) >= 3:
        smiles_col = sys.argv[2]
    else:
        smiles_col = "First(SMILES)"
    main(csv_path, smiles_col)
