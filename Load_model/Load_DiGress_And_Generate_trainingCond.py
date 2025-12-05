# Load_DiGress_And_Generate.py
# -*- coding: utf-8 -*-
"""
1) 기존 스펙트럼 CSV + CSVSpecDataset 을 이용해서
   - y(스펙트럼 + global 인코딩)를 그대로 꺼내어
   - condY_from_dataset.csv 로 저장한다.
   - 이때 lambda_max_nm 과 spectrum_list, 간단한 메타 정보(pH_label, DB, ID, type)를 함께 저장한다.

2) 방금 만든 condY_from_dataset.csv / cond_y_batch 를 이용해서
   - y_0 ~ y_{y_dim-1} 를 cond_y_batch 텐서로 만들고
   - 학습된 DiGress(DiscreteDenoisingDiffusion) 모델에 넣어
     조건부로 분자를 생성하고 SMILES + 조건 정보를 CSV로 저장한다.

※ stats.json 은 CSVSpecDataset 에서 처리하며,
   numeric 값들은 학습 때와 동일한 규칙(z-score 등)으로 인코딩된다.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import tqdm
import yaml
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # RDKit WARNING 숨기기

# ------------------------------------------------------------------
# 0. 프로젝트 루트 / sys.path / ckpt alias 설정
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Load_model 상위가 프로젝트 루트라고 가정
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[DEBUG] PROJECT_ROOT added to sys.path:", PROJECT_ROOT)

# === DiGress 관련 모듈 ===
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
import src.analysis
import src.analysis.visualization
import src.diffusion

# Dataset (훈련 때 쓰던 것)
from src.datasets.csv_spectrum_dataset import CSVSpecDataset

# Lightning ckpt 가 예전에 'analysis', 'diffusion' 이라는 모듈 경로를 썼으므로
# 현재 src.analysis / src.diffusion 을 alias 로 등록
sys.modules["analysis"] = src.analysis
sys.modules["diffusion"] = src.diffusion


# ------------------------------------------------------------------
# 1. 경로 / 설정
# ------------------------------------------------------------------
# (1) 학습된 DiGress 체크포인트(.ckpt) 경로 ★★ 여기만 본인 경로 확인
CKPT_PATH = r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\last.ckpt"

# (2) condY DB CSV 저장 위치
CONDY_CSV_PATH = Path(
    r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\CondY_from_dataset.csv"
)

# (3) 생성된 분자 출력 폴더
OUT_DIR = Path(
    r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\generated_from_condY_trainingset"
)

# (4) 각 조건당 몇 개 분자를 생성할지
NUM_MOLS_PER_CONDITION = 1000
MAX_BATCH_SIZE = 1000

# (5) 조건으로 사용할 원본 스펙트럼 CSV 경로 ★★ (훈련 때 쓴 것과 동일하게 맞추는 것이 좋음)
DATASET_CSV = Path(
    r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
)
DATASET_STAGE = "train"   # "train" / "test" 등
DATASET_STATS: Optional[Path] = None  # None 이면 <csv_stem>_stats.json 자동 사용

# (6) YAML 설정 경로
YAML_PATH = r"C:\Users\analcheminfo\PycharmProjects\DiGress\configs\dataset\csv_spec.yaml"


# ------------------------------------------------------------------
# 2. YAML 로부터 global/spectrum 설정 로드
# ------------------------------------------------------------------
def load_global_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    configs/dataset/csv_spec.yaml 에서
    - spectrum_start / spectrum_end
    - global_cols
    - fixed_vocabs
    - boolean_cols
    - smiles_col / inchi_col
    - spectrum_fill_eps / add_h
    만 추출해서 딕셔너리로 반환.
    (train_csv / test_csv 는 여기서 무시)
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    spec_start = int(cfg.get("spectrum_start", 200))
    spec_end = int(cfg.get("spectrum_end", 800))

    global_cols = cfg.get("global_cols", [])
    fixed_vocabs = cfg.get("fixed_vocabs", {}) or {}
    boolean_cols = cfg.get("boolean_cols", []) or []

    smiles_col = cfg.get("smiles_col", None)
    inchi_col = cfg.get("inchi_col", "InChI")

    spectrum_fill_eps = float(cfg.get("spectrum_fill_eps", 1e-8))
    add_h = bool(cfg.get("add_h", False))

    return {
        "spectrum_start": spec_start,
        "spectrum_end": spec_end,
        "global_cols": global_cols,
        "fixed_vocabs": fixed_vocabs,
        "boolean_cols": boolean_cols,
        "smiles_col": smiles_col,
        "inchi_col": inchi_col,
        "spectrum_fill_eps": spectrum_fill_eps,
        "add_h": add_h,
    }


# ------------------------------------------------------------------
# 3. DiGress 모델 로드 & 샘플링
# ------------------------------------------------------------------
def load_trained_model(ckpt_path: str, device: torch.device) -> DiscreteDenoisingDiffusion:
    """
    학습된 DiGress(DiscreteDenoisingDiffusion) LightningModule 체크포인트를 로드.
    sampling_metrics 관련 unexpected key 는 strict=False 로 무시한다.
    """
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        ckpt_path,
        train_metrics=None,
        sampling_metrics=None,
        strict=False,  # sampling_metrics.* 같은 불필요 키 무시
    )
    model.disable_sampling_visualization = True

    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")
    print(f"[INFO] diffusion steps (T): {model.T}")
    return model


def sample_molecules_with_conditions_chunked(
    model: DiscreteDenoisingDiffusion,
    cond_y_batch: torch.Tensor,           # [num_conds, y_dim]
    num_mols_per_condition: int,
    chunk_cond_size: int = 8,             # 한 번에 처리할 cond 개수
):
    """
    cond_y_batch 를 여러 chunk 로 나눠서
    - 각 chunk 안에서, 다시 여러 sub-batch 로 나눠 sample_batch 호출
    - chunk 진행도 + chunk 내부 배치 진행도 둘 다 tqdm 으로 표시

    VRAM 이 부족할 때 batch_size 를 직접 줄이는 용도.
    """
    device = next(model.parameters()).device
    num_conds = cond_y_batch.size(0)
    all_molecules = []

    number_chain_steps = int(model.T - 1)

    # 청크 시작 인덱스 리스트 & 전체 청크 수
    chunk_starts = list(range(0, num_conds, chunk_cond_size))
    num_chunks = len(chunk_starts)
    print(f"[INFO] Total conditions: {num_conds}, "
          f"chunk_cond_size={chunk_cond_size}, num_chunks={num_chunks}")
    print(f"[INFO] Each condition → {num_mols_per_condition} molecules "
          f"→ Total molecules = {num_conds * num_mols_per_condition}")

    # chunk 진행도 tqdm
    for chunk_idx, start in enumerate(
        tqdm.tqdm(chunk_starts, desc="Sampling chunks", unit="chunk"),
        start=1
    ):
        end = min(num_conds, start + chunk_cond_size)
        cond_chunk = cond_y_batch[start:end]              # [C, y_dim]
        C = cond_chunk.size(0)

        # 이 chunk 에서 전체 생성해야 할 y 벡터
        full_batch = cond_chunk.repeat_interleave(
            repeats=num_mols_per_condition,
            dim=0
        )                                                 # [C * N, y_dim]
        total = full_batch.size(0)

        # sub-batch 개수 계산
        num_sub_batches = (total + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

        print(f"[INFO] [chunk {chunk_idx}/{num_chunks}] "
              f"cond {start} ~ {end-1} (chunk_size={C}, total_batch={total}, "
              f"sub_batches={num_sub_batches}, MAX_BATCH_SIZE={MAX_BATCH_SIZE})")

        # sub-batch 진행도 tqdm
        for sub_idx in tqdm.tqdm(
            range(num_sub_batches),
            desc=f"  Batches in chunk {chunk_idx}",
            unit="batch",
            leave=False,   # chunk bar 아래에 깔끔하게
        ):
            sub_start = sub_idx * MAX_BATCH_SIZE
            sub_end = min(total, (sub_idx + 1) * MAX_BATCH_SIZE)
            cond_sub = full_batch[sub_start:sub_end]      # [B_sub, y_dim]

            with torch.no_grad():
                molecules = model.sample_batch(
                    batch_id=sub_start,              # 대충 인덱스로 사용
                    batch_size=cond_sub.size(0),
                    keep_chain=0,
                    number_chain_steps=number_chain_steps,
                    save_final=cond_sub.size(0),
                    num_nodes=None,
                    cond_y_base=cond_sub,
                )

            all_molecules.extend(molecules)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_molecules


# ------------------------------------------------------------------
# 4. SMILES + cond 정보 저장
# ------------------------------------------------------------------
def save_smiles_with_conditions(
    model: DiscreteDenoisingDiffusion,
    molecule_list,
    cond_df: pd.DataFrame,
    num_mols_per_condition: int,
    out_dir: Path,
):
    """
    molecule_list (길이 = num_conds * num_mols_per_condition)를
    cond_df (condY + 메타 정보)와 매칭해서

    - CSV: generated_molecules_with_conditions.csv
      (SMILES + lambda_max_nm + spectrum_list + pH_label/DB/ID/type 등)

    로 저장하면서, tqdm 으로 진행률도 보여준다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = model.visualization_tools
    if vis is None:
        raise RuntimeError("visualization_tools 가 None 입니다.")

    rows = []
    global_idx = 0

    num_conds = len(cond_df)
    total = num_conds * num_mols_per_condition
    print(f"[INFO] Saving {total} molecules "
          f"(num_conds={num_conds}, reps={num_mols_per_condition})")

    # tqdm 으로 전체 진행률 표시
    with tqdm.tqdm(total=total, desc="Saving molecules", unit="mol") as pbar:
        for cond_idx in range(num_conds):
            row_cond = cond_df.iloc[cond_idx]
            cond_id = row_cond.get("cond_id", cond_idx + 1)

            for rep in range(num_mols_per_condition):
                if global_idx >= len(molecule_list):
                    break

                atom_types, edge_types = molecule_list[global_idx]
                global_idx += 1

                mol = vis.mol_from_graphs(
                    atom_types.cpu().numpy(),
                    edge_types.cpu().numpy(),
                )
                if mol is None:
                    smi = None
                else:
                    smi = Chem.MolToSmiles(mol)

                row_out = {
                    "cond_row_index": cond_idx,
                    "cond_id": cond_id,
                    "rep_idx": rep,
                    "SMILES": smi,
                    "lambda_max_nm": row_cond.get("lambda_max_nm", None),
                    "spectrum_list": row_cond.get("spectrum_list", None),
                    "pH_label": row_cond.get("pH_label", None),
                    "DB": row_cond.get("DB", None),
                    "ID": row_cond.get("ID", None),
                    "type": row_cond.get("type", None),
                }
                rows.append(row_out)

                pbar.update(1)  # 한 개 저장할 때마다 +1
                pbar.set_postfix(cond_id=int(cond_id), rep=rep)

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "generated_molecules_with_conditions.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] SMILES + cond 정보를 {csv_path} 에 저장했습니다.")


# ------------------------------------------------------------------
# 5. main: 전체 파이프라인 실행
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -------------------------------------------------------
    # ① YAML 설정 로드
    # -------------------------------------------------------
    gcfg = load_global_config_from_yaml(YAML_PATH)

    print("=== YAML 기반 설정 ===")
    print(f"spectrum_start..end : {gcfg['spectrum_start']}..{gcfg['spectrum_end']}")
    print(f"global_cols         : {gcfg['global_cols']}")
    print(f"fixed_vocabs        : {gcfg['fixed_vocabs']}")
    print(f"boolean_cols        : {gcfg['boolean_cols']}")
    print()

    # -------------------------------------------------------
    # ② CSVSpecDataset 생성 (학습 때와 동일한 규칙으로 y 인코딩)
    # -------------------------------------------------------
    dataset = CSVSpecDataset(
        csv_path=str(DATASET_CSV),
        stage=DATASET_STAGE,
        smiles_col=gcfg["smiles_col"],
        inchi_col=gcfg["inchi_col"],
        spectrum_start=gcfg["spectrum_start"],
        spectrum_end=gcfg["spectrum_end"],
        global_cols=gcfg["global_cols"],
        stats_path=str(DATASET_STATS) if DATASET_STATS is not None else None,
        spectrum_fill_eps=gcfg["spectrum_fill_eps"],
        fixed_vocabs=gcfg["fixed_vocabs"],
        boolean_cols=gcfg["boolean_cols"],
        add_h=gcfg["add_h"],
    )
    print("dataset.boolean_cols",dataset.boolean_cols)
    print(dataset.fixed_vocabs)

    if len(dataset) == 0:
        print("[ERROR] Dataset length is 0. CSV / InChI / SMILES 컬럼을 확인하세요.")
        return

    spec_len = dataset.spec_len
    y_dim = dataset.y_dim
    global_dim = y_dim - spec_len

    print(f"[INFO] Dataset built. len={len(dataset)}")
    print(f"[INFO] spec_len={spec_len}, global_dim={global_dim}, y_dim={y_dim}")

    # ================== 디버그: CSVSpecDataset 통과 직후 y / spectrum / global dimension ==================
    # 첫 번째 샘플만 기준으로 확인
    g0 = dataset[0]  # PyG Data 객체
    y0 = g0.y.view(-1)  # (L,) 벡터로 펴기

    print("[DEBUG] single sample g0.y shape:", g0.y.shape)  # 보통 (1, L) 혹은 (L,)
    print("[DEBUG] length of y0:", y0.shape[0])  # L = spec_len + global_dim

    y_spec0 = y0[:spec_len]
    y_global0 = y0[spec_len:]

    print("[DEBUG] spectrum part (y_spec0) shape:", y_spec0.shape)  # (spec_len,)
    print("[DEBUG] global part   (y_global0) shape:", y_global0.shape)  # (global_dim,)
    print("====================================================================")

    # 원본 CSV (메타 정보용)
    df_src = pd.read_csv(DATASET_CSV).reset_index(drop=True)

    # -------------------------------------------------------
    # ③ Dataset에서 cond_y_batch + cond_df(메타정보) 생성
    # -------------------------------------------------------
    cond_rows: List[Dict[str, Any]] = []
    y_list: List[np.ndarray] = []

    spec_start_nm = gcfg["spectrum_start"]

    for i in range(len(dataset)):
        g = dataset[i]
        y_vec = g.y.view(-1).cpu().numpy().astype(float)  # (y_dim,)
        y_list.append(y_vec)

        y_spec = y_vec[:spec_len]

        # λmax 계산
        max_idx = int(np.argmax(y_spec))
        lambda_max_nm = spec_start_nm + max_idx

        # 전체 스펙트럼 리스트
        spec_list = y_spec.tolist()

        # 메타 정보: 원본 CSV의 같은 인덱스 행을 사용 (필요시 ID 매핑 로직으로 확장 가능)
        src_row = df_src.iloc[i]

        row_out: Dict[str, Any] = {
            "cond_id": i,
            "lambda_max_nm": float(lambda_max_nm),
            "spectrum_list": spec_list,
        }
        for col in ["pH_label", "DB", "ID", "type"]:
            if col in df_src.columns:
                row_out[col] = src_row[col]

        cond_rows.append(row_out)

    # cond_y_batch 텐서로 변환 (spec + global 전체)
    y_mat = np.stack(y_list, axis=0)  # (N, y_dim)
    cond_y_batch = torch.tensor(y_mat, dtype=torch.float32, device=device)
    print(f"[INFO] cond_y_batch shape = {cond_y_batch.shape}")

    # cond_df (메타 정보 부분)
    cond_df = pd.DataFrame(cond_rows)

    # (옵션) condY 전체(y_0..y_n)까지 포함한 CSV 저장 - 디버깅/기록용
    df_y = pd.DataFrame(y_mat, columns=[f"y_{j}" for j in range(y_dim)])
    condy_full = pd.concat([cond_df.copy(), df_y], axis=1)
    CONDY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    condy_full.to_csv(CONDY_CSV_PATH, index=False)
    print(f"[INFO] condY(from dataset) CSV saved to: {CONDY_CSV_PATH}")
    print(f"       y_dim = {y_dim} (spec_len={spec_len}, global_dim={global_dim})")

    # -------------------------------------------------------
    # ④ DiGress 모델 로드
    # -------------------------------------------------------
    model = load_trained_model(CKPT_PATH, device=device)

    # -------------------------------------------------------
    # ⑤ 조건부 DiGress 샘플링
    # -------------------------------------------------------
    molecule_list = sample_molecules_with_conditions_chunked(
        model=model,
        cond_y_batch=cond_y_batch,
        num_mols_per_condition=NUM_MOLS_PER_CONDITION,
        chunk_cond_size=16,
    )

    # -------------------------------------------------------
    # ⑥ SMILES + cond 정보 함께 저장
    # -------------------------------------------------------
    save_smiles_with_conditions(
        model=model,
        molecule_list=molecule_list,
        cond_df=cond_df,
        num_mols_per_condition=NUM_MOLS_PER_CONDITION,
        out_dir=OUT_DIR,
    )

    print("\n[Done] Molecule generation finished.")


if __name__ == "__main__":
    main()
