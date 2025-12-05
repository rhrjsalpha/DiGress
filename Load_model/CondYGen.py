# generate_from_condY_yaml_pipeline.py
# -*- coding: utf-8 -*-
"""
1) configs/dataset/csv_spec.yaml 을 읽어서
   - Gaussian 스펙트럼 + global 조건으로 이루어진 condY 벡터들을 만들고
   - condY_manual_from_yaml.csv 로 저장한다.

2) 방금 만든 condY_manual_from_yaml.csv 를 읽어서
   - y_0 ~ y_{y_dim-1} 를 cond_y_batch 텐서로 만들고
   - 학습된 DiGress(DiscreteDenoisingDiffusion) 모델에 넣어
     조건부로 분자를 생성하고 SMILES/PNG 로 저장한다.

※ stats.json 은 사용하지 않으며,
   numeric 값들은 YAML 규칙에 따라 raw 값 그대로 사용한다.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# ------------------------------------------------------------------
# 0. 프로젝트 루트 / sys.path / ckpt alias 설정
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Load_model 상위가 프로젝트 루트라고 가정
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[DEBUG] PROJECT_ROOT added to sys.path:", PROJECT_ROOT)

import os
import numpy as np
import pandas as pd
import yaml
import torch
from rdkit import Chem
from rdkit.Chem import Draw

# === DiGress 관련 모듈 ===
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
import src.analysis
import src.analysis.visualization
import src.diffusion

# Lightning ckpt 가 예전에 'analysis', 'diffusion' 이라는 모듈 경로를 썼으므로
# 현재 src.analysis / src.diffusion 을 alias 로 등록
sys.modules["analysis"] = src.analysis
sys.modules["diffusion"] = src.diffusion


# ------------------------------------------------------------------
# 1. 경로 / 설정
# ------------------------------------------------------------------
# (1) 학습된 DiGress 체크포인트(.ckpt) 경로
CKPT_PATH = r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\last.ckpt"

# (2) YAML 설정 경로 (csv_spec.yaml)
YAML_PATH = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\configs\dataset\csv_spec.yaml")

# (3) condY DB CSV 저장 위치
CONDY_CSV_PATH = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\condY_manual_from_yaml.csv")

# (4) 생성된 분자 출력 폴더
OUT_DIR = Path(r"C:\Users\analcheminfo\PycharmProjects\DiGress\Load_model\generated_from_condY")

# (5) 각 조건당 몇 개 분자를 생성할지
NUM_MOLS_PER_CONDITION = 1


# ------------------------------------------------------------------
# 2. YAML 로드 및 인코딩 유틸
# ------------------------------------------------------------------
def load_dataset_cfg(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_gaussian_spectrum(
    center_nm: float,
    spectrum_start: int,
    spectrum_end: int,
    sigma_nm: float = 20.0,
    height: float = 1.0,
) -> np.ndarray:
    """
    [spectrum_start, spectrum_end] 구간에서 center_nm 중심의 Gaussian 생성.
    return: shape = (spec_len,)
    """
    spec_len = spectrum_end - spectrum_start + 1
    nm_grid = np.linspace(spectrum_start, spectrum_end, spec_len, dtype=np.float32)
    x = (nm_grid - float(center_nm)) / float(sigma_nm)
    gauss = np.exp(-0.5 * x * x)
    if gauss.max() > 0:
        gauss = gauss / gauss.max() * float(height)
    return gauss.astype(np.float32)


def encode_globals_from_yaml(
    cfg: Dict[str, Any],
    cond: Dict[str, Any],
) -> np.ndarray:
    """
    YAML 의 global_cols / fixed_vocabs / boolean_cols / numeric_cols 규칙에 따라
    cond dict 의 값들을 하나의 글로벌 벡터로 인코딩.
    """
    global_cols   = cfg.get("global_cols", []) or []
    fixed_vocabs  = cfg.get("fixed_vocabs", {}) or {}
    boolean_cols  = cfg.get("boolean_cols", []) or []
    numeric_cols  = cfg.get("numeric_cols", []) or []

    outs: List[float] = []

    for col in global_cols:
        if col not in cond:
            raise KeyError(f"[encode_globals_from_yaml] 조건 dict 에 '{col}' 값이 없습니다.")

        val = cond[col]

        # 1) boolean
        if col in boolean_cols:
            outs.append(1.0 if bool(val) else 0.0)
            continue

        # 2) categorical (fixed vocab → one-hot)
        if col in fixed_vocabs:
            vocab = fixed_vocabs[col]
            for v in vocab:
                outs.append(1.0 if str(val) == str(v) else 0.0)
            continue

        # 3) numeric
        if col in numeric_cols:
            outs.append(float(val))
            continue

        # 4) 기타 (숫자로 캐스팅 시도)
        try:
            outs.append(float(val))
        except Exception:
            print(f"[WARN] 컬럼 '{col}' 값 '{val}' 을 float 로 바꿀 수 없어 0.0 으로 대체합니다.")
            outs.append(0.0)

    return np.array(outs, dtype=np.float32)


def make_CONDITIONS(
    center_nm_list: List[float],
    sigma_nm_list: List[float],
    pH_label,
    dielectric_constant_avg_list: List[float],
    is_qm,
    type,
) -> List[Dict[str, Any]]:
    """
    center_nm, sigma_nm, dielectric_constant_avg 조합으로 조건 dict 리스트 생성.
    나머지 pH_label, is_qm, type 은 고정값 사용.
    """
    lists = []
    cond_id = 0
    for center_nm in center_nm_list:
        for sigma_nm in sigma_nm_list:
            for dielectric_constant_avg in dielectric_constant_avg_list:
                dicts: Dict[str, Any] = {}
                cond_id += 1
                dicts["cond_id"] = cond_id
                dicts["center_nm"] = center_nm
                dicts["sigma_nm"] = sigma_nm
                dicts["height"] = 1.0
                dicts["pH_label"] = pH_label
                dicts["dielectric_constant_avg"] = dielectric_constant_avg
                dicts["is_qm"] = is_qm
                dicts["type"] = type
                lists.append(dicts)
    return lists


# ------------------------------------------------------------------
# 3. 1단계: YAML 기반으로 condY DB(CSV) 생성
# ------------------------------------------------------------------
def build_condY_csv_from_yaml(
    yaml_path: Path,
    out_csv: Path,
    CONDITIONS: List[Dict[str, Any]],
) -> int:
    """
    csv_spec.yaml 과 CONDITIONS 를 이용하여
    - Gaussian 스펙트럼 + global 인코딩 → y_0..y_{y_dim-1}
    - cond_id, center_nm, sigma_nm, height, global raw 값 포함
    형태의 DataFrame 을 만들고 out_csv 로 저장.

    return: y_dim (스펙트럼+글로벌 전체 길이)
    """
    cfg = load_dataset_cfg(yaml_path)

    spectrum_start = int(cfg.get("spectrum_start", 200))
    spectrum_end   = int(cfg.get("spectrum_end",   800))
    spec_len = spectrum_end - spectrum_start + 1

    global_cols = cfg.get("global_cols", []) or []

    print("=== YAML 기반 설정 ===")
    print(f"spectrum_start..end : {spectrum_start}..{spectrum_end} (spec_len={spec_len})")
    print(f"global_cols         : {global_cols}")
    print(f"fixed_vocabs        : {cfg.get('fixed_vocabs', {}) or {}}")
    print(f"boolean_cols        : {cfg.get('boolean_cols', []) or []}")
    print(f"numeric_cols        : {cfg.get('numeric_cols', []) or []}")
    print()

    rows: List[Dict[str, Any]] = []
    y_dim = None

    for cond in CONDITIONS:
        cond_id   = cond.get("cond_id", "")
        center_nm = float(cond["center_nm"])
        sigma_nm  = float(cond.get("sigma_nm", 20.0))
        height    = float(cond.get("height", 1.0))

        # 1) spectrum part
        spec = make_gaussian_spectrum(
            center_nm=center_nm,
            spectrum_start=spectrum_start,
            spectrum_end=spectrum_end,
            sigma_nm=sigma_nm,
            height=height,
        )  # (spec_len,)

        # 2) global part
        g = encode_globals_from_yaml(cfg, cond)  # (global_dim,)

        # 3) concat
        y_vec = np.concatenate([spec, g], axis=0)  # (y_dim,)

        if y_dim is None:
            y_dim = y_vec.shape[0]
        elif y_dim != y_vec.shape[0]:
            raise ValueError(f"y_dim mismatch: {y_dim} vs {y_vec.shape[0]}")

        # CSV 한 row
        row: Dict[str, Any] = {
            "cond_id": cond_id,
            "center_nm": center_nm,
            "sigma_nm": sigma_nm,
            "height": height,
        }
        # raw global 값도 같이 저장해 두기
        for col in global_cols:
            row[col] = cond[col]

        # y_0 ~ y_{y_dim-1}
        for i in range(y_dim):
            row[f"y_{i}"] = float(y_vec[i])

        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    global_dim = y_dim - spec_len
    print(f"[OK] {len(df)}개의 condY 를 저장했습니다 → {out_csv.resolve()}")
    print(f"     y_dim = {y_dim} (spec_len={spec_len}, global_dim={global_dim})")

    return y_dim


# ------------------------------------------------------------------
# 4. 2단계: condY CSV 읽어서 텐서(batch)로 변환
# ------------------------------------------------------------------
def load_condY_batch_from_csv(
    csv_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """
    condY_manual_from_yaml.csv 에서 y_0..y_n 컬럼만 골라서
    (num_conds, y_dim) 텐서로 변환.
    """
    df = pd.read_csv(csv_path)
    y_cols = [c for c in df.columns if c.startswith("y_")]
    y_cols = sorted(y_cols, key=lambda x: int(x.split("_")[1]))  # y_0, y_1, ... 순서 정렬

    y = torch.tensor(df[y_cols].values, dtype=torch.float32, device=device)
    print(f"[INFO] cond_y_batch loaded from CSV: shape = {y.shape}")
    return y


# ------------------------------------------------------------------
# 5. DiGress 모델 로드 & 샘플링 & 그림/SMILES 저장
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
        visualization_tools=None,
        extra_features=None,
        domain_features=None,
        strict=False,  # sampling_metrics.* 같은 불필요 키 무시
    )
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")
    print(f"[INFO] diffusion steps (T): {model.T}")
    return model


def sample_molecules_with_conditions(
    model: DiscreteDenoisingDiffusion,
    cond_y_batch: torch.Tensor,
    num_mols_per_condition: int = 1,
):
    """
    cond_y_batch (B, y_dim)을 받아 DiGress의 sample_batch로 분자 샘플링.
    각 조건당 num_mols_per_condition 개씩 생성.
    """
    device = next(model.parameters()).device
    B = cond_y_batch.size(0)

    cond_y_expanded = cond_y_batch.repeat_interleave(num_mols_per_condition, dim=0)
    batch_size = cond_y_expanded.size(0)

    number_chain_steps = int(model.T - 1)

    with torch.no_grad():
        molecule_list = model.sample_batch(
            batch_id=0,
            batch_size=batch_size,
            keep_chain=0,
            number_chain_steps=number_chain_steps,
            save_final=batch_size,
            num_nodes=None,
            cond_y_base=cond_y_expanded,
        )

    print(f"[INFO] Sampled {len(molecule_list)} graphs from model.")
    return molecule_list


def graphs_to_smiles_and_png(
    model: DiscreteDenoisingDiffusion,
    molecule_list,
    out_dir: Path,
):
    """
    sample_batch로 얻은 molecule_list ([(atom_types, edge_types), ...]) 를
    RDKit Mol 및 SMILES 로 변환하고, PNG 로 저장.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = model.visualization_tools
    if vis is None:
        raise RuntimeError(
            "model.visualization_tools 가 None 입니다. "
            "training 때 MolecularVisualization 을 넘겨줬는지 확인하세요."
        )

    smiles_list = []
    for idx, (atom_types, edge_types) in enumerate(molecule_list):
        atom_types_np = atom_types.numpy()
        edge_types_np = edge_types.numpy()

        mol = vis.mol_from_graphs(atom_types_np, edge_types_np)
        if mol is None:
            print(f"[WARN] idx={idx} 에서 Mol 생성 실패, 건너뜀.")
            continue

        smi = Chem.MolToSmiles(mol)
        smiles_list.append(smi)

        png_path = out_dir / f"mol_{idx:03d}.png"
        try:
            Draw.MolToFile(mol, str(png_path))
        except Exception as e:
            print(f"[WARN] MolToFile 실패 (idx={idx}): {e}")

    # SMILES 텍스트 저장
    txt_path = out_dir / "generated_smiles.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    print(f"[INFO] Saved {len(smiles_list)} SMILES to {txt_path}")
    print(f"[INFO] PNG images saved under {out_dir.resolve()}")


# ------------------------------------------------------------------
# 6. main: 전체 파이프라인 실행
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # (1) condY 를 만들 조건들 정의
    #     필요에 따라 center_nm_list / sigma_nm_list / dielectric_constant 리스트 수정 가능
    center_nm_list = [350.0, 400.0, 450.0]  # 피크 위치 후보
    sigma_nm_list  = [15.0, 20.0, 25.0]     # 폭 후보

    base_pH_label            = "neutral"
    base_dielectric_constant = [1.0, 80.1]  # 예: 진공/물 등
    base_is_qm               = 1
    base_type_label          = "EM"

    CONDITIONS = make_CONDITIONS(
        center_nm_list=center_nm_list,
        sigma_nm_list=sigma_nm_list,
        pH_label=base_pH_label,
        dielectric_constant_avg_list=base_dielectric_constant,
        is_qm=base_is_qm,
        type=base_type_label,
    )

    # (2) YAML 규칙대로 condY DB를 만들고 CSV로 저장
    _ = build_condY_csv_from_yaml(
        yaml_path=YAML_PATH,
        out_csv=CONDY_CSV_PATH,
        CONDITIONS=CONDITIONS,
    )

    # (3) 방금 만든 condY CSV에서 cond_y_batch 텐서 읽기
    cond_y_batch = load_condY_batch_from_csv(
        csv_path=CONDY_CSV_PATH,
        device=device,
    )

    # (4) DiGress 모델 로드
    model = load_trained_model(CKPT_PATH, device=device)

    # (5) 조건을 넣어 분자 샘플링
    molecule_list = sample_molecules_with_conditions(
        model=model,
        cond_y_batch=cond_y_batch,
        num_mols_per_condition=NUM_MOLS_PER_CONDITION,
    )

    # (6) SMILES 및 PNG 저장
    graphs_to_smiles_and_png(
        model=model,
        molecule_list=molecule_list,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
