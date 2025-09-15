# src/analysis/gen_metrics.py
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import csv, os, math

def _morgan_fp(smiles: str, radius=2, nBits=2048):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits)

def snn_avg(gen_smiles: list[str], ref_smiles: list[str]) -> float:
    """Similarity to Nearest Neighbor: 각 생성 분자의 max Tanimoto 평균"""
    ref_fps = [_morgan_fp(s) for s in (ref_smiles or [])]
    ref_fps = [fp for fp in ref_fps if fp is not None]
    if not ref_fps:
        return float("nan")
    sims = []
    for s in gen_smiles or []:
        fp = _morgan_fp(s)
        if fp is None:
            continue
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)))
    return float(np.mean(sims)) if sims else float("nan")

def _murcko(s: str) -> str | None:
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)

def scaffold_jaccard_and_recovery(gen_smiles: list[str], ref_smiles: list[str]) -> tuple[float,float]:
    """Jaccard = |G∩R|/|G∪R|, Recovery = |G∩R|/|G|"""
    G = {sc for sc in (_murcko(s) for s in (gen_smiles or [])) if sc}
    R = {sc for sc in (_murcko(s) for s in (ref_smiles or [])) if sc}
    if not G:
        return float("nan"), float("nan")
    inter = len(G & R); union = len(G | R)
    jaccard = inter/union if union else 1.0
    recovery = inter/len(G)
    return float(jaccard), float(recovery)

def kl_divergence(p, q, eps=1e-12) -> float:
    """히스토그램 분포 기반 KL(P||Q), p/q는 배열형"""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps); q = q / (q.sum() + eps)
    return float((p * (np.log(p + eps) - np.log(q + eps))).sum())

def sa_score(smiles: str) -> float | None:
    """Ertl SA가 있으면 사용, 없으면 QED로 1~10 근삿값."""
    try:
        # 원하는 위치에 sascorer.py를 두면 사용됨 (예: src/analysis/third_party/sascorer.py)
        from analysis.third_party import sascorer
        m = Chem.MolFromSmiles(smiles)
        return float(sascorer.calculateScore(m)) if m is not None else None
    except Exception:
        try:
            from rdkit.Chem import QED
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return None
            q = float(QED.qed(m))  # 0..1
            return float(max(1.0, min(10.0, 10.0 - 9.0*q)))
        except Exception:
            return None

def compute_generation_metrics(gen_smiles: list[str],
                               ref_smiles: list[str] | None = None,
                               histos: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """생성 분자 리스트와 레퍼런스(보통 train set), (선택)히스토 쌍(p, q)을 받아 추가 지표 계산."""
    d = {}
    d["SNN"] = snn_avg(gen_smiles, ref_smiles or [])
    j, rec = scaffold_jaccard_and_recovery(gen_smiles, ref_smiles or [])
    d["scaffold_jaccard"] = j
    d["scaffold_recovery"] = rec

    sa_vals = [sa_score(s) for s in (gen_smiles or [])]
    sa_vals = [v for v in sa_vals if v is not None and not math.isnan(v)]
    if sa_vals:
        arr = np.array(sa_vals, dtype=float)
        d.update({
            "SA_mean": float(arr.mean()),
            "SA_median": float(np.median(arr)),
            "SA_q25": float(np.percentile(arr, 25)),
            "SA_q75": float(np.percentile(arr, 75)),
        })
    else:
        d.update({"SA_mean": float("nan"), "SA_median": float("nan"), "SA_q25": float("nan"), "SA_q75": float("nan")})

    if histos:
        for k, (p, q) in histos.items():
            d[f"KL_{k}"] = kl_divergence(p, q)
    return d

def write_metrics_csv(out_csv: str, base_row: dict, extra_metrics: dict) -> str:
    """한 줄 append 저장(헤더 자동). base_row + extra_metrics 병합."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    row = {**base_row, **extra_metrics}
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    return out_csv
