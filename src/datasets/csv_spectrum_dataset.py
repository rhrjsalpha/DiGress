# -*- coding: utf-8 -*-
# src/datasets/csv_spectrum_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit import RDLogger

# RDKit 경고 숨김
RDLogger.DisableLog('rdApp.*')

# --------------------------
# 기본 원자/결합 피처 정의
# --------------------------
ALLOWED_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]  # 필요시 확장
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

def split_y(batch_y: torch.Tensor, spec_len: int):
    # batch_y: (B, L)
    y_spec = batch_y[:, :spec_len]
    cond = batch_y[:, spec_len:] if batch_y.size(1) > spec_len else None
    return y_spec, cond

def one_hot(x, choices):
    v = [0] * len(choices)
    if x in choices:
        v[choices.index(x)] = 1
    return v


def atom_feature(atom: Chem.Atom) -> torch.Tensor:
    sym = atom.GetSymbol()
    return torch.tensor(one_hot(sym, ALLOWED_ATOMS), dtype=torch.float32)


def edge_feature(bond: Chem.Bond) -> torch.Tensor:
    bt = BOND_TYPES.get(bond.GetBondType(), 0)
    # one-hot(4) + is_conjugated + is_in_ring => 4 + 1 + 1 = 6
    oh = [0, 0, 0, 0]
    oh[bt] = 1
    feat = oh + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    return torch.tensor(feat, dtype=torch.float32)


def mol_from_row(row: pd.Series, smiles_col: Optional[str], inchi_col: Optional[str]) -> Optional[Chem.Mol]:
    """SMILES 우선, 실패 시 InChI로 시도."""
    mol = None
    if smiles_col and isinstance(row.get(smiles_col), str):
        mol = Chem.MolFromSmiles(row[smiles_col])
    if mol is None and inchi_col and isinstance(row.get(inchi_col), str):
        mol = Chem.MolFromInchi(row[inchi_col])
    if mol is not None:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            pass
    return mol


def build_graph(mol: Chem.Mol) -> Data:
    atoms = [atom_feature(a) for a in mol.GetAtoms()]
    x = torch.stack(atoms, dim=0) if len(atoms) > 0 else torch.zeros((0, len(ALLOWED_ATOMS)), dtype=torch.float32)

    rows, cols, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ef = edge_feature(b)
        rows += [i, j]
        cols += [j, i]
        eattr += [ef, ef]

    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.stack(eattr, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def infer_numeric(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna())
        return True
    except Exception:
        return False


class CSVSpecDataset(InMemoryDataset):
    """
    CSV → Graph + Spectrum(y) + Optional Global Condition(뒤에 concat)
    - y = [spec(200..800), global_cond...]
    - 스펙트럼 결측치(NaN/±inf/비수치)는 작은 값(spectrum_fill_eps)으로 채움.
    - 범주형 고정 vocab/불리언 컬럼을 사용자 지정 가능(fixed_vocabs, boolean_cols).
    """

    def __init__(
        self,
        csv_path: str,
        stage: str = "train",  # "train" | "val" | "test"
        smiles_col: Optional[str] = None,
        inchi_col: Optional[str] = "InChI",
        spectrum_start: int = 200,
        spectrum_end: int = 800,
        global_cols: Optional[List[str]] = None,  # 예) ["solvent_phase","is_qm","dielectric_constant_avg","pH_label"]
        stats_path: Optional[str] = None,
        transform=None,
        pre_transform=None,
        spectrum_fill_eps: float = 1e-8,  # 결측 채움용 작은 값
        fixed_vocabs: Optional[Dict[str, List[str]]] = None,  # 예) {"solvent_phase":["solid","liquid","gas"], "pH_label":["acidic","basic","neutral"]}
        boolean_cols: Optional[List[str]] = None,             # 예) ["is_qm"]
    ):
        self.csv_path = str(csv_path)
        self.stage = stage
        self.smiles_col = smiles_col
        self.inchi_col = inchi_col
        self.spec_cols = [str(i) for i in range(spectrum_start, spectrum_end + 1)]
        self.global_cols = global_cols or []
        self.stats_path = stats_path or (str(Path(csv_path).with_suffix("")) + "_stats.json")
        self.spectrum_fill_eps = spectrum_fill_eps
        # 사용자 지정 인코딩 설정
        self.fixed_vocabs: Dict[str, List[str]] = {k: [str(t) for t in v] for k, v in (fixed_vocabs or {}).items()}
        self.boolean_cols: List[str] = list(boolean_cols or [])

        # root는 CSV가 있는 폴더를 사용 → processed 파일을 같은 폴더 아래에 생성
        root = str(Path(self.csv_path).parent)
        self._ensure_stats()
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        # PyTorch 2.6: weights_only=False 명시
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def _ensure_stats(self) -> None:
        """stats.json이 없으면 stage와 상관없이 즉시 생성한다."""
        sp = Path(self.stats_path)
        if sp.exists():
            return
        df = pd.read_csv(self.csv_path)
        df = df.reset_index(drop=True)  # 행 위치 인덱스 안정화

        # 통계 산출 (불리언/고정 vocab 규칙 반영)
        stats = self._fit_stats(df)
        for col, vocab in self.fixed_vocabs.items():
            stats.setdefault("categorical", {})
            stats["categorical"][col] = {"vocab": list(vocab)}

        sp.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    # raw 체크 우회(원시 파일을 따로 요구하지 않도록 빈 리스트 반환)
    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{Path(self.csv_path).stem}_{self.stage}.pt"]

    def process(self) -> None:
        df = pd.read_csv(self.csv_path)

        # 스펙트럼 컬럼 존재 확인
        missing = [c for c in self.spec_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing spectrum columns ({len(missing)}): {missing[:5]} ...")

        # === 스펙트럼 결측/비정상 값 처리: NaN/±inf/비수치 → self.spectrum_fill_eps ===
        spec = df[self.spec_cols].apply(pd.to_numeric, errors="coerce")  # 비수치 → NaN
        spec = spec.replace([np.inf, -np.inf], np.nan)
        spec = spec.fillna(self.spectrum_fill_eps)  # 결측만 아주 작은 값으로 대체 (정상 0.0은 유지)
        df[self.spec_cols] = spec.astype(np.float32)

        spec_mat = df[self.spec_cols].to_numpy(dtype=np.float32)  # (N, 601)

        # 통계 산출/로드 (수치형: z-score, 범주형: one-hot)
        if self.stage == "train":
            stats = self._fit_stats(df)
            Path(self.stats_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            stats = json.loads(Path(self.stats_path).read_text())

        # 고정 vocab이 있다면(train/val/test 모두) stats에 강제 주입하여 순서를 보장
        for col, vocab in self.fixed_vocabs.items():
            stats.setdefault("categorical", {})
            stats["categorical"][col] = {"vocab": vocab}

        data_list: List[Data] = []
        for ridx, row in df.iterrows():
            mol = mol_from_row(row, self.smiles_col, self.inchi_col)
            if mol is None:
                continue
            g = build_graph(mol)

            # ✅ object → float32 문제 회피: 미리 만든 2D 매트릭스에서 꺼냄
            y_spec_np = spec_mat[ridx]
            # (안전) 만약 혹시라도 NaN/inf가 섞였으면 eps로 치환
            y_spec_np = np.nan_to_num(y_spec_np, nan=self.spectrum_fill_eps,
                                      posinf=self.spectrum_fill_eps, neginf=self.spectrum_fill_eps)
            y_spec = torch.from_numpy(y_spec_np.copy())  # contiguous 보장용 copy()

            y_globals = self._encode_globals(row, stats)
            y = torch.cat([y_spec, y_globals], dim=0) if y_globals.numel() > 0 else y_spec

            # g.y = y
            # y를 **1D 벡터 (L,)**로 저장하면, PyG는 배치 시 dim=0으로 그냥 이어 붙임
            # 학습 코드에서는 (B, L)을 기대하고 batch_y[:, :spec_len]처럼 2D 인덱싱을 하니 에러(too many indices for tensor of dimension 1)
            g.y = y.unsqueeze(0) # 1D -> 2D

            # (옵션) sample id
            for key in ("ID", "id", "Name", "name"):
                if key in row and isinstance(row[key], str):
                    g.sample_id = row[key]
                    break

            data_list.append(g)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # -------- helpers --------
    def _fit_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"numeric": {}, "categorical": {}, "boolean": sorted(self.boolean_cols)}
        for col in self.global_cols:
            if col not in df.columns:
                continue
            # 불리언 강제 컬럼은 통계 수집 생략(0/1 직접 인코딩)
            if col in self.boolean_cols:
                continue
            # 고정 vocab이 지정된 범주형
            if col in self.fixed_vocabs:
                stats["categorical"][col] = {"vocab": list(self.fixed_vocabs[col])}
                continue

            s = df[col]
            if infer_numeric(s):
                v = pd.to_numeric(s, errors="coerce")
                m = float(np.nanmean(v))
                sd = float(np.nanstd(v) + 1e-12)
                stats["numeric"][col] = {"mean": m, "std": sd}
            else:
                cats = sorted([str(x) for x in s.dropna().unique().tolist()])
                stats["categorical"][col] = {"vocab": cats}
        return stats

    def _encode_globals(self, row: pd.Series, stats: Dict[str, Any]) -> torch.Tensor:
        outs: List[float] = []
        for col in self.global_cols:
            if col not in row.index:
                continue
            val = row[col]

            # 1) 불리언 컬럼 0/1 인코딩
            if col in self.boolean_cols:
                outs.append(self._to_bool01(val))
                continue

            # 2) 수치형 z-score
            if "numeric" in stats and col in stats["numeric"]:
                m = stats["numeric"][col]["mean"]
                sd = stats["numeric"][col]["std"]
                try:
                    x = float(val)
                except Exception:
                    x = m
                if abs(sd) < 1e-12:
                    outs.append(0.0)
                else:
                    outs.append((x - m) / sd)
                continue

            # 3) 범주형 one-hot (고정 vocab 포함)
            if "categorical" in stats and col in stats["categorical"]:
                vocab = stats["categorical"][col]["vocab"]
                token = str(val) if pd.notna(val) else ""
                outs += [1.0 if token == v else 0.0 for v in vocab]
                continue

            # 4) 기타 fallback
            if isinstance(val, (bool, np.bool_)):
                outs.append(1.0 if bool(val) else 0.0)
            elif isinstance(val, str):
                outs += [1.0, 0.0] if val.lower() in ("true", "yes", "y", "1") else [0.0, 1.0]
            else:
                try:
                    outs.append(float(val))
                except Exception:
                    outs.append(0.0)

        return torch.tensor(outs, dtype=torch.float32)

    @staticmethod
    def _to_bool01(v) -> float:
        if pd.isna(v):
            return 0.0
        if isinstance(v, (bool, np.bool_)):
            return 1.0 if bool(v) else 0.0
        # 숫자: 0/1 or 0.0/1.0 등
        try:
            fv = float(v)
            return 1.0 if fv >= 0.5 else 0.0
        except Exception:
            pass
        # 문자열: 다양한 진리값 처리
        s = str(v).strip().lower()
        if s in {"1", "t", "true", "y", "yes"}:
            return 1.0
        if s in {"0", "f", "false", "n", "no"}:
            return 0.0
        return 0.0


# ==== PyCharm에서 바로 실행 가능한 러너 ====
def main():
    import argparse
    from pathlib import Path
    import torch
    from torch_geometric.loader import DataLoader

    # ===== 코드 내부 고정 설정 =====
    GLOBAL_COLS   = ["solvent_phase", "is_qm", "dielectric_constant_avg", "pH_label"]
    FIXED_VOCABS  = {
        "solvent_phase": ["solid", "liquid", "gas"],
        "pH_label": ["acidic", "basic", "neutral"],
    }
    BOOLEAN_COLS  = ["is_qm"]  # 0/1로 직접 인코딩

    # ---- helpers ----
    def find_default_csv() -> Path:
        root_candidates = [
            Path("./data/csv/EM_stratified_train_clustered_resplit_with_mu_eps.csv"),
            Path("./data/csv/QM_EM_ABS_stratified_train_resplit_with_mu_eps.csv"),
            Path("./data/csv/EM_stratified_test_clustered_resplit_with_mu_eps.csv"),
            Path("./data/csv/QM_EM_ABS_stratified_test_resplit_with_mu_eps.csv"),
        ]
        for c in root_candidates:
            if c.exists():
                return c.resolve()
        alt_candidates = [
            Path("../../data/csv/EM_stratified_train_clustered_resplit_with_mu_eps.csv"),
            Path("../../data/csv/QM_EM_ABS_stratified_train_resplit_with_mu_eps.csv"),
            Path("../../data/csv/EM_stratified_test_clustered_resplit_with_mu_eps.csv"),
            Path("../../data/csv/QM_EM_ABS_stratified_test_resplit_with_mu_eps.csv"),
        ]
        for c in alt_candidates:
            if c.exists():
                return c.resolve()
        for base in (Path("./data/csv"), Path("../../data/csv")):
            if base.exists():
                any_csv = next(base.glob("*.csv"), None)
                if any_csv:
                    return any_csv.resolve()
        raise FileNotFoundError("CSV를 찾지 못했습니다. ./data/csv 폴더에 csv를 두거나 --csv로 지정하세요.")

    def build_args():
        p = argparse.ArgumentParser(description="CSV → Graph + Spectrum(y) 드라이런 (코드내 전역설정 고정)", add_help=True)
        # 전역 피처/보캡/불리언은 코드 내에서 고정하므로 CLI 인자에서 제거
        p.add_argument("--csv", default=None)
        p.add_argument("--stage", choices=["train", "val", "test"], default="train",
                       help="미지정 시 파일명에 test가 들어가면 test, 아니면 train")
        p.add_argument("--inchi-col", default="InChI")
        p.add_argument("--smiles-col", default=None)
        p.add_argument("--spec-start", type=int, default=200)
        p.add_argument("--spec-end", type=int, default=800)
        p.add_argument("--stats", default=None,
                       help="train에서 생성된 *_stats.json 경로. 미지정 시 <csv_stem>_stats.json")
        p.add_argument("--spec-fill-eps", type=float, default=1e-8,
                       help="스펙트럼 결측치(NaN/±inf/비수치)를 채울 작은 값")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--show-n", type=int, default=10, help="데이터셋에서 앞 N개 샘플 상세 출력")
        p.add_argument("--show-batches", type=int, default=0, help="앞쪽 배치 K개 요약 출력")
        return p.parse_args()

    args = build_args()
    print(args)
    # CSV 자동 결정
    csv_path = Path(args.csv).resolve() if args.csv else find_default_csv()
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    # stage 자동 결정
    stage = args.stage if args.stage is not None else ("test" if "test" in csv_path.name.lower() else "train")

    # stats 경로 기본값
    stats_path = args.stats or (str(csv_path.with_suffix("")) + "_stats.json")

    print("== CONFIG ==")
    print(f"csv            : {csv_path}")
    print(f"stage          : {stage}")
    print(f"spectrum range : {args.spec_start}..{args.spec_end}  (len={args.spec_end-args.spec_start+1})")
    print(f"globals (fixed): {GLOBAL_COLS}")
    print(f"solvent vocab  : {FIXED_VOCABS['solvent_phase']}")
    print(f"pH vocab       : {FIXED_VOCABS['pH_label']}")
    print(f"boolean cols   : {BOOLEAN_COLS}")
    print(f"spec_fill_eps  : {args.spec_fill_eps}")
    print(f"stats_path     : {stats_path}")
    print()

    # ---- Dataset 생성 (처음이면 processed/*.pt 생성) ----
    ds = CSVSpecDataset(
        csv_path=str(csv_path),
        stage=stage,
        smiles_col=args.smiles_col,
        inchi_col=args.inchi_col,
        spectrum_start=args.spec_start,
        spectrum_end=args.spec_end,
        global_cols=GLOBAL_COLS,           # 코드 내 고정
        stats_path=stats_path,
        spectrum_fill_eps=args.spec_fill_eps,
        fixed_vocabs=FIXED_VOCABS,         # 코드 내 고정
        boolean_cols=BOOLEAN_COLS,         # 코드 내 고정
    )
    print(f"[OK] Dataset built. len={len(ds)}")
    processed_pt = Path(ds.processed_paths[0])
    print(f"[INFO] processed cache: {processed_pt}")
    print()

    if len(ds) == 0:
        print("[WARN] 데이터셋 길이가 0입니다. InChI/SMILES 파싱 실패 혹은 필터링 문제를 확인하세요.")
        raise SystemExit(0)

    spec_len = args.spec_end - args.spec_start + 1

    # ---- N개 샘플 상세 출력 ----
    to_show = min(max(args.show_n, 0), len(ds))
    if to_show > 0:
        print(f"== FIRST {to_show} SAMPLES ==")
        for idx in range(to_show):
            d = ds[idx]
            y_dim = int(d.y.numel())
            g_len = y_dim - spec_len
            print(f"[{idx}] sample_id={getattr(d, 'sample_id', '')}")
            print(f"  x.shape={None if d.x is None else tuple(d.x.shape)} "
                  f"edge_index={tuple(d.edge_index.shape)} "
                  f"edge_attr={None if d.edge_attr is None else tuple(d.edge_attr.shape)} "
                  f"y.shape={tuple(d.y.shape)} (spec={spec_len}, globals={g_len})")
            spec_head = d.y[:min(5, spec_len)].tolist()
            spec_tail = d.y[max(0, spec_len - 5):spec_len].tolist()
            print(f"  y.spec head={spec_head} ... tail={spec_tail}")
            if g_len > 0:
                g_head = d.y[spec_len:spec_len + min(10, g_len)].tolist()
                print(f"  y.globals head={g_head}")

            d = ds[idx]
            y_vec = d.y.view(-1)  # (L,)
            spec_len = args.spec_end - args.spec_start + 1
            g_len = y_vec.numel() - spec_len
            print(f"y.shape={tuple(d.y.shape)}  (stored as (1, L))")
            print(f"(spec={spec_len}, globals={g_len})")

            spec_head = y_vec[:min(5, spec_len)].tolist()
            spec_tail = y_vec[max(0, spec_len - 5):spec_len].tolist()
            print(f"  y.spec head={spec_head} ... tail={spec_tail}")
            if g_len > 0:
                g_head = y_vec[spec_len:spec_len + min(10, g_len)].tolist()
                print(f"  y.globals head={g_head}")

    # ---- 한 배치 로딩 & 요약 ----
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(dl))
    print("\n== BATCH ==")
    print(f"batch.x          : {None if batch.x is None else tuple(batch.x.shape)}")
    print(f"batch.edge_index : {tuple(batch.edge_index.shape)}")
    print(f"batch.edge_attr  : {None if batch.edge_attr is None else tuple(batch.edge_attr.shape)}")
    print(f"batch.y          : {tuple(batch.y.shape)}")
    print(f"batch.batch(shape): {tuple(batch.batch.shape)}")

    # ---- K개 배치 요약 ----
    if args.show_batches and args.show_batches > 0:
        from itertools import islice
        print(f"\n== FIRST {args.show_batches} BATCHES ==")
        for b_idx, b in enumerate(islice(DataLoader(ds, batch_size=args.batch_size, shuffle=False), args.show_batches)):
            print(f"[batch {b_idx}] "
                  f"x={None if b.x is None else tuple(b.x.shape)} "
                  f"edge_index={tuple(b.edge_index.shape)} "
                  f"edge_attr={None if b.edge_attr is None else tuple(b.edge_attr.shape)} "
                  f"y={tuple(b.y.shape)}")

    # ---- GPU 테스트(옵션) ----
    if torch.cuda.is_available():
        _ = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.__dict__.items()}
        print("\n[CUDA] 첫 배치를 GPU로 이동 테스트 완료.")

    print("\n[Done] CSV → Graph + Spectrum 파이프라인 드라이런 완료.")

if __name__ == "__main__":
    main()
