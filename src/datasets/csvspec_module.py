# src/datasets/csvspec_module.py
from __future__ import annotations

import os
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from omegaconf import OmegaConf
# RDKit 경고 숨김
RDLogger.DisableLog('rdApp.*')

# --------------------------
# 기본 원자/결합 피처 정의
# --------------------------
ALLOWED_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"]  # 필요시 확장

### atom 정의 되지 않은 것으로 오류 날시 UNK 사용해 보기 ###
UNK_TOKEN = None
# UNK_TOKEN = "<UNK>"
# ATOM_VOCAB = ALLOWED_ATOMS + [UNK_TOKEN]
# ATOM2IDX = {sym: i for i, sym in enumerate(ATOM_VOCAB)}

ATOM_VOCAB = ALLOWED_ATOMS[:]  # UNK 제거
ATOM2IDX = {sym: i for i, sym in enumerate(ATOM_VOCAB)}

BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

# 금지할 원소 기본값 (필요시 수정/확장 가능)
FORBIDDEN_ATOMS_DEFAULT = {"As", "Si"}

def has_forbidden_atoms(mol: Chem.Mol, forbidden: set[str]) -> bool:
    """분자에 금지 원소가 하나라도 있으면 True."""
    if mol is None:
        return True  # 파싱 실패도 학습 대상에서 제외
    for a in mol.GetAtoms():
        if a.GetSymbol() in forbidden:
            return True
    return False

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
    try:
        if not UNK_TOKEN == None:
            idx = ATOM2IDX.get(sym, ATOM2IDX[UNK_TOKEN])  # vocab 밖이면 UNK로
        else:
            idx = ATOM2IDX[sym]
    except KeyError or NameError:
        if sym not in ATOM2IDX:
            raise KeyError(f"OOV atom symbol: {sym}")
        idx = ATOM2IDX[sym]

    v = torch.zeros(len(ATOM_VOCAB), dtype=torch.float32)
    v[idx] = 1.0
    return v


def edge_feature(bond: Chem.Bond) -> torch.Tensor:
    bt = BOND_TYPES.get(bond.GetBondType(), 0)
    # one-hot(4) + is_conjugated + is_in_ring => 4 + 1 + 1 = 6
    oh = [0, 0, 0, 0]
    oh[bt] = 1
    feat = oh + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    return torch.tensor(feat, dtype=torch.float32)


def mol_from_row(row, smiles_col: Optional[str], inchi_col: Optional[str], add_h: bool = False) -> Optional[Chem.Mol]:
    """SMILES 우선, 실패 시 InChI. add_h=True면 수소 추가."""
    mol = None
    if smiles_col and isinstance(row.get(smiles_col), str):
        mol = Chem.MolFromSmiles(row[smiles_col])
    if mol is None and inchi_col and isinstance(row.get(inchi_col), str):
        mol = Chem.MolFromInchi(row[inchi_col])

    try:
        if add_h:
            mol = Chem.AddHs(mol)
        else:
            # SMILES에 있던 명시적 H까지 제거(암시적 H는 그대로 유지)
            mol = Chem.RemoveHs(mol)
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

def _get_oov_atoms(mol) -> tuple[list[int], list[str]]:
    unk_idx, unk_syms = [], set()
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        # ATOM2IDX는 ["H","C","N","O","F","P","S","Cl","Br","I","<UNK>"]만 키로 가짐
        if s not in ATOM2IDX:  # vocab 밖 → UNK 대상
            unk_idx.append(a.GetIdx())
            unk_syms.add(s)
    return unk_idx, sorted(unk_syms)

def _save_unk_viz(mol, out_dir: Path, ridx: int, smiles: str, unk_syms: list[str], unk_idx: list[int]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 파일명은 행번호 중심으로 안전하게
    fname = f"{ridx:06d}.png"
    legend = f"UNK={','.join(unk_syms)} | row={ridx}"
    Draw.MolToFile(mol, str(out_dir / fname), size=(500, 400), highlightAtoms=unk_idx, legend=legend)
    return fname

def _canon_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def _canon_cat(x):
    s = _canon_str(x).lower()  # 소문자 + 공백제거
    # pH_label 표준화
    if s in {"neutral", "neu"}:
        s = "neutral"
    elif s in {"basic", "base", "alkaline"}:
        s = "basic"
    elif s in {"acidic", "acid"}:
        s = "acidic"
    # solvent_phase 표준화
    if s in {"g", "gas"}:
        s = "gas"
    elif s in {"l", "liq", "liquid"}:
        s = "liquid"
    elif s in {"s", "sol", "solid"}:
        s = "solid"
    elif s in {"qm", "quantum", "calc"}:
        s = "QM"
    return s

class CSVSpecDataset_for_Diffusion(InMemoryDataset):
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
        add_h: bool = False,
        forbidden_atoms: Optional[List[str]] = None,
        unk_vis_dir: Optional[str] = None,
    ):
        self.unk_vis_dir = str(unk_vis_dir) if unk_vis_dir else None
        self.csv_path = str(csv_path)
        self.stage = stage
        self.smiles_col = smiles_col
        self.inchi_col = inchi_col
        self.spec_cols = [str(i) for i in range(spectrum_start, spectrum_end + 1)]
        self.global_cols = global_cols or []
        self.stats_path = stats_path or (str(Path(csv_path).with_suffix("")) + "_stats.json")
        self.spectrum_fill_eps = spectrum_fill_eps
        self.add_h = bool(add_h)
        # 사용자 지정 인코딩 설정
        self.fixed_vocabs: Dict[str, List[str]] = {k: [str(t) for t in v] for k, v in (fixed_vocabs or {}).items()}
        self.boolean_cols: List[str] = list(boolean_cols or [])

        # 금지 원소 설정(기본: As, Si)
        if forbidden_atoms is None:
            self.forbidden_atoms = set(FORBIDDEN_ATOMS_DEFAULT)
        else:
            self.forbidden_atoms = set(str(x) for x in forbidden_atoms)

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

    @property
    def y_dim(self) -> int:
        """전역 y(스펙트럼+글로벌) 차원. g.y는 (1, L)로 저장되므로 L을 반환."""
        y0 = self[0].y
        return int(y0.size(-1) if y0.dim() >= 2 else y0.numel())

    @property
    def spec_len(self) -> int:
        """스펙트럼 길이(파장 구간 개수)."""
        return len(self.spec_cols)

    @property
    def num_node_features(self) -> int:
        """노드 특성 차원(one-hot 원자 등). 일부 코드가 기대할 수 있어 제공."""
        x0 = self[0].x
        return int(x0.size(-1)) if x0 is not None else 0

    # raw 체크 우회(원시 파일을 따로 요구하지 않도록 빈 리스트 반환)
    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        stem = f"{Path(self.csv_path).stem}_{self.stage}_addH{int(self.add_h)}.pt"
        return [stem]


    def process(self) -> None:
        df = pd.read_csv(self.csv_path, low_memory=False)

        # 금지 원소 포함 행 제거(통계/정규화 전에 적용)
        if getattr(self, "forbidden_atoms", None):
            keep_mask = []
            drop_rows = 0
            for _, row in df.iterrows():
                mol = mol_from_row(row, self.smiles_col, self.inchi_col, add_h=False)
                keep = not has_forbidden_atoms(mol, self.forbidden_atoms)
                keep_mask.append(keep)
                if not keep:
                    drop_rows += 1
            if drop_rows > 0:
                print(f"[FILTER] Dropped {drop_rows} molecules due to forbidden atoms: {sorted(self.forbidden_atoms)}")
            df = df.loc[keep_mask].reset_index(drop=True)

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
        log_rows = []
        for ridx, row in df.iterrows():
            mol = mol_from_row(row, self.smiles_col, self.inchi_col, add_h=self.add_h)
            if mol is None:
                continue
            # 안전상, 여기서도 한 번 더 가드(이상치가 들어오지 않게)
            if self.forbidden_atoms and has_forbidden_atoms(mol, self.forbidden_atoms):
                continue

            # ★ UNK 후보 검사 및 시각화
            if self.unk_vis_dir:
                unk_idx, unk_syms = _get_oov_atoms(mol)
                if unk_idx:  # vocab 밖 원소가 하나라도 있으면
                    img_name = _save_unk_viz(
                        mol, Path(self.unk_vis_dir), ridx,
                        row.get(self.smiles_col, "") if self.smiles_col else "",
                        unk_syms, unk_idx
                    )
                    log_rows.append({
                        "row_index": ridx,
                        "smiles": row.get(self.smiles_col, "") if self.smiles_col else "",
                        "unk_symbols": ",".join(unk_syms),
                        "image": img_name
                    })

            g = build_graph(mol)

            try:
                # 우선 CSV의 원문을 쓰고, 없으면 RDKit로 생성한 canonical SMILES를 넣음
                if self.smiles_col and isinstance(row.get(self.smiles_col), str):
                    g.smiles = str(row[self.smiles_col])
                else:
                    g.smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

                if self.inchi_col and isinstance(row.get(self.inchi_col), str):
                    g.inchi = str(row[self.inchi_col])
            except Exception:
                pass

            # ✅ object → float32 문제 회피: 미리 만든 2D 매트릭스에서 꺼냄
            y_spec_np = spec_mat[ridx]
            y_spec_np = np.nan_to_num(y_spec_np, nan=self.spectrum_fill_eps,
                                      posinf=self.spectrum_fill_eps, neginf=self.spectrum_fill_eps)
            y_spec = torch.from_numpy(y_spec_np.copy())  # contiguous 보장용 copy()

            y_globals = self._encode_globals(row, stats)
            y = torch.cat([y_spec, y_globals], dim=0) if y_globals.numel() > 0 else y_spec

            g.y = y.unsqueeze(0)  # (L,) → (1, L)

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
                s_norm = s.dropna().apply(_canon_cat)
                cats = sorted([str(x) for x in s_norm.unique().tolist()])
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
                token = _canon_cat(val) if pd.notna(val) else ""
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

def to_digress_edge5(data):
    if getattr(data, "edge_attr", None) is None or data.edge_attr.numel() == 0:
        data.edge_attr = torch.zeros((0, 5), dtype=torch.float32, device=data.x.device)
        return data
    types4 = data.edge_attr[:, :4]
    t_idx = types4.argmax(dim=-1)  # 0..3
    data.edge_attr = F.one_hot(t_idx + 1, num_classes=5).to(torch.float32)
    return data

# === NEW: PyG Data 배치를 (B,N,dx)/(B,N,N,de)/(B,N)로 패딩하는 collate ===
def collate_dense_padded(batch, MAX_N=None):
    """
    batch: List[torch_geometric.data.Data] with fields:
      - x: (n_i, dx)
      - edge_index: (2, m_i)
      - edge_attr: (m_i, de)  (여기선 to_digress_edge5로 5채널 원핫)
      - y: (1, L)  또는 (L,)
    return:
      X: (B, N, dx), E: (B, N, N, de), Y: (B, L), M: (B, N)
    """
    import torch

    # feature 크기 파악
    d_x = int(batch[0].x.size(-1)) if getattr(batch[0], "x", None) is not None else 0
    if getattr(batch[0], "edge_attr", None) is not None and batch[0].edge_attr.numel() > 0:
        d_e = int(batch[0].edge_attr.size(-1))
    else:
        d_e = 5  # DiGress 규약 기본(무결합 포함 5채널)

    # 배치 내 최대 노드 수 N 결정 (또는 고정 MAX_N)
    n_list = [int(d.x.size(0)) for d in batch]
    N = int(MAX_N) if (MAX_N is not None) else max(n_list)

    Xs, Es, Ms, Ys = [], [], [], []
    for data in batch:
        n = int(data.x.size(0))

        X_pad = torch.zeros(N, d_x, dtype=data.x.dtype)
        E_pad = torch.zeros(N, N, d_e, dtype=(data.edge_attr.dtype if getattr(data, "edge_attr", None) is not None and data.edge_attr.numel() > 0 else torch.float32))
        M_pad = torch.zeros(N, dtype=torch.float32)

        # 노드/마스크
        X_pad[:n] = data.x
        M_pad[:n] = 1.0

        # 엣지(N×N×de) 채우기
        if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.long()  # (2, m)
            ea = data.edge_attr
            # (i,j)가 n 범위 안에 있다고 가정 (RDKit 그래프)
            E_pad[ei[0], ei[1]] = ea

        # y: (1,L) → (L,)
        y = data.y
        y = y.view(-1).to(torch.float32)

        Xs.append(X_pad)
        Es.append(E_pad)
        Ms.append(M_pad)
        Ys.append(y)

    X = torch.stack(Xs, dim=0)      # (B,N,dx)
    E = torch.stack(Es, dim=0)      # (B,N,N,de)
    M = torch.stack(Ms, dim=0)      # (B,N)
    Y = torch.stack(Ys, dim=0)      # (B,L)

    return X, E, Y, M

# --- 2) DataModule: CSVSpecDataset_for_Diffusion 그대로 감싸서 train/val/test 구성 ---
class CSVSpecDataModule(AbstractDataModule):
    def __init__(
        self,
        cfg,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        smiles_col: Optional[str] = None,
        inchi_col: Optional[str] = "InChI",
        spectrum_start: int = 200,
        spectrum_end: int = 800,
        global_cols: Optional[List[str]] = None,
        spectrum_fill_eps: float = 1e-8,
        fixed_vocabs: Optional[Dict[str, List[str]]] = None,
        boolean_cols: Optional[List[str]] = None,
        add_h: bool = False,
        batch_size: int = 128,
        num_workers: int = 4,
        forbidden_atoms: Optional[List[str]] = None,
        unk_vis_dir=True
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

        spec_kwargs = dict(
            smiles_col=smiles_col,
            inchi_col=inchi_col,
            spectrum_start=spectrum_start,
            spectrum_end=spectrum_end,
            global_cols=global_cols or [],
            spectrum_fill_eps=spectrum_fill_eps,
            fixed_vocabs=fixed_vocabs or {},
            boolean_cols=boolean_cols or [],
            add_h=add_h,
            transform=to_digress_edge5,  # ★ 핵심: 여기서 5채널로 변환
            forbidden_atoms=forbidden_atoms,
            unk_vis_dir=unk_vis_dir
        )

        # 빈 문자열/None → 비활성화 또는 재사용 선택
        if not val_csv or str(val_csv).strip() == "":
            # 1) 검증 완전 비활성화
            val_csv = None
            print("[INFO] val_csv not provided → validation disabled")
            # 2) 또는 train을 검증으로 재사용하고 싶다면:
            # val_csv = train_csv
            # print("[INFO] val_csv not provided → using train_csv as validation set")

        if not test_csv or str(test_csv).strip() == "":
            test_csv = None
            print("[INFO] test_csv not provided → test disabled")

        # (선택) 경로 검사 (활성화된 split만)
        for pth, tag in [(train_csv, "train"), (val_csv, "val"), (test_csv, "test")]:
            if pth is None:
                continue
            if not (isinstance(pth, str) and os.path.exists(pth)):
                raise FileNotFoundError(f"[{tag}] CSV not found: {pth}")

        # 동일한 stats(json) 공유
        train_ds = CSVSpecDataset_for_Diffusion(
            csv_path=train_csv, stage="train", **spec_kwargs
        )

        val_ds = None
        if val_csv is not None:
            val_ds = CSVSpecDataset_for_Diffusion(
                csv_path=val_csv,
                stage="val",
                stats_path=train_ds.stats_path,
                **spec_kwargs,
            )

        test_ds = None
        if test_csv is not None:
            test_ds = CSVSpecDataset_for_Diffusion(
                csv_path=test_csv,
                stage="test",
                stats_path=train_ds.stats_path,
                **spec_kwargs,
            )

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

        super().__init__(cfg, datasets={"train": train_ds, "val": val_ds, "test": test_ds})
        self.inner = self.train_dataset  # Spectre 모듈과 동일 패턴

        # ---- 입력/출력 차원 추정 (dx, de, dy) ----
        sample = None
        if getattr(self, "train_dataset", None) is not None and len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
        elif getattr(self, "val_dataset", None) is not None and len(self.val_dataset) > 0:
            sample = self.val_dataset[0]
        elif getattr(self, "test_dataset", None) is not None and len(self.test_dataset) > 0:
            sample = self.test_dataset[0]

        def _safe_size_last(t):
            try:
                return int(t.size(-1))
            except Exception:
                return 0

        import torch

        if sample is None:
            self.dx = 0
            self.de = 0
            self.dy = 0
        else:
            # X 차원
            x_attr = getattr(sample, "x", None)
            if isinstance(x_attr, torch.Tensor):
                self.dx = _safe_size_last(x_attr)
            else:
                self.dx = 0

            # E 차원
            e_attr = getattr(sample, "edge_attr", None)
            if isinstance(e_attr, torch.Tensor):
                if e_attr.ndim >= 2:
                    self.de = int(e_attr.size(-1))
                elif e_attr.ndim == 1:
                    self.de = 1
                else:
                    self.de = 0
            else:
                self.de = 0

            # y 차원
            y_attr = getattr(sample, "y", None)
            if isinstance(y_attr, torch.Tensor):
                if y_attr.ndim == 1:
                    self.dy = int(y_attr.size(0))
                elif y_attr.ndim >= 2:
                    self.dy = int(y_attr.size(-1))
                else:
                    self.dy = int(y_attr.numel())
            else:
                self.dy = 0

        # 하위호환 별칭(선택)
        self.x_dim = self.dx
        self.e_dim = self.de
        self.y_dim = self.dy

        # ✅ pad_to_n을 안전하게 읽어서 멤버로 저장
        self.pad_to_n = None
        try:
            if self.cfg is not None and OmegaConf.is_config(self.cfg):
                # data.pad_to_n 키가 없으면 None
                self.pad_to_n = OmegaConf.select(self.cfg, "data.pad_to_n", default=None)
            elif isinstance(self.cfg, dict):
                self.pad_to_n = (self.cfg.get("data") or {}).get("pad_to_n")
        except Exception:
            self.pad_to_n = None

    # 필요시 DataLoader를 커스터마이즈
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
#
    def val_dataloader(self):
        if getattr(self, "val_dataset", None) is None:
            return DataLoader([], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
#
    def test_dataloader(self):
        if getattr(self, "test_dataset", None) is None:
            return DataLoader([], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    #def train_dataloader(self):
    #    return TorchDataLoader(
    #        self.train_dataset,
    #        batch_size=self.batch_size,
    #        shuffle=True,
    #        num_workers=self.num_workers,
    #        pin_memory=True,
    #        collate_fn=lambda b: collate_dense_padded(b, MAX_N=self.pad_to_n),
    #    )
#
    #def val_dataloader(self):
    #    if getattr(self, "val_dataset", None) is None:
    #        return TorchDataLoader([], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    #    return TorchDataLoader(
    #        self.val_dataset,
    #        batch_size=self.batch_size,
    #        shuffle=False,
    #        num_workers=self.num_workers,
    #        pin_memory=True,
    #        collate_fn=lambda b: collate_dense_padded(b, MAX_N=self.pad_to_n),
    #    )
#
    #def test_dataloader(self):
    #    if getattr(self, "test_dataset", None) is None:
    #        return TorchDataLoader([], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    #    return TorchDataLoader(
    #        self.test_dataset,
    #        batch_size=self.batch_size,
    #        shuffle=False,
    #        num_workers=self.num_workers,
    #        pin_memory=True,
    #        collate_fn=lambda b: collate_dense_padded(b, MAX_N=self.pad_to_n),
    #    )


# --- 3) Dataset Infos: 원자/결합/노드 통계를 구성 (DiGress 모델 입출력 차원 계산에 필요) ---
class CSVSpecInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: CSVSpecDataModule, remove_h: bool = False):
        # 원자 타입: csv_spectrum_dataset.py의 ALLOWED_ATOMS와 일치
        # (여기선 고정 맵; 필요한 경우 datamodule에서 등장한 타입만 추려써도 OK)
        atom_decoder = ATOM_VOCAB[:]  # ["H","C","N","O","F","P","S","Cl","Br","I"]
        self.atom_encoder = {sym: i for i, sym in enumerate(atom_decoder)}
        self.atom_decoder = atom_decoder
        self.num_atom_types = len(self.atom_decoder)
        self.remove_h = bool(remove_h)

        # 결합 타입은 DiGress 규약: 0=no-bond, 1=single, 2=double, 3=triple, 4=aromatic
        base_valencies = [1, 4, 3, 2, 1, 3, 2, 1, 1, 1]  # H..I
        if datamodule.dx > len(base_valencies):
            base_valencies = base_valencies + [0] * (datamodule.dx - len(base_valencies))  # UNK=0
        self.valencies = base_valencies[:datamodule.dx]

        base_weights = { 0: 1, 1: 12, 2: 14, 3: 16, 4: 19, 5: 31, 6: 32, 7: 35, 8: 80, 9: 127, 10: 11 }
        self.atom_weights = {i: float(base_weights.get(i, 0.0)) for i in range(datamodule.dx)}

        self.max_weight = 600  # 데이터에 맞게 넉넉히

        # 통계(노드 수/노드 타입/엣지 타입) 대략 계산
        # n_nodes 분포 및 max_n_nodes
        import numpy as np
        sizes = []
        node_type_hist = torch.zeros(datamodule.dx, dtype=torch.float)  # 11dim
        edge_type_hist = torch.zeros(5, dtype=torch.float) # 5dim
        for g in datamodule.train_dataset:
            sizes.append(g.x.size(0))
            node_type_hist += g.x.sum(dim=0).to(torch.float)  # g.x.shape[-1] == datamodule.dx
            if g.edge_attr.numel() > 0:
                edge_type_hist += g.edge_attr.sum(dim=0).to(torch.float)

        self.max_n_nodes = int(max(sizes)) if len(sizes) else 64
        # 히스토그램을 분포로 정규화(0번째 버킷은 '0개 노드' 자리이므로 길이 max_n_nodes+1)
        n_hist = np.bincount(np.array(sizes), minlength=self.max_n_nodes + 1)
        self.n_nodes   = torch.tensor(n_hist, dtype=torch.float)
        self.node_types = (node_type_hist / max(node_type_hist.sum(), torch.tensor(1.0))).clone()
        self.edge_types = (edge_type_hist / max(edge_type_hist.sum(), torch.tensor(1.0))).clone()

        # ---- Valency target distribution (Aromatic = 1.5, soft binning) ------------
        # SamplingMolecularMetrics가 참조하는 타깃 분포
        self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2, dtype=torch.float32)

        # 결합 차수: 0=no, 1=single, 2=double, 3=triple, 4=aromatic(=1.5)
        bond_order = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.5], dtype=torch.float32)

        for g in datamodule.train_dataset:
            n = int(g.x.size(0))
            if getattr(g, "edge_index", None) is None or g.edge_index.numel() == 0:
                # 엣지가 없으면 모든 노드의 발런시가 0
                self.valency_distribution[0] += n
                continue

            # 엣지 타입(one-hot 5ch) → 타입 인덱스 → 결합 차수(실수; 방향족=1.5)
            etype = g.edge_attr.argmax(dim=-1)  # (m,)
            eord = bond_order[etype]  # (m,)

            # 노드별 발런시 합(실수)
            src = g.edge_index[0].to(torch.long)  # (m,)
            node_val = torch.zeros(n, dtype=torch.float32)
            node_val.scatter_add_(0, src, eord)  # sum of bond orders per node

            # 소프트 할당: 정수 bin 사이에 선형 분배 (정수면 lo==hi가 되어 1이 정확히 들어감)
            lo = torch.floor(node_val).to(torch.long)
            hi = torch.ceil(node_val).to(torch.long)
            frac = (node_val - lo.to(torch.float32)).clamp_(0, 1)

            max_idx = self.valency_distribution.numel() - 1
            lo = lo.clamp_(0, max_idx)
            hi = hi.clamp_(0, max_idx)

            buf = torch.zeros_like(self.valency_distribution)
            buf.scatter_add_(0, lo, (1 - frac))
            buf.scatter_add_(0, hi, frac)
            self.valency_distribution += buf
        # ---------------------------------------------------------------------------

        # 입력/출력 차원 (모델에서 compute_input_output_dims 호출 시 참고)
        self.input_dims = {"X": datamodule.dx, "E": datamodule.de, "y": datamodule.dy}
        self.output_dims = {"X": datamodule.dx, "E": datamodule.de, "y": datamodule.dy}
        self.remove_h = bool(remove_h)
        self.y_dim = datamodule.dy

        # 부모 클래스 헬퍼로 기본 채워넣기(Spectre 예시와 동일 패턴)
        super().complete_infos(self.n_nodes, self.node_types)  # 【Spectre 예시】complete_infos 사용【turn23file2†spectre_dataset.py†L45-L52】

    # 필요시 추가: compute_input_output_dims는 main에서 호출됩니다
    # main.py는 extra_features/domain_features와 함께 입출력 차원을 계산해요【turn23file7†main.py†L13-L19】.


# ============================
# Standalone runner (PyCharm)
# ============================
def _parse_forbidden_atoms(text):
    if not text:
        return None
    return [t.strip() for t in text.split(",") if t.strip()]

def main():
    """
    PyCharm에서 바로 실행 가능한 UNK 시각화용 main()
    - CONFIG만 수정하고 실행하세요.
    - 각 split별 폴더에 UNK로 매핑된 분자 이미지를 저장하고, unk_molecules.csv 로그를 남깁니다.
    """
    # -----------------------
    # CONFIG (여기만 수정)
    # -----------------------
    MODE = "single"  # "single" | "multi"
    CSV  = "/root/PycharmProjects/DiGress/data/csv/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
    SPLIT = "train"  # "train" | "val" | "test"

    TRAIN_CSV = "/root/PycharmProjects/DiGress/data/csv/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
    VAL_CSV   = None
    TEST_CSV  = None

    SMILES_COL = "SMILES"
    INCHI_COL  = "InChI"   # 없으면 None
    ADD_H = False          # 암시적 H를 명시화(AddHs) 후 처리할지
    FORBIDDEN_ATOMS = ["As"]  # 금지(드롭)할 원소 리스트. []면 금지 없음
    CSV_ENCODING = None    # None이면 자동 시도
    UNK_VIS_DIR = "viz_unk_demo"  # 출력 루트 (split 하위 폴더 생성)
    IMG_SIZE = (500, 400)
    # -----------------------

    # 지연 import (모듈 내 정의를 그대로 사용)
    import os
    from pathlib import Path
    import pandas as pd
    from rdkit.Chem import Draw

    # 모듈 전역 정의 사용 (이미 파일에 존재)
    # - ATOM2IDX, FORBIDDEN_ATOMS_DEFAULT, has_forbidden_atoms, mol_from_row
    global ATOM2IDX, FORBIDDEN_ATOMS_DEFAULT
    global has_forbidden_atoms, mol_from_row

    # 금지 집합 구성
    forbidden = set(FORBIDDEN_ATOMS) if FORBIDDEN_ATOMS is not None else set(FORBIDDEN_ATOMS_DEFAULT)

    # 입력 목록 구성
    triples = []
    if MODE.lower() == "single":
        if not CSV:
            raise SystemExit("[CONFIG] MODE='single'에서는 CSV 경로가 필요합니다.")
        triples.append((SPLIT.lower(), CSV))
    else:
        if TRAIN_CSV: triples.append(("train", TRAIN_CSV))
        if VAL_CSV:   triples.append(("val",   VAL_CSV))
        if TEST_CSV:  triples.append(("test",  TEST_CSV))
        if not triples:
            raise SystemExit("[CONFIG] MODE='multi'에서는 TRAIN_CSV/VAL_CSV/TEST_CSV 중 하나 이상 지정하세요.")

    out_root = Path(UNK_VIS_DIR)
    print("[INFO] 실행 설정")
    for sp, path in triples:
        print(f"  - {sp}: {path}")
    print(f"  - smiles_col={SMILES_COL}, inchi_col={INCHI_COL}, add_h={ADD_H}")
    print(f"  - forbidden_atoms={sorted(list(forbidden)) if forbidden else []}")
    print(f"  - out_dir={out_root.resolve()}")

    total_rows = 0
    total_parsed = 0
    total_forbidden_dropped = 0
    total_unk = 0

    for sp, csv_path in triples:
        if not (isinstance(csv_path, str) and os.path.exists(csv_path)):
            raise FileNotFoundError(f"[{sp}] CSV not found: {csv_path}")

        out_dir = out_root / sp
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSV 로드
        try:
            df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")

        log_rows = []
        rows = len(df)
        parsed = 0
        forbidden_dropped = 0
        unk_count = 0

        for ridx, row in df.iterrows():
            total_rows += 1

            mol = mol_from_row(row, SMILES_COL, INCHI_COL, add_h=ADD_H)  # SMILES 우선, 실패 시 InChI; AddHs 옵션 지원 :contentReference[oaicite:3]{index=3}
            if mol is None:
                continue
            parsed += 1
            total_parsed += 1

            # 금지 원소 드롭
            if forbidden and has_forbidden_atoms(mol, forbidden):  # 금지 원소 판정 로직은 모듈 내 정의 사용 :contentReference[oaicite:4]{index=4}
                forbidden_dropped += 1
                total_forbidden_dropped += 1
                continue

            # UNK 후보 탐지: ATOM_VOCAB(ATOM2IDX)에 없는 심볼
            unk_idx = []
            unk_syms = set()
            for a in mol.GetAtoms():
                s = a.GetSymbol()
                if s not in ATOM2IDX:  # vocab 밖 → UNK 매핑 대상 (ATOM2IDX/UNK 정의는 모듈 상단에 존재) :contentReference[oaicite:5]{index=5}
                    unk_idx.append(a.GetIdx())
                    unk_syms.add(s)

            if unk_idx:
                unk_count += 1
                total_unk += 1
                # PNG 저장
                fname = f"{ridx:06d}.png"
                legend = f"UNK={','.join(sorted(unk_syms))} | row={ridx}"
                Draw.MolToFile(mol, str(out_dir / fname), size=IMG_SIZE, highlightAtoms=unk_idx, legend=legend)
                # 로그 적재
                log_rows.append({
                    "row_index": ridx,
                    "smiles": row.get(SMILES_COL, "") if SMILES_COL else "",
                    "inchi": row.get(INCHI_COL, "") if INCHI_COL else "",
                    "unk_symbols": ",".join(sorted(unk_syms)),
                    "image": fname,
                })

        # split별 요약 및 로그 저장
        if log_rows:
            pd.DataFrame(log_rows).to_csv(out_dir / "unk_molecules.csv", index=False)
        print(f"[OK] {sp}: rows={rows}, parsed={parsed}, forbidden_dropped={forbidden_dropped}, unk_mols={unk_count} → {out_dir}")

    print(f"\n[DONE] total_rows={total_rows}, total_parsed={total_parsed}, total_forbidden_dropped={total_forbidden_dropped}, total_unk_mols={total_unk}")

if __name__ == "__main__":
    main()