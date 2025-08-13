import sys
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from functools import partial
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from rdkit.Chem import AllChem
import io
from rdkit.Chem.MolStandardize import rdMolStandardize

PREDEFINED_VOCAB = {
    'Solvent': ["water", "ethanol", "methanol", "acetonitrile"],
    'pH': ["acidic", "neutral", "basic"],
}

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def _get_feature_index(value, vocab):
    if value in vocab:
        return vocab.index(value)  # 값이 존재하면 그 정확한 위치(인덱스) 를 돌려줌
    return vocab.index(vocab[0])  # 값이 안 찾아졌을 경우 이것을 vocab 0 으로 놓는다.


def _compute_shortest_paths(adj):
    num_nodes = adj.shape[0]  # 인접 행렬에서 Node
    dist = np.full((num_nodes, num_nodes), -1, dtype=int)  # 거리행렬, -1로 초기화 (미방문은 -1)
    np.fill_diagonal(dist, 0)  # 자기 자신 까지 거리 = 0
    for i in range(num_nodes):
        q = [(i, 0)]
        visited = {i}
        head = 0
        while head < len(q):
            u, d = q[head]
            head += 1
            dist[i, u] = d
            for v in np.where(adj[u])[0]:
                if v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))
    dist[dist == -1] = -1
    return dist


def get_global_feature_info(global_feature_names, PREDEFINED_VOCAB):
    nominal_feature_vocab = {k: v for k, v in PREDEFINED_VOCAB.items() if k in global_feature_names}
    continuous_feature_names_list = [name for name in global_feature_names if name not in nominal_feature_vocab]

    global_cat_dim = 0
    for name in nominal_feature_vocab:  # 명목형
        global_cat_dim += len(nominal_feature_vocab[name])

    global_cont_dim = len(continuous_feature_names_list)  # 수치형

    return nominal_feature_vocab, continuous_feature_names_list, global_cat_dim, global_cont_dim

def smiles2graph_customized(
        smiles: str,
        multi_hop_max_dist: int,
        ATOM_FEATURES_VOCAB,
        float_feature_keys,
        BOND_FEATURES_VOCAB,
        ):
    """
    SMILES 1 개를 Graphormer용 그래프 딕셔너리로
    """
    # InChI 문자열이면
    # print("smiles2graph_customized, smiles",smiles)
    if smiles.startswith("InChI="):
        try:
            mol = Chem.MolFromInchi(smiles)
        except Exception as e:
            print(f"[ERROR] MolFromInchi failed: {e}")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            print(f"[ERROR] MolFromSmiles failed: {e}")

    if mol is None:
        return None

    # 혹시 혼합물 있을 경우 염 제거 시도후 안되면 배제
    try:
        if len(Chem.GetMolFrags(mol)) > 1:
            lfc = rdMolStandardize.LargestFragmentChooser()
            mol = lfc.choose(mol)
    except Exception as e:
        print(f"[Fragment Clean Failed] {smiles} → {e}")
        return None

    # 부분 전하 계산
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        print("partial charge calculation failed")
        pass

    # 다양한 feature를 smiles 로부터 계산 #
    num_nodes = mol.GetNumAtoms()
    global_idx = num_nodes
    adj = np.zeros((num_nodes, num_nodes), dtype=bool)  # Flase 로 차있는 adj

    node_features_cat = {key: [] for key in ATOM_FEATURES_VOCAB if isinstance(ATOM_FEATURES_VOCAB[key], list)}
    node_features_cont = {key: [] for key in float_feature_keys}

    for atom in mol.GetAtoms():
        for key, vocab_or_type in ATOM_FEATURES_VOCAB.items():
            if isinstance(vocab_or_type, list):
                if key == 'atomic_num':
                    prop = atom.GetAtomicNum()
                elif key == 'formal_charge':
                    prop = atom.GetFormalCharge()
                elif key == 'hybridization':
                    prop = atom.GetHybridization()
                elif key == 'is_aromatic':
                    prop = int(atom.GetIsAromatic())
                elif key == 'total_num_hs':
                    prop = atom.GetTotalNumHs()
                elif key == 'explicit_valence':
                    prop = atom.GetExplicitValence()
                elif key == 'total_bonds':
                    prop = atom.GetTotalDegree()
                node_features_cat[key].append(_get_feature_index(prop, vocab_or_type))
            elif vocab_or_type is float:
                if key == 'atomic_mass':
                    node_features_cont[key].append(atom.GetMass())
                elif key == 'partial_charge':
                    try:
                        charge = float(atom.GetProp('_GasteigerCharge'))
                        node_features_cont[key].append(charge)
                    except(KeyError, ValueError):  # 오류가 난 경우 0
                        node_features_cont[key].append(0.0)

    # Combine categorical features into a single integer array : 정수형 또는 범주형
    x_cat = np.stack(list(node_features_cat.values()), axis=-1)

    # Combine continuous features into a single float array : 실수형만
    x_cont = np.stack(list(node_features_cont.values()), axis=-1)

    ### Bond Feature 들 ###
    attn_edge_type = {
        k: np.zeros((num_nodes, num_nodes, len(vocab)), dtype=np.int64)  # (N,N,D)
        for k, vocab in BOND_FEATURES_VOCAB.items()
    }
    edge_indices = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = True  # bond 에 있는 i위치와 j 위치를 사용하여 adj 인접행렬 완성 (N,N)
        edge_indices.extend([[i, j], [j, i]])

        for key, vocab in BOND_FEATURES_VOCAB.items():
            if key == 'bond_type':
                prop = bond.GetBondType()
            elif key == 'stereo':
                prop = bond.GetStereo()
            elif key == 'is_conjugated':
                prop = int(bond.GetIsConjugated())
            elif key == 'is_in_ring':
                prop = int(bond.IsInRing())

            # bond type 에는 단일, 이중, 삼중 ... 있다 가정
            idx = _get_feature_index(prop, vocab)  # bond type에서 단일 = 0 이중=1, 삼중=2 이런식으로 되어 있다 가정
            attn_edge_type[key][i, j, idx] = 1  # 결합이 단일이면 정사각형 i,j,0 을 1로 채우고 나머지 i,j,1, i,j,2 등은 0으로 놔둠
            attn_edge_type[key][j, i, idx] = 1  # attn_edge_type = (N,N,D)

    spatial_pos = _compute_shortest_paths(adj)  # 가장 짧은 경로를 나타내는 spatial pos 를 만듦

    # multi hop #
    edge_input = {
        key: np.zeros(
            (num_nodes, num_nodes, multi_hop_max_dist, len(vocab)),  # ← 4-D 로!
            dtype=np.int64
        )
        for key, vocab in BOND_FEATURES_VOCAB.items()
    }
    # 2) 값 복사
    # multi hop N,N,D,C
    # 출발노드, 목적지노드, hop 별 슬롯 개수, Edge-type-one hot
    # 3 hop 이 max 일 경우 : D 에 대해서 (1,0,0) (0,1,0) (0,0,1)이렇게 가능함
    # 만약 3hop 인 경우 그 사이에 두개의 결합 -> C(=3) 길이의 one-hot 벡터가 hop 수(2개)만큼 “이어 붙여진” 구조
    # multi_hop.png 참조
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = spatial_pos[i, j]
            if 1 <= dist < multi_hop_max_dist:
                for key in BOND_FEATURES_VOCAB.keys():
                    # print(edge_input[key].shape)
                    # print(attn_edge_type[key].shape)
                    edge_input[key][i, j, dist - 1, :] = attn_edge_type[key][i, j, :]

    # edge_input[key][i, j, dist - 1] = attn_edge_type[key][i, j]
    #print("customized",type(edge_input))
    return {
        'x_cat': x_cat,
        'x_cont': x_cont,
        'adj': adj,
        'edge_index': np.array(edge_indices).T if edge_indices else np.empty((2, 0), dtype=int),
        'attn_edge_type': attn_edge_type,  # Now a dict of arrays
        'spatial_pos': spatial_pos,
        'edge_input': edge_input,  # Now a dict of arrays
        'num_nodes': num_nodes,
    }

def smiles2graph_with_global(
        smiles: str,
        global_cat_idx: list[int],     # 각 범주형 feature의 ‘정수 index’
        global_cont_val: list[float],  # 각 연속형 feature의 실수 값
        ATOM_FEATURES_VOCAB,
        float_feature_keys,
        BOND_FEATURES_VOCAB,
        multi_hop_max_dist: int,
):
    #print("smiles2graph_with_global, smiles:", smiles)
    if smiles.startswith("InChI="):
        try:
            mol = Chem.MolFromInchi(smiles)
        except Exception as e:
            print(f"[ERROR] MolFromInchi failed: {e}")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            print(f"[ERROR] MolFromSmiles failed: {e}")

    if mol is None or len(Chem.GetMolFrags(mol)) > 1:
        return None

    # ---------- 1) 기존 원자 정보 ----------
    orig = smiles2graph_customized(smiles, multi_hop_max_dist, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB,)     # ← 기존 함수 재활용
    if orig is None: return None

    n               = orig["num_nodes"]            # 원자 개수
    global_idx      = n                            # 새 노드 인덱스
    atom_cat_dim    = orig["x_cat"].shape[1]
    atom_cont_dim   = orig["x_cont"].shape[1]
    g_cat_dim       = len(global_cat_idx)
    g_cont_dim      = len(global_cont_val)

    # ---------- 2) 노드-특성 행렬 확장 ----------
    x_cat  = np.zeros((n + 1, atom_cat_dim + g_cat_dim), dtype=np.int64)
    x_cont = np.zeros((n + 1, atom_cont_dim + g_cont_dim), dtype=np.float32)
    # 원자용 열(atom_cat/cont)만 값
    x_cat[:n, :atom_cat_dim] = orig["x_cat"]
    x_cont[:n, :atom_cont_dim] = orig["x_cont"]

    # Global 노드: 원자 슬롯 0, 글로벌 슬롯만 값
    x_cat[global_idx, atom_cat_dim:]   = np.array(global_cat_idx, dtype=np.int64)
    x_cont[global_idx, atom_cont_dim:] = np.array(global_cont_val, dtype=np.float32)

    # ---------- 3) Global Edge 추가 ----------
    D = orig["attn_edge_type"]["bond_type"].shape[-1]
    adj = np.zeros((n + 1, n + 1), dtype=bool)
    adj[:n, :n] = orig["adj"]
    adj[global_idx, :n] = adj[:n, global_idx] = True

    # ===== ① edge_index 확장 =====
    orig_e = orig["edge_index"]  # (2, E)
    extra = [[i, global_idx] for i in range(n)] +             [[global_idx, i] for i in range(n)]
    edge_index = np.concatenate([orig_e,np.array(extra).T], axis=1)  # (2, E+2n)

    # ---------- 4) attn_edge_type / edge_input 확장 ----------
    attn_edge_type = {}
    for key, orig_t in orig["attn_edge_type"].items():
        t = np.zeros((n + 1, n + 1, orig_t.shape[-1]), dtype=np.int64)
        t[:n, :n] = orig_t
        if key == "bond_type":  # Global-edge 표시
            t[global_idx, :n, 0] = 1
            t[:n, global_idx, 0] = 1
        attn_edge_type[key] = t

    edge_input = {}
    for key, orig_e in orig["edge_input"].items():
        max_dist, Dk = orig_e.shape[2], orig_e.shape[3]
        e = np.zeros((n + 1, n + 1, max_dist, Dk), dtype=np.int64)
        e[:n, :n] = orig_e
        if key == "bond_type":
            e[global_idx, :n, 0, 0] = 1
            e[:n, global_idx, 0, 0] = 1
        edge_input[key] = e

    # Global-edge one-hot 추가 ----------------------------
    max_dist = multi_hop_max_dist  # 이미 함수 인자로 있음

    # 2-채널 one-hot: [not-connected, connected]
    attn_edge_type["is_global"] = np.zeros((n + 1, n + 1, 2), np.int64)
    attn_edge_type["is_global"][global_idx, :n, 1] = 1  # g → atom
    attn_edge_type["is_global"][:n, global_idx, 1] = 1  # atom → g

    edge_input["is_global"] = np.zeros((n + 1, n + 1, max_dist, 2), np.int64)
    edge_input["is_global"][global_idx, :n, 0, 1] = 1  # 1-hop 슬롯
    edge_input["is_global"][:n, global_idx, 0, 1] = 1

    # ---------- 4) 최단거리 ----------
    spatial_pos = _compute_shortest_paths(adj)
    spatial_pos[spatial_pos == 0] = 1   # self-loop convention
    #print("smiles to graph with global",type(edge_input))

    return {
        "x_cat": x_cat,
        "x_cont": x_cont,
        "adj": adj,
        "edge_index": edge_index,
        "attn_edge_type": attn_edge_type,
        "spatial_pos": spatial_pos,
        "edge_input": edge_input,
        "num_nodes": n + 1,
        "global_features_cat": np.array(global_cat_idx, dtype=np.int64),
        "global_features_cont": np.array(global_cont_val, dtype=np.float32),
    }

class UnifiedSMILESDataset(Dataset):
    def __init__(
        self,
        nominal_feature_vocab,
        continuous_feature_names,
        global_cat_dim,
        global_cont_dim,
        ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB,
        mol_col: str = 'smiles',
        mode="cls",  # "cls", "cls+global_data", "cls+global_model"
        max_nodes: int = 128,
        multi_hop_max_dist: int = 5,
        # 여러 CSV 옵션 — 들어온 개수로 멀티/싱글 판단
        uvvis_csv: str = None,
        fluorescence_csv: str = None,
        qm_nm_csv: str = None,
        qm_eVOsc_csv: str = None,
        # 공통 스펙트럼 그리드
        target_grid: tuple[int, int] = (200, 800),
        # 관측 가능한 원래 범위(없으면 자동 판단 가능하지만 기본값 둠)
        uv_range: tuple[int, int] = (200, 800),
        fl_range: tuple[int, int] = (300, 700),
        qm_nm_range : tuple[int, int] = (300, 700),

        attn_bias_w: float = 0.0,

        ex_normalize: str = None,
        prob_normalize: str = None,

        # nm_dist_mode: str = "hist",
        # nm_gauss_sigma: float = 10.0,

    ):
        self.mode = mode
        self.is_global = mode in ("cls_global_data", "cls_global_model")
        self.nominal_feature_vocab = nominal_feature_vocab
        self.continuous_feature_names = continuous_feature_names
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        # self.target_type = target_type
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.attn_bias_weight = attn_bias_w
        self.mol_col = mol_col

        # 공통 파장 그리드
        self.gmin, self.gmax = target_grid
        self.grid_cols = [str(w) for w in range(self.gmin, self.gmax + 1)]

        # multi task Data
        inputs = []
        if uvvis_csv:
            inputs.append(("UV", pd.read_csv(uvvis_csv).copy()))
        if fluorescence_csv:
            inputs.append(("FL", pd.read_csv(fluorescence_csv).copy()))
        if qm_nm_csv:
            inputs.append(("QM_NM", pd.read_csv(qm_nm_csv).copy()))
        if qm_eVOsc_csv:
            inputs.append(("QM_EVP", pd.read_csv(qm_eVOsc_csv).copy()))

        self.is_multitask = len(inputs) > 1 # single / multitask

        dfs = []
        for tag, df in inputs:
            kind = self._sniff_kind(df)
            if kind == "spectrum_nm":
                # 원본 관측 범위 선택
                if tag == "UV":
                    pr = uv_range
                elif tag == "FL":
                    pr = fl_range
                elif tag == "QM_NM":
                    pr = qm_nm_range
                else:
                    pr = (self.gmin, self.gmax)
                df2 = self._ensure_grid(df, present_range=pr)

            elif kind == "exprob":
                # eV/osc
                df2 = df.copy()  # 변환 없이 그대로 사용
                df2["row_kind"] = "exprob"

            else:
                # 모르는 형식이면 빈 스펙트럼(NaN)로
                df2 = df.copy()
                for c in self.grid_cols:
                    if c not in df2.columns:
                        df2[c] = np.nan
                df2["row_kind"] = "spec"

            # 태스크 라벨
            df2["task_name"] = {"UV": "UV", "FL": "FL", "QM_NM": "QM_NM", "QM_EVP": "QM_EVP"}.get(tag, "SINGLE")

            # QM 소스면 글로벌 기본값 주입
            if tag in ("QM_NM", "QM_EVP"):
                # 열이 없어도 생성되며, assign은 단편화 유발 안 함
                df2 = df2.assign(
                    **({"Solvent": "QM"} if "Solvent" in (self.nominal_feature_vocab or {}) else {}),
                    **({"pH": "neutral"} if "pH" in (self.nominal_feature_vocab or {}) else {}),
                )
            dfs.append(df2)

        # 4) 결합 + 셔플(고정 시드)
        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        self.data = self.data.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # target_type ex_prob #
        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize

        # 5) 글로벌 피처 사전(있으면 사용, 없어도 통과)
        self.nominal_feature_info = self._build_nominal_feature_info()

        # Solvent에 "QM"이 반드시 들어가도록
        if "Solvent" in self.nominal_feature_vocab:
            if "QM" not in self.nominal_feature_vocab["Solvent"]:
                self.nominal_feature_vocab["Solvent"] = list(self.nominal_feature_vocab["Solvent"]) + ["QM"]

        # pH에 "neutral"이 반드시 들어가도록
        if "pH" in self.nominal_feature_vocab:
            if "neutral" not in self.nominal_feature_vocab["pH"]:
                self.nominal_feature_vocab["pH"] = list(self.nominal_feature_vocab["pH"]) + ["neutral"]

        ########

        # 유효 SMILES → 그래프 생성/필터
        self.raw_graphs, valid_idx = [], []
        for i, s in enumerate(self.data[self.mol_col].astype(str)):
            g = smiles2graph_customized(
                s, self.multi_hop_max_dist,
                ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                float_feature_keys=float_feature_keys,
                BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB
            )
            if g is not None:
                self.raw_graphs.append(g)
                valid_idx.append(i)

        # 유효 인덱스로 슬라이스
        self.data = self.data.iloc[valid_idx].reset_index(drop=True)
        # self.sample_targets = [self.sample_targets[i] for i in valid_idx]
        self.sample_targets = self.process_targets()

        self.graphs = [
            self._preprocess_graph_with_optional_global(i, g, ATOM_FEATURES_VOCAB, float_feature_keys,
                                                        BOND_FEATURES_VOCAB)
            for i, g in enumerate(self.raw_graphs)
        ]

        # task id
        task_names = self.data["task_name"].tolist() if "task_name" in self.data.columns else ["SINGLE"] * len(
            self.data)
        uniq = {name: i for i, name in enumerate(sorted(set(task_names)))}
        self.task_ids = torch.tensor([uniq[n] for n in task_names], dtype=torch.long)
        self.is_multitask = len(uniq) > 1

        self._validate_columns(self.data)
        # self.data = self.data.loc[:, self._get_all_cols_to_load()]

        #self.raw_graphs = [g for g in [smiles2graph_customized(s, self.multi_hop_max_dist ,ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB, float_feature_keys=float_feature_keys, BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB) for s in self.data["smiles"]] if g is not None]

        # 1. targets, mask 분리

        # spectrum, mask_tensor = self.process_targets()

        # 2. 유효 SMILES 기준으로 graph 추출
        self.raw_graphs = []
        valid_indices = []
        # --- 그래프 처리 ---
        # raw_g = self.raw_graphs[idx]
        # g_processed = self.preprocess_graph(raw_g)
        # task_id = self.task_ids[idx]
        for i, s in enumerate(self.data[self.mol_col]):
            g = smiles2graph_customized(
                s,
                self.multi_hop_max_dist,
                ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                float_feature_keys=float_feature_keys,
                BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB
            )
            if g is not None:
                self.raw_graphs.append(g)
                valid_indices.append(i)

        self.proc_graphs = [self.preprocess_graph(g) for g in self.raw_graphs]

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)

        self.graphs = [self._preprocess_graph_with_optional_global(i, g, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB,) for i, g in enumerate(self.raw_graphs)]
        for g in self.graphs:
            print("",g.keys())

    def __getitem__(self, idx):
        #print("idx", idx)
        tgt = self.sample_targets[idx]
        task_id = self.task_ids[idx]

        # --- Always prepare global features (in case model-internal needs it) ---
        global_feat_cat_indices = []
        for name in self.nominal_feature_vocab:
            val = self.data.loc[idx, name]
            vocab_info = self.nominal_feature_info[name]
            global_feat_cat_indices.append(vocab_info['value_to_idx'].get(val, 0))

        global_feat_cont_values = [float(self.data.loc[idx, name]) for name in self.continuous_feature_names]

        global_feat_cat_tensor = torch.tensor(global_feat_cat_indices, dtype=torch.long)
        global_feat_cont_tensor = torch.tensor(global_feat_cont_values, dtype=torch.float32)

        base_g = self.proc_graphs[idx]

        # global node를 그래프 안에 “주입”해야 하는 모드라면,
        # 원본 dict를 망가뜨리지 않도록 얕은 복사 후 필드만 추가하세요.
        g_processed = dict(base_g)

        if self.mode == "cls_global_data":
            g_processed["global_features_cat"] = global_feat_cat_tensor
            g_processed["global_features_cont"] = global_feat_cont_tensor

            # 반환 형식: 한 샘플을 dict로
        out = {
            "graph": g_processed,
            "task_id": task_id,
            "kind": tgt["kind"],  # "spec" 또는 "exprob"
        }
        if tgt["kind"] == "spec":
            out["y"] = tgt["y"]  # [P]
            out["mask"] = tgt["mask"]  # [P]
        else:
            out["ex"] = tgt["ex"]  # [K]
            out["prob"] = tgt["prob"]  # [K]
            out["mask"] = tgt["mask"]  # [K]
        return out

    def __len__(self):
        return len(self.graphs)

    def _sniff_kind(self, df: pd.DataFrame) -> str:
        # 스펙트럼형? (그리드 컬럼 중 하나라도 있으면)
        if any(c in df.columns for c in self.grid_cols):
            return "spectrum_nm"
        # ex/prob 리스트형?
        ex_cols = sorted([c for c in df.columns if c.lower().startswith("ex")], key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        prob_cols = sorted([c for c in df.columns if c.lower().startswith("prob")], key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        if ex_cols and prob_cols and len(ex_cols) == len(prob_cols):
            return "exprob"
        return "unknown"

    #def _ensure_grid(self, df: pd.DataFrame, *, present_range: tuple[int,int]) -> pd.DataFrame:
    #    df = df.copy()
    #    pmin, pmax = present_range
    #    # 없으면 NaN 생성
    #    for c in self.grid_cols:
    #        if c not in df.columns:
    #            df[c] = np.nan
    #    # 관측 범위 밖 NaN 강제
    #    for w in range(self.gmin, self.gmax + 1):
    #        if not (pmin <= w <= pmax):
    #            df[str(w)] = np.nan
    #    # 메타→파장 정렬
    #    meta = [c for c in df.columns if c not in self.grid_cols]
    #    return df[meta + self.grid_cols]

    def _ensure_grid(self, df: pd.DataFrame, *, present_range: tuple[int, int]) -> pd.DataFrame:
        pmin, pmax = present_range

        # 1) 메타/그리드 분리 및 한 번에 재색인 (+ copy로 블록 통합)
        grid_set = set(self.grid_cols)
        meta_cols = [c for c in df.columns if c not in grid_set]
        df = df.reindex(columns=meta_cols + self.grid_cols).copy()

        # 2) 관측 범위 밖은 일괄 NaN (넘파이 배열로 한 번에)
        wavelengths = np.arange(self.gmin, self.gmax + 1)
        outside = (wavelengths < pmin) | (wavelengths > pmax)
        if outside.any():
            grid_vals = df[self.grid_cols].to_numpy(copy=True)
            grid_vals[:, outside] = np.nan
            df[self.grid_cols] = grid_vals

        # 3) 라벨 컬럼도 "assign"으로 한 번에 추가(새 프레임 반환 → 단편화 X)
        return df.assign(row_kind="spec")

    def _build_nominal_feature_info(self):
        return {
            name: {
                'unique_values': vocab,
                'value_to_idx': {val: i for i, val in enumerate(vocab)}
            } for name, vocab in self.nominal_feature_vocab.items()
        }

    def _validate_columns(self, csv_file):
        # 1) SMILES(or mol) 컬럼만은 필수
        if self.mol_col not in self.data.columns:
            raise ValueError(f"Missing required column '{self.mol_col}' in {csv_file}")

        # 2) 최소한 하나의 타깃 표현이 있어야 함:
        #    - nm 스펙트럼 그리드(공통 grid 중 하나라도 존재) 또는
        #    - ex*/prob* 페어 중 일부라도 존재
        has_grid_any = any(c in self.data.columns for c in self.grid_cols)

        all_cols = list(self.data.columns)
        has_ex = any(str(c).lower().startswith("ex") for c in all_cols)
        has_prob = any(str(c).lower().startswith("prob") for c in all_cols)
        has_exprob_any = has_ex and has_prob

        if not (has_grid_any or has_exprob_any):
            raise ValueError(
                "No detectable target columns. "
                "Need at least some wavelength grid columns (e.g., '200'..'800') "
                "or paired ex*/prob* columns."
            )

        # 3) 글로벌 피처는 있으면 사용, 없으면 통과 (강제 X)
        #    단, nominal/continuous 이름이 지정되었는데 실제 컬럼이 없으면 경고만
        missing_nominal = [n for n in (self.nominal_feature_vocab or {}).keys()
                           if n not in self.data.columns]
        missing_cont = [n for n in (self.continuous_feature_names or [])
                        if n not in self.data.columns]
        if missing_nominal or missing_cont:
            print(f"[Warn] Missing global feature columns. "
                  f"nominal missing={missing_nominal}, continuous missing={missing_cont}")

    def process_targets(self):
        """
        행 단위로 타깃을 자동 판별해 리스트로 반환/보관.
          - 스펙트럼형(row_kind='spec' 또는 그리드 컬럼 존재 & ex/prob 유효쌍 없음):
              관측값만 min-max 정규화, 관측 외 구간은 0으로 채우고 mask=False.
              -> {'kind':'spec', 'y':Tensor[P], 'mask':Tensor[P]}
          - eV/osc형(ex1..K & prob1..K 유효쌍 존재):
              eV/osc를 변환 없이 그대로 사용(옵션 정규화만 적용 가능).
              -> {'kind':'exprob', 'ex':Tensor[K], 'prob':Tensor[K], 'mask':Tensor[K]}
        반환값: self.sample_targets (list of dict)
        """
        # 공통 그리드 컬럼
        if not hasattr(self, "grid_cols"):
            self.gmin, self.gmax = getattr(self, "gmin", 200), getattr(self, "gmax", 800)
            self.grid_cols = [str(w) for w in range(self.gmin, self.gmax + 1)]

        def _order_keys(keys):
            def keynum(k):
                n = ''.join(ch for ch in k if ch.isdigit())
                return int(n) if n else 0

            return sorted(keys, key=keynum)

        # ex/prob 컬럼 목록(있으면)
        all_cols = list(self.data.columns)
        ex_cols_all = _order_keys([c for c in all_cols if c.lower().startswith("ex")])
        prob_cols_all = _order_keys([c for c in all_cols if c.lower().startswith("prob")])

        sample_targets = []

        for i in range(len(self.data)):
            row = self.data.loc[i]

            # 1) eV/osc 유효쌍 추출
            ex_vals, prob_vals = None, None
            if ex_cols_all and prob_cols_all:
                K = min(len(ex_cols_all), len(prob_cols_all))
                if K > 0:
                    ex_arr = row[ex_cols_all[:K]].to_numpy(dtype=float)
                    prob_arr = row[prob_cols_all[:K]].to_numpy(dtype=float)
                    m = np.isfinite(ex_arr) & np.isfinite(prob_arr)
                    ex_vals = ex_arr[m]
                    prob_vals = prob_arr[m]

            # 2) eV/osc형인지 판단 (유효쌍 ≥ 1)
            if ex_vals is not None and prob_vals is not None and ex_vals.size > 0:
                # (옵션) 행 단위 정규화: 요청 없으면 None 유지
                if getattr(self, "ex_normalize", None) == "min_max":
                    emn, emx = np.min(ex_vals), np.max(ex_vals)
                    if emx - emn > 1e-8:
                        ex_vals = (ex_vals - emn) / (emx - emn)
                elif getattr(self, "ex_normalize", None) == "std":
                    mu, sd = np.mean(ex_vals), np.std(ex_vals) + 1e-8
                    ex_vals = (ex_vals - mu) / sd

                if getattr(self, "prob_normalize", None) == "min_max":
                    pmn, pmx = np.min(prob_vals), np.max(prob_vals)
                    if pmx - pmn > 1e-8:
                        prob_vals = (prob_vals - pmn) / (pmx - pmn)
                elif getattr(self, "prob_normalize", None) == "std":
                    mu, sd = np.mean(prob_vals), np.std(prob_vals) + 1e-8
                    prob_vals = (prob_vals - mu) / sd

                K = ex_vals.size
                sample_targets.append({
                    "kind": "exprob",
                    "ex": torch.tensor(ex_vals, dtype=torch.float32),  # [K] (eV)
                    "prob": torch.tensor(prob_vals, dtype=torch.float32),  # [K]
                    "mask": torch.ones(K, dtype=torch.bool),  # [K]
                })
            else:
                # 3) 스펙트럼형: 공통 그리드에서 관측값만 정규화
                try:
                    arr = row[self.grid_cols].to_numpy(dtype=float)
                except KeyError:
                    arr = row.reindex(self.grid_cols).to_numpy(dtype=float)
                m = ~np.isnan(arr)
                y = np.zeros_like(arr, dtype=float)
                if m.any():
                    v = arr[m]
                    vmin, vmax = np.min(v), np.max(v)
                    scale = max(vmax - vmin, 1e-8)
                    y[m] = (v - vmin) / scale
                sample_targets.append({
                    "kind": "spec",
                    "y": torch.tensor(y, dtype=torch.float32),  # [P]
                    "mask": torch.tensor(m.astype(bool)),  # [P]
                })

        # 보관 및 반환
        self.sample_targets = sample_targets
        return sample_targets

    def preprocess_graph(self, graph):
        num_nodes = graph["num_nodes"]

        x_cat = torch.from_numpy(graph['x_cat']).long()
        x_cont = torch.from_numpy(graph['x_cont']).float()

        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        in_deg = torch.bincount(edge_index[1], minlength=num_nodes).long()
        out_deg = torch.bincount(edge_index[0], minlength=num_nodes).long()

        adj = torch.from_numpy(graph['adj'])

        attn_edge_type_tensor_dict = {
            k: torch.from_numpy(v).long()
            for k, v in graph['attn_edge_type'].items()
        }

        spatial_pos = torch.from_numpy(graph['spatial_pos']).float()
        attn_bias = torch.zeros((num_nodes, num_nodes), dtype=torch.float)  # Placeholder

        edge_input_tensor_dict = {
            k: torch.from_numpy(v).long()
            for k, v in graph['edge_input'].items()
        }

        g = {
            "x_cat": x_cat,
            "x_cont": x_cont,
            "adj": adj,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "attn_edge_type": attn_edge_type_tensor_dict,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "edge_input": edge_input_tensor_dict,
            "num_nodes": num_nodes,
        }

        # 오직 cls+global_data 모드일 때만 포함
        if self.mode == "cls_global_data": # 글로벌 노드
            global_cat = torch.tensor(graph.get("global_features_cat", []), dtype=torch.long)
            global_cont = torch.tensor(graph.get("global_features_cont", []), dtype=torch.float32)
            g["global_features_cat"] = global_cat
            g["global_features_cont"] = global_cont
        #else:
        #    print("cls only OR cls+global_data mode")
        print(g)
        return g

    def _preprocess_graph_with_optional_global(self, idx, graph, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB):
        if self.mode == "cls_global_data": # global node를 실제로 생성
            global_cat = self._get_global_feature_cat_tensor(idx).tolist()
            global_cont = self._get_global_feature_cont_tensor(idx).tolist()
            return smiles2graph_with_global(
                self.data.loc[idx, self.mol_col],
                global_cat,
                global_cont,
                multi_hop_max_dist=self.multi_hop_max_dist,
                ATOM_FEATURES_VOCAB= ATOM_FEATURES_VOCAB,
                float_feature_keys =float_feature_keys,
                BOND_FEATURES_VOCAB = BOND_FEATURES_VOCAB
            )
        else:
            # print(print("cls only OR cls+global_model mode"))
            return graph

    def _get_global_feature_cat_tensor(self, idx):
        indices = []
        for name in self.nominal_feature_vocab:
            val = self.data.loc[idx, name]
            indices.append(self.nominal_feature_info[name]['value_to_idx'].get(val, 0))
        return torch.tensor(indices, dtype=torch.long)

    def _get_global_feature_cont_tensor(self, idx):
        return torch.tensor([float(self.data.loc[idx, name]) for name in self.continuous_feature_names], dtype=torch.float32)


def collate_fn(batch, ds, n_pairs=None, min_max=None):
    batch = [b for b in batch if b is not None and b[0] is not None]

    if not batch:
        return None

    graphs = [b[0] for b in batch]
    tgt_idx = [b[2] for b in batch]
    task_ids_list = [b[-1] if isinstance(b[-1], torch.Tensor) else torch.tensor(b[-1]) for b in batch]
    task_ids = torch.stack(task_ids_list)
    max_nodes = max(g['num_nodes'] for g in graphs) if graphs else 0

    # ─────────────────────────────────────
    # Global features 처리
    if ds.mode == "cls_global_model":
        global_feat_cat_list = [b[3]["global_features_cat"] for b in batch]
        global_feat_cont_list = [b[3]["global_features_cont"] for b in batch]

    elif ds.mode == "cls_global_data":
        global_feat_cat_list = [g.get('global_features_cat', torch.empty(0, dtype=torch.long)) for g in graphs]
        global_feat_cont_list = [g.get('global_features_cont', torch.empty(0, dtype=torch.float32)) for g in graphs]

    else:
        global_feat_cat_list = []
        global_feat_cont_list = []

    def stack_or_empty(tensor_list, dtype):
        if len(tensor_list) == 0 or tensor_list[0].numel() == 0:
            return torch.zeros((len(tensor_list), 0), dtype=dtype)
        return torch.stack(tensor_list)

    collated_global_features_cat = stack_or_empty(global_feat_cat_list, torch.long)
    collated_global_features_cont = stack_or_empty(global_feat_cont_list, torch.float32)

    # ─────────────────────────────────────
    # Node features
    collated_x_cat = torch.stack([
        torch.nn.functional.pad(g['x_cat'], (0, 0, 0, max_nodes - g['num_nodes']), value=0)
        for g in graphs
    ])
    collated_x_cont = torch.stack([
        torch.nn.functional.pad(g['x_cont'], (0, 0, 0, max_nodes - g['num_nodes']), value=0)
        for g in graphs
    ])

    # ─────────────────────────────────────
    # 기타 그래프 텐서
    adj_list, spatial_pos_list, attn_bias_list, in_degree_list, out_degree_list = [], [], [], [], []
    collated_attn_edge_type = {key: [] for key in graphs[0]['attn_edge_type'].keys()}
    collated_edge_input = {key: [] for key in graphs[0]['edge_input'].keys()}

    for g in graphs:
        pad_len = max_nodes - g['num_nodes']
        adj_list.append(torch.nn.functional.pad(g['adj'], (0, pad_len, 0, pad_len)))
        spatial_pos_list.append(torch.nn.functional.pad(g['spatial_pos'], (0, pad_len, 0, pad_len), value=510))
        attn_bias_list.append(torch.nn.functional.pad(g['attn_bias'], (0, pad_len, 0, pad_len)))
        in_degree_list.append(torch.nn.functional.pad(g['in_degree'], (0, pad_len)))
        out_degree_list.append(torch.nn.functional.pad(g['out_degree'], (0, pad_len)))

        for key, t in g['attn_edge_type'].items():
            D = t.shape[-1]
            pad_t = torch.zeros((max_nodes, max_nodes, D), dtype=torch.long)
            pad_t[:g['num_nodes'], :g['num_nodes'], :] = t
            collated_attn_edge_type[key].append(pad_t)

        for key, t in g['edge_input'].items():
            max_dist = t.shape[2]
            D = t.shape[-1]
            pad_t = torch.zeros((max_nodes, max_nodes, max_dist, D), dtype=t.dtype)
            pad_t[:g['num_nodes'], :g['num_nodes'], :, :] = t
            collated_edge_input[key].append(pad_t)
    collated_attn_edge_type_tensor = torch.cat(
        [torch.stack(collated_attn_edge_type[key]) for key in collated_attn_edge_type], dim=-1
    )
    collated_edge_input_tensor = torch.cat(
        [torch.stack(collated_edge_input[key]) for key in collated_edge_input], dim=-1
    )

    # ─────────────────────────────────────
    res = {
        "x_cat": collated_x_cat,
        "x_cont": collated_x_cont,
        "adj": torch.stack(adj_list),
        "in_degree": torch.stack(in_degree_list),
        "out_degree": torch.stack(out_degree_list),
        "spatial_pos": torch.stack(spatial_pos_list),
        "attn_bias": torch.stack(attn_bias_list),
        "attn_edge_type": collated_attn_edge_type_tensor,
        "edge_input": collated_edge_input_tensor,
        "targets": torch.stack([ds.targets[i] for i in tgt_idx]),
        "task_id": task_ids,              # ★ 추가
    }

    #if ds.target_type == "exp_spectrum":
    #    print("masks in collate_fn", ds.masks.shape)
    #    res.update({"masks":ds.masks})

    if ds.target_type == "exp_spectrum":
        # 배치에 들어온 인덱스(tgt_idx)에 맞춰 마스크만 추출
        mask_batch = torch.as_tensor(ds.masks[tgt_idx], dtype=torch.bool)  # [B, P]

        # y_pred 가 [B, P, 1] 형태라면 차원 맞추기
        mask_batch = mask_batch.unsqueeze(-1)  # [B, P, 1]

        res["masks"] = mask_batch

    # ─────────────────────────────────────
    # Global Feature 모드일 경우만 추가
    if ds.mode in ["cls_global_data", "cls_global_model"]:
        if collated_global_features_cat.numel() > 0:
            res["global_features_cat"] = collated_global_features_cat
        if collated_global_features_cont.numel() > 0:
            res["global_features_cont"] = collated_global_features_cont

    return res

################################################

# 경로
# --- 아주 작은 vocab (원자/결합) ---
ATOM_FEATURES_VOCAB = {
    'atomic_num': list(range(1, 119)),
    'formal_charge': list(range(-3, 4)),
    'hybridization': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.OTHER,
    ],
    'is_aromatic': [0, 1],
    'total_num_hs': list(range(0, 9)),
    'explicit_valence': list(range(0, 9)),
    'total_bonds': list(range(0, 9)),
    # 연속형은 float_feature_keys로 분리
}
float_feature_keys = ['atomic_mass', 'partial_charge']

BOND_FEATURES_VOCAB = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY,
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1],
}

uvvis_csv_path = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\uvvis_fake_train.csv"
fluorescence_csv_path = r'C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\fluorescence_fake_train.csv'
qm_nm_csv_path = r'C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\qm_nm_fake_train.csv'
print("dataset")
ds = UnifiedSMILESDataset(
    nominal_feature_vocab={},             # 글로벌 카테고리 없음
    continuous_feature_names=[],          # 글로벌 연속형 없음
    global_cat_dim=0,
    global_cont_dim=0,
    ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
    float_feature_keys=float_feature_keys,
    BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
    mode="cls",                           # 글로벌 노드 주입 안 함
    max_nodes=128,
    multi_hop_max_dist=5,
    uvvis_csv=uvvis_csv_path,                   # ← 여기만 주면 됩니다
    uv_range=(200, 800),
    fluorescence_csv=fluorescence_csv_path,
    fl_range= (300,700),
    qm_nm_csv= qm_nm_csv_path,
    qm_nm_range=(200,800),
    target_grid=(200, 800),
)

print(f"dataset length: {len(ds)}")
sample = ds[0]

print("== sample keys ==")
for k in sample:
    print("key", k)
    if k == "graph":
        print("graph: {...}")
    else:
        v = sample[k]
        print(k, (tuple(v.shape) if torch.is_tensor(v) else type(v)),)

g = sample["graph"]
print("\n== graph tensor shapes ==")
for k, v in g.items():
    if torch.is_tensor(v):
        print(f"{k:15s} torch.is_tensor", tuple(v.shape), v.dtype)
    elif isinstance(v, dict):
        # dict 텐서들 (attn_edge_type / edge_input) 내부 shape 요약
        inner = {kk: tuple(vv.shape) for kk, vv in v.items()}
        print(f"{k:15s} inner", inner)
    else:
        print(f"{k:15s} else", type(v))

print("\n== target ==")
if sample["kind"] == "spec":
    print("kind = spec")
    print("y:", tuple(sample["y"].shape), sample["y"].dtype)
    print("mask:", tuple(sample["mask"].shape), sample["mask"].dtype)
else:
    print("kind = exprob")
    print("ex:", tuple(sample["ex"].shape))
    print("prob:", tuple(sample["prob"].shape))
    print("mask:", tuple(sample["mask"].shape))
