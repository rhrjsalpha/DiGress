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
    'Solvent': [
        '1,4-Dioxane', 'Acetonitrile', 'Benzene', 'Chloroform', 'Cyclohexane',
        'Dichloromethane', 'Dimethylformamide', 'Dimethylsulfoxide', 'Ethanol',
        'Ethylacetate', 'Heptane', 'Hexane', 'Methanol', 'N-Methyl-2-pyrrolidone',
        'Tetrahydrofuran', 'Toluene', 'Water', "DMSO", "Acetone"
    ],
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
    mol = Chem.MolFromSmiles(smiles)
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
    mol = Chem.MolFromSmiles(smiles)
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
        csv_file,
        nominal_feature_vocab,
        continuous_feature_names,
        global_cat_dim,
        global_cont_dim,
        ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB,
        mode="cls",  # "cls", "cls+global_data", "cls+global_model"
        max_nodes: int = 128,
        multi_hop_max_dist: int = 5,
        target_type: str = "default",
        intensity_normalize: str = "min_max", #
        intensity_range: tuple = (200, 800),  #
        attn_bias_w: float = 0.0,
        ex_normalize: str = None,
        prob_normalize: str = None,
        nm_dist_mode: str = "hist",
        nm_gauss_sigma: float = 10.0,

    ):
        self.mode = mode
        self.is_global = mode in ("cls_global_data", "cls_global_model")
        self.nominal_feature_vocab = nominal_feature_vocab
        self.continuous_feature_names = continuous_feature_names
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.target_type = target_type
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist

        # target_type ex_prob #
        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize
        self.nm_dist_mode = nm_dist_mode
        self.nm_gauss_sigma = nm_gauss_sigma

        # target_type experiment #
        self.intensity_normalize = intensity_normalize
        self.intensity_range = intensity_range

        self.attn_bias_weight = attn_bias_w

        self.data = pd.read_csv(csv_file)
        self.nominal_feature_info = self._build_nominal_feature_info()

        self._validate_columns(csv_file)
        self.data = self.data.loc[:, self._get_all_cols_to_load()]

        # Stastics to Normalize ex_prob
        if target_type in ["ex_prob", "nm_distribution"]:
            ex_data = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob_data = self.data[[f"prob{i}" for i in range(1, 51)]].values

            self.global_ex_min = float(np.min(ex_data))
            self.global_ex_max = float(np.max(ex_data))
            self.global_ex_mean = float(np.mean(ex_data))
            self.global_ex_std = float(np.std(ex_data))

            self.global_prob_min = float(np.min(prob_data))
            self.global_prob_max = float(np.max(prob_data))
            self.global_prob_mean = float(np.mean(prob_data))
            self.global_prob_std = float(np.std(prob_data))

        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range
            target_cols = [str(i) for i in range(nm_min, nm_max + 1)]
            existing_cols = [col for col in target_cols if col in self.data.columns]
            missing_cols = set(target_cols) - set(existing_cols)
            for col in missing_cols:
                self.data[col] = 0.0

        else:
            self.global_ex_min = self.global_ex_max = self.global_ex_mean = self.global_ex_std = None
            self.global_prob_min = self.global_prob_max = self.global_prob_mean = self.global_prob_std = None

        #self.raw_graphs = [g for g in [smiles2graph_customized(s, self.multi_hop_max_dist ,ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB, float_feature_keys=float_feature_keys, BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB) for s in self.data["smiles"]] if g is not None]

        # 1. targets, mask 분리
        if self.target_type == "exp_spectrum":
            spectrum, mask_tensor = self.process_targets()
        else:
            spectrum = self.process_targets()
            mask_tensor = None

        # 2. 유효 SMILES 기준으로 graph 추출
        self.raw_graphs = []
        valid_indices = []

        for i, s in enumerate(self.data["smiles"]):
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

        # 3. 대상 필터링
        self.targets = spectrum[valid_indices]
        if mask_tensor is not None:
            self.masks = mask_tensor[valid_indices]  # ← 필요 시 사용

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)

        self.graphs = [self._preprocess_graph_with_optional_global(i, g, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB, ) for i, g in enumerate(self.raw_graphs)]
        for g in self.graphs:
            print("",g.keys())

    def __getitem__(self, idx):
        #print("idx", idx)
        tgt = self.targets[idx]

        # --- Always prepare global features (in case model-internal needs it) ---
        global_feat_cat_indices = []
        for name in self.nominal_feature_vocab:
            val = self.data.loc[idx, name]
            vocab_info = self.nominal_feature_info[name]
            global_feat_cat_indices.append(vocab_info['value_to_idx'].get(val, 0))

        global_feat_cont_values = [float(self.data.loc[idx, name]) for name in self.continuous_feature_names]

        global_feat_cat_tensor = torch.tensor(global_feat_cat_indices, dtype=torch.long)
        global_feat_cont_tensor = torch.tensor(global_feat_cont_values, dtype=torch.float32)

        # --- 그래프 처리 ---
        raw_g = self.raw_graphs[idx]
        g_processed = self.preprocess_graph(raw_g)

        # CLS + GlobalNode를 그래프에 미리 넣은 경우 → g_processed 안에 포함
        if self.mode == "cls_global_data":
            g_processed["global_features_cat"] = global_feat_cat_tensor
            g_processed["global_features_cont"] = global_feat_cont_tensor
            return g_processed, tgt, idx

        # CLS + GlobalNode를 모델에서 처리 → 그래프에는 안넣음
        elif self.mode == "cls_global_model":
            return g_processed, tgt, idx, {
                "global_features_cat": global_feat_cat_tensor,
                "global_features_cont": global_feat_cont_tensor
            }

        # CLS-only → global_feature 사용하지 않음
        else:
            return g_processed, tgt, idx

    def __len__(self):
        return len(self.graphs)

    def _preprocess_graph_with_optional_global(self, idx, graph, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB):
        if self.mode == "cls_global_data":
            global_cat = self._get_global_feature_cat_tensor(idx).tolist()
            global_cont = self._get_global_feature_cont_tensor(idx).tolist()
            return smiles2graph_with_global(
                self.data.loc[idx, "smiles"],
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

    def _build_nominal_feature_info(self):
        return {
            name: {
                'unique_values': vocab,
                'value_to_idx': {val: i for i, val in enumerate(vocab)}
            } for name, vocab in self.nominal_feature_vocab.items()
        }

    def _get_all_cols_to_load(self):
        if self.target_type in ["ex_prob", "nm_distribution"]:
            target_cols = [f"ex{i}" for i in range(1, 51)] + [f"prob{i}" for i in range(1, 51)]
        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range  # 예: (200, 800)
            all_columns = self.data.columns
            target_cols = []
            for i in range(nm_min, nm_max + 1):
                target_cols.append(str(i))
        else:
            target_cols = []
        required_cols = ["smiles"] + target_cols

        return required_cols + list(self.nominal_feature_vocab.keys()) + self.continuous_feature_names

    def _validate_columns(self, csv_file):
        for col in self._get_all_cols_to_load():
            #print(self._get_all_cols_to_load())
            if col not in self.data.columns:
                #print(self.data.columns)
                raise ValueError(f"Missing required column '{col}' in {csv_file}")

    def process_targets(self, n_pairs=None):
        if self.target_type == "default":
            arr = self.data.iloc[:, 1:101].values
            return torch.tensor(arr, dtype=torch.float32)

        elif self.target_type == "ex_prob":
            arr = self.data.iloc[:, 1:101].values
            max_pairs = arr.shape[1] // 2
            if n_pairs is None or n_pairs > max_pairs:
                n_pairs = max_pairs
            ex = arr[:, :max_pairs]
            prob = arr[:, max_pairs:]

            # 1. intensity 기준 정렬
            sorted_idx = np.argsort(-prob, axis=1)
            top_idx = sorted_idx[:, :n_pairs]
            ex_top = np.take_along_axis(ex, top_idx, axis=1)
            prob_top = np.take_along_axis(prob, top_idx, axis=1)

            # 2. eV 기준 정렬
            asc_idx = np.argsort(ex_top, axis=1)
            ex_top = np.take_along_axis(ex_top, asc_idx, axis=1)
            prob_top = np.take_along_axis(prob_top, asc_idx, axis=1)

            # 4. ex 정규화
            if self.ex_normalize == "ex_min_max":
                ex_top = (ex_top - self.global_ex_min) / (self.global_ex_max - self.global_ex_min + 1e-8)
            elif self.ex_normalize == "ex_std":
                ex_top = (ex_top - self.global_ex_mean) / (self.global_ex_std + 1e-8)
            elif self.ex_normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown ex_normalize: {self.ex_normalize}")

            # 5. prob 정규화
            if self.prob_normalize == "prob_min_max":
                prob_top = (prob_top - self.global_prob_min) / (self.global_prob_max - self.global_prob_min + 1e-8)
            elif self.prob_normalize == "prob_std":
                prob_top = (prob_top - self.global_prob_mean) / (self.global_prob_std + 1e-8)
            elif self.prob_normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown prob_normalize: {self.prob_normalize}")

            stacked = np.stack((ex_top, prob_top), axis=-1)
            return torch.tensor(stacked, dtype=torch.float32)

        elif self.target_type == "nm_distribution":
            ex = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob = self.data[[f"prob{i}" for i in range(1, 51)]].values
            nm = (1239.841984 / ex).round().astype(int)

            if self.intensity_normalize == "min_max":
                prob = (prob - self.global_prob_min) / (self.global_prob_max - self.global_prob_min + 1e-8)

            nm_min, nm_max = self.intensity_range
            nm = np.clip(nm, nm_min, nm_max)
            spec_len = nm_max - nm_min + 1
            out = np.zeros((len(self.data), spec_len), dtype=np.float32)

            if self.nm_dist_mode == "hist":
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    for λ, p in zip(row_nm, row_p):
                        if 150 <= λ <= 600:
                            out[i, λ - 150] += p

            elif self.nm_dist_mode == "gauss":
                bins = np.arange(150, 601)
                σ = self.nm_gauss_sigma
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    spec = np.zeros_like(bins, dtype=np.float32)
                    for λ, p in zip(row_nm, row_p):
                        if 150 <= λ <= 600 and p > 0:
                            kernel = np.exp(-0.5 * ((bins - λ) / σ) ** 2)
                            kernel /= (kernel.sum() + 1e-8)
                            spec += p * kernel
                    out[i] = spec
            else:
                raise ValueError(f"Unknown nm_dist_mode: {self.nm_dist_mode}, use 'hist' or 'gauss'")
            return torch.tensor(out, dtype=torch.float32)

        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range
            target_cols = [str(i) for i in range(nm_min, nm_max + 1)]

            # 존재하는 컬럼만 선택
            existing_cols = [col for col in target_cols if col in self.data.columns]
            missing_cols = set(target_cols) - set(existing_cols)
            if missing_cols:
                print(f"[Warning] {len(missing_cols)} missing columns will be filled with zeros")

            # 누락된 컬럼 0으로 채워 넣기
            for col in missing_cols:
                self.data[col] = 0.0

            spectrum = self.data[target_cols].fillna(0.0).values

            # 개별 스펙트럼별 정규화
            normed = []
            masks = []
            for row in spectrum:
                mask = (row != 0)
                if np.sum(mask) == 0:
                    normed.append(np.zeros_like(row))
                else:
                    valid_vals = row[mask]
                    row_min, row_max = np.min(valid_vals), np.max(valid_vals)
                    row_range = row_max - row_min + 1e-8
                    norm_row = np.zeros_like(row)
                    norm_row[mask] = (valid_vals - row_min) / row_range
                    normed.append(norm_row)

                masks.append(mask.astype(np.float32))

            spectrum = np.stack(normed)
            spectrum = torch.tensor(spectrum, dtype=torch.float32)
            mask_tensor = torch.tensor(np.stack(masks), dtype=torch.float32)  # 또는 dtype=torch.bool
            return spectrum, mask_tensor

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

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
        if self.mode == "cls_global_data":
            global_cat = torch.tensor(graph.get("global_features_cat", []), dtype=torch.long)
            global_cont = torch.tensor(graph.get("global_features_cont", []), dtype=torch.float32)
            g["global_features_cat"] = global_cat
            g["global_features_cont"] = global_cont
        #else:
        #    print("cls only OR cls+global_data mode")

        return g

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
        "targets": torch.stack([ds.targets[i] for i in tgt_idx])
    }

    if ds.target_type == "exp_spectrum":
        res.update({"masks":ds.masks})

    # ─────────────────────────────────────
    # Global Feature 모드일 경우만 추가
    if ds.mode in ["cls_global_data", "cls_global_model"]:
        if collated_global_features_cat.numel() > 0:
            res["global_features_cat"] = collated_global_features_cat
        if collated_global_features_cont.numel() > 0:
            res["global_features_cont"] = collated_global_features_cont

    return res

#### 실행 코드 ####


def show_batch_shapes(batch, title="Batch"):
    print(f"  ▶ {title}")
    for k, v in batch.items():
        if isinstance(v, dict):
            print(f"    {k:16s} (dict of tensors)")
            for sub_k, sub_v in v.items():
                if torch.is_tensor(sub_v):
                    print(f"      {sub_k:14s} {tuple(sub_v.shape)}")
        elif torch.is_tensor(v):
            print(f"    {k:16s} {tuple(v.shape)}")

def build_parser():
    p = argparse.ArgumentParser("UnifiedSMILESDataset pipeline")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["cls", "cls_global_data", "cls_global_model"], default="cls")
    p.add_argument("--target_type", choices=["default", "ex_prob", "nm_distribution"], default="default")
    p.add_argument("--ex_norm", choices=["ex_min_max", "ex_std", "none"], default="none")
    p.add_argument("--prob_norm", choices=["prob_min_max", "prob_std", "none"], default="none")
    p.add_argument("--nm_dist_mode", choices=["hist", "gauss"], default="hist")
    p.add_argument("--nm_gauss_sigma", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_nodes", type=int, default=128)
    p.add_argument("--multi_hop_max_dist", type=int, default=5)
    return p

def run_pipeline(args):
    ATOM_FEATURES_VOCAB = {
        'atomic_num': list(range(1, 119)),  # TODO I need to decrease the range
        'formal_charge': list(range(-5, 6)),  # increase range when diffusion / or add threshold
        'hybridization': [
            Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER,
            # Chem.rdchem.HybridizationType.UNSPECIFIED # add this when diffusion
        ],
        'is_aromatic': [0, 1],
        'total_num_hs': list(range(0, 9)),  # increase it when diffusion
        'explicit_valence': list(range(0, 8)),  # increase range when diffusion / or add threshold of valence encoding
        'total_bonds': list(range(0, 8)),  # increase range when diffusion / or add threshold
        'partial_charge': float,  # check error and change code when diffusion
        'atomic_mass': float,  # OK when diffusion
    }

    float_feature_keys = ['partial_charge', 'atomic_mass']

    BOND_FEATURES_VOCAB = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
            # Chem.rdchem.BondType.UNSPECIFIED # add this when diffusion
        ],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,  # OK when diffusion
        ],
        'is_conjugated': [0, 1],  # OK when diffusion
        'is_in_ring': [0, 1],  # OK when diffusion
    }

    from types import SimpleNamespace
    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']
    nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(
        GLOBAL_FEATURE_NAMES, PREDEFINED_VOCAB
    )

    if args.mode not in ["cls", "cls_global_data", "cls+global_model"]:
        raise ValueError("Invalid mode: choose from 'cls', 'cls+global_data', 'cls+global_model'")

    ds = UnifiedSMILESDataset(
        csv_file=args.train_file,
        nominal_feature_vocab=nominal_feature_vocab,
        continuous_feature_names=continuous_feature_names,
        global_cat_dim=global_cat_dim,
        global_cont_dim=global_cont_dim,
        mode=args.mode,
        max_nodes=args.max_nodes,
        multi_hop_max_dist=args.multi_hop_max_dist,
        target_type=args.target_type,
        ex_normalize=args.ex_norm,
        prob_normalize=args.prob_norm,
        nm_dist_mode=args.nm_dist_mode,
        nm_gauss_sigma=args.nm_gauss_sigma,
        ATOM_FEATURES_VOCAB = ATOM_FEATURES_VOCAB,
        float_feature_keys = float_feature_keys,
        BOND_FEATURES_VOCAB = BOND_FEATURES_VOCAB,
        intensity_normalize =  args.intensity_normalize,  #
        intensity_range =  args.intensity_range,  #
        attn_bias_w = args.attn_bias_w,
    )


    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b, _ds=ds: collate_fn(b, _ds)
    )

    for i, batch in enumerate(dl):
        print(f"\n===== Mode: {args.mode} | Batch {i + 1} =====")
        show_batch_shapes(batch, f"Batch {i + 1}")
        break

from types import SimpleNamespace
if __name__ == "__main__":
    file_path_1 = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\train_50_with_features.csv"
    file_path_2 = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\GP5\data_prepare\fake_exp_like_data_from_QM9Snm_1nm_last_withGlobalFeature_100data.csv"
    file_paht_exp = r"C:\Users\analcheminfo\PycharmProjects\DiGress\Graphormer\graphormer_data\NIST_with_fake_golbal.csv"
    #df = pd.read_csv(file_path_2)
    #df[0:100].to_csv("fake_exp_like_data_from_QM9Snm_1nm_last_withGlobalFeature_100data.csv")
    #print(df.head())
    if len(sys.argv) == 1:
        args = SimpleNamespace(
            train_file= file_paht_exp,
            mode="cls_global_data",
            target_type="exp_spectrum", # nm_distribution, ex_prob, exp_spectrum
            ex_norm="none",
            prob_norm="none",
            nm_dist_mode="hist",
            nm_gauss_sigma=10.0,
            batch_size=10,
            max_nodes=128,
            multi_hop_max_dist=5,
            nominal_feature_vocab=PREDEFINED_VOCAB,
            continuous_feature_names=['pressure_atm', 'temperature_K'],
            intensity_normalize="min_max",  #
            intensity_range=(1, 10),  #
            attn_bias_w=0.0,
        )
    else:
        args = build_parser().parse_args()

    run_pipeline(args)


