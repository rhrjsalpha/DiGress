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

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==============================================================================
#  Graph Generation Logic (Previously in Smiles_to_Graph.py)
# ==============================================================================

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
    'global_node': [0, 1],
}


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


def smiles2graph_customized(smiles: str, multi_hop_max_dist: int = 5):
    """
    SMILES 1 개를 Graphormer용 그래프 딕셔너리로
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 혹시 혼합물 있을 경우 배제
    if len(Chem.GetMolFrags(mol)) > 1:
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
        multi_hop_max_dist: int = 5,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) > 1:
        return None

    # ---------- 1) 기존 원자 정보 ----------
    orig = smiles2graph_customized(smiles, multi_hop_max_dist)     # ← 기존 함수 재활용
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
    }

# ==============================================================================
#  Dataloader specific logic starts here
# ==============================================================================

PREDEFINED_VOCAB = {
    'Solvent': [
        '1,4-Dioxane', 'Acetonitrile', 'Benzene', 'Chloroform', 'Cyclohexane',
        'Dichloromethane', 'Dimethylformamide', 'Dimethylsulfoxide', 'Ethanol',
        'Ethylacetate', 'Heptane', 'Hexane', 'Methanol', 'N-Methyl-2-pyrrolidone',
        'Tetrahydrofuran', 'Toluene', 'Water', "DMSO", "Acetone"
    ],
}

def get_global_feature_info(global_feature_names):
    nominal_feature_vocab = {k: v for k, v in PREDEFINED_VOCAB.items() if k in global_feature_names}
    continuous_feature_names_list = [name for name in global_feature_names if name not in nominal_feature_vocab]

    global_cat_dim = 0
    for name in nominal_feature_vocab:  # 명목형
        global_cat_dim += len(nominal_feature_vocab[name])

    global_cont_dim = len(continuous_feature_names_list)  # 수치형

    return nominal_feature_vocab, continuous_feature_names_list, global_cat_dim, global_cont_dim


# ==============================================================================
#  2. Updated SMILESDataset Class
# ==============================================================================
class SMILESDataset(Dataset):
    def __init__(
            self,
            csv_file,
            nominal_feature_vocab,
            continuous_feature_names,
            global_cat_dim,
            global_cont_dim,
            is_global: bool = False,  # <-- New parameter
            max_nodes: int = 128,
            multi_hop_max_dist: int = 5,
            target_type: str = "default",
            attn_bias_w: float = 0.0,
            ex_normalize: str = None,
            prob_normalize: str = None,
            nm_dist_mode: str = "hist",
            nm_gauss_sigma: float = 10.0,
    ):
        try:
            self.data = pd.read_csv(csv_file, encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {csv_file}")

        self.is_global = is_global
        self.nominal_feature_vocab = nominal_feature_vocab
        self.continuous_feature_names = continuous_feature_names
        self.global_feature_names = list(
            nominal_feature_vocab.keys()) + continuous_feature_names  # Reconstruct for _get_all_cols_to_load
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.nominal_feature_info = self._build_nominal_feature_info()

        self._validate_columns(csv_file)
        self.data = self.data.loc[:, self._get_all_cols_to_load()]

        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize
        self.data.iloc[:, 1:101] = self.data.iloc[:, 1:101].apply(pd.to_numeric, errors="coerce").fillna(0)

        # 정규화 관련 #
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
        self.nm_dist_mode = nm_dist_mode
        self.nm_gauss_sigma = nm_gauss_sigma

        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.target_type = target_type
        self.attn_bias_weight = attn_bias_w

        # Generate raw graphs first, filtering out None values from invalid SMILES
        self.raw_graphs = []
        for _, row in self.data.iterrows():
            smi = row["smiles"]

            # ---------- 범주형 글로벌 피처 인덱스 ----------
            g_cat = []
            for feat_name in self.nominal_feature_vocab.keys():
                vocab_info = self.nominal_feature_info[feat_name]
                val = row[feat_name]
                g_cat.append(vocab_info["value_to_idx"].get(val, 0))  # UNK→0

            # ---------- 연속형 글로벌 피처 값 ----------
            g_cont = [float(row[feat]) for feat in self.continuous_feature_names]

            # ---------- 그래프 생성 ----------
            g = smiles2graph_with_global(
                smi, g_cat, g_cont,
                multi_hop_max_dist=self.multi_hop_max_dist
            )
            if g is not None and g["num_nodes"] <= self.max_nodes:
                self.raw_graphs.append(g)  # smiles2graph_customized : smiles -> graph and graph feature

        # Preprocess graphs (will be modified in __getitem__ if is_global is True)
        self.graphs = [self.preprocess_graph(g) for g in self.raw_graphs]  # preprocess_graph : graph_feautre to tensor
        self.targets = self.process_targets()  # tragets to tensor, Normalize targets

    # Removed _add_global_node_raw function as it's handled by the model

    def __getitem__(self, idx):
        tgt = self.targets[idx]

        if self.is_global:
            g_processed = self.preprocess_graph(self.raw_graphs[idx])
            return g_processed, tgt, idx

    def __len__(self):
        return len(self.data)

    def _build_nominal_feature_info(self):
        info = {}
        for feat_name, vocab_list in self.nominal_feature_vocab.items():
            info[feat_name] = {
                'unique_values': vocab_list,
                'value_to_idx': {val: i for i, val in enumerate(vocab_list)}
            }
        return info

    def _get_all_cols_to_load(self):
        required_cols = ["smiles"] + [f"ex{i}" for i in range(1, 51)] + [f"prob{i}" for i in range(1, 51)]
        return required_cols + self.global_feature_names

    def _validate_columns(self, csv_file):
        for col in self._get_all_cols_to_load():
            if col not in self.data.columns:
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
            sorted_idx = np.argsort(-prob, axis=1)
            top_idx = sorted_idx[:, :n_pairs]
            ex_top = np.take_along_axis(ex, top_idx, axis=1)
            prob_top = np.take_along_axis(prob, top_idx, axis=1)
            asc_idx = np.argsort(ex_top, axis=1)
            ex_top = np.take_along_axis(ex_top, asc_idx, axis=1)
            prob_top = np.take_along_axis(prob_top, asc_idx, axis=1)
            stacked = np.stack((ex_top, prob_top), axis=-1)
            return torch.tensor(stacked, dtype=torch.float32)

        elif self.target_type == "nm_distribution":
            ex = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob = self.data[[f"prob{i}" for i in range(1, 51)]].values
            nm = (1239.841984 / ex).round().astype(int)
            nm = np.clip(nm, 150, 600)
            out = np.zeros((len(self.data), 451), dtype=np.float32)

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

        attn_edge_type_tensor_dict = {k: torch.from_numpy(v).long()
                                      for k, v in graph['attn_edge_type'].items()}

        spatial_pos = torch.from_numpy(graph['spatial_pos']).float()

        attn_bias = torch.zeros((num_nodes, num_nodes), dtype=torch.float)  # Placeholder
        print(type(graph['edge_input']))
        edge_input_tensor_dict = {k: torch.from_numpy(v).long() for k, v in graph['edge_input'].items()}

        return {
            "x_cat": x_cat,
            "x_cont": x_cont,
            "adj": adj,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "attn_edge_type": attn_edge_type_tensor_dict,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "edge_input": edge_input_tensor_dict,
            "num_nodes": num_nodes
        }

    # The following methods are no longer needed in SMILESDataset as they are in smiles2graph_customized
    # def generate_edge_input(self, spatial_pos, attn_edge_type, max_dist):
    #     pass

    # @staticmethod
    # def compute_shortest_paths(adj):
    #     pass


# ==============================================================================
#  3. Collate Function and Utils
# ==============================================================================
def pad_tensor_x_dict(x_dict_item, max_n):  # Not Used
    padded_x_dict = {}
    for key, tensor in x_dict_item.items():
        pad_len = max_n - tensor.size(0)
        padded_x_dict[key] = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len), 'constant', 0)
    return padded_x_dict


def pad_tensor_dict(tensor_dict_item, max_n, pad_dim_start, pad_dim_end):  # Not used
    padded_dict = {}
    for key, tensor in tensor_dict_item.items():
        # Assuming tensor shape is (num_elements, feature_dim)
        # Need to pad num_elements dimension
        current_len = tensor.size(pad_dim_start)
        pad_len = max_n - current_len

        # Construct padding tuple based on tensor dimensions
        # (padding_left, padding_right, padding_top, padding_bottom, ...) for each dimension
        padding_tuple = [0] * (tensor.dim() * 2)
        padding_tuple[pad_dim_start * 2 + 1] = pad_len  # Pad at the end of the specified dimension

        padded_dict[key] = torch.nn.functional.pad(tensor, padding_tuple, 'constant', 0)
    return padded_dict


def collate_fn(batch, ds, is_global=False, n_pairs=None, min_max=None):
    """
    1. 그래프마다 노드 수가 다르므로 패딩으로 크기를 맞추고
    2. dict 형태로 흩어져 있던 edge-feature 들을 하나의 큰 Tensor 로 합칩니다.
    """
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None

    graphs = [b[0] for b in batch]
    tgt_idx = [b[2] for b in batch]

    # Always get global_features from g_processed (b[0])
    # g_processed will contain 'global_features' whether is_global is True or False
    global_feat_cat_list = [g.get('global_features_cat', torch.empty(0, dtype=torch.long)) for g in graphs]
    global_feat_cont_list = [g.get('global_features_cont', torch.empty(0, dtype=torch.float32)) for g in graphs]

    max_nodes = max(g['num_nodes'] for g in graphs) if graphs else 0  # 최대 노드 수

    # Collate node features
    collated_x_cat = torch.stack(
        [torch.nn.functional.pad(g['x_cat'], (0, 0, 0, max_nodes - g['num_nodes']), 'constant', 0) for g in
         graphs])  # (B, N_max, F_cat)
    collated_x_cont = torch.stack(
        [torch.nn.functional.pad(g['x_cont'], (0, 0, 0, max_nodes - g['num_nodes']), 'constant', 0) for g in
         graphs])  # (B, N_max, F_cat)

    # Collate global features
    collated_global_features_cat = torch.stack(global_feat_cat_list) if global_feat_cat_list and global_feat_cat_list[
        0].numel() > 0 else torch.empty(0)
    collated_global_features_cont = torch.stack(global_feat_cont_list) if global_feat_cont_list and \
                                                                          global_feat_cont_list[
                                                                              0].numel() > 0 else torch.empty(0)

    # Collate other graph tensors (adj, spatial_pos, attn_bias, attn_edge_type, edge_input)
    adj_list, spatial_pos_list, attn_bias_list, in_degree_list, out_degree_list = [], [], [], [], []
    collated_attn_edge_type = {key: [] for key in graphs[0]['attn_edge_type'].keys()}
    collated_edge_input = {key: [] for key in graphs[0]['edge_input'].keys()}

    for g in graphs:
        pad_len = max_nodes - g['num_nodes']
        adj_list.append(torch.nn.functional.pad(g['adj'], (0, pad_len, 0, pad_len)))  # (N_max,N_max)
        spatial_pos_list.append(
            torch.nn.functional.pad(g['spatial_pos'], (0, pad_len, 0, pad_len), value=510))  # 510 = “없음” 토큰
        attn_bias_list.append(torch.nn.functional.pad(g['attn_bias'], (0, pad_len, 0, pad_len)))
        in_degree_list.append(torch.nn.functional.pad(g['in_degree'], (0, pad_len)))
        out_degree_list.append(torch.nn.functional.pad(g['out_degree'], (0, pad_len)))

        # --- 변경 시작: attn_edge_type 딕셔너리를 단일 텐서로 통합 ---
        for key, t in g['attn_edge_type'].items():
            D = len(BOND_FEATURES_VOCAB[key])
            pad_t = torch.zeros((max_nodes, max_nodes, D), dtype=torch.long)
            pad_t[:g['num_nodes'], :g['num_nodes'], :] = t  # t 는 이미 (N,N,D)
            collated_attn_edge_type[key].append(pad_t)
    feature_tensors_attn_edge_type = []
    for key in collated_attn_edge_type:
        feature_tensors_attn_edge_type.append(torch.stack(collated_attn_edge_type[key]))
    collated_attn_edge_type_tensor = torch.cat(feature_tensors_attn_edge_type, dim=-1)
    # --- 변경 끝 ---

    # --- 변경 시작: edge_input 딕셔너리를 단일 텐서로 통합 ---
    for key, t in g['edge_input'].items():
        max_dist = t.shape[2]
        D = t.shape[-1]
        pad_t = torch.zeros((max_nodes, max_nodes, max_dist, D), dtype=t.dtype)
        pad_t[:g['num_nodes'], :g['num_nodes'], :, :] = t
        collated_edge_input[key].append(pad_t)
    feature_tensors_edge_input = []
    for key in collated_edge_input:
        feature_tensors_edge_input.append(torch.stack(collated_edge_input[key]))
    collated_edge_input_tensor = torch.cat(feature_tensors_edge_input, dim=-1)
    # --- 변경 끝 ---

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
    }

    # Target processing
    targets = torch.stack([ds.targets[i] for i in tgt_idx])
    res['targets'] = targets

    if collated_global_features_cat.numel() > 0:
        res["global_features_cat"] = collated_global_features_cat
    if collated_global_features_cont.numel() > 0:
        res["global_features_cont"] = collated_global_features_cont

    return res


# ==============================================================================
#  4. Simplified Main Execution Block
# ==============================================================================
DEFAULTS = dict(
    mode="both",
    train_file=r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\train_50_with_features.csv",
    test_file=r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\test_10_with_features.csv",
    is_global=True,  # <-- New default
    target_type="nm_distribution",
    batch_size=4,
    ex_norm="none",
    prob_norm="none",
    nm_dist_mode="hist",
    nm_gauss_sigma=10,
    n_pairs=5
)


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
    p = argparse.ArgumentParser("SMILES data pipeline")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action='store_true', default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--target_type", choices=["default", "ex_prob", "nm_distribution"], default=DEFAULTS["target_type"])
    p.add_argument("--ex_norm", choices=["ex_min_max", "ex_std", "none"], default=DEFAULTS["ex_norm"])
    p.add_argument("--prob_norm", choices=["prob_min_max", "prob_std", "none"], default=DEFAULTS["prob_norm"])
    p.add_argument("--nm_dist_mode", choices=["hist", "gauss"], default=DEFAULTS["nm_dist_mode"])
    return p


def run_pipeline(args):
    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']
    nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(
        GLOBAL_FEATURE_NAMES)

    splits = [("train", args.train_file)] if args.mode in ("train", "both") else []
    if args.mode in ("test", "both"):
        splits.append(("test", args.test_file))

    for split, csv in splits:
        print(f"\n===== {split.upper()} | {csv} | is_global={args.is_global} ======")
        ds = SMILESDataset(
            csv_file=csv,
            nominal_feature_vocab=nominal_feature_vocab,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            is_global=args.is_global,
            target_type=args.target_type,
            ex_normalize=args.ex_norm,
            prob_normalize=args.prob_norm,
            nm_dist_mode=args.nm_dist_mode,
            nm_gauss_sigma=args.nm_gauss_sigma,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=(split == "train"),
            collate_fn=lambda b, _ds=ds, _is_global=args.is_global: collate_fn(b, _ds, is_global=_is_global,
                                                                               n_pairs=args.n_pairs)
        )

        for i, batch in enumerate(dl):
            show_batch_shapes(batch, f"Batch {i + 1}")
            # --- 변경 시작: 최종 텐서 shape 출력 ---
            print("\n--- Final Collated Tensor Shapes (from collate_fn) ---")
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"  {k:16s}: {tuple(v.shape)}")
                    print(f"  {k:16s}: {v.dtype}")
                elif isinstance(v, dict):
                    print(f"  {k:16s}: (dict of tensors)")
                    for sub_k, sub_v in v.items():
                        if torch.is_tensor(sub_v):
                            print(f"    {sub_k:14s}: {tuple(sub_v.shape)}")
            # --- 변경 끝 ---
            break


def show_feature_info(csv_path):
    names = [col for col in ['Solvent', 'Temperature', 'Pressure'] if col in pd.read_csv(csv_path).columns]
    nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(names)
    print("\n=== Global-feature info ===")
    print("Nominal feature vocab:", nominal_feature_vocab)
    print("Continuous feature names:", continuous_feature_names)
    print("Global categorical dimension:", global_cat_dim)
    print("Global continuous dimension:", global_cont_dim)


from types import SimpleNamespace

if __name__ == "__main__":
    if len(sys.argv) == 1:
        args = SimpleNamespace(**DEFAULTS)
        # For testing the new mode in IDE
        # args.is_global = True
    else:
        args = build_parser().parse_args()

    run_pipeline(args)
    show_feature_info(args.train_file)

# attn_edge_type : “1-hop” 엣지의 one-hot 예: bond_type, stereo,
# spatial_pos : 두 노드 최단 거리
# attn_bias : 가상노드·거리·엣지 임베딩을 모두 합산하는 “빈 캔버스” 역할
# edge_input : Multi-hop Edge feature 스택

