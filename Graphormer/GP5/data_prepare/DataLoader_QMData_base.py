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
    'atomic_num': list(range(1, 119)),
    'formal_charge': list(range(-5, 6)),
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER
    ],
    'is_aromatic': [0, 1],
    'total_num_hs': list(range(0, 9)),
    'explicit_valence': list(range(0, 8)),
    'total_bonds'     : list(range(0, 8)),
    'partial_charge': float,
    'atomic_mass': float,
}

float_feature_keys = ['partial_charge', 'atomic_mass']

BOND_FEATURES_VOCAB = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1],
}

def _get_feature_index(value, vocab):
    if value in vocab:
        return vocab.index(value)
    return vocab.index(vocab[0]) # Default to the first element if not found

def _compute_shortest_paths(adj):
    num_nodes = adj.shape[0]
    dist = np.full((num_nodes, num_nodes), -1, dtype=int)
    np.fill_diagonal(dist, 0)
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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 혹시 혼합물 있을 경우 배제
    if len(Chem.GetMolFrags(mol)) > 1:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except:
        print("partial charge calculation failed")
        pass

    num_nodes = mol.GetNumAtoms()
    adj = np.zeros((num_nodes, num_nodes), dtype=bool)

    node_features_cat = {key: [] for key in ATOM_FEATURES_VOCAB if isinstance(ATOM_FEATURES_VOCAB[key], list)}
    node_features_cont = {key: [] for key in float_feature_keys}


    for atom in mol.GetAtoms():
        for key, vocab_or_type in ATOM_FEATURES_VOCAB.items():
            if isinstance(vocab_or_type, list):
                if key == 'atomic_num':   prop = atom.GetAtomicNum()
                elif key == 'formal_charge': prop = atom.GetFormalCharge()
                elif key == 'hybridization': prop = atom.GetHybridization()
                elif key == 'is_aromatic':   prop = int(atom.GetIsAromatic())
                elif key == 'total_num_hs':  prop = atom.GetTotalNumHs()
                elif key == 'explicit_valence': prop = atom.GetExplicitValence()
                elif key == 'total_bonds':
                    prop = atom.GetTotalDegree()
                node_features_cat[key].append(_get_feature_index(prop, vocab_or_type))
            elif vocab_or_type is float:
                if key == 'atomic_mass': node_features_cont[key].append(atom.GetMass())
                elif key == 'partial_charge':
                    try:
                        charge = float(atom.GetProp('_GasteigerCharge'))
                        node_features_cont[key].append(charge)
                    except(KeyError, ValueError):
                        node_features_cont[key].append(0.0)

    # Combine categorical features into a single integer array
    x_cat = np.stack(list(node_features_cat.values()), axis=-1)

    # Combine continuous features into a single float array
    x_cont = np.stack(list(node_features_cont.values()), axis=-1)

    attn_edge_type = {
        k: np.zeros((num_nodes, num_nodes, len(vocab)), dtype=np.int64)  # (N,N,D)
        for k, vocab in BOND_FEATURES_VOCAB.items()
    }
    edge_indices = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = True
        edge_indices.extend([[i, j], [j, i]])

        for key, vocab in BOND_FEATURES_VOCAB.items():
            if key == 'bond_type':     prop = bond.GetBondType()
            elif key == 'stereo':        prop = bond.GetStereo()
            elif key == 'is_conjugated': prop = int(bond.GetIsConjugated())
            elif key == 'is_in_ring':    prop = int(bond.IsInRing())

            idx = _get_feature_index(prop, vocab)
            attn_edge_type[key][i, j] = 1
            attn_edge_type[key][j, i] = 1

    spatial_pos = _compute_shortest_paths(adj)
    #print("spatial_pos",spatial_pos)

    # np.inf 값이 포함된 분자(연결되지 않은 구성 요소를 가진 분자)는 제거
    #if np.isinf(spatial_pos).any():
    #    return None

    edge_input = {
        key: np.zeros(
            (num_nodes, num_nodes, multi_hop_max_dist, len(vocab)),  # ← 4-D 로!
            dtype=np.int64
        )
        for key, vocab in BOND_FEATURES_VOCAB.items()
    }
    # 2) 값 복사
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = spatial_pos[i, j]
            if 1 <= dist < multi_hop_max_dist:
                for key in BOND_FEATURES_VOCAB.keys():
                    print(edge_input[key].shape)
                    print(attn_edge_type[key].shape)
                    edge_input[key][i, j, dist - 1, :] = attn_edge_type[key][i, j, :]

    edge_input[key][i, j, dist - 1] = attn_edge_type[key][i, j]

    return {
        'x_cat': x_cat,
        'x_cont': x_cont,
        'adj': adj,
        'edge_index': np.array(edge_indices).T if edge_indices else np.empty((2, 0), dtype=int),
        'attn_edge_type': attn_edge_type, # Now a dict of arrays
        'spatial_pos': spatial_pos,
        'edge_input': edge_input, # Now a dict of arrays
        'num_nodes': num_nodes,
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
    for name in nominal_feature_vocab:
        global_cat_dim += len(nominal_feature_vocab[name])

    global_cont_dim = len(continuous_feature_names_list)

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
        is_global: bool = False, #<-- New parameter
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
        self.global_feature_names = list(nominal_feature_vocab.keys()) + continuous_feature_names # Reconstruct for _get_all_cols_to_load
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.nominal_feature_info = self._build_nominal_feature_info()

        self._validate_columns(csv_file)
        self.data = self.data.loc[:, self._get_all_cols_to_load()]

        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize
        self.data.iloc[:, 1:101] = self.data.iloc[:, 1:101].apply(pd.to_numeric, errors="coerce").fillna(0)
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
        self.raw_graphs = [g for g in [smiles2graph_customized(s) for s in self.data["smiles"]] if g is not None]

        # Preprocess graphs (will be modified in __getitem__ if is_global is True)
        self.graphs = [self.preprocess_graph(g) for g in self.raw_graphs]
        self.targets = self.process_targets()

    # Removed _add_global_node_raw function as it's handled by the model

    def __getitem__(self, idx):
        tgt = self.targets[idx]

        # --- Process global features ---
        global_feat_cat_indices = []
        for name in self.nominal_feature_vocab.keys():
            val = self.data.loc[idx, name]
            vocab_info = self.nominal_feature_info[name]
            if val in vocab_info['value_to_idx']:
                global_feat_cat_indices.append(vocab_info['value_to_idx'][val])
            else:
                # Handle unseen nominal values by mapping to a default (e.g., 0 or a special UNK token)
                global_feat_cat_indices.append(0) # Assuming 0 is a safe default/padding_idx

        global_feat_cont_values = []
        for name in self.continuous_feature_names:
            val = self.data.loc[idx, name]
            global_feat_cont_values.append(float(val))

        global_feat_cat_tensor = torch.tensor(global_feat_cat_indices, dtype=torch.long)
        global_feat_cont_tensor = torch.tensor(global_feat_cont_values, dtype=torch.float32)

        # --- Main logic based on is_global ---
        if self.is_global:
            raw_g = self.raw_graphs[idx]
            g_processed = self.preprocess_graph(raw_g)
            g_processed['global_features_cat'] = global_feat_cat_tensor
            g_processed['global_features_cont'] = global_feat_cont_tensor
            return g_processed, tgt, idx
        else:
            g_processed = self.graphs[idx]
            return g_processed, tgt, idx, {'global_features_cat': global_feat_cat_tensor, 'global_features_cont': global_feat_cont_tensor}

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

        attn_bias = torch.zeros((num_nodes, num_nodes), dtype=torch.float) # Placeholder

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
def pad_tensor_x_dict(x_dict_item, max_n):
    padded_x_dict = {}
    for key, tensor in x_dict_item.items():
        pad_len = max_n - tensor.size(0)
        padded_x_dict[key] = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len), 'constant', 0)
    return padded_x_dict

def pad_tensor_dict(tensor_dict_item, max_n, pad_dim_start, pad_dim_end):
    padded_dict = {}
    for key, tensor in tensor_dict_item.items():
        # Assuming tensor shape is (num_elements, feature_dim)
        # Need to pad num_elements dimension
        current_len = tensor.size(pad_dim_start)
        pad_len = max_n - current_len
        
        # Construct padding tuple based on tensor dimensions
        # (padding_left, padding_right, padding_top, padding_bottom, ...) for each dimension
        padding_tuple = [0] * (tensor.dim() * 2)
        padding_tuple[pad_dim_start * 2 + 1] = pad_len # Pad at the end of the specified dimension
        
        padded_dict[key] = torch.nn.functional.pad(tensor, padding_tuple, 'constant', 0)
    return padded_dict


def collate_fn(batch, ds, is_global=False, n_pairs=None, min_max=None):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None

    graphs = [b[0] for b in batch]
    tgt_idx = [b[2] for b in batch]

    # Always get global_features from g_processed (b[0])
    # g_processed will contain 'global_features' whether is_global is True or False
    global_feat_cat_list = [g.get('global_features_cat', torch.empty(0, dtype=torch.long)) for g in graphs]
    global_feat_cont_list = [g.get('global_features_cont', torch.empty(0, dtype=torch.float32)) for g in graphs]

    max_nodes = max(g['num_nodes'] for g in graphs) if graphs else 0

    # Collate node features
    collated_x_cat = torch.stack([torch.nn.functional.pad(g['x_cat'], (0, 0, 0, max_nodes - g['num_nodes']), 'constant', 0) for g in graphs])
    collated_x_cont = torch.stack([torch.nn.functional.pad(g['x_cont'], (0, 0, 0, max_nodes - g['num_nodes']), 'constant', 0) for g in graphs])

    # Collate global features
    collated_global_features_cat = torch.stack(global_feat_cat_list) if global_feat_cat_list and global_feat_cat_list[0].numel() > 0 else torch.empty(0)
    collated_global_features_cont = torch.stack(global_feat_cont_list) if global_feat_cont_list and global_feat_cont_list[0].numel() > 0 else torch.empty(0)

    # Collate other graph tensors (adj, spatial_pos, attn_bias, attn_edge_type, edge_input)
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

        # --- 변경 시작: attn_edge_type 딕셔너리를 단일 텐서로 통합 ---
        for key, t in g['attn_edge_type'].items():
            D = len(BOND_FEATURES_VOCAB[key])
            pad_t = torch.zeros((max_nodes, max_nodes, D), dtype=torch.long)
            pad_t[:g['num_nodes'], :g['num_nodes'], :] = t   # t 는 이미 (N,N,D)
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
    mode          = "both",
    train_file    = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\train_50_with_features.csv",
    test_file     = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\test_10_with_features.csv",
    is_global     = True, #<-- New default
    target_type   = "nm_distribution",
    batch_size    = 4,
    ex_norm       = "none",
    prob_norm     = "none",
    nm_dist_mode  = "hist",
    nm_gauss_sigma= 10,
    n_pairs       = 5
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
    p.add_argument("--target_type", choices=["default","ex_prob","nm_distribution"], default=DEFAULTS["target_type"])
    p.add_argument("--ex_norm",   choices=["ex_min_max","ex_std","none"], default=DEFAULTS["ex_norm"])
    p.add_argument("--prob_norm", choices=["prob_min_max","prob_std","none"], default=DEFAULTS["prob_norm"])
    p.add_argument("--nm_dist_mode", choices=["hist","gauss"], default=DEFAULTS["nm_dist_mode"])
    return p

def run_pipeline(args):
    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']
    nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(GLOBAL_FEATURE_NAMES)

    splits = [("train", args.train_file)] if args.mode in ("train","both") else []
    if args.mode in ("test","both"):
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
            ds, batch_size=args.batch_size, shuffle=(split=="train"),
            collate_fn=lambda b, _ds=ds, _is_global=args.is_global: collate_fn(b, _ds, is_global=_is_global, n_pairs=args.n_pairs)
        )

        for i, batch in enumerate(dl):
            show_batch_shapes(batch, f"Batch {i+1}")
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