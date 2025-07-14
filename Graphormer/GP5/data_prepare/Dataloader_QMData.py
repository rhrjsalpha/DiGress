import sys
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from functools import partial

# Add project root to Python path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from Graphormer.GP5.data_prepare.Smiles_to_Graph import smiles2graph as smiles2graph_customized

# ==============================================================================
#  1. Predefined Vocabulary for Nominal Features
# ==============================================================================
# 용매 종류들
PREDEFINED_VOCAB = {
    'Solvent': [
        '1,4-Dioxane', 'Acetonitrile', 'Benzene', 'Chloroform', 'Cyclohexane',
        'Dichloromethane', 'Dimethylformamide', 'Dimethylsulfoxide', 'Ethanol',
        'Ethylacetate', 'Heptane', 'Hexane', 'Methanol', 'N-Methyl-2-pyrrolidone',
        'Tetrahydrofuran', 'Toluene', 'Water', "DMSO", "Acetone"
    ],
}
# 여러개인 경우 다음과 같이 정의
#  PREDEFINED_VOCAB = {
#    2         'Solvent': ['1,4-Dioxane', ..., 'Acetone'],
#    3         'Catalyst': ['Acid', 'Base', 'None'], # 또 다른 명목형 특성 예시
#    4         # ... 필요한 만큼 명목형 특성 추가
#    5     }

def get_global_feature_info(csv_file, global_feature_names):
    # 미리 정의된 값이 없을 경우를 대비해, 데이터의 일부분만 불러와 명목형(feature) 값(범주)을 추론할 수도 있습니다.
    # 지금은 명목형 특성에 대해 PREDEFINED_VOCAB 사전에만 의존합니다.
    # 더 견고한 시스템이라면, PREDEFINED_VOCAB에 없는 경우
    # 'global_feature_names' 열들을 직접 읽어 고유 값을 추출해
    # 범주 목록을 동적으로 생성할 수 있습니다.

    nominal_feature_vocab = {k: v for k, v in PREDEFINED_VOCAB.items() if k in global_feature_names}
    # k = norminal feature column name, v = feature class
    #for k, v in PREDEFINED_VOCAB.items():
    #    if k in global_feature_names:
    #        print(k, v)

    global_dim = 0
    for name in global_feature_names:
        if name in nominal_feature_vocab:
            global_dim += len(nominal_feature_vocab[name]) # 0 또는 1
        else:
            # nominal_feature_vocab 사전에 없으면 숫자형 특성으로 간주
            global_dim += 1 # Each numerical feature adds 1 to dimension

    # 범주1 = [a,b,c] 범주2 = [e,d,f] 숫자특징1, 숫자특징2
    # global_dim = 범주특징1 3개 + 범주특징2 3개 + 숫자특징1 + 숫자특징2 = 8 dimension
    # 범주 특징이 숫자특징에 비해 더 강조되는 문제점 생김 이것을 추후 해결해야함
    # 한 범주형(feature)을 원-핫(one-hot)으로 펼치면 곧바로 N 개의 차원이 생기는데,
    # 숫자형은 그대로 1 차원이라서 모델이 “범주형 정보에 더 많은 가중치를 주기 쉽다” 는 문제가 발생
    return global_dim, nominal_feature_vocab

# ==============================================================================
#  2. Updated SMILESDataset Class
# ==============================================================================
class SMILESDataset(Dataset):
    def __init__(
        self,
        csv_file,
        nominal_feature_vocab,
        global_feature_names,
        max_nodes: int = 128,
        multi_hop_max_dist: int = 5,
        target_type: str = "default",
        attn_bias_w: float = 0.0,
        ex_normalize: str = None,
        prob_normalize: str = None,
        nm_dist_mode: str = "hist",  # "hist" | "gauss"
        nm_gauss_sigma: float = 10.0,  # 가우시안 σ [nm
    ):
        try:
            self.data = pd.read_csv(csv_file, encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {csv_file}")

        self.nominal_feature_vocab = nominal_feature_vocab # 범주형(명목형) 특성 사전
        self.global_feature_names = global_feature_names or [] # 전역(Global) 특성으로 사용할 컬럼명 리스트
        self.nominal_feature_info = self._build_nominal_feature_info() # 위 사전을 바탕으로 unique_values : 범주 목록 , value_to_idx : 값→인덱스 매핑 두 항목을 갖는 dict 생성.

        self._validate_columns(csv_file) # CSV에 필수 열(smiles, ex/ prob, global features) 이 있는지 검사
        self.data = self.data.loc[:, self._get_all_cols_to_load()] # 필요 없는 열 제거

        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize
        self.data.iloc[:, 1:101] = self.data.iloc[:, 1:101].apply(pd.to_numeric, errors="coerce").fillna(0) # 숫자로 강제 변환 숫자로 변환 안 되면 NaN → 0으로
        ex_data = self.data[[f"ex{i}" for i in range(1, 51)]].values # 모든 excitation 값의 전역 min/max 계산
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

        self.max_nodes = max_nodes                   # 한 그래프(분자)의 최대 노드 수 패딩 한계
        self.multi_hop_max_dist = multi_hop_max_dist # attention-bias 입력용 multi-hop 거리 한계
        self.target_type = target_type               # "default", "ex_prob", "nm_distribution" 중 선택
        self.attn_bias_weight = attn_bias_w          # edge 특성을 attention bias로 옮겨올 때 곱해 줄 가중치

        self.graphs = [smiles2graph_customized(s) for s in self.data["smiles"]] # 각 SMILES → 원시 그래프 dict, keys : ['edge_index', 'edge_feat', 'node_feat', 'num_nodes']

        all_edge_feats = torch.cat([
            torch.tensor(g["edge_feat"], dtype=torch.long)
            for g in self.graphs if g["edge_feat"] is not None
        ]) # 모든 분자의 edge_feat 모아서 하나의 텐서로 연결
        self.num_edge_types = torch.unique(all_edge_feats).numel()  # 고유 edge type 개수

        self.graphs = [self.preprocess_graph(g) for g in self.graphs] # Graphormer 입력용 텐서(dict)로 전처리
        self.targets = self.process_targets()

    def _build_nominal_feature_info(self):
        # 범주형 특징 -> 수치형 index 생성
        info = {}
        for feat_name, vocab_list in self.nominal_feature_vocab.items():
            info[feat_name] = {
                'unique_values': vocab_list,
                'value_to_idx': {val: i for i, val in enumerate(vocab_list)}
            }
        print("info:", info)
        return info

    def _get_all_cols_to_load(self):
        # 로딩 해야 하는 컬럼들 선택
        # eV, Osc 이외에 실험 스펙트럼에도 적용 가능하도록 추후 수정해야 함
        required_cols = ["smiles"] + [f"ex{i}" for i in range(1, 51)] + [f"prob{i}" for i in range(1, 51)]
        return required_cols + self.global_feature_names

    def _validate_columns(self, csv_file):
        # csv에서 읽어온 컬럼중 필요한 컬럼이 존재하는지 안하는지 검증
        for col in self._get_all_cols_to_load():
            if col not in self.data.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_file}")

    def __getitem__(self, idx):
        g = self.graphs[idx]
        tgt = self.targets[idx]
        
        proc_globals = []
        for name in self.global_feature_names:
            val = self.data.loc[idx, name]
            if name in self.nominal_feature_info: # 범주형(명목형)일 경우
                vocab_info = self.nominal_feature_info[name]
                vec = torch.zeros(len(vocab_info['unique_values']), dtype=torch.float32) # 범주내 모든 feature에 대해 onehot 벡터 생성, 일단 전부 0으로 처리
                if val in vocab_info['value_to_idx']: # 어떤 값이 존재하면 1, 그렇지 않으면 0
                    vec[vocab_info['value_to_idx'][val]] = 1.0
                else: # 어떤 값이 있는데 predefined vocab 에 없다면 프린트
                    print(f"Warning: Value '{val}' for feature '{name}' not in predefined vocab. Treating as zeros.")
                proc_globals.append(vec)
            else: # 수치형 global feature 일 경우
                proc_globals.append(torch.tensor([float(val)], dtype=torch.float32))

        global_feat = torch.cat(proc_globals) if proc_globals else torch.empty(0)
        return g, tgt, idx, global_feat

    def __len__(self):
        return len(self.data)

    # ... (Rest of the class methods: process_targets, preprocess_graph, etc. are unchanged)
    def process_targets(self, n_pairs=None):
        if self.target_type == "default":
            arr = self.data.iloc[:, 1:101].values
            return torch.tensor(arr, dtype=torch.float32)

        elif self.target_type == "ex_prob": # 흡수강도 상위 n 쌍 만 반환 함
            arr = self.data.iloc[:, 1:101].values
            max_pairs = arr.shape[1] // 2
            if n_pairs is None or n_pairs > max_pairs:
                n_pairs = max_pairs
            ex = arr[:, :max_pairs]
            prob = arr[:, max_pairs:]
            sorted_idx = np.argsort(-prob, axis=1) # 흡수강도 순서로 정렬
            top_idx = sorted_idx[:, :n_pairs] # 가장강한 n쌍만 선택
            ex_top = np.take_along_axis(ex, top_idx, axis=1)
            prob_top = np.take_along_axis(prob, top_idx, axis=1)
            asc_idx = np.argsort(ex_top, axis=1) # eV 순으로 정렬
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

            if self.nm_dist_mode == "hist":  # ── 기존 방식
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    for λ, p in zip(row_nm, row_p):
                        if 150 <= λ <= 600:
                            out[i, λ - 150] += p

            elif self.nm_dist_mode == "gauss":  # ── 가우시안 브로드닝
                bins = np.arange(150, 601)  # (451,)
                σ = self.nm_gauss_sigma
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    spec = np.zeros_like(bins, dtype=np.float32)
                    for λ, p in zip(row_nm, row_p):
                        if 150 <= λ <= 600 and p > 0:
                            kernel = np.exp(-0.5 * ((bins - λ) / σ) ** 2)
                            kernel /= (kernel.sum() + 1e-8)  # 면적=1
                            spec += p * kernel
                    out[i] = spec

            else:
                raise ValueError(f"Unknown nm_dist_mode: {self.nm_dist_mode}, use 'hist' or 'gauss'")
            return torch.tensor(out, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}, use 'default' or 'ex_prob' or 'nm_distribution'")

    def preprocess_graph(self, graph):
        num_nodes = graph["num_nodes"]
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        edge_attr = graph.get("edge_feat")
        x = torch.tensor(graph["node_feat"], dtype=torch.long)

        in_deg = torch.bincount(edge_index[1], minlength=num_nodes)
        out_deg = torch.bincount(edge_index[0], minlength=num_nodes)

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True

        if edge_attr is not None:
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        else:
            edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.long)

        attn_edge_type = torch.zeros(
            (num_nodes, num_nodes, edge_attr.size(-1)), dtype=torch.long
        )
        attn_edge_type[edge_index[0], edge_index[1]] = edge_attr + 1

        spatial_pos = torch.tensor(self.compute_shortest_paths(adj.numpy()), dtype=torch.long)

        attn_bias = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for e_idx, (src, tgt) in enumerate(zip(edge_index[0], edge_index[1])):
            attn_bias[src, tgt] = self.attn_bias_weight * edge_attr[e_idx].sum().float()

        edge_input = self.generate_edge_input(spatial_pos, attn_edge_type, self.multi_hop_max_dist)

        return {
            "x": x,
            "adj": adj,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "edge_input": edge_input,
        }

    def generate_edge_input(self, spatial_pos, attn_edge_type, max_dist):
        n = spatial_pos.size(0)
        edge_in = torch.zeros((n, n, max_dist, attn_edge_type.size(-1)), dtype=torch.long)
        for i in range(n):
            for j in range(n):
                d = spatial_pos[i, j]
                if 1 <= d <= max_dist:
                    edge_in[i, j, d - 1] = attn_edge_type[i, j]
        return edge_in

    @staticmethod
    def compute_shortest_paths(adj):
        # 분자들 내 원자 사이의 최단 거리 계산,
        # 가상 노드, 글로벌 노드등은 Graphormer 내에 존재하며 이들과 원자 사이의 거리는 Grahpormer 내에서 1로 처리됨
        num_nodes = adj.shape[0]
        dist = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(dist, 0)
        for i, j in zip(*np.where(adj)):
            dist[i, j] = 1
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
        return dist

# ==============================================================================
#  3. Collate Function and Utils (Unchanged)
# ==============================================================================
def pad_tensor_x(t, max_n):
    out = torch.zeros((max_n, t.size(1)), dtype=t.dtype)
    out[: t.size(0)] = t
    return out

def pad_tensor_1d(t, max_n):
    out = torch.zeros((max_n,), dtype=t.dtype)
    out[: t.size(0)] = t
    return out

def pad_tensor(t, max_len, pad_dim):
    pad_sizes = [max_len] * pad_dim + list(t.shape[pad_dim:])
    out = torch.zeros(pad_sizes, dtype=t.dtype)
    slices = tuple(slice(0, min(d, max_len)) for d in t.shape)
    out[slices] = t
    return out

def collate_fn(batch, ds, n_pairs=None, min_max=None):
    graphs = [b[0] for b in batch]
    tgt_idx = [b[2] for b in batch]
    globals_feat = torch.stack([b[3] for b in batch]) if batch and batch[0][3].numel() > 0 else None

    max_nodes = max(g["x"].size(0) for g in graphs) if graphs else 0

    x = torch.stack([pad_tensor_x(g["x"], max_nodes) for g in graphs])
    adj = torch.stack([pad_tensor(g["adj"], max_nodes, pad_dim=2) for g in graphs])
    in_deg = torch.stack([pad_tensor_1d(g["in_degree"], max_nodes) for g in graphs])
    out_deg = torch.stack([pad_tensor_1d(g["out_degree"], max_nodes) for g in graphs])
    spatial = torch.stack([pad_tensor(g["spatial_pos"], max_nodes, 2) for g in graphs])
    attn_bias = torch.stack([pad_tensor(g["attn_bias"], max_nodes, 2) for g in graphs])
    attn_et = torch.stack([pad_tensor(g["attn_edge_type"], max_nodes, 3) for g in graphs])
    edge_in = torch.stack([pad_tensor(g["edge_input"], max_nodes, 4) for g in graphs])

    if ds.target_type == "ex_prob":
        all_tg = ds.process_targets(n_pairs=n_pairs)
        targets = torch.stack([all_tg[i] for i in tgt_idx])
        ex, prob = targets[..., 0], targets[..., 1]

        if ds.ex_normalize == "ex_min_max":
            min_v = ds.global_ex_min if min_max is None else min_max[0]
            max_v = ds.global_ex_max if min_max is None else min_max[1]
            ex_norm = (ex - min_v) / (max_v - min_v)
        elif ds.ex_normalize == "ex_std":
            ex_norm = (ex - ds.global_ex_mean) / (ds.global_ex_std + 1e-8)
        elif ds.ex_normalize == "ex":
            ex_norm = ex
        else:
            raise ValueError(f"Unknown ex_normalize: {ds.ex_normalize} \n Avaliable list is ex_min_max, ex_std, ex")

        if ds.prob_normalize == "prob_min_max":
            prob_norm = (prob - ds.global_prob_min) / (ds.global_prob_max - ds.global_prob_min + 1e-8)
        elif ds.prob_normalize == "prob_std":
            prob_norm = (prob - ds.global_prob_mean) / (ds.global_prob_std + 1e-8)
        elif ds.prob_normalize == "prob":
            prob_norm = prob
        else:
            raise ValueError(f"Unknown prob_normalize: {ds.prob_normalize} \n Avaliable list is prob_min_max, prob_std, prob")

        targets = torch.stack([ex_norm, prob_norm], dim=-1)

    else:
        targets = torch.stack([b[1] for b in batch])

    res = {
        "x": x, "adj": adj, "in_degree": in_deg, "out_degree": out_deg,
        "spatial_pos": spatial, "attn_bias": attn_bias,
        "attn_edge_type": attn_et, "edge_input": edge_in, "targets": targets,
    }
    if globals_feat is not None:
        res["global_features"] = globals_feat
    return res




# ==============================================================================
#  4. Simplified Main Execution Block
# ==============================================================================
# ───────────────── default 값 한곳에 모아두기 ─────────────
DEFAULTS = dict(
    mode          = "both",
    train_file    = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\train_50_with_features.csv",
    test_file     = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\test_10_with_features.csv",
    target_type   = "nm_distribution",          # default | ex_prob | nm_distribution
    batch_size    = 4,
    ex_norm       = "none",             # ex_min_max | ex_std | none
    prob_norm     = "none",             # prob_min_max | prob_std | none
    nm_dist_mode  = "hist",            # hist | gauss
    nm_gauss_sigma= 10,                 # 5 | 10 | 15
    n_pairs       = 5                   # ex_prob 에서 상위 몇 쌍
)

# ───────────────── 헬퍼: 배치별 텐서 shape 출력 ────────────────────
def show_batch_shapes(batch, title="Batch"):
    print(f"  ▶ {title}")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"    {k:16s} {tuple(v.shape)}")

# ───────────────── 파서 빌더 (변경 없음) ───────────────────────────
def build_parser():
    p = argparse.ArgumentParser("SMILES data pipeline")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--target_type", choices=["default","ex_prob","nm_distribution"],
                   default=DEFAULTS["target_type"])
    p.add_argument("--ex_norm",   choices=["ex_min_max","ex_std","none"],
                   default=DEFAULTS["ex_norm"])
    p.add_argument("--prob_norm", choices=["prob_min_max","prob_std","none"],
                   default=DEFAULTS["prob_norm"])
    p.add_argument("--nm_dist_mode", choices=["hist","gauss"],
                   default=DEFAULTS["nm_dist_mode"])
    return p

# ───────────────── 메인 파이프라인 ────────────────────────────────
def run_pipeline(args):
    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']
    vocab = {k: v for k, v in PREDEFINED_VOCAB.items() if k in GLOBAL_FEATURE_NAMES}

    splits = [("train", args.train_file)] if args.mode in ("train","both") else []
    if args.mode in ("test","both"):
        splits.append(("test", args.test_file))

    for split, csv in splits:
        print(f"\n===== {split.upper()} | {csv} =====")
        ds = SMILESDataset(
            csv_file=csv,
            nominal_feature_vocab=vocab,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            target_type=args.target_type,
            ex_normalize=args.ex_norm,
            prob_normalize=args.prob_norm,
            nm_dist_mode=args.nm_dist_mode,
            nm_gauss_sigma=args.nm_gauss_sigma,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=(split=="train"),
            collate_fn=lambda b,_ds=ds: collate_fn(b,_ds,n_pairs=args.n_pairs)
        )

        for i, batch in enumerate(dl):
            show_batch_shapes(batch, f"Batch {i+1}")      # ← 추가
            break                                         # 첫 배치만 확인

# ───────────────── 글로벌 feature info ────────────────────────────
def show_feature_info(csv_path):
    df = pd.read_csv(csv_path)
    names = df.columns[-3:]
    dim, vocab = get_global_feature_info(csv_path, names)
    print("\n=== Global-feature info ===")
    print("global_dim :", dim)
    print("vocab      :", vocab)

# ───────────────── entry point ────────────────────────────────────
from types import SimpleNamespace
if __name__ == "__main__":
    if len(sys.argv) == 1:              # IDE 버튼 실행
        args = SimpleNamespace(**DEFAULTS)
    else:                               # 터미널 실행
        args = build_parser().parse_args()

    run_pipeline(args)                  # 파이프라인 + shape 로그
    show_feature_info(args.train_file)  # global feature info 함께 출력