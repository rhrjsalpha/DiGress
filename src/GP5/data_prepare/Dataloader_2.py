import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ogb.utils.mol import smiles2graph
from Graphormer.GP5.data_prepare.Smiles_to_Graph import smiles2graph as smiles2graph_customized
import numpy as np
import torch.nn as nn

class SMILESDataset(Dataset):
    def __init__(self, csv_file, max_nodes=128, multi_hop_max_dist=5, target_type="default", attn_bias_w=0):
        """
        A dataset to handle SMILES strings and their target values.

        Args:
            csv_file: Path to the CSV file containing 'smiles' and target values.
            max_nodes: Maximum number of nodes in a graph (used for padding).
            multi_hop_max_dist: Maximum distance for multi-hop edges.
            attn_bias_w: Weight for attention bias based on edge features.
        """
        self.data = pd.read_csv(csv_file)
        print(self.data['smiles'].loc[1])

        # Only keep the relevant columns: 'smiles', 'ex1~50', 'prob1~50'
        self.data = self.data.loc[:, ["smiles"] + [f"ex{i}" for i in range(1, 51)] + [f"prob{i}" for i in range(1, 51)]]

        # Ensure target columns are numeric and fill NaN values with 0
        self.data.iloc[:, 1:] = self.data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0)

        # "ex" 데이터 전체를 numpy 배열로 불러오기
        ex_data = self.data[[f"ex{i}" for i in range(1, 51)]].values

        # global min과 global max 계산
        self.global_ex_min = np.min(ex_data)
        self.global_ex_max = np.max(ex_data)

        print(f"Global ex min: {self.global_ex_min}, max: {self.global_ex_max}")

        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.target_type = target_type
        self.attn_bias_weight = attn_bias_w

        # Process SMILES into graph structures
        self.graphs = [smiles2graph_customized(smiles) for smiles in self.data["smiles"]] ## 변경 ##

        # Compute unique edge types across all graphs
        all_edge_feats = torch.cat([
            torch.tensor(graph["edge_feat"], dtype=torch.long) for graph in self.graphs
            if graph["edge_feat"] is not None
        ])
        self.num_edge_types = torch.unique(all_edge_feats).numel()

        # Initialize edge encoder based on unique edge types
        self.edge_encoder = nn.Embedding(self.num_edge_types + 1, 128, padding_idx=0)

        # Preprocess graphs and process targets
        self.graphs = [self.preprocess_graph(graph) for graph in self.graphs]
        self.targets = self.process_targets()


    def process_targets(self,  n_pairs=None):
        """
        Process the targets based on the target_type.

        Returns:
            Processed targets as a torch tensor.
        """
        if self.target_type == "default": #새TOP N 개 뽑는 코드 필요
            # Default targets split into ex and prob
            targets = self.data.iloc[:, 1:].values  # Extract all target columns
            print("shape default",torch.tensor(targets, dtype=torch.float32).shape)
            return torch.tensor(targets, dtype=torch.float32)

        elif self.target_type == "ex_prob":
            # Case 2: Targets as [ex, prob] pairs
            targets = self.data.iloc[:, 1:].values
            max_pairs = targets.shape[1] // 2  # Maximum available pairs (e.g., 50 if 100 columns)
            if n_pairs is None or n_pairs > max_pairs:
                n_pairs = max_pairs  # Use all pairs if n_pairs is not specified or exceeds max_pairs

            ex = targets[:, :max_pairs]  # First max_pairs columns for 'ex'
            prob = targets[:, max_pairs:]  # Corresponding max_pairs columns for 'prob'

            # Sort by prob values in descending order and take top n_pairs
            sorted_indices = np.argsort(-prob, axis=1)  # Sort indices by descending prob
            #print(sorted_indices)
            top_indices = sorted_indices[:, :n_pairs]  # Select top n_pairs indices

            # Select corresponding ex and prob values
            sorted_ex = np.take_along_axis(ex, top_indices, axis=1) # top_indices 에는 몇번째인지 인덱스 정보가 존재
            sorted_prob = np.take_along_axis(prob, top_indices, axis=1)

            # 다시 ex 기준으로 오름차순 정렬
            ascending_order_indices = np.argsort(sorted_ex, axis=1)
            sorted_ex = np.take_along_axis(sorted_ex, ascending_order_indices, axis=1)
            sorted_prob = np.take_along_axis(sorted_prob, ascending_order_indices, axis=1)

            # Stack along the last dimension to create [ex, prob] pairs
            stacked_targets = np.stack((sorted_ex, sorted_prob), axis=-1)  # Shape: [num_samples, n_pairs, 2]
            #print("SMILES_Dataset npair stacked_targets",stacked_targets)
            return torch.tensor(stacked_targets, dtype=torch.float32)

        elif self.target_type == "nm_distribution":
            # Case 3: Convert eV to nm and create distribution from 100nm to 800nm
            ex = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob = self.data[[f"prob{i}" for i in range(1, 51)]].values
            nm = (1239.841984 / ex).round().astype(int)  # Convert eV to nm and round
            nm = np.clip(nm, 150, 600)  # Clip to range 150-600 nm
            targets = np.zeros((len(self.data), 451), dtype=np.float32)
            for i, (nm_row, prob_row) in enumerate(zip(nm, prob)):
                for nm_val, prob_val in zip(nm_row, prob_row):
                    if 150 <= nm_val <= 650:
                        targets[i, nm_val - 150] += prob_val
            new_tensor = torch.tensor(targets, dtype=torch.float32)
            print(new_tensor.shape)
            return new_tensor

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

    def preprocess_graph(self, graph):
        """
        Preprocess a single graph and return its components.

        Args:
            graph: A dictionary returned by smiles2graph.

        Returns:
            A dictionary containing preprocessed graph data.
        """
        num_nodes = graph["num_nodes"]
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        edge_attr = graph.get("edge_feat", None)

        # Base features (node features)
        x = torch.tensor(graph["node_feat"], dtype=torch.long)

        # In-degree and out-degree calculation
        in_degree = torch.bincount(edge_index[1], minlength=num_nodes)
        out_degree = torch.bincount(edge_index[0], minlength=num_nodes)

        # Adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True

        # Edge features
        if edge_attr is not None:
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        else:
            edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.long)

        attn_edge_type = torch.zeros((num_nodes, num_nodes, edge_attr.size(-1)), dtype=torch.long)
        attn_edge_type[edge_index[0], edge_index[1]] = edge_attr + 1

        # Shortest paths (used for spatial_pos)
        spatial_pos = torch.tensor(self.compute_shortest_paths(adj.numpy()), dtype=torch.long)

        # Modify attn_bias: Add edge information with weight
        edge_weight = self.attn_bias_weight
        attn_bias = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for i, (src, tgt) in enumerate(zip(edge_index[0], edge_index[1])):
            attn_bias[src, tgt] = edge_weight * edge_attr[i].sum().float()

        # Generate edge_input for multi-hop distances
        edge_input = self.generate_edge_input(spatial_pos, attn_edge_type, self.multi_hop_max_dist)

        return {
            "x": x,
            "adj": adj,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "edge_input": edge_input,
        }

    #def generate_edge_input(self, spatial_pos, attn_edge_type, max_dist):
    #    """
    #    Generate edge_input tensor for multi-hop edges.

    #    Args:
    #        spatial_pos: Shortest path distance matrix.
    #        attn_edge_type: Attention edge type tensor.
    #        max_dist: Maximum allowed distance for multi-hop edges.

    #    Returns:
    #        edge_input tensor.
    #    """
    #    num_nodes = spatial_pos.size(0)
    #    edge_input = torch.zeros((num_nodes, num_nodes, max_dist, attn_edge_type.size(-1)), dtype=torch.long)

    #    for i in range(num_nodes):
    #        for j in range(num_nodes):
    #            if 1 <= spatial_pos[i, j] <= max_dist:  # Only consider paths within max_dist
    #                edge_input[i, j, spatial_pos[i, j] - 1] = attn_edge_type[i, j]

    #    # Add edge encoding
    #    edge_input = self.edge_encoder(edge_input)  # Encode edge_input using embedding
    #    return edge_input

    def generate_edge_input(self, spatial_pos, attn_edge_type, max_dist):
        """
        Generate edge_input tensor for multi-hop edges.

        Args:
            spatial_pos: Shortest path distance matrix.
            attn_edge_type: Attention edge type tensor.
            max_dist: Maximum allowed distance for multi-hop edges.

        Returns:
            edge_input tensor.
        """
        num_nodes = spatial_pos.size(0)
        edge_input = torch.zeros((num_nodes, num_nodes, max_dist, attn_edge_type.size(-1)), dtype=torch.long)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if 1 <= spatial_pos[i, j] <= max_dist:  # Only consider paths within max_dist
                    edge_input[i, j, spatial_pos[i, j] - 1] = attn_edge_type[i, j]

        return edge_input

    @staticmethod
    def compute_shortest_paths(adj):
        """
        Compute shortest paths using the Floyd-Warshall algorithm.

        Args:
            adj: Adjacency matrix.

        Returns:
            Shortest path distance matrix.
        """
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # 이 SMILESDataset의 __getitem__ 함수가 나중에 DataLoader에 들어가면 이부분 자동 실행됨 -> 그래프, target, idx가 나올것임 -> idx를 통해 batch를 나눌것
        #print(f"Fetching sample index: {idx}")
        return self.graphs[idx], self.targets[idx], idx


def collate_fn(batch, dataset, n_pairs=None, min_max=None): # DataLoader에서 collatefn 에 의해 SMILESDataset의 __getitem__에 의해 불려온 것들의 처리가 일어남, 이때 idx는 기존 전체의 idx 기
    """
    Collate a batch of data and process targets dynamically based on target type.

    Args:
        batch: List of (graph, target, index) tuples.
        dataset: The dataset object with `process_targets` method.
        n_pairs: Number of [ex, prob] pairs to include (only for `ex_prob` target type).

    Returns:
        A dictionary containing batched and padded data.
    """
    max_nodes = max(graph["x"].size(0) for graph, _, _ in batch)
    batch_indices = [index for _, _, index in batch]  # 인덱스 추출

    x = torch.stack([
        pad_tensor_x(graph["x"], max_nodes) for graph, _, _ in batch
    ])
    adj = torch.stack([
        pad_tensor(graph["adj"], max_nodes, pad_dim=2) for graph, _, _ in batch
    ])
    in_degree = torch.stack([
        pad_tensor_1d(graph["in_degree"], max_nodes) for graph, _, _ in batch
    ])
    out_degree = torch.stack([
        pad_tensor_1d(graph["out_degree"], max_nodes) for graph, _, _ in batch
    ])
    spatial_pos = torch.stack([
        pad_tensor(graph["spatial_pos"], max_nodes, pad_dim=2) for graph, _, _ in batch
    ])
    attn_bias = torch.stack([
        pad_tensor(graph["attn_bias"], max_nodes, pad_dim=2) for graph, _, _ in batch
    ])
    attn_edge_type = torch.stack([
        pad_tensor(graph["attn_edge_type"], max_nodes, pad_dim=3) for graph, _, _ in batch
    ])
    edge_input = torch.stack([
        pad_tensor(graph["edge_input"], max_nodes, pad_dim=4) for graph, _, _ in batch
    ])

    # 올바른 target 선택
    if dataset.target_type == "ex_prob":
        all_targets = dataset.process_targets(n_pairs=n_pairs)
        targets = torch.stack([all_targets[i] for i in batch_indices])

        #  여기서 ex 데이터만 따로 normalize 수행
        ex = targets[:, :, 0]  # ex만 추출
        prob = targets[:, :, 1]  # prob 그대로 유지

        if min_max is not None:
            min_val = torch.tensor(min_max[0]).to(ex.device)  # train_min 값
            max_val = torch.tensor(min_max[1]).to(ex.device)  # train_max 값
            #  min_val, max_val이 scalar 값일 경우 브로드캐스팅
            if min_val.numel() == 1:
                min_val = min_val.expand_as(ex)
                max_val = max_val.expand_as(ex)
            #  (2,) -> (1, 50)으로 reshape
            else:
                min_val = min_val[0].view(1, 1).expand(ex.size(0), ex.size(1))
                max_val = max_val[0].view(1, 1).expand(ex.size(0), ex.size(1))
        else:
            min_val = getattr(dataset, 'global_ex_min', None)
            max_val = getattr(dataset, 'global_ex_max', None)
        #print(min_val, max_val)

        if min_val is not None and max_val is not None:
            # ✅ min-max normalization 수행
            ex_norm = (ex - min_val) / (max_val - min_val)
        else:
            print("[Warning] min-max 값이 제공되지 않았습니다. Normalization을 건너뜁니다.")
            ex_norm = ex
        # min-max normalization (global min/max)
        # ex_norm = (ex - dataset.global_ex_min) / (dataset.global_ex_max - dataset.global_ex_min)

        # normalized ex와 기존 prob 재결합
        targets = torch.stack([ex_norm, prob], dim=-1)

    elif dataset.target_type in ["default", "nm_distribution"]:
        targets = torch.stack([target for _, target, _ in batch])
        targets = QM2D_to_1D(targets, eV_interval=0.01)
        targets = targets.T

    return {
        "x": x,
        "adj": adj,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "spatial_pos": spatial_pos,
        "attn_bias": attn_bias,
        "attn_edge_type": attn_edge_type,
        "edge_input": edge_input,
        "targets": targets,
    }






def pad_tensor_x(tensor, max_nodes):
    """
    Pad a node feature tensor to the specified number of nodes.

    Args:
        tensor: Input tensor of shape [num_nodes, feature_dim].
        max_nodes: Target number of nodes.

    Returns:
        Padded tensor of shape [max_nodes, feature_dim].
    """
    num_nodes, feature_dim = tensor.size()
    padded = torch.zeros((max_nodes, feature_dim), dtype=tensor.dtype)
    padded[:num_nodes, :] = tensor
    return padded

def pad_tensor_1d(tensor, max_nodes):
    """
    Pad a 1D tensor to the specified length.

    Args:
        tensor: Input tensor of shape [num_nodes].
        max_nodes: Target length.

    Returns:
        Padded tensor of shape [max_nodes].
    """
    padded = torch.zeros((max_nodes,), dtype=tensor.dtype)
    padded[:tensor.size(0)] = tensor
    return padded

def pad_tensor(tensor, max_len, pad_dim):
    """
    Pad a tensor to the specified length.

    Args:
        tensor: Input tensor.
        max_len: Target length.
        pad_dim: Dimensionality of padding.

        Returns:
            Padded tensor.
    """
    pad_size = [max_len] * pad_dim + list(tensor.shape[pad_dim:])
    padded = torch.zeros(pad_size, dtype=tensor.dtype)
    slices = tuple(slice(0, min(dim, max_len)) for dim in tensor.shape)
    padded[slices] = tensor
    return padded


def QM2D_to_1D(tensor: torch.Tensor, eV_interval):
    #print("QM2Dto1D")
    #print("Input tensor shape:", tensor.shape)

    ex = tensor[:, 0:50]  # 유지: [batch_size, 50]
    prob = tensor[:, 50:100]  # 유지: [batch_size, 50]

    combined = torch.stack((ex, prob), dim=-1)  # [batch_size, 50, 2]
    #print("Combined tensor shape:", combined.shape)

    # 최종 결과를 저장할 딕셔너리
    final_dict = ev_prob_dict_maker(eV_interval)

    # 배치마다 개별 dict_base 생성 후 병합
    for i in range(combined.shape[0]):  # 배치 크기만큼 반복
        dict_base = ev_prob_dict_maker(eV_interval)  # 배치마다 새로운 딕셔너리 생성

        for one_ex, one_prob in combined[i]:
            round_ex = round(float(one_ex), 2)
            if round_ex in dict_base:
                dict_base[round_ex].append(one_prob.item())

        # 빈 값 채우기 및 평균 계산
        for key in dict_base:
            if len(dict_base[key]) == 0:
                dict_base[key].append(0.0)
            elif len(dict_base[key]) > 1:
                dict_base[key] = [sum(dict_base[key]) / len(dict_base[key])]

        # 개별 dict_base 값을 최종 dict에 병합
        for key, value in dict_base.items():
            final_dict[key].extend(value)  # 모든 배치에서 병합

    #print("Final dict keys:", final_dict.keys())

    # 최종 딕셔너리에서 데이터를 추출하여 리스트 생성
    tensor_values = [final_dict[key] for key in sorted(final_dict.keys())]
    #print("Final tensor values length:", len(tensor_values))
    #print("Each value count:", len(tensor_values[0]))

    # 텐서 변환
    output_tensor = torch.tensor(tensor_values, dtype=torch.float32)
    output_tensor = output_tensor.T  # 크기 변경 [5, 451]
    #print("Output tensor shape:", output_tensor.shape)

    return output_tensor


def ev_prob_dict_maker(eV_interval):
    """1.7 (약700)- 6.2 eV(약200nm)"""
    start_ev = 1.69
    end_ev = 6.19
    ev = start_ev
    ev_dict = {}
    while ev <= end_ev:
        ev += eV_interval
        ev = round(ev, 2)
        ev_dict[ev] = []
    return ev_dict

def ev_to_nm(energy_ev):
    """ 전자볼트(eV)를 나노미터(nm)로 변환 """
    return 1239.841984 / energy_ev

def nm_to_ev(wavelength_nm):
    """ 나노미터(nm)를 전자볼트(eV)로 변환 """
    return 1239.841984 / wavelength_nm


def encode_global(features: dict, schema: dict) -> torch.Tensor:
    """
    임의의 전역 특성(features)을 하나의 Dense 벡터로 인코딩.

    Parameters
    ----------
    features : dict
        {"solvent": 3, "temp": 298, "pressure": 1.0, ...}
    schema : dict
        각 특성의 인코딩 규칙.
        형식:
        {
            "solvent":  {"type": "onehot",     "num_classes": 8},
            "temp":     {"type": "continuous", "mean": 298, "std": 50},
            "pressure": {"type": "continuous", "min": 0.8, "max": 1.2},
            ...
        }

    Returns
    -------
    torch.Tensor  # shape = (총 채널 수,)
    """
    chunks = []
    for name, rule in schema.items():
        value = features[name]

        if rule["type"] == "onehot":
            # 단일 정수 → one-hot
            vec = F.one_hot(torch.tensor(value),
                            num_classes=rule["num_classes"]).float()
            chunks.append(vec)

        elif rule["type"] == "multihot":
            # 다중 라벨(list[int]) → multi-hot
            vec = torch.zeros(rule["num_classes"], dtype=torch.float)
            vec[torch.tensor(value, dtype=torch.long)] = 1.0
            chunks.append(vec)

        elif rule["type"] == "continuous":
            # 연속값 스케일링 (표준화 또는 min-max 둘 다 지원)
            if "mean" in rule and "std" in rule:
                norm = (value - rule["mean"]) / rule["std"]
            elif "min" in rule and "max" in rule:
                norm = (value - rule["min"]) / (rule["max"] - rule["min"])
            else:
                raise ValueError(f"continuous rule for '{name}' "
                                 "must have (mean,std) or (min,max)")
            chunks.append(torch.tensor([norm], dtype=torch.float))

        else:
            raise ValueError(f"Unknown encode type: {rule['type']}")

    return torch.cat(chunks, dim=0)

# Example Usage
if __name__ == "__main__":
    a = 2
    if a == 0:
        dataset_default = SMILESDataset(csv_file="../../data/train_50.csv", target_type="default", attn_bias_w=1.0)  # 100개 target
        dataloader = DataLoader(dataset_default, batch_size=5, collate_fn=lambda batch: collate_fn(batch, dataset_default, n_pairs=5))

        count=0
        for batch in dataloader:
            count+=1
            print(f"{count}_batch")
            print("dataset_default")
            print(batch.keys())
            print(len(batch["targets"]))
            print(batch["adj"].size())
            print(batch["in_degree"].size())
            print(batch["out_degree"].size())
            print(batch["spatial_pos"].size())
            print(batch["attn_bias"].size())
            print(batch["attn_edge_type"].size())
            print(batch["edge_input"].size())
            print("target_size",batch["targets"].size())
            #print(batch["targets"])
        #if count == 1: # 첫번째 batch 만 나오도록 함
        #    break
    elif a == 1:
        count = 0
        dataset_ex_prob = SMILESDataset(csv_file="../../data/train_50.csv", target_type="ex_prob",
                                        attn_bias_w=1.0)  # [ex, prob] 50개
        dataloader = DataLoader(dataset_ex_prob, batch_size=8, collate_fn=lambda batch: collate_fn(batch, dataset_ex_prob, n_pairs=5))
        for batch in dataloader:
            count += 1
            print("dataset_ex_prob")
            print(batch.keys())
            print(batch["x"].size())
            print(batch["adj"].size())
            print(batch["in_degree"].size())
            print(batch["out_degree"].size())
            print(batch["spatial_pos"].size())
            print(batch["attn_bias"].size())
            print(batch["attn_edge_type"].size())
            print(batch["edge_input"].size())
            print("targets_size",batch["targets"].size())
            #for i in batch['targets']:
            #    print(i)
            #if count == 5:
            #    break

    elif a == 2:
        dataset_nm_dist = SMILESDataset(csv_file="../../data/train_50.csv", target_type="nm_distribution",
                                        attn_bias_w=1.0)  # 801개 target
        dataloader = DataLoader(dataset_nm_dist, batch_size=32, collate_fn=lambda batch: collate_fn(batch, dataset_nm_dist, n_pairs=5))
        count = 0
        for batch in dataloader:
            count += 1
            print("dataset_nm_dist")
            print(batch.keys())
            print(batch["x"].size())
            print(batch["adj"].size())
            print(batch["in_degree"].size())
            print(batch["out_degree"].size())
            print(batch["spatial_pos"].size())
            print(batch["attn_bias"].size())
            print(batch["attn_edge_type"].size())
            print(batch["edge_input"].size())
            print(batch["targets"].size())
            if count == 5:
                break

# attn_bias 초기화시 0으로
# spatial_pos 노드간 최단경로 거리
# edge_input : multi_hop 고려 edge feature

#dict_keys(['x', 'adj', 'in_degree', 'out_degree', 'spatial_pos', 'attn_bias', 'attn_edge_type', 'edge_input', 'targets'])
#torch.Size([32, 35, 9]) Node features 32: 배치 사이즈 35: 각 분자의 (패딩 포함) 최대 노드 수 9: 노드 feature 차원
#torch.Size([32, 35, 35]) # adj, spatial_pos, attn_bias, attn_edge_type, edge_input
#torch.Size([32, 35])
#torch.Size([32, 35])
#torch.Size([32, 35, 35])
#torch.Size([32, 35, 35])
#torch.Size([32, 35, 35, 35])
#torch.Size([32, 35, 35, 35, 35])
#torch.Size([451, 32])