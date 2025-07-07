import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from ogb.utils.mol import smiles2graph
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

        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.target_type = target_type
        self.attn_bias_weight = attn_bias_w

        # Process SMILES into graph structures
        self.graphs = [smiles2graph(smiles) for smiles in self.data["smiles"]]

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


    def process_targets(self):
        """
        Process the targets based on the target_type.

        Returns:
            Processed targets as a torch tensor.
        """
        if self.target_type == "default":
            # Case 1: Default target with 100 values
            return torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)

        elif self.target_type == "ex_prob":
            # Case 2: Target as [ex, prob] pairs with 50 values each
            targets = self.data.iloc[:, 1:].values.reshape(-1, 2, 50)
            return torch.tensor(targets, dtype=torch.float32)

        elif self.target_type == "nm_distribution":
            # Case 3: Convert eV to nm and create distribution from 100nm to 800nm
            ex = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob = self.data[[f"prob{i}" for i in range(1, 51)]].values
            nm = (1239.841984 / ex).round().astype(int)  # Convert eV to nm and round
            nm = np.clip(nm, 100, 800)  # Clip to range 100-800 nm
            targets = np.zeros((len(self.data), 801), dtype=np.float32)
            for i, (nm_row, prob_row) in enumerate(zip(nm, prob)):
                for nm_val, prob_val in zip(nm_row, prob_row):
                    if 100 <= nm_val <= 800:
                        targets[i, nm_val - 100] += prob_val
            return torch.tensor(targets, dtype=torch.float32)

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

        # Add edge encoding
        edge_input = self.edge_encoder(edge_input)  # Encode edge_input using embedding
        return edge_input

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

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


def collate_fn(batch):
    """
    Collate a batch of data and pad the graphs to the maximum size in the batch.

    Args:
        batch: List of (graph, target) tuples.

    Returns:
        A dictionary containing batched and padded data.
    """
    max_nodes = max(graph["x"].size(0) for graph, _ in batch)

    x = torch.stack([
        pad_tensor_x(graph["x"], max_nodes) for graph, _ in batch
    ])
    adj = torch.stack([
        pad_tensor(graph["adj"], max_nodes, pad_dim=2) for graph, _ in batch
    ])
    in_degree = torch.stack([
        pad_tensor_1d(graph["in_degree"], max_nodes) for graph, _ in batch
    ])
    out_degree = torch.stack([
        pad_tensor_1d(graph["out_degree"], max_nodes) for graph, _ in batch
    ])
    spatial_pos = torch.stack([
        pad_tensor(graph["spatial_pos"], max_nodes, pad_dim=2) for graph, _ in batch
    ])
    attn_bias = torch.stack([
        pad_tensor(graph["attn_bias"], max_nodes, pad_dim=2) for graph, _ in batch
    ])
    attn_edge_type = torch.stack([
        pad_tensor(graph["attn_edge_type"], max_nodes, pad_dim=3) for graph, _ in batch
    ])  # New padding for attn_edge_type
    edge_input = torch.stack([
        pad_tensor(graph["edge_input"], max_nodes, pad_dim=4) for graph, _ in batch  # Multi-hop padding
    ])
    targets = torch.stack([target for _, target in batch])

    return {
        "x": x,
        "adj": adj,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "spatial_pos": spatial_pos,
        "attn_bias": attn_bias,
        "attn_edge_type": attn_edge_type,  # Ensure attn_edge_type is included
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

# Example Usage
if __name__ == "__main__":
    dataset_default = SMILESDataset(csv_file="../../data/test_1000.csv", target_type="default", attn_bias_w=1.0)  # 100개 target
    dataset_ex_prob = SMILESDataset(csv_file="../../data/test_1000.csv", target_type="ex_prob", attn_bias_w=1.0)  # [ex, prob] 50개
    dataset_nm_dist = SMILESDataset(csv_file="../../data/test_1000.csv", target_type="nm_distribution", attn_bias_w=1.0)  # 801개 target

    dataloader = DataLoader(dataset_default, batch_size=32, collate_fn=collate_fn)

    count=1
    for batch in dataloader:
        print("dataset_default")
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
        if count == 1:
            break

    dataloader = DataLoader(dataset_ex_prob, batch_size=32, collate_fn=collate_fn)
    for batch in dataloader:
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
        print(batch["targets"].size())
        if count == 1:
            break

    dataloader = DataLoader(dataset_nm_dist, batch_size=32, collate_fn=collate_fn)
    for batch in dataloader:
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
        if count == 1:
            break

# attn_bias 초기화시 0으로
# spatial_pos 노드간 최단경로 거리
# edge_input : multi_hop 고려 edge feature