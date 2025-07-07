from rdkit import Chem  # 🔧 RDKit 추가
import torch
import numpy as np
from ogb.utils.mol import smiles2graph

class GraphDataset:
    def __init__(self, smiles_list, max_nodes=128, multi_hop_max_dist=5):
        self.smiles_list = smiles_list
        self.graphs = [self.validate_graph(smiles2graph(smiles)) for smiles in smiles_list]
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist

    @staticmethod
    def validate_graph(graph):
        required_keys = ['num_nodes', 'edge_index', 'edge_feat', 'node_feat']
        for key in required_keys:
            if key not in graph:
                raise ValueError(f"Graph is missing required key: {key}")
        if graph['edge_feat'] is None or len(graph['edge_feat']) == 0:
            raise ValueError("Graph has invalid or missing edge features.")
        return graph

    def extract_node_features(self, smiles):  # 🔧 실제 노드 feature 추출용 함수 추가
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        node_feats = []
        for atom in mol.GetAtoms():
            atom_type = atom.GetAtomicNum()                # 원자 번호 (e.g., C=6)
            formal_charge = atom.GetFormalCharge()         # 형식 전하
            hybrid = int(atom.GetHybridization())          # Hybridization: SP=1, SP2=2, ...
            aromatic = int(atom.GetIsAromatic())           # 방향족 여부
            num_H = atom.GetTotalNumHs()

            node_feats.append([atom_type, formal_charge, hybrid, aromatic, num_H])
        return torch.tensor(node_feats, dtype=torch.long)

    def preprocess_graph(self, graph, smiles=None):  # 🔧 smiles 인자 추가
        num_nodes = graph['num_nodes']
        edge_index = graph['edge_index']
        edge_attr = graph.get('edge_feat', None)
        if edge_attr is None:
            raise ValueError("Missing edge features")

        # 🔧 RDKit 기반 실제 노드 feature 사용
        if smiles is not None:
            node_features = self.extract_node_features(smiles)
        else:
            raise ValueError("SMILES string is required to extract RDKit features")

        # Adjacency matrix 생성
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True

        # Edge feature matrix
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros((num_nodes, num_nodes, edge_attr.shape[-1]), dtype=torch.long)
        attn_edge_type[edge_index[0], edge_index[1]] = torch.tensor(edge_attr, dtype=torch.long) + 1

        # 최단 경로 거리 계산
        shortest_path = self.compute_shortest_paths(adj.numpy())
        max_dist = min(int(np.amax(shortest_path)), self.multi_hop_max_dist)

        # multi-hop edge input 생성
        edge_input = self.generate_edge_input(shortest_path, attn_edge_type.numpy(), max_dist)

        return {
            'x': node_features,  # 🔧 RDKit 기반 feature
            'adj': adj,
            'attn_edge_type': attn_edge_type,
            'shortest_path': torch.tensor(shortest_path, dtype=torch.long),
            'edge_input': torch.tensor(edge_input, dtype=torch.long),
        }

    @staticmethod
    def compute_shortest_paths(adj):
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

    def generate_edge_input(self, shortest_path, attn_edge_type, max_dist):
        num_nodes = shortest_path.shape[0]
        edge_input = np.zeros((num_nodes, num_nodes, max_dist, attn_edge_type.shape[-1]), dtype=np.int64)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if shortest_path[i, j] < max_dist:
                    edge_input[i, j, int(shortest_path[i, j]) - 1] = attn_edge_type[i, j]
        return edge_input

    def __getitem__(self, index):
        return self.preprocess_graph(self.graphs[index], self.smiles_list[index])  # 🔧 smiles 추가 전달

    def __len__(self):
        return len(self.graphs)

    def collate(self, batch):
        max_nodes = min(self.max_nodes, max([b['x'].size(0) for b in batch]))
        x = torch.stack([self.pad_tensor(b['x'], max_nodes) for b in batch])
        adj = torch.stack([self.pad_tensor(b['adj'], max_nodes) for b in batch])
        edge_input = torch.stack([self.pad_tensor(b['edge_input'], max_nodes, pad_dim=3) for b in batch])
        return {'x': x, 'adj': adj, 'edge_input': edge_input}

    @staticmethod
    def pad_tensor(tensor, max_len, pad_dim=2):
        pad_size = [max_len] * pad_dim + list(tensor.shape[pad_dim:])
        padded = torch.zeros(pad_size, dtype=tensor.dtype)
        padded[:tensor.shape[0], :tensor.shape[1]] = tensor
        return padded


# Example Usage
if __name__ == "__main__":
    smiles_list = ["CCO", "CCN", "CCC"]
    dataset = GraphDataset(smiles_list)
    graph = dataset[0]  # Access the first graph
    print(graph)
