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

    def get_bond_feature_id(self, bond):
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }
        
        bond_type = bond_type_map.get(bond.GetBondType(), 0)
        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())
        
        stereo_map = {
            Chem.rdchem.BondStereo.STEREONONE: 0,
            Chem.rdchem.BondStereo.STEREOANY: 0, # Treat ANY as NONE
            Chem.rdchem.BondStereo.STEREOZ: 1,
            Chem.rdchem.BondStereo.STEREOE: 2,
            Chem.rdchem.BondStereo.STEREOCIS: 3,
            Chem.rdchem.BondStereo.STEREOTRANS: 4,
        }
        stereo = stereo_map.get(bond.GetStereo(), 0)

        # Combine features to create a unique ID
        # 4 (bond_type) * 2 (is_conjugated) * 2 (is_in_ring) * 6 (stereo) = 96
        feature_id = bond_type + is_conjugated * 4 + is_in_ring * 8 + stereo * 16
        return feature_id + 1 # Add 1 to reserve 0 for padding

    def preprocess_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_nodes = mol.GetNumAtoms()

        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append([
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs()
            ])
        node_features = torch.tensor(node_features, dtype=torch.long)

        # Adjacency matrix and Edge feature matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        attn_edge_type = torch.zeros((num_nodes, num_nodes, 1), dtype=torch.long)

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = True
            bond_id = self.get_bond_feature_id(bond)
            attn_edge_type[i, j, 0] = attn_edge_type[j, i, 0] = bond_id

        # ... (rest of the function remains the same)

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
        return self.preprocess_graph(self.smiles_list[index])

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
