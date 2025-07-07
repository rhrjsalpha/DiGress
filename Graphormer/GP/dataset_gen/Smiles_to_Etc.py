import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from rdkit.Chem import rdmolops
import numpy as np

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 노드(원자) 특성: 정수형으로 설정
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features = torch.tensor(atom_features, dtype=torch.long).unsqueeze(-1)  # torch.float -> torch.long

    # 엣지(결합) 정보
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
    edge_indices = np.array(adjacency_matrix.nonzero())
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)

    # 엣지 특성
    num_nodes = len(atom_features)
    bond_types = torch.zeros((num_nodes, num_nodes, 1), dtype=torch.float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_types[i, j, 0] = bond.GetBondTypeAsDouble()
        bond_types[j, i, 0] = bond.GetBondTypeAsDouble()


    # in_degree와 out_degree 계산
    degree_matrix = adjacency_matrix.sum(axis=0)
    in_degree = torch.tensor(degree_matrix, dtype=torch.long)
    out_degree = torch.tensor(degree_matrix, dtype=torch.long)


    return {
        "x": atom_features,  # 정수형 텐서
        "edge_index": edge_indices,
        "edge_attr": bond_types,
        "in_degree": in_degree,
        "out_degree": out_degree,
    }



def extract_molecule_info(smiles):
    # RDKit 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 모든 원자 정보 추출
    atom_info = []
    for atom in mol.GetAtoms():
        atom_info.append({
            "atom_idx": atom.GetIdx(),  # 원자의 인덱스
            "atomic_num": atom.GetAtomicNum(),  # 원자 번호
            "symbol": atom.GetSymbol(),  # 원소 기호
            "degree": atom.GetDegree(),  # 연결된 결합 수
            "formal_charge": atom.GetFormalCharge(),  # 형식 전하
            "hybridization": str(atom.GetHybridization()),  # 혼성화 상태
            "is_aromatic": atom.GetIsAromatic(),  # 방향족 여부
        })

    # 모든 결합 정보 추출
    bond_info = []
    for bond in mol.GetBonds():
        bond_info.append({
            "bond_idx": bond.GetIdx(),  # 결합 인덱스
            "begin_atom_idx": bond.GetBeginAtomIdx(),  # 시작 원자 인덱스
            "end_atom_idx": bond.GetEndAtomIdx(),  # 끝 원자 인덱스
            "bond_type": str(bond.GetBondType()),  # 결합 종류
            "is_aromatic": bond.GetIsAromatic(),  # 방향족 여부
        })

    return atom_info, bond_info

def smiles_to_graphormer_input(smiles_list):
    # 각 분자의 데이터를 저장할 리스트
    node_features, edge_indices, edge_attrs = [], [], []
    max_nodes = 0  # 가장 큰 분자의 노드 수를 추적

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # 노드(원자) 특성
        nodes = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
        node_features.append(nodes)

        # 엣지(결합) 정보
        edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
        edge_types = [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]

        edge_indices.append(torch.tensor(edges, dtype=torch.long).t())
        edge_attrs.append(torch.tensor(edge_types, dtype=torch.float).view(-1, 1))

        # 최대 노드 수 갱신
        max_nodes = max(max_nodes, len(nodes))

    # 패딩 처리
    batch_size = len(smiles_list)
    padded_node_features = torch.zeros(batch_size, max_nodes, 1)  # 노드 특성 패딩
    padded_edge_indices = []  # 엣지 인덱스는 개별적으로 처리
    padded_edge_attrs = torch.zeros(batch_size, max_nodes, max_nodes, 1)  # 엣지 특성 패딩

    for i, (nodes, edges, edge_attr) in enumerate(zip(node_features, edge_indices, edge_attrs)):
        num_nodes = nodes.size(0)

        # 노드 특성 복사
        padded_node_features[i, :num_nodes, :] = nodes

        # 엣지 인덱스는 그대로 추가 (배치 처리 시 합칠 필요 있음)
        padded_edge_indices.append(edges)

        # 엣지 특성 복사 (스파스 구조가 아닌 경우)
        for j, (start, end) in enumerate(edges.t()):
            padded_edge_attrs[i, start, end, :] = edge_attr[j]

    # 최종 batched_data 생성
    batched_data = {
        "x": padded_node_features,
        "edge_index": padded_edge_indices,  # 패딩 없이 원래의 구조로 유지
        "edge_attr": padded_edge_attrs,
        # 추가 정보는 필요에 따라 생성
    }
    return batched_data


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        # SMILES를 그래프 데이터로 변환
        smiles = self.smiles_list[idx]
        return smiles_to_graphormer_input([smiles])  # 위에서 정의한 변환 함수 사용

# batched_data["x"], batched_data["edge_attr"], batched_data["edge_index"]와 같은 필드의 텐서 크기가 각 분자마다 다릅니다.
# 예를 들어: batched_data["x"]에서 한 분자의 노드 개수는 6, 다른 분자는 3입니다.
# PyTorch는 이러한 크기 불일치를 허용하지 않으므로, torch.stack을 사용할 수 없습니다. -> collate_fn 이 이를 해결시켜 줌