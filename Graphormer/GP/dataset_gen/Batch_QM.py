import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset
from Graphormer.GP.dataset_gen.Smiles_to_Etc import smiles_to_graph
import time

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


class MoleculeDataset:
    def __init__(self, dataframe):
        self.data = dataframe
        self.invalid_smiles_list = []  # 오류가 발생한 SMILES 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            smiles = self.data.loc[idx, "smiles"]
            target_ev = self.data.loc[idx, [f"ex{i}" for i in range(1, 51)]].values
            target_prob = self.data.loc[idx, [f"prob{i}" for i in range(1, 51)]].values

            try:
                # Graph input from SMILES
                graph_data = smiles_to_graph(smiles)  # Using the previously defined function
                if graph_data is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")

                return {
                    "x": graph_data["x"],
                    "edge_index": graph_data["edge_index"],
                    "edge_attr": graph_data["edge_attr"],
                    "in_degree": graph_data["in_degree"],
                    "out_degree": graph_data["out_degree"],
                    "target": torch.tensor(list(zip(target_ev, target_prob)), dtype=torch.float),
                }
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                return None

        except KeyError as e1:
            print(e1)


# batched_data["x"], batched_data["edge_attr"], batched_data["edge_index"]와 같은 필드의 텐서 크기가 각 분자마다 다릅니다.
# 예를 들어: batched_data["x"]에서 한 분자의 노드 개수는 6, 다른 분자는 3입니다.
# PyTorch는 이러한 크기 불일치를 허용하지 않으므로, torch.stack을 사용할 수 없습니다. -> collate_fn 이 이를 해결시켜 줌


def collate_fn(batch):
    print("Before processing:", batch[0]['x'].dtype)

    # None 데이터 제거
    batch = [data for data in batch if data is not None]

    # 최대 노드 수와 최대 엣지 수 계산
    max_nodes = max(data["x"].size(0) for data in batch)
    max_edges = max(data["edge_index"].size(1) if data["edge_index"].numel() > 0 else 0 for data in batch)

    # 패딩을 위한 딕셔너리 초기화 (데이터 유형 명시)
    padded_batch = {
        "x": torch.zeros(len(batch), max_nodes, batch[0]["x"].size(-1), dtype=torch.int64),
        "edge_index": torch.zeros(len(batch), 2, max_edges, dtype=torch.long),
        "edge_attr": torch.zeros(len(batch), max_nodes, max_nodes, batch[0]["edge_attr"].size(-1), dtype=torch.float32),
        "in_degree": torch.zeros(len(batch), max_nodes, dtype=torch.int64),
        "out_degree": torch.zeros(len(batch), max_nodes, dtype=torch.int64),
        "target": torch.zeros(len(batch), 50, 2, dtype=torch.float32),
    }

    for i, data in enumerate(batch):
        print(data.keys())
        print(data['x'].dtype)
        num_nodes = data["x"].size(0)
        num_edges = data["edge_index"].size(1)

        # 노드 특성 패딩
        padded_batch["x"][i, :num_nodes, :] = data["x"]

        # 엣지 인덱스 패딩
        padded_batch["edge_index"][i, :, :num_edges] = data["edge_index"]

        # 엣지 특성 패딩
        padded_batch["edge_attr"][i, :num_nodes, :num_nodes, :] = data["edge_attr"]

        # in_degree와 out_degree 패딩
        padded_batch["in_degree"][i, :num_nodes] = data["in_degree"]
        padded_batch["out_degree"][i, :num_nodes] = data["out_degree"]

        # 타겟 데이터 추가
        padded_batch["target"][i] = data["target"]

    print("After processing:", padded_batch["x"].dtype)
    return padded_batch

