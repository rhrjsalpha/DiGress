import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

def collate_fn(batch):
    max_nodes = max(data["x"].size(0) for data in batch)  # 최대 노드 수
    max_edges = max(len(data["edge_index"][1]) for data in batch)  # 최대 엣지 수

    # 패딩을 위한 빈 텐서 형성 max size 에 맞도록 빈 텐서가 만들어 진다.
    padded_batch = {
        "x": torch.zeros(len(batch), max_nodes, 1),
        "edge_index": [],
        "edge_attr": torch.zeros(len(batch), max_edges, 1),
    }

    # padded_batch의 빈 부분에 적절하게 복사 붙여넣기 (padding)
    for i, data in enumerate(batch):
        num_nodes = data["x"].size(0)
        num_edges = data["edge_index"].size(1)

        # 노드 특성 패딩
        padded_batch["x"][i, :num_nodes] = data["x"]

        # 엣지 인덱스 패딩
        edge_index_padded = torch.zeros(2, max_edges, dtype=torch.long)
        edge_index_padded[:, :num_edges] = data["edge_index"]
        padded_batch["edge_index"].append(edge_index_padded)

        # 엣지 특성 패딩
        padded_batch["edge_attr"][i, :num_edges] = data["edge_attr"]

    # edge_index를 텐서로 변환
    padded_batch["edge_index"] = torch.stack(padded_batch["edge_index"], dim=0)
    return padded_batch

def collate_fn(batch):


    # 최대 노드 수와 최대 엣지 수 계산
    max_nodes = max(data["x"].size(1) for data in batch)  # 최대 노드 수
    nodes = [data["x"].size(0) for data in batch]
    print(nodes)
    max_edges = max(data["edge_index"][0].size(1) for data in batch) # 최대 엣지 수
    edges = [data["edge_index"][0].size(0) for data in batch]
    print(edges)
    print('max_nodes, max_edges',max_nodes, max_edges)

    # 패딩을 위한 딕셔너리 초기화
    padded_batch = {
        "x": torch.zeros(len(batch), max_nodes, batch[0]["x"].size(1)),  # 노드 특성
        "edge_index": torch.zeros(len(batch), 2, max_edges, dtype=torch.long),  # 엣지 인덱스
        "edge_attr": torch.zeros(len(batch), max_edges, batch[0]["edge_attr"].size(1)),  # 엣지 특성
    }

    for i, data in enumerate(batch):
        num_nodes = data["x"].size(0)
        num_edges = data["edge_index"][0].size(1)
        edge_index = data["edge_index"][0]
        print("num_nodes and num_edges",num_nodes, num_edges)

        print(f"data['x'] shape: {data['x'].shape}")
        print(f"padded_batch['x'][i, :num_nodes] shape: {padded_batch['x'][i, :num_nodes].shape}")
        # 노드 특성 패딩
        padded_batch["x"][i, :num_nodes, :] = data["x"].squeeze(0).transpose(0, 1)

        # 엣지 인덱스 패딩
        print(f"data['edge_index']: {data['edge_index']}, type: {type(data['edge_index'])}")
        padded_batch["edge_index"][i, :, :num_edges] = edge_index

        # 엣지 특성 패딩
        print(padded_batch["edge_attr"][i, :num_edges, :].shape)
        print(data["edge_attr"].squeeze(-1).shape)
        padded_batch["edge_attr"][i, :num_edges, :] = data["edge_attr"].squeeze(-1)


    return padded_batch