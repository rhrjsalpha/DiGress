import torch
from Smiles_to_Etc import smiles_to_graphormer_input


def collate_fn(batch):
    # 최대 노드 수와 최대 엣지 수 계산
    max_nodes = max(data["x"].size(1) for data in batch)  # 최대 노드 수
    max_edges = max(data["edge_index"][0].size(1) for data in batch)  # 최대 엣지 수

    print(f"max_nodes: {max_nodes}, max_edges: {max_edges}")

    # 패딩을 위한 딕셔너리 초기화
    padded_batch = {
        "x": torch.zeros(len(batch), max_nodes, batch[0]["x"].size(-1)),  # 노드 특성
        "edge_index": torch.zeros(len(batch), 2, max_edges, dtype=torch.long),  # 엣지 인덱스
        "edge_attr": torch.zeros(len(batch), max_nodes, max_nodes, batch[0]["edge_attr"].size(-1)),  # 엣지 특성
    }

    for i, data in enumerate(batch):
        num_nodes = data["x"].size(1)
        num_edges = data["edge_index"][0].size(1)
        edge_index = data["edge_index"][0]

        print(f"num_nodes: {num_nodes}, num_edges: {num_edges}")

        # 노드 특성 패딩
        padded_batch["x"][i, :num_nodes, :] = data["x"].squeeze(0)

        # 엣지 인덱스 패딩
        padded_batch["edge_index"][i, :, :num_edges] = edge_index

        # 엣지 특성 패딩
        padded_batch["edge_attr"][i, :num_nodes, :num_nodes, :] = data["edge_attr"].squeeze(0)

    return padded_batch



# 예제 실행
smiles_list = ["C1=CC=CC=C1", "CCO", "CC(=O)O"]  # 벤젠, 에탄올, 아세트산
batched_data = smiles_to_graphormer_input(smiles_list)

#print(type(batched_data), batched_data)
#print("Batched Node Features (x):", batched_data["x"].shape, batched_data["x"].type()) # Batched Node Features (x): torch.Size([3, 6, 1])  # 3개의 분자, 최대 6개의 노드
#print("Batched Edge Attributes (edge_attr):", batched_data["edge_attr"].shape, batched_data["edge_attr"].type()) # Batched Edge Attributes (edge_attr): torch.Size([3, 6, 6, 1])  # 3개의 분자, 최대 6개의 노드 간 엣지
#print("Batched Edge Indices (edge_index):", [e.shape for e in batched_data["edge_index"]], type(batched_data["edge_index"])) # Batched Edge Indices (edge_index): [torch.Size([2, 12]), torch.Size([2, 3]), torch.Size([2, 4])]

