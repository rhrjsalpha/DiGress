from Smiles_to_Etc import MoleculeDataset
from Batch_generater import collate_fn
from torch.utils.data import DataLoader

# SMILES 리스트
smiles_list = ["C1=CC=CC=C1", "CCO", "CC(=O)O"] * 20  # 예제: 100개 분자

# 데이터셋 및 DataLoader 생성
dataset = MoleculeDataset(smiles_list)
print(dataset)
for i in dataset:
    print(i['x']) # i['x']: [1, num_nodes, features]
    print("print(i['x'])",i['x'].shape)
    print(i['edge_index'][0]) # i['edge_index'][0]:[2, num_edges]
    print("i['edge_index']",i['edge_index'][0].shape)
    print(i['edge_attr']) # i['edge_attr']: [1, num_nodes, num_nodes, features]
    print("i['edge_attr']",i['edge_attr'].shape)
dataloader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn, shuffle=True)

# 배치 확인
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"x shape: {batch['x'].shape}")
    print(f"edge_index shape: {batch['edge_index'].shape}")
    print(f"edge_attr shape: {batch['edge_attr'].shape}")