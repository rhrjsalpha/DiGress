from Batch_QM import MoleculeDataset, collate_fn
from torch.utils.data import DataLoader
import pandas as pd

# SMILES 리스트
smiles_list = ["C1=CC=CC=C1", "CCO", "CC(=O)O"] * 20  # 예제: 100개 분자

# 데이터셋 및 DataLoader 생성
dataset = MoleculeDataset(smiles_list)
print(dataset)

# CSV 파일 경로
file_path = r'..\..\data/test_1000.csv'

# 데이터 로드 및 필요한 컬럼 추출
data = pd.read_csv(file_path)
selected_columns = ['smiles'] + [f'ex{i}' for i in range(1, 51)] + [f'prob{i}' for i in range(1, 51)]
processed_data = data[selected_columns]

# 데이터셋 생성
dataset = MoleculeDataset(processed_data)
print("Dataset created.")

# 데이터셋 내부 확인
#for i in dataset:
#    print(f"x (node features): {i['x'].shape}")  # [1, num_nodes, features]
#    print(f"edge_index (adjacency list): {i['edge_index'][0].shape}")  # [2, num_edges]
#    print(f"edge_attr (edge features): {i['edge_attr'].shape}")  # [1, num_nodes, num_nodes, features]
#    print(f"target (eV and probabilities): {i['target'].shape}")  # [50, 2]
#
# DataLoader 생성
dataloader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn, shuffle=True)

# DataLoader 반복
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    print(f"x shape: {batch['x'].shape}")
    print(f"edge_index shape: {batch['edge_index'].shape}")
    print(f"edge_attr shape: {batch['edge_attr'].shape}")
    print(f"target shape: {batch['target'].shape}")

# 배치 확인
for i, batch in enumerate(dataloader):
    try:
        print(f"Dataset length: {len(dataset)}")
        print(f"Batch {i+1}:")
        print(f"x shape: {batch['x'].shape}")  # [batch_size, max_nodes, features]
        print(f"edge_index shape: {batch['edge_index'].shape}")  # [batch_size, 2, max_edges]
        print(f"edge_attr shape: {batch['edge_attr'].shape}")  # [batch_size, max_nodes, max_nodes, features]
        print(f"target shape: {batch['target'].shape}")  # [batch_size, 50, 2]
    except KeyError:
        break
