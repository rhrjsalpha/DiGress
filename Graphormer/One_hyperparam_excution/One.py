import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Graphormer.GP.dataset_gen.Batch_QM import MoleculeDataset, collate_fn
from Graphormer.GP.models.graphormer_torch import GraphormerModel, get_default_args

import pandas as pd

# 1. 데이터셋 준비
file_path = r'..\data/test_1000.csv'

# 데이터 로드 및 필요한 컬럼 추출
data = pd.read_csv(file_path)
selected_columns = ['smiles'] + [f'ex{i}' for i in range(1, 51)] + [f'prob{i}' for i in range(1, 51)]
processed_data = data[selected_columns]

# MoleculeDataset 생성
dataset = MoleculeDataset(processed_data)
print(dataset)

a=0
for i in dataset:
    if a > len(dataset)-1:
        break
    print(a)
    try:
        print(i.keys())
        print(i['x'].dtype)
    except KeyError:
        pass
    a+=1

dataloader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn, shuffle=True)

# 2. 모델 및 하이퍼파라미터 설정
args = get_default_args()
args.num_classes = 50 * 2  # [eV, prob] 50개의 페어
args.max_nodes = 128  # 최대 노드 수 (데이터셋에 따라 조정)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphormerModel(args).to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()  # eV와 prob 예측을 위한 MSE 손실
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # 데이터 배치 로드
        x = batch["x"].to(device)  # [batch_size, max_nodes, features]
        edge_index = batch["edge_index"].to(device)  # [batch_size, 2, max_edges]
        edge_attr = batch["edge_attr"].to(device)  # [batch_size, max_nodes, max_nodes, features]
        in_degree = batch["in_degree"].to(device)  # [batch_size, max_nodes]
        out_degree = batch["out_degree"].to(device)  # [batch_size, max_nodes]
        target = batch["target"].to(device)  # [batch_size, 50, 2]

        # 모델 입력 준비
        batched_data = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "in_degree": in_degree,
            "out_degree": out_degree,
        }

        # 모델 예측 및 손실 계산
        optimizer.zero_grad()
        output = model(batched_data)  # [batch_size, 50 * 2]

        # 손실 함수 적용
        loss = criterion(output.view(-1, 50, 2), target)  # Output과 target 크기 맞춤
        loss.backward()
        optimizer.step()

        # 손실 누적
        running_loss += loss.item()

    # 에포크 종료 후 평균 손실 출력
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

# 4. 모델 저장
torch.save(model.state_dict(), "graphormer_model.pth")
print("Model saved to graphormer_model.pth")
