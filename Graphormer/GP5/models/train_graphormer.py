# 혼자실행 #

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from graphormer import GraphormerModel
import os
from Graphormer.GP5.Custom_Loss.custom_loss import fastdtw_loss
from Graphormer.GP5.Custom_Loss.sdtw_cuda_loss import SoftDTW
#SoftDTWLoss gamma: 부드러움 정도를 제어. 값이 작으면 기존 DTW와 유사, 값이 크면 더 부드럽게 작동. normalize: 정규화 여부 (True/False).


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Configuration dictionary for GraphormerModel

config = {
    "num_atoms": 100,
    "num_in_degree": 10,
    "num_out_degree": 10,
    "num_edges": 50,
    "num_spatial": 100, # 만약 분자에서 계산된 값이 이것보다 더 크면 오류남
    "num_edge_dis": 10,
    "edge_type": "multi_hop",
    "multi_hop_max_dist": 5,
    "num_encoder_layers": 6,
    "embedding_dim": 128,
    "ffn_embedding_dim": 256,
    "num_attention_heads": 8,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.1,
    "activation_fn": "gelu",
    "pre_layernorm": False,
    "q_noise": 0.0,
    "qn_block_size": 8,
    "output_size":100,
}

target_type = 'default' # 'default' "ex_prob" "nm_distribution"
print("target_type: ",target_type)

# Initialize dataset and DataLoader
dataset = SMILESDataset(csv_file="../../data/data_example.csv", attn_bias_w=1.0,target_type=target_type)  # Include edge weight
#  DataLoader가 생성한 데이터 배치로, 보통 (input_data, target) 형태의 튜플
dataloader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=1)
)

# Initialize the model, loss function, and optimizer
model = GraphormerModel(config)

# 손실 함수 정의 두개에 대해서 다르게
# nn.SmoothL1Loss() nn.L1Loss() nn.MSELoss() log_cosh_loss(outputs_prob, targets_prob)
# SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)
SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def loss_fn_gen(loss_fn):
    if loss_fn == 'MSE':
        return nn.MSELoss()
    elif loss_fn == 'MAE':
        return nn.L1Loss()
    elif loss_fn == 'SoftDTW':
        return SoftDTWLoss
    elif loss_fn == 'Huber':
        return nn.SmoothL1Loss()

loss_function = 'MSE'
loss_function_ex = 'SoftDTW'
loss_function_prob = 'SoftDTW'
print("loss_function_ex:", loss_function_ex)
print("loss_function_prob:", loss_function_prob)

criterion = loss_fn_gen(loss_function)
criterion_ex = loss_fn_gen(loss_function_ex)  # For 'ex' part
criterion_prob = loss_fn_gen(loss_function_prob)  # For 'prob' part (예: Binary Cross Entropy)

def move_to_device(batch, device):
    """
    Move batch data to the specified device.
    """
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# Training loop
num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 초기 가중치 설정 (합이 1이 되도록 설정)
weight_ex = 0.5  # ex 손실의 초기 가중치
weight_prob = 1 - weight_ex  # prob 손실의 초기 가중치

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # Move batch data to the device
        batch = move_to_device(batch, device)

        # Extract inputs and targets
        batched_data = {
            "x": batch["x"],
            "adj": batch["adj"],
            "in_degree": batch["in_degree"],
            "out_degree": batch["out_degree"],
            "spatial_pos": batch["spatial_pos"],
            "attn_bias": batch["attn_bias"],
            "edge_input": batch["edge_input"],
            "attn_edge_type": batch["attn_edge_type"],
        }
        targets = batch["targets"]

        # Forward pass
        outputs = model(batched_data, targets=targets, target_type=target_type)

        # Compute loss
        if target_type == "ex_prob":
            if loss_function_ex == 'SoftDTW':
                # SoftDTW를 배치 내 각 샘플에 대해 계산
                outputs_ex = outputs[:, :, 0:1]  # 예측된 'ex'
                targets_ex = targets[:, :, 0:1]  # 실제 'ex'
                loss_ex = torch.stack([
                    criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                    for i in range(outputs_ex.size(0))
                ]).mean()
            else:
                # Outputs for 'ex_prob' are 3D: [batch_size, num_pairs, 2]
                outputs_ex = outputs[:, :, 0]  # First part for 'ex'
                targets_ex = targets[:, :, 0]
                # Compute individual losses
                loss_ex = criterion_ex(outputs_ex, targets_ex)

            # 'prob'에 대해 손실 계산
            if loss_function_prob == 'SoftDTW':
                # SoftDTW를 배치 내 각 샘플에 대해 계산
                outputs_prob = outputs[:, :, 1:2]  # 예측된 'prob'
                targets_prob = targets[:, :, 1:2]  # 실제 'prob'
                loss_prob = torch.stack([
                    criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                    for i in range(outputs_prob.size(0))
                ]).mean()

            else:
                outputs_prob = outputs[:, :, 1]  # Second part for 'prob'
                targets_prob = targets[:, :, 1]
                loss_prob = criterion_prob(outputs_prob, targets_prob)
            print(loss_ex, loss_prob)
            print(weight_ex, weight_prob)
            loss = weight_ex * loss_ex + weight_prob * loss_prob




        elif target_type == "default":
            # Outputs for 'default' are 2D: [batch_size, num_features]
            #print("outputs.size default",outputs.size())
            outputs_ex = outputs[:, :outputs.size(1) // 2]  # First half for 'ex'
            outputs_prob = outputs[:, outputs.size(1) // 2:]  # Second half for 'prob'
            #print("default targets.shape", targets.shape)

            # Corresponding targets split for 'ex' and 'prob'
            #print("outputs.size default",targets.size())
            targets_ex = targets[:, :targets.size(1) // 2]  # First half for 'ex'
            targets_prob = targets[:, targets.size(1) // 2:]  # Second half for 'prob'

            # Compute individual losses
            if loss_function_ex == 'SoftDTW':
                outputs_ex = outputs_ex.unsqueeze(-1)
                targets_ex = targets_ex.unsqueeze(-1)
                # Apply SoftDTW for each batch sample
                loss_ex = torch.stack([
                    criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                    for i in range(outputs_ex.size(0))
                ]).mean()  # 평균 계산
            else:
                # Use standard loss (e.g., MSE, L1)
                loss_ex = criterion_ex(outputs_ex, targets_ex)

            if loss_function_prob == 'SoftDTW':
                outputs_prob = outputs_prob.unsqueeze(-1)
                targets_prob = targets_prob.unsqueeze(-1)
                # Apply SoftDTW for each batch sample
                loss_prob = torch.stack([
                    criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                    for i in range(outputs_prob.size(0))
                ]).mean()  # 평균 계산
            else:
                # Use standard loss (e.g., MSE, L1)
                loss_prob = criterion_prob(outputs_prob, targets_prob)
            # Combine the losses
            print(loss_ex, loss_prob)
            print(weight_ex, weight_prob)
            loss = weight_ex * loss_ex + weight_prob * loss_prob

        elif target_type == "nm_distribution":
            loss = criterion(outputs, targets)

        else:
            raise ValueError("Invalid target type")
        print(loss)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

print("Training complete.")

# [[eV_1,Osc_1] [eV_2,Osc_2].. [eV_50, Osc_50]] 이런 형태로 학습이 진행됨



