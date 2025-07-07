# 다른 곳 import #

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from graphormer import GraphormerModel
import os
from Graphormer.GP5.Custom_Loss.custom_loss import fastdtw_loss
from Graphormer.GP5.Custom_Loss.sdtw_cuda_loss import SoftDTW
import json
from Graphormer.GP5.models.graphormer_train import train_model

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train_model(
    config,
    target_type="default",
    loss_function = "MSE",
    loss_function_ex="SoftDTW",
    loss_function_prob="SoftDTW",
    weight_ex=0.5,
    num_epochs=10,
    batch_size=2,
    n_pairs=1,
    learning_rate=0.001,
    dataset_path="../../data/data_example.csv",
):
    """
    Train the Graphormer model with specified configurations and return the final loss.

    Args:
        config (dict): Configuration for the Graphormer model.
        target_type (str): Target type ("default", "ex_prob", "nm_distribution").
        loss_function_ex (str): Loss function for 'ex'.
        loss_function_prob (str): Loss function for 'prob'.
        weight_ex (float): Weight for 'ex' loss. Weight for 'prob' will be 1 - weight_ex.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        n_pairs (int): Number of pairs for 'ex_prob' targets.
        learning_rate (float): Learning rate for the optimizer.
        dataset_path (str): Path to the dataset CSV file.

    Returns:
        float: Final average loss.
    """
    # Initialize dataset and DataLoader
    dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )

    # Initialize the model, loss function, and optimizer
    model = GraphormerModel(config)
    SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)

    def loss_fn_gen(loss_fn):
        if loss_fn == 'MSE':
            return nn.MSELoss()
        elif loss_fn == 'MAE':
            return nn.L1Loss()
        elif loss_fn == 'SoftDTW':
            return SoftDTWLoss
        elif loss_fn == 'Huber':
            return nn.SmoothL1Loss()

    criterion = loss_fn_gen(loss_function)
    criterion_ex = loss_fn_gen(loss_function_ex)
    criterion_prob = loss_fn_gen(loss_function_prob)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight_prob = 1 - weight_ex

    # 손실 값 저장을 위한 리스트 초기화
    loss_history = []
    weight_history = []
    output_dir = "./training_logs"
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    total_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_weights = {}

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)

            # Compute loss
            if target_type == "ex_prob":
                outputs_ex = outputs[:, :, 0:1]
                targets_ex = targets[:, :, 0:1]
                loss_ex = torch.stack([criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0)) for i in range(outputs_ex.size(0))]).mean()

                outputs_prob = outputs[:, :, 1:2]
                targets_prob = targets[:, :, 1:2]
                loss_prob = torch.stack([criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0)) for i in range(outputs_prob.size(0))]).mean()

                loss = weight_ex * loss_ex + weight_prob * loss_prob
            elif target_type == "default":
                outputs_ex = outputs[:, :outputs.size(1) // 2]
                outputs_prob = outputs[:, outputs.size(1) // 2:]
                targets_ex = targets[:, :targets.size(1) // 2]
                targets_prob = targets[:, targets.size(1) // 2:]

                loss_ex = torch.stack([criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0)) for i in range(outputs_ex.size(0))]).mean()
                loss_prob = torch.stack([criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0)) for i in range(outputs_prob.size(0))]).mean()

                loss = weight_ex * loss_ex + weight_prob * loss_prob
            else:
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        total_loss += epoch_loss / len(dataloader)

        loss_history.append(total_loss)

        # 매 epoch마다 모델 가중치 저장
        for name, param in model.named_parameters():
            if param.requires_grad:
                epoch_weights[name] = param.data.cpu().numpy().tolist()

        weight_history.append(epoch_weights)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth"))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    # 손실 값 및 가중치 기록 저장
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)

    with open(os.path.join(output_dir, "weight_history.json"), "w") as f:
        json.dump(weight_history, f)

    return total_loss / num_epochs


###############################################################################
config = {
    "num_atoms": 100,              # 분자의 최대 원자 수 (그래프의 노드 개수)
    "num_in_degree": 10,           # 그래프 노드의 최대 in-degree
    "num_out_degree": 10,          # 그래프 노드의 최대 out-degree
    "num_edges": 50,               # 그래프의 최대 엣지 개수
    "num_spatial": 100,            # 공간적 위치 인코딩을 위한 최대 값
    "num_edge_dis": 10,            # 엣지 거리 인코딩을 위한 최대 값
    "edge_type": "multi_hop",      # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
    "multi_hop_max_dist": 5,       # Multi-hop 엣지의 최대 거리
    "num_encoder_layers": 6,       # Graphormer 모델에서 사용할 인코더 레이어 개수
    "embedding_dim": 128,          # 임베딩 차원 크기 (노드, 엣지 등)
    "ffn_embedding_dim": 256,      # Feedforward Network의 임베딩 크기
    "num_attention_heads": 8,      # Multi-head Attention에서 헤드 개수
    "dropout": 0.1,                # 드롭아웃 비율
    "attention_dropout": 0.1,      # Attention 레이어의 드롭아웃 비율
    "activation_dropout": 0.1,     # 활성화 함수 이후 드롭아웃 비율
    "activation_fn": "gelu",       # 활성화 함수 ("gelu", "relu" 등)
    "pre_layernorm": False,        # LayerNorm을 Pre-Normalization으로 사용할지 여부
    "q_noise": 0.0,                # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
    "qn_block_size": 8,            # Quantization block 크기
    "output_size": 100,            # 모델 출력 크기
}

# Example usage
if __name__ == "__main__":
    final_loss = train_model(config=config, target_type="ex_prob", num_epochs=1000, n_pairs=5)
    print(f"Final Average Loss: {final_loss:.4f}")