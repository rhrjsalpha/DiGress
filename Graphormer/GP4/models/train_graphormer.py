import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP4.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from graphormer import GraphormerModel
import os

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

# Initialize dataset and DataLoader
dataset = SMILESDataset(csv_file="../../data/data_example.csv", attn_bias_w=1.0)  # Include edge weight
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Initialize the model, loss function, and optimizer
model = GraphormerModel(config)

# 손실 함수 정의 두개에 대해서 다르게
# nn.SmoothL1Loss() nn.L1Loss() nn.MSELoss() log_cosh_loss(outputs_prob, targets_prob)
criterion_ex = nn.MSELoss()  # For 'ex' part
criterion_prob = nn.L1Loss()  # For 'prob' part (예: Binary Cross Entropy)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def move_to_device(batch, device):
    """
    Move batch data to the specified device.
    """
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # Move batch data to the device
        batch = move_to_device(batch, device)

        # Extract inputs and targets from the batch
        batched_data = {
            "x": batch["x"],
            "adj": batch["adj"],
            "in_degree": batch["in_degree"],
            "out_degree": batch["out_degree"],
            "spatial_pos": batch["spatial_pos"],
            "attn_bias": batch["attn_bias"],
            "edge_input": batch["edge_input"],  # Include edge_input
            "attn_edge_type": batch["attn_edge_type"],  # Ensure attn_edge_type is included
        }
        targets = batch["targets"]

        # Forward pass
        outputs = model(batched_data)  # Outputs shape: [batch_size, 100]

        # Split outputs and targets
        outputs_ex = outputs[:, :50, 0]  # Take the 'ex' part (first column if [ex, prob])
        outputs_prob = outputs[:, :50, 1]  # Take the 'prob' part (second column)
        targets_ex = targets[:, :50]  # 'ex' targets
        targets_prob = targets[:, 50:]  # 'prob' targets

        # Compute individual losses
        #print("loss_ex",outputs_ex.shape,targets_ex.shape)
        loss_ex = criterion_ex(outputs_ex, targets_ex)  # Compute MSE loss for 'ex'
        #print("loss_prob",outputs_prob.shape,targets_prob.shape)
        loss_prob = criterion_prob(outputs_prob, targets_prob)  # Compute BCE loss for 'prob'
        # Combine losses
        total_loss = loss_ex + loss_prob  # Adjust weights if necessary

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss += total_loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

print("Training complete.")

# [[eV_1,Osc_1] [eV_2,Osc_2].. [eV_50, Osc_50]] 이런 형태로 학습이 진행됨



