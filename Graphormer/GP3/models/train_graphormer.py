import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP3.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
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
criterion = nn.MSELoss()  # Assuming a regression task
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
        outputs, _ = model(batched_data)

        # Compute loss
        print(outputs.shape, targets.shape)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

print("Training complete.")

# [eV_1, eV_2,.. eV_50, Osc_1, .... Osc_50] 이런 형태로 학습이 진행됨



