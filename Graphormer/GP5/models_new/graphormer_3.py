import torch
import torch.nn as nn
from Graphormer.GP5.modules.graphormer_graph_encoder import GraphormerGraphEncoder


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer model.
    """
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class GraphormerModel(nn.Module):
    def __init__(self, config, target_type="default"):
        """
        Graphormer model with dynamically adjustable output sizes.

        Args:
            config: Configuration dictionary with model hyperparameters.
            target_type: Type of target (e.g., 'default', 'ex_prob', 'nm_distribution').
        """
        super(GraphormerModel, self).__init__()
        self.target_type = target_type
        self.embedding_dim = config["embedding_dim"]

        # Encoder initialization
        self.encoder = GraphormerGraphEncoder(
            num_atoms=config["num_atoms"],
            num_in_degree=config["num_in_degree"],
            num_out_degree=config["num_out_degree"],
            num_edges=config["num_edges"],
            num_spatial=config["num_spatial"],
            num_edge_dis=config["num_edge_dis"],
            edge_type=config["edge_type"],
            multi_hop_max_dist=config["multi_hop_max_dist"],
            num_encoder_layers=config["num_encoder_layers"],
            embedding_dim=config["embedding_dim"],
            ffn_embedding_dim=config["ffn_embedding_dim"],
            num_attention_heads=config["num_attention_heads"],
            dropout=config["dropout"],
            attention_dropout=config["attention_dropout"],
            activation_dropout=config["activation_dropout"],
            activation_fn=config["activation_fn"],
            pre_layernorm=config.get("pre_layernorm", False),
            q_noise=config.get("q_noise", 0.0),
            qn_block_size=config.get("qn_block_size", 8),
        )

        # --- MODIFICATION START ---
        # Add a linear layer for global features # 전역 특성(Solvent, Temperature, Pressure)을 처리하기 위한 선형 레이어 추가
        # global_feature_dim is now passed directly in config
        global_feature_dim = config.get("global_feature_dim", 0) 
        self.global_feature_projection = nn.Linear(global_feature_dim, self.embedding_dim)
        # --- MODIFICATION END ---

        if config.get("out_of_training", False):
            # out_of_training=True인 경우 output_size를 명시적으로 요구
            if "output_size" not in config:
                raise ValueError("config에 'output_size'를 반드시 명시해야 합니다 (out_of_training=True 일때).")
            self.output_size = config["output_size"]
            self.output_layer = nn.Linear(self.embedding_dim, self.output_size)
        else:
            self.output_size = None
            # --- MODIFICATION START ---
            # Output layer will now take concatenated embedding_dim * 2 (graph + global) # 그래프 임베딩과 전역 특성 임베딩을 연결한 후의 차원을 고려하여 출력 레이어 초기화
            # This will be dynamically initialized in forward pass, but the input size will be embedding_dim * 2
            self.output_layer = None # Will be initialized dynamically based on combined embedding
            # --- MODIFICATION END ---

        self.apply(init_graphormer_params)

    def forward(self, batched_data, targets=None, target_type=None):

        """
        Forward pass through the Graphormer model.

        Args:
            batched_data: Dictionary containing input graph graphormer_data.
            targets: Optional target tensor to dynamically adjust output size.
            target_type: Type of target (e.g., 'default', 'ex_prob', 'nm_distribution').

        Returns:
            Final output of the model.
        """
        # Use passed target_type or fallback to default
        target_type = target_type if target_type is not None else self.target_type
        #print("Current target_type:", target_type)

        # Pass through the encoder
        node_embeddings, _ = self.encoder(batched_data)

        # --- MODIFICATION START ---
        # Process global features # batched_data에서 전역 특성 추출 및 임베딩
        global_features = batched_data["global_features"]
        global_features_embedding = self.global_feature_projection(global_features)

        # Concatenate node embeddings and global features embedding # 그래프 노드 임베딩과 전역 특성 임베딩을 연결
        # Assuming node_embeddings is (batch_size, embedding_dim)
        # If node_embeddings is (batch_size, num_nodes, embedding_dim), you might need to average or pool it first
        # For Graphormer, node_embeddings is typically (batch_size, embedding_dim) after CLS token or pooling
        # Based on graphormer_graph_encoder.py, graph_rep is (B, C), so it's (batch_size, embedding_dim)
        combined_embedding = torch.cat([node_embeddings, global_features_embedding], dim=1)
        # --- MODIFICATION END ---

        #print("target type", target_type, self.target_type)
        # Dynamically initialize the output layer
        if self.output_layer is None:
            # --- MODIFICATION START ---
            # Output size for the linear layer should be based on the combined embedding dimension # 결합된 임베딩 차원을 기반으로 출력 레이어의 입력 차원 설정
            combined_embedding_dim = combined_embedding.size(-1)
            # --- MODIFICATION END ---
            if targets is not None:
                # Infer output size dynamically based on target shape
                if target_type == "default":
                    print(targets.shape)
                    output_size = targets.size(-1)  # Default target's last dimension
                    print("output_size is ",output_size)
                elif target_type == "ex_prob":
                    output_size = targets.size(1) * 2  # Number of pairs * 2 ([ex, prob] pairs)
                elif target_type == "nm_distribution":
                    output_size = targets.size(-1)  # Distribution size (e.g., 801)
                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

                #print("Dynamically determined output_size:", output_size)
            else:
                # Default sizes based on target_type when targets are not provided
                output_size = 100 if self.target_type == "default" else (
                    50 * 2 if self.target_type == "ex_prob" else 801
                )
                #print("Fallback output_size (default):", output_size)
            print("output_size.shape",output_size)
            # Initialize the output layer with dynamically determined output size
            # --- MODIFICATION START ---
            self.output_layer = nn.Linear(combined_embedding_dim, output_size).to(node_embeddings.device)
            # --- MODIFICATION END ---

        # Apply the output layer
        # --- MODIFICATION START ---
        output = self.output_layer(combined_embedding) # 결합된 임베딩을 출력 레이어에 전달
        # --- MODIFICATION END ---
        # Clamp to positive using ReLU or Softplus
        output = nn.functional.softplus(output)
        #print("Output shape before adjustment:", output.shape)


        #print("shape output", output.size(), output.shape)

        # Reshape output for 'ex_prob'
        if target_type == "ex_prob":
            output = output.view(output.size(0), -1, 2)  # [batch_size, num_pairs, 2]
            #print("Output shape after adjustment for 'ex_prob':", output.shape)


        return output



# Example Usage
if __name__ == "__main__":
    # Example configuration dictionary
    config = {
        "num_atoms": 100,
        "num_in_degree": 10,
        "num_out_degree": 10,
        "num_edges": 50,
        "num_spatial": 20,
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
        "num_unique_solvents": 5, # Added for global features # 전역 특성 사용을 위해 추가된 설정
    }

    # Instantiate and test the model
    model = GraphormerModel(config)
    batched_data = {
        "x": torch.randint(0, 100, (8, 16)),
        "in_degree": torch.randint(0, 10, (8, 16)),
        "out_degree": torch.randint(0, 10, (8, 16)),
        "edge_attr": torch.randint(0, 50, (8, 16, 16)),
        "spatial_pos": torch.randint(0, 20, (8, 16, 16)),
        "global_features": torch.randn(8, 5 + 2), # Example global features (batch_size, num_solvents + 2) # 전역 특성 예시 데이터 추가
    }
    output = model(batched_data)
    print(output.shape) # Should be [batch_size, output_size] or [batch_size, num_pairs, 2] for ex_prob
