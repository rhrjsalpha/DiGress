import torch
import torch.nn as nn
from Graphormer.GP3.modules.graphormer_graph_encoder import GraphormerGraphEncoder


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
    def __init__(self, config):
        """
        Graphormer model with an encoder based on the provided configuration.

        Args:
            config: Configuration dictionary with model hyperparameters.
        """
        self.output_size = config.get("output_size", 100)

        super(GraphormerModel, self).__init__()

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
        # Output Layer to match target size
        self.output_layer = nn.Linear(config["embedding_dim"], self.output_size)  # Target size = 100

        # Parameter initialization
        self.apply(init_graphormer_params)

    def forward(self, batched_data):
        """
        Forward pass through the Graphormer model.

        Args:
            batched_data: Dictionary containing input graph data.

        Returns:
            Final node embeddings and intermediate states from the encoder.
        """

        node_embeddings, _ = self.encoder(batched_data)
        # Apply the output layer
        final_output = self.output_layer(node_embeddings)  # Adjust output size to [batch_size, 100]

        return final_output


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
    }

    # Instantiate and test the model
    model = GraphormerModel(config)
    batched_data = {
        "x": torch.randint(0, 100, (8, 16)),
        "in_degree": torch.randint(0, 10, (8, 16)),
        "out_degree": torch.randint(0, 10, (8, 16)),
        "edge_attr": torch.randint(0, 50, (8, 16, 16)),
        "spatial_pos": torch.randint(0, 20, (8, 16, 16)),
    }
    output = model(batched_data)
    print(output)
