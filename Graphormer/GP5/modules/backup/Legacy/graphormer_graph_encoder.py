import torch
import torch.nn as nn
from Graphormer.GP3.modules.graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer
from Graphormer.GP3.modules.graphormer_layers import GraphNodeFeature, GraphAttnBias

class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms,
        num_in_degree,
        num_out_degree,
        num_edges,
        num_spatial,
        num_edge_dis,
        edge_type="multi_hop",
        multi_hop_max_dist=5,
        num_encoder_layers=12,
        embedding_dim=768,
        ffn_embedding_dim=768,
        num_attention_heads=32,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_fn="gelu",
        pre_layernorm=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()

        # Node and Edge feature extraction
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            GraphormerGraphEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                pre_layernorm=pre_layernorm,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            for _ in range(num_encoder_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, batched_data):
        """
        Args:
            batched_data: Dictionary containing graph data
                - x: Node features
                - in_degree: In-degree of nodes
                - out_degree: Out-degree of nodes
                - edge_input: Multi-hop edge input
                - spatial_pos: Spatial position features
                - attn_bias: Precomputed attention bias

        Returns:
            Final node embeddings and intermediate states.
        """
        # Extract node features
        node_features = self.graph_node_feature(batched_data)

        # Compute attention bias
        attn_bias = self.graph_attn_bias(batched_data)

        # Apply layer normalization and dropout
        x = self.dropout(self.layernorm(node_features))

        # Transpose for encoder layers: (B, N, C) -> (N, B, C)
        x = x.transpose(0, 1)

        # Apply encoder layers
        intermediate_states = []
        for layer in self.layers:
            x, _ = layer(x, self_attn_bias=attn_bias)
            intermediate_states.append(x)

        # Final representation
        x = x.transpose(0, 1)  # Back to (B, N, C)
        graph_rep = x[:, 0, :]  # Graph-level representation (CLS token)

        return graph_rep, intermediate_states

