# graphormer_torch.py

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from Graphormer.GP.modules.graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params

logger = logging.getLogger(__name__)

class GraphormerModel(nn.Module):
    def __init__(self, args):
        super(GraphormerModel, self).__init__()
        self.args = args

        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)

        self.encoder = GraphormerEncoder(args)

        self.encoder_embed_dim = args.encoder_embed_dim
        if args.pretrained_model_name != "none":
            self.load_state_dict(torch.load(args.pretrained_model_name))
            if not args.load_pretrained_model_output_layer:
                self.encoder.reset_output_layer_parameters()

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(nn.Module):
    def __init__(self, args):
        super(GraphormerEncoder, self).__init__()
        self.max_nodes = args.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            # Graphormer-specific arguments
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # General model parameters
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
        )

        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)
        self.activation_fn = self.get_activation_fn(args.activation_fn)

        self.embed_out = nn.Linear(args.encoder_embed_dim, args.num_classes, bias=False)

    def reset_output_layer_parameters(self):
        self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        # Apply activation and layer normalization
        x = self.layer_norm(self.activation_fn(x))

        # Output projection
        x = self.embed_out(x)

        return x

    @staticmethod
    def get_activation_fn(activation_fn):
        if activation_fn == "gelu":
            return F.gelu
        elif activation_fn == "relu":
            return F.relu
        elif activation_fn == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")


# Argument parsing and default values
def get_default_args():
    class Args:
        pass

    args = Args()
    args.num_atoms = 512
    args.num_in_degree = 256
    args.num_out_degree = 256
    args.num_edges = 128
    args.num_spatial = 128
    args.num_edge_dis = 32
    args.edge_type = "type"
    args.multi_hop_max_dist = 5
    args.encoder_layers = 6
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 3072
    args.encoder_attention_heads = 8
    args.dropout = 0.1
    args.attention_dropout = 0.1
    args.act_dropout = 0.0
    args.encoder_normalize_before = True
    args.pre_layernorm = False
    args.apply_graphormer_init = True
    args.activation_fn = "gelu"
    args.num_classes = 10
    args.max_nodes = 128
    args.pretrained_model_name = "none"
    args.load_pretrained_model_output_layer = False
    return args

# Example usage
if __name__ == "__main__":
    args = get_default_args()
    model = GraphormerModel(args)
    dummy_data = {"x": torch.randint(0, 10, (8, 128, 16))}
    output = model(dummy_data)
    print(output.shape)
