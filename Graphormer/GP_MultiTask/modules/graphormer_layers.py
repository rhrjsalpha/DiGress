import math
import torch
import torch.nn as nn
import time

def init_params(module, n_layers):
    """
    Initialize parameters for Linear and Embedding layers.
    """
    if isinstance(module, nn.Linear):
        #print(f"Initializing Linear: weight {module.weight.shape}, bias {module.bias.shape if module.bias is not None else 'None'}")
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        #print(f"Initializing Embedding: weight {module.weight.shape}")
        module.weight.data.normal_(mean=0.0, std=0.02)
        #print(f"Weight after initialization: {module.weight.shape}")

class GraphNodeFeature(nn.Module):
    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 각각의 feature encoding
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        # MLP: 입력 feature들을 concat해서 통합
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # 각각 임베딩
        atom_feat = self.atom_encoder(x).sum(dim=-2)  # [B, N, D]
        in_deg_feat = self.in_degree_encoder(in_degree)
        out_deg_feat = self.out_degree_encoder(out_degree)

        # [B, N, D*3]로 concat 후 MLP
        node_feature = torch.cat([atom_feat, in_deg_feat, out_deg_feat], dim=-1)  # [B, N, D*3]
        node_feature = self.feature_mlp(node_feature)  # [B, N, D]

        # 가상 노드 추가
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        # Embeddings for edge features and spatial positions
        #print("GraphAttnBias num_spatial",num_spatial)
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"], # In Ring
        )
        #print("-----GraphAttnBias_inputs------")
        #print('attn_bias', attn_bias.shape, attn_bias.dtype)
        #print('spatial_pos', spatial_pos.shape, spatial_pos.dtype)
        #print("spatial_pos min:", spatial_pos.min().item())
        #print("spatial_pos max:", spatial_pos.max().item())
        #print("embedding weight size:", self.spatial_pos_encoder.weight.size(0))
        #print('x', x.shape, x.dtype)
        #print('edge_input', edge_input.shape, edge_input.dtype)
        #print('attn_edge_type', attn_edge_type.shape, attn_edge_type.dtype)
        #print("--------------------------------")
        ##################################################
        #### 적절한 텐서 크기를 가지도록 텐서를 생성하는 과정 ####
        ##################################################
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone() # attn_bias를 복사해 graph_attn_bias에 넣는다. [batch,node,node]

        #print("축 추가전",graph_attn_bias.shape)
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        ) # multi-head attention을 위한 head 축을 추가 [batch,head,node,node]
        #print("축 추가후",graph_attn_bias.shape)

        # Encode spatial positions
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        #print("spatial_pos_bias.shape",spatial_pos_bias.shape) #[batch,head,node,node]
        #print("graph_attn_bias[:, :, :, :]",graph_attn_bias[:, :, :, :].shape) #[batch,head,node,node]

        #### attn_bias 에 spatial_pos_bias 를 더함 ####
        graph_attn_bias[:, :, :, :] += spatial_pos_bias
        ############################################## [batch,head,node,node] + [batch,head,node,node]

        ### Add virtual distance for the graph token 가상노드와의 distance 인 1을 추가 ###
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        #print("t, Add virtual distance for the graph token", t.shape) # [1, head, 1]

        # 기존 graph_attn_bias 크기: [batch_size, num_heads, num_nodes, num_nodes]
        batch_size, num_heads, num_nodes, _ = graph_attn_bias.size()

        # 가상 노드를 포함하도록 크기 확장
        new_bias = torch.zeros(batch_size, num_heads, num_nodes + 1, num_nodes + 1, device=graph_attn_bias.device)
        new_bias[:, :, 1:, 1:] = graph_attn_bias  # 기존 bias를 새로운 위치에 복사

        # 가상 노드와의 거리 추가
        new_bias[:, :, 1:, 0] = t  # 세로축 가상 distance , virtual node 정보가 0번째 세로축에 추가
        new_bias[:, :, 0, 1:] = t  # 가로축 가상 distance , virtual node 정보가 0번째 가로축에 추가
        #print("new_bias", new_bias.shape) # [batch_size, num_heads, num_nodes + 1, num_nodes + 1]

        #print("spatial_pos shape before:", spatial_pos.shape)

        ######################################
        #### 준비된 텐서에 값을 채워 넣는 과정 ####
        ######################################
        # Encode edge features
        if hasattr(self, "edge_dis_encoder"):
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            edge_input = self.edge_encoder(edge_input).sum(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )

            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).sum(-2).permute(0, 3, 1, 2)
        #print("spatial_pos shape after:", spatial_pos.shape)

        #print("edge_input, last", edge_input.shape)
        #print("new_bias[:, :, :, :], last", new_bias[:, :, :, :].shape)
        new_bias[:, :, 1:, 1:] += edge_input
        #print("new_bias += edge_input", new_bias.shape)

        #print("attn_bias.shape", attn_bias.shape)
        new_bias[:, :, 1:, 1:] += attn_bias.unsqueeze(1)
        #print("new_bias += attn_bias", new_bias.shape)

        return new_bias
