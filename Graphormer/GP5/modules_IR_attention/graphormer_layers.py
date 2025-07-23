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
    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, global_cat_dim=0, global_cont_dim=0, num_categorical_features=7, num_continuous_features=2):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.num_categorical_features = num_categorical_features
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree

        # Encoders for categorical features
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        
        # Encoder for continuous features
        self.cont_input_dim = num_continuous_features + global_cont_dim
        self.continuous_encoder = nn.Linear(self.cont_input_dim, hidden_dim)

        # Global feature encoders
        if self.global_cat_dim > 0:
            self.global_cat_encoder = nn.Embedding(self.global_cat_dim + 1, hidden_dim, padding_idx=0) # +1 for padding
        else:
            self.global_cat_encoder = None

        if self.global_cont_dim > 0:
            self.global_cont_encoder = nn.Linear(self.global_cont_dim, hidden_dim)
        else:
            self.global_cont_encoder = None

        # MLP to combine all features
        # The input dimension will be hidden_dim (from summed categorical) + hidden_dim (from continuous)
        # + hidden_dim (in_degree) + hidden_dim (out_degree) + hidden_dim (global_cat) + hidden_dim (global_cont)
        mlp_input_dim = hidden_dim * 4 # Initial: categorical, in_deg, out_deg, continuous
        if self.global_cat_encoder is not None: mlp_input_dim += hidden_dim
        if self.global_cont_encoder is not None: mlp_input_dim += hidden_dim

        self.feature_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        # self.global_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):

        x_cat, x_cont, in_degree, out_degree = (
            batched_data["x_cat"],
            batched_data["x_cont"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        in_degree  = in_degree.clamp_(0, self.num_in_degree - 1)
        out_degree = out_degree.clamp_(0, self.num_in_degree - 1)

        n_graph, n_node, _ = x_cat.size()

        # Embed categorical features
        categorical_feat = self.atom_encoder(x_cat).sum(dim=-2) # Summing embeddings of all categorical features

        in_deg_feat = self.in_degree_encoder(in_degree)
        out_deg_feat = self.out_degree_encoder(out_degree)
        
        # Encode continuous features
        continuous_feat = self.continuous_encoder(x_cont)

        # Collect all features to concatenate
        all_features = [categorical_feat, in_deg_feat, out_deg_feat, continuous_feat]

        # Process global features if they exist
        if self.global_cat_encoder is not None:
            global_features_cat = batched_data["global_features_cat"]
            global_cat_feat = self.global_cat_encoder(global_features_cat).sum(dim=-2) # Summing embeddings
            # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim] -> [batch_size, num_nodes, hidden_dim]
            global_cat_feat = global_cat_feat.unsqueeze(1).repeat(1, n_node, 1)
            all_features.append(global_cat_feat)

        if self.global_cont_encoder is not None:
            global_features_cont = batched_data["global_features_cont"]
            global_cont_feat = self.global_cont_encoder(global_features_cont)
            # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim] -> [batch_size, num_nodes, hidden_dim]
            global_cont_feat = global_cont_feat.unsqueeze(1).repeat(1, n_node, 1)
            all_features.append(global_cont_feat)

        node_feature = torch.cat(all_features, dim=-1)
        node_feature = self.feature_mlp(node_feature)

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        #global_token_feature = self.global_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

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
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        # self.edge_encoder = nn.Linear(num_edges, hidden_dim)
        # spatial_pos_encoder를 nn.Linear로 변경
        self.spatial_pos_encoder = nn.Linear(1, num_heads)

        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )

        ## Edge Flag / 가상 거리 (Virtual Distance) 정의 및 적용 ##
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        # self.new_global_node_virtual_distance = nn.Embedding(1, num_heads) # Added for new global node

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x_cat = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x_cat"],
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
        #print('x_cat', x_cat.shape, x_cat.dtype) # Changed from x
        #print('edge_input', edge_input.shape, edge_input.dtype)
        #print('attn_edge_type', attn_edge_type.shape, attn_edge_type.dtype)
        #print("--------------------------------")
        ##################################################
        #### 적절한 텐서 크기를 가지도록 텐서를 생성하는 과정 ####
        ##################################################
        n_graph, n_node, _ = x_cat.size() # Changed from x.size()[:2]
        graph_attn_bias = attn_bias.clone() # attn_bias를 복사해 graph_attn_bias에 넣는다. [batch,node,node]

        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        ) # multi-head attention을 위한 head 축을 추가 [batch,head,node,node]
        #print("축 추가후",graph_attn_bias.shape)

        # Encode spatial positions
        # np.inf 값을 처리하고 nn.Linear에 통과시키기 위해 unsqueeze(-1) 추가
        spatial_pos_processed = spatial_pos.clone()
        spatial_pos_processed[torch.isinf(spatial_pos_processed)] = 0 # inf 값을 0으로 임시 대체
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos_processed.unsqueeze(-1)).permute(0, 3, 1, 2)
        # inf였던 위치에 매우 작은 음수 값 할당
        inf_mask = torch.isinf(spatial_pos).unsqueeze(1).expand_as(spatial_pos_bias)
        spatial_pos_bias[inf_mask] = -1e9 # 연결되지 않은 노드에 매우 작은 값 할당

        #print("spatial_pos_bias.shape",spatial_pos_bias.shape) #[batch,head,node,node]
        #print("graph_attn_bias[:, :, :, :]",graph_attn_bias[:, :, :, :].shape) #[batch,head,node,node]

        #### attn_bias 에 spatial_pos_bias 를 더함 ###
        graph_attn_bias[:, :, :, :] += spatial_pos_bias
        ############################################## [batch,head,node,node] + [batch,head,node,node]

        # 기존 graph_attn_bias 크기: [batch_size, num_heads, num_nodes, num_nodes]
        batch_size, num_heads, num_nodes, _ = graph_attn_bias.size()

        # 가상 노드를 포함하도록 크기 확장 (CLS + Nodes)
        new_bias = torch.full((batch_size, num_heads,num_nodes + 1, num_nodes + 1), -1e9, device=graph_attn_bias.device)
        new_bias[:, :, 1:, 1:] = graph_attn_bias

        ### 가상거리를 `new_bias` 행렬에 적용하는 부분 ###
        # CLS/VNode (인덱스 0)와의 거리 추가
        t_cls = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        new_bias[:, :, 1:, 0] = t_cls  # CLS -> New Global Node, Nodes
        new_bias[:, :, 0, 1:] = t_cls  # New Global Node, Nodes -> CLS

        #print("new_bias", new_bias.shape) # [batch_size, num_heads, num_nodes + 2, num_nodes + 2]

        #print("spatial_pos shape before:", spatial_pos.shape)

        ######################################
        #### 준비된 텐서에 값을 채워 넣는 과정 ####
        ######################################
        # Encode edge features
        if hasattr(self, "edge_dis_encoder"):
            spatial_pos_ = spatial_pos.clone()

            spatial_pos_ = torch.where(torch.isinf(spatial_pos_), torch.tensor(0.0, device=spatial_pos_.device), spatial_pos_) # inf 값을 0으로 임시 대체 # inf → 0으로 대체 (안전하게 처리)

            # self-loop 포함: 거리 0이었던 것들은 모두 1로 간주 (softmax 분모 방지용)
            spatial_pos_[spatial_pos_ == 0] = 1

            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)

            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            edge_input = self.edge_encoder(edge_input).sum(-2)

            # ── NEW: 현재 텐서에서 실제 크기를 읽어온다 ─────────────────
            B, N, _, D, H = edge_input.shape  # B=batch, N=nodes, D=max_dist, H=num_heads

            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )

            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                D, -1, self.num_heads  # (D, B*N*N, H)
            )

            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.view(-1, H, H)[:D]
            )

            edge_input = edge_input_flat.reshape(
                D, B, N, N, H
            ).permute(1, 2, 3, 0, 4)  # (B, N, N, D, H)

            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).sum(-2).permute(0, 3, 1, 2)

        new_bias[:, :, 1:, 1:] += edge_input

        return new_bias.view(batch_size * num_heads, num_nodes + 1, num_nodes + 1)
