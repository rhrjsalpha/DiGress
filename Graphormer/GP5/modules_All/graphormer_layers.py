import math
import torch
import torch.nn as nn
import time


def init_params(module, n_layers):
    """
    Initialize parameters for Linear and Embedding layers.
    """
    if isinstance(module, nn.Linear):
        # print(f"Initializing Linear: weight {module.weight.shape}, bias {module.bias.shape if module.bias is not None else 'None'}")
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        # print(f"Initializing Embedding: weight {module.weight.shape}")
        module.weight.data.normal_(mean=0.0, std=0.02)
        # print(f"Weight after initialization: {module.weight.shape}")


class GraphNodeFeature(nn.Module):
    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, global_cat_dim=0,
                 global_cont_dim=0, num_categorical_features=7, num_continuous_features=2, mode="cls_only"):
        super(GraphNodeFeature, self).__init__()
        self.mode = mode

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.num_categorical_features = num_categorical_features

        # Encoders for categorical features
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        # Encoder for continuous features
        self.continuous_encoder = nn.Linear(num_continuous_features, hidden_dim)

        # Global feature encoders
        if self.global_cat_dim > 0:
            self.global_cat_encoder = nn.Embedding(self.global_cat_dim + 1, hidden_dim, padding_idx=0)  # +1 for padding
        else:
            self.global_cat_encoder = None

        if self.global_cont_dim > 0:
            self.global_cont_encoder = nn.Linear(self.global_cont_dim, hidden_dim)
        else:
            self.global_cont_encoder = None

        # MLP to combine all features
        # The input dimension will be hidden_dim (from summed categorical) + hidden_dim (from continuous)
        # + hidden_dim (in_degree) + hidden_dim (out_degree) + hidden_dim (global_cat) + hidden_dim (global_cont)
        mlp_input_dim = hidden_dim * 4  # Initial: categorical, in_deg, out_deg, continuous
        if self.mode == "cls_global_data":
            if self.global_cat_encoder is not None:
                mlp_input_dim += hidden_dim
            if self.global_cont_encoder is not None:
                mlp_input_dim += hidden_dim
        print("graphormer_layers, mlp_input_dim:", mlp_input_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        if self.mode == 'cls_global_model':
            self.global_token = nn.Embedding(1, hidden_dim)

        # new_global_node_embedding은 이제 사용하지 않습니다. global_cat/cont_encoder가 그 역할을 대신합니다.
        self.new_global_node_embedding = None

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x_cat, x_cont, in_degree, out_degree = (
            batched_data["x_cat"],
            batched_data["x_cont"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node, _ = x_cat.size()

        # Embed categorical features
        categorical_feat = self.atom_encoder(x_cat).sum(dim=-2)  # Summing embeddings of all categorical features

        in_deg_feat = self.in_degree_encoder(in_degree)
        out_deg_feat = self.out_degree_encoder(out_degree)

        # Encode continuous features
        continuous_feat = self.continuous_encoder(x_cont)

        # Collect all features to concatenate
        all_features = [categorical_feat, in_deg_feat, out_deg_feat, continuous_feat]

        ### Process global features ###

        print("GraphNodeFeature self.mode", self.mode)
        if self.mode  == "cls_global_data":
            if self.global_cat_encoder is not None:
                print("graphormer_layers, batched_data.keys()",batched_data.keys())
                global_features_cat = batched_data["global_features_cat"]
                global_cat_feat = self.global_cat_encoder(global_features_cat).sum(dim=-2)  # Summing embeddings
                # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim] -> [batch_size, num_nodes, hidden_dim]
                global_cat_feat = global_cat_feat.unsqueeze(1).repeat(1, n_node, 1)
                all_features.append(global_cat_feat)
                print("graphormer_layers, global_cat_feat:", global_cat_feat.size())
                print(torch.cat(all_features, dim=-1).size())

            if self.global_cont_encoder is not None:
                global_features_cont = batched_data["global_features_cont"]
                global_cont_feat = self.global_cont_encoder(global_features_cont)
                # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim] -> [batch_size, num_nodes, hidden_dim]
                global_cont_feat = global_cont_feat.unsqueeze(1).repeat(1, n_node, 1)
                all_features.append(global_cont_feat)
                print("graphormer_layers, global_cont_feat:", global_cont_feat.size())
                print(torch.cat(all_features, dim=-1).size())

        # Concatenate all features and process through MLP
        # for f in all_features:
        #     print(f.shape)
        node_feature = torch.cat(all_features, dim=-1)
        print("graphormer_layers, node feature shape:", node_feature.shape)
        node_feature = self.feature_mlp(node_feature)

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        print("graphormer_layers, mode", self.mode)
        if self.mode == "cls_only":
            graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
            return graph_node_feature

        elif self.mode == "cls_global_data":
            # global_cat/cont feature가 이미 global node 형태로 처리됨
            graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
            return graph_node_feature

        elif self.mode == "cls_global_model":
            # Global node is now integrated via global_cat_encoder/global_cont_encoder
            # The CLS token (graph_token_feature) is concatenated with node_feature
            # Global token을 global_cat/cont 기반으로 생성
            global_features = []
            if self.global_cat_encoder is not None:
                global_features_cat = batched_data["global_features_cat"]  # [B, F]
                global_cat_feat = self.global_cat_encoder(global_features_cat).sum(dim=-2)  # [B, H]
                global_features.append(global_cat_feat)

            if self.global_cont_encoder is not None:
                global_features_cont = batched_data["global_features_cont"]  # [B, F]
                global_cont_feat = self.global_cont_encoder(global_features_cont)  # [B, H]
                global_features.append(global_cont_feat)

            if global_features:
                # Combine global_cat + global_cont
                combined_global_feat = sum(global_features)  # [B, H]
            else:
                # fallback: just use learnable embedding if no global feature
                combined_global_feat = self.global_token.weight.squeeze(0).expand(n_graph, self.hidden_dim)

            global_token_feature = combined_global_feat.unsqueeze(1)  # [B, 1, H]

            # Concatenate: [CLS] + [GLOBAL] + [NODE]
            graph_node_feature = torch.cat([graph_token_feature, global_token_feature, node_feature], dim=1)
            return graph_node_feature
        else:
            raise ValueError(f"Invalid mode '{self.mode}' in GraphNodeFeature. "
                             "Expected one of: 'cls_only', 'cls_global_data', 'cls_global_model'.")


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
            mode="cls_only"
    ):
        super(GraphAttnBias, self).__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.num_atoms = num_atoms
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
        if self.mode == "cls_global_model":
            self.global_node_virtual_distance = nn.Embedding(1, num_heads)  # Added for new global node

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x_cat = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x_cat"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],  # In Ring
        )

        ##################################################
        #### 적절한 텐서 크기를 가지도록 텐서를 생성하는 과정 ####
        ##################################################
        n_graph, n_node, _ = x_cat.size()  # Changed from x.size()[:2]
        graph_attn_bias = attn_bias.clone()  # attn_bias를 복사해 graph_attn_bias에 넣는다. [batch,node,node]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # multi-head attention을 위한 head 축을 추가 [batch,head,node,node]

        # Encode spatial positions
        # np.inf 값을 처리하고 nn.Linear에 통과시키기 위해 unsqueeze(-1) 추가
        spatial_pos_processed = spatial_pos.clone()
        spatial_pos_processed[torch.isinf(spatial_pos_processed)] = 0  # inf 값을 0으로 임시 대체
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos_processed.unsqueeze(-1)).permute(0, 3, 1, 2)
        # inf였던 위치에 매우 작은 음수 값 할당
        inf_mask = torch.isinf(spatial_pos).unsqueeze(1).expand_as(spatial_pos_bias)
        spatial_pos_bias[inf_mask] = -1e9  # 연결되지 않은 노드에 매우 작은 값 할당

        #### attn_bias 에 spatial_pos_bias 를 더함 ###
        graph_attn_bias[:, :, :, :] += spatial_pos_bias
        ############################################## [batch,head,node,node] + [batch,head,node,node]

        ### Add virtual distance for the graph token ###

        # 가상 노드 수 설정 (모드에 따라)
        if self.mode == "cls_only":
            total_virtual = 1
        elif self.mode == "cls_global_data":
            total_virtual = 1  # Global node already in data
        elif self.mode == "cls_global_model":
            total_virtual = 2  # Add one more virtual node
        else:
            raise ValueError(f"Invalid GraphAttnBias mode: {self.mode}")

        # 전체 bias 크기 재설정
        batch_size = graph_attn_bias.size(0)
        new_bias = torch.full(
            (batch_size, self.num_heads, n_node + total_virtual, n_node + total_virtual),
            -1e9, device=graph_attn_bias.device
        )

        ######################################
        #### 준비된 텐서에 값을 채워 넣는 과정 ####
        ######################################
        # Encode edge features
        if hasattr(self, "edge_dis_encoder"):
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_ = torch.where(torch.isinf(spatial_pos_), torch.tensor(0.0, device=spatial_pos_.device),
                                       spatial_pos_)  # inf 값을 0으로 임시 대체
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

        print("graph_attn_bias:", graph_attn_bias.shape)
        print("edge_input:", edge_input.shape)
        print("new_bias:", new_bias.shape)
        print("total_virtual:", total_virtual)
        new_bias[:, :, total_virtual:, total_virtual:] = graph_attn_bias + edge_input

        # CLS virtual 연결
        t_cls = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        new_bias[:, :, total_virtual:, 0] = t_cls
        new_bias[:, :, 0, total_virtual:] = t_cls

        # GlobalNode 연결 (모델 내 삽입 모드일 때만)
        if self.mode == "cls_global_model":
            t_global = self.global_node_virtual_distance.weight.view(1, self.num_heads, 1)
            new_bias[:, :, total_virtual:, 1] = t_global
            new_bias[:, :, 1, total_virtual:] = t_global
            new_bias[:, :, 0, 1] = t_global.squeeze(-1)
            new_bias[:, :, 1, 0] = t_global.squeeze(-1)

        #new_bias[:, :, 2:, 2:] += edge_input

        return new_bias #new_bias.view(batch_size * self.num_heads, self.num_atoms + total_virtual, self.num_atoms + total_virtual)
