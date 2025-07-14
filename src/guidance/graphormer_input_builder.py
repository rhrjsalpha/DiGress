import torch
import numpy as np
from src.diffusion.extra_features_RDKit import RDKitExtraFeatures
from src.diffusion.extra_features import ExtraFeatures


class GraphormerInputBuilder:
    def __init__(self, cfg, dataset_info):
        self.cfg = cfg
        self.dataset_info = dataset_info
        self.rdkit_features_extractor = RDKitExtraFeatures(dataset_info)
        self.graph_features_extractor = ExtraFeatures('all', dataset_info)
        self.multi_hop_max_dist = cfg.model.multi_hop_max_dist
        self.spatial_pos_max = cfg.model.spatial_pos_max

    def build_input(self, noisy_data, extra_conditions=None):
        # 1. Extract base features
        rdkit_placeholder = self.rdkit_features_extractor(noisy_data)
        rdkit_node_features = rdkit_placeholder.X

        graph_placeholder = self.graph_features_extractor(noisy_data)
        graph_node_features = graph_placeholder.X
        graph_global_features = graph_placeholder.y

        # 2. Process extra conditions
        condition_features = self._encode_conditions(extra_conditions, batch_size=noisy_data['X_t'].size(0))

        # 3. Combine features
        final_node_features = torch.cat([rdkit_node_features, graph_node_features], dim=-1)
        virtual_node_features = torch.cat([graph_global_features, condition_features], dim=-1).unsqueeze(1)
        final_x = torch.cat([virtual_node_features, final_node_features], dim=1)

        # 4. Build Graphormer-specific graph structures from noisy_data
        batched_data_list = []
        for i in range(noisy_data['X_t'].size(0)):
            n_nodes = int(noisy_data['node_mask'][i].sum())
            adj = (noisy_data['E_t'][i, :n_nodes, :n_nodes].argmax(dim=-1) > 0).int()
            attn_edge_type = torch.zeros(n_nodes, n_nodes, 1, dtype=torch.long)
            
            shortest_path_result = self._compute_shortest_paths(adj.cpu().numpy())
            spatial_pos = torch.from_numpy(shortest_path_result).long()

            attn_bias = torch.zeros(n_nodes + 1, n_nodes + 1, dtype=torch.float)
            attn_bias[1:, 1:][spatial_pos >= self.spatial_pos_max] = float('-inf')

            edge_input = self._generate_edge_input(shortest_path_result, attn_edge_type.cpu().numpy())

            batched_data_list.append({
                'x': final_x[i],
                'attn_bias': attn_bias,
                'attn_edge_type': attn_edge_type,
                'spatial_pos': spatial_pos,
                'in_degree': adj.long().sum(dim=0),
                'out_degree': adj.long().sum(dim=1),
                'edge_input': torch.from_numpy(edge_input).long()
            })

        # 5. Collate the batch
        return self._collate_batch(batched_data_list)

    def _encode_conditions(self, conditions, batch_size):
        # Placeholder for condition encoding
        return torch.zeros(batch_size, 10)

    def _compute_shortest_paths(self, adj):
        num_nodes = adj.shape[0]
        dist = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(dist, 0)
        for i, j in np.argwhere(adj == 1):
            dist[i, j] = 1
        for k in range(num_nodes):
            dist = np.minimum(dist, dist[np.newaxis, k, :] + dist[:, k, np.newaxis])
        return dist

    def _generate_edge_input(self, shortest_path, attn_edge_type):
        num_nodes = shortest_path.shape[0]
        edge_input = np.zeros((num_nodes, num_nodes, self.multi_hop_max_dist, attn_edge_type.shape[-1]), dtype=np.int64)
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist = int(shortest_path[i, j])
                if dist < self.multi_hop_max_dist:
                    edge_input[i, j, dist - 1] = attn_edge_type[i, j]
        return edge_input

    def _collate_batch(self, items):
        max_node_num = max(item['x'].size(0) for item in items)
        
        # Padding functions from collator.py logic
        def pad_1d(x, padlen): return torch.cat([x, x.new_zeros(padlen - x.size(0))])
        def pad_2d(x, padlen): return torch.cat([x, x.new_zeros(padlen - x.size(0), x.size(1))], dim=0)
        def pad_attn_bias(x, padlen): 
            new_x = x.new_zeros([padlen, padlen], dtype=torch.float).fill_(float("-inf"))
            new_x[:x.size(0), :x.size(1)] = x
            return new_x
        def pad_edge_type(x, padlen): 
            new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
            new_x[:x.size(0), :x.size(1), :] = x
            return new_x
        def pad_spatial_pos(x, padlen): 
            new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
            new_x[:x.size(0), :x.size(1)] = x
            return new_x
        def pad_3d(x, p1, p2, p3): 
            new_x = x.new_zeros([p1, p2, p3, x.size(-1)], dtype=x.dtype)
            new_x[:x.size(0), :x.size(1), :x.size(2), :] = x
            return new_x

        x = torch.stack([pad_2d(item['x'], max_node_num) for item in items])
        attn_bias = torch.stack([pad_attn_bias(item['attn_bias'], max_node_num) for item in items])
        attn_edge_type = torch.stack([pad_edge_type(item['attn_edge_type'], max_node_num) for item in items])
        spatial_pos = torch.stack([pad_spatial_pos(item['spatial_pos'], max_node_num) for item in items])
        in_degree = torch.stack([pad_1d(item['in_degree'], max_node_num) for item in items])
        edge_input = torch.stack([pad_3d(item['edge_input'], max_node_num, max_node_num, self.multi_hop_max_dist) for item in items])

        return {
            'x': x,
            'attn_bias': attn_bias,
            'attn_edge_type': attn_edge_type,
            'spatial_pos': spatial_pos,
            'in_degree': in_degree,
            'out_degree': in_degree, # Undirected graph
            'edge_input': edge_input
        }

