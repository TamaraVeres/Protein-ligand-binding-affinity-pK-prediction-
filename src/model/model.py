import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, BatchNorm
from torch_geometric.utils import degree


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(BatchNorm(hidden_dim))

        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def encode_nodes(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new
        return h

    def pool(self, h, batch):
        g = global_max_pool(h, batch)
        return F.relu(self.proj(g))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encode_nodes(x, edge_index, edge_attr)
        return self.pool(h, batch)


def pocket_shape(pocket_data) -> torch.Tensor:
    x = pocket_data.x
    pos = pocket_data.pos
    if hasattr(pocket_data, "batch") and pocket_data.batch is not None:
        batch = pocket_data.batch
    else:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    n_graphs = int(batch.max().item()) + 1
    num_res = degree(batch, num_nodes=n_graphs, dtype=torch.float).unsqueeze(1)

    mean_pos = global_mean_pool_(pos, batch, n_graphs)
    node_center = mean_pos[batch]
    dists = torch.norm(pos - node_center, dim=1, keepdim=True)
    mean_dist = global_mean_pool_(dists, batch, n_graphs)
    mean_dist2 = global_mean_pool_(dists ** 2, batch, n_graphs)
    std_dist = torch.sqrt((mean_dist2 - mean_dist ** 2).clamp(min=0))
    mean_x = global_mean_pool_(x, batch, n_graphs)
    return torch.cat([num_res, mean_dist, std_dist, mean_x], dim=1)


def global_mean_pool_(x, batch, size):
    from torch_geometric.nn import global_mean_pool
    return global_mean_pool(x, batch, size=size)


class AffinityModel(nn.Module):
    """3-branch affinity model: ligand GNN + pocket GNN + global features.

    Reconstructed to match the trained checkpoint exactly (strict load).
    """

    def __init__(self, ligand_in_dim: int = 32, pocket_in_dim: int = 47,
                 lig_graph_feat_in_dim: int = 51, poc_graph_feat_in_dim: int = 79,
                 hidden_dim: int = 256, ligand_edge_dim: int = 8, pocket_edge_dim: int = 9,
                 aa_embed_dim: int = 8, ablate=None, feature_stats=None):
        super().__init__()
        self.aa_embed_dim = aa_embed_dim
        self.ablate = ablate if ablate is not None else []
        self.aa_embedding = nn.Embedding(21, aa_embed_dim)

        self.ligand_gnn = GNNEncoder(ligand_in_dim, hidden_dim, ligand_edge_dim, num_layers=4)
        self.pocket_gnn = GNNEncoder(pocket_in_dim + aa_embed_dim, hidden_dim, pocket_edge_dim, num_layers=4)

        shape_in_dim = 3 + pocket_in_dim
        global_in_dim = lig_graph_feat_in_dim + poc_graph_feat_in_dim + shape_in_dim
        self.global_fc = nn.Linear(global_in_dim, hidden_dim)

        fusion_in_dim = 3 * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim // 2, 1),
        )

 
        for name, dim in [
            ("lig_x", ligand_in_dim), ("lig_edge", ligand_edge_dim), ("lig_graph", lig_graph_feat_in_dim),
            ("poc_x", pocket_in_dim), ("poc_edge", pocket_edge_dim), ("poc_graph", poc_graph_feat_in_dim),
        ]:
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))

  
        if feature_stats is not None:
            for name, (mean, std) in feature_stats.items():
                getattr(self, f"{name}_mean").copy_(mean)
                getattr(self, f"{name}_std").copy_(std)

    def _get_batch(self, data):
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch
        return torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    @staticmethod
    def _std(x, mean, std):
        return (x - mean) / std

    def forward(self, ligand_data, pocket_data) -> torch.Tensor:
        lig_x = self._std(ligand_data.x, self.lig_x_mean, self.lig_x_std)
        if ligand_data.edge_attr.size(0) > 0:
            lig_edge = self._std(ligand_data.edge_attr, self.lig_edge_mean, self.lig_edge_std)
        else:
            lig_edge = ligand_data.edge_attr
        h_lig = self.ligand_gnn(lig_x, ligand_data.edge_index, lig_edge, self._get_batch(ligand_data))

        poc_x_norm = self._std(pocket_data.x, self.poc_x_mean, self.poc_x_std)
        if pocket_data.edge_attr.size(0) > 0:
            poc_edge = self._std(pocket_data.edge_attr, self.poc_edge_mean, self.poc_edge_std)
        else:
            poc_edge = pocket_data.edge_attr
        aa_emb = self.aa_embedding(pocket_data.aa_idx)
        poc_x = torch.cat([poc_x_norm, aa_emb], dim=-1)
        h_poc = self.pocket_gnn(poc_x, pocket_data.edge_index, poc_edge, self._get_batch(pocket_data))

        n_lig = int(self._get_batch(ligand_data).max().item()) + 1
        n_poc = int(self._get_batch(pocket_data).max().item()) + 1
        lig_graph = self._std(ligand_data.lig_graph_feat.view(n_lig, -1), self.lig_graph_mean, self.lig_graph_std)
        poc_graph = self._std(pocket_data.poc_graph_feat.view(n_poc, -1), self.poc_graph_mean, self.poc_graph_std)
        shape_vec = pocket_shape(pocket_data)
        global_vec = torch.cat([lig_graph, poc_graph, shape_vec], dim=-1)
        h_global = F.relu(self.global_fc(global_vec))

        # L2-normalize each branch before fusion 
        h_lig = F.normalize(h_lig, dim=-1)
        h_poc = F.normalize(h_poc, dim=-1)
        h_global = F.normalize(h_global, dim=-1)


        if "lig" in self.ablate:
            h_lig = torch.zeros_like(h_lig)
        if "poc" in self.ablate:
            h_poc = torch.zeros_like(h_poc)
        if "global" in self.ablate:
            h_global = torch.zeros_like(h_global)

        h_all = torch.cat([h_lig, h_poc, h_global], dim=-1)
        return self.fusion(h_all).squeeze(-1)
