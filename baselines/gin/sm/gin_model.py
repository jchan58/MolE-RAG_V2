"""
GIN model — faithful port of snap-stanford/pretrain-gnns `chem/model.py` (GINConv + GNN +
GNN_graphpred, gnn_type='gin') to modern PyTorch Geometric (2.x).

The architecture, featurization vocabulary, and forward computation are identical to the original;
only three PyG-1.x -> 2.x API differences are adapted:
  * `add_self_loops(...)` now returns `(edge_index, edge_attr)` -> we unpack `edge_index, _`.
  * `self.propagate(aggr, edge_index, ...)` -> aggregation is set in `super().__init__(aggr=...)`
    and `propagate(edge_index, ...)` is called without the positional aggr.
  * `torch_scatter` is not needed for GIN (only GCN's norm used it), so it is dropped.

Original: https://github.com/snap-stanford/pretrain-gnns  (Hu et al., ICLR 2020).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops

# Featurization vocabulary (must match featurize.py / original loader.py)
num_atom_type = 120       # 118 elements + mask + self-loop headroom (original value)
num_chirality_tag = 3
num_bond_type = 6         # 4 bond types + aromatic/self-loop + mask (original value)
num_bond_direction = 3


class GINConv(MessagePassing):
    """GIN aggregation that incorporates edge features by concatenation (edge embeddings added
    to neighbour messages). Identical to pretrain-gnns GINConv."""

    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # features for self-loop edges: bond type idx 4 (self-loop), bond dir idx 0
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """5-layer (default) GIN node encoder with atom-type + chirality input embeddings, BatchNorm,
    ReLU+dropout (no ReLU on the final layer), JK readout. Identical to pretrain-gnns GNN."""

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin"):
        super().__init__()
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        if gnn_type != "gin":
            raise ValueError("This port supports gnn_type='gin' only.")
        self.gnns = torch.nn.ModuleList([GINConv(emb_dim, aggr="add") for _ in range(num_layer)])
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x, edge_index, edge_attr):
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:  # no ReLU on last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        elif self.JK == "last":
            return h_list[-1]
        elif self.JK == "max":
            return torch.max(torch.cat([h.unsqueeze(0) for h in h_list], dim=0), dim=0)[0]
        elif self.JK == "sum":
            return torch.sum(torch.cat([h.unsqueeze(0) for h in h_list], dim=0), dim=0)
        raise ValueError("Invalid JK.")


class GNN_graphpred(torch.nn.Module):
    """Graph-level prediction head: GIN encoder -> graph pooling -> linear to num_tasks.
    Random init (trained from scratch). Identical to pretrain-gnns GNN_graphpred with gnn_type='gin'."""

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0.0,
                 graph_pooling="mean", gnn_type="gin"):
        super().__init__()
        self.num_layer = num_layer
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        pools = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}
        if graph_pooling not in pools:
            raise ValueError("This port supports graph_pooling in {mean,sum,max}.")
        self.pool = pools[graph_pooling]
        in_dim = (self.num_layer + 1) * emb_dim if JK == "concat" else emb_dim
        self.graph_pred_linear = torch.nn.Linear(in_dim, num_tasks)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_rep = self.gnn(x, edge_index, edge_attr)
        return self.graph_pred_linear(self.pool(node_rep, batch))
