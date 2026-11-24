"""Self-contained MolLM (Gerstein Lab, 2D+3D) fine-tuning helpers.

Builds the released MolLM molecular graph encoder (TransformerM / Transformer-M graph
transformer over OGB 2D features + RDKit ETKDG 3D conformers), loads the released pretrained
graph-encoder weights, and attaches a fresh prediction head -- faithful to the repo's
downstream/MoleculePrediction/{model.py,loader.py}.  We instantiate TransformerM directly and
copy `preprocess_item` / `collator_3d` verbatim so we avoid importing torch_geometric / torch_scatter
(the repo's loader.MoleculeDataset is tied to an old PyG API we don't need: we feed OUR molecules).
"""
import os, sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# ---- locate bundled graph-transformer source + build the compiled `algos` (floyd-warshall) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.environ.get("GT_DIR", os.path.join(_HERE, "mollm_src", "graph-transformer"))
_DATA_DIR = os.path.join(GT_DIR, "Transformer_M", "data")
if GT_DIR not in sys.path:
    sys.path.insert(0, GT_DIR)

import pyximport  # noqa: E402
os.makedirs("/tmp/pyxbld", exist_ok=True)
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  build_dir="/tmp/pyxbld", language_level=3)
sys.path.insert(0, _DATA_DIR)
import algos  # noqa: E402  (Transformer_M/data/algos.pyx)

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit import RDLogger  # noqa: E402
RDLogger.DisableLog("rdApp.*")


class AttrDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self


# ============================================================================
# Featurisation -- faithful to loader.mol_to_graph_data_obj_simple (OGB 9-dim atom /
# 3-dim bond features + ETKDG 3D conformer + MMFF optimise), returned as an AttrDict.
# ============================================================================
MAX_EMBED_ATOMS = int(os.environ.get("MAX_EMBED_ATOMS", "128"))  # skip 3D embed for huge molecules


def mol_to_item(smiles, y, embed_seed=0xf00d):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    bad = False
    # Skip 3D embedding for very large molecules: ETKDG/MMFF on them is slow and can crash/OOM a
    # worker; fall back to zero coords (2D graph kept), same as the repo's embed-failure path.
    if mol.GetNumAtoms() > MAX_EMBED_ATOMS:
        bad = True
    try:
        if not bad:
            params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDG()
            params.randomSeed = int(embed_seed)
            params.maxIterations = 200
            if AllChem.EmbedMolecule(mol, params) == -1:
                bad = True
    except Exception:
        bad = True
    if not bad:
        try:
            m2 = Chem.Mol(mol)
            AllChem.MMFFOptimizeMolecule(m2)
            mol = m2
        except Exception:
            pass
    mol = Chem.RemoveHs(mol)

    atom_feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    if len(atom_feats) == 0:
        return None
    x = torch.tensor(np.array(atom_feats), dtype=torch.long)  # [N,9]

    if len(mol.GetBonds()) > 0:
        edges, efeat = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            f = bond_to_feature_vector(b)
            edges += [(i, j), (j, i)]
            efeat += [f, f]
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(efeat, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 3), dtype=np.int64)

    try:
        if not bad:
            pos = mol.GetConformer().GetPositions()
        else:
            pos = np.zeros((mol.GetNumAtoms(), 3))
    except Exception:
        pos = np.zeros((mol.GetNumAtoms(), 3))
    if pos.shape[0] != x.size(0):
        pos = np.zeros((x.size(0), 3))

    item = AttrDict()
    item.x = x
    item.edge_index = torch.from_numpy(edge_index).to(torch.int64)
    item.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
    item.pos = torch.from_numpy(np.asarray(pos)).to(torch.float32)
    item.y = torch.tensor(np.asarray(y), dtype=torch.float).view(-1)  # [T]
    return preprocess_item(item)


# ---- from Transformer_M/data/{wrapper,collator}.py (avoids importing torch_geometric).
# Uses size(-1) (the collator.py variant) so it is correct for both the 2D node/edge feature
# tensors in preprocess_item AND the 4D node_type_edge tensor in collator_3d. NOT jit-scripted
# (plain eager) to avoid any TorchScript fusion.
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x
    N = x.size(0)
    x = convert_to_single_emb(x)
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree
    item.edge_input = torch.from_numpy(edge_input).long()
    return item


# ---- copied verbatim from Transformer_M/data/collator.py (pure torch) ----
def pad_1d_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        nx = x.new_zeros([padlen], dtype=x.dtype); nx[:xlen] = x; x = nx
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        nx = x.new_zeros([padlen, xdim], dtype=x.dtype); nx[:xlen, :] = x; x = nx
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        nx = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        nx[:xlen, :xlen] = x; nx[xlen:, :xlen] = 0; x = nx
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        nx = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype); nx[:xlen, :xlen, :] = x; x = nx
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        nx = x.new_zeros([padlen, padlen], dtype=x.dtype); nx[:xlen, :xlen] = x; x = nx
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        nx = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        nx[:xlen1, :xlen2, :xlen3, :] = x; x = nx
    return x.unsqueeze(0)


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        nx = x.new_zeros([padlen, xdim], dtype=x.dtype); nx[:xlen, :] = x; x = nx
    return x.unsqueeze(0)


def collator_3d(items, max_node=512, multi_hop_max_dist=5, spatial_pos_max=1024):
    items = [it for it in items if it is not None and it.x.size(0) <= max_node]
    items = [(it.idx, it.attn_bias, it.attn_edge_type, it.spatial_pos, it.in_degree,
              it.out_degree, it.x, it.edge_input[:, :, :multi_hop_max_dist, :], it.y, it.pos)
             for it in items]
    (idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees,
     xs, edge_inputs, ys, poses) = zip(*items)
    for i, _ in enumerate(attn_biases):
        attn_biases[i][1:, 1:][spatial_poses[i] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    node_type_edges = []
    for k in range(len(items)):
        node_atom_type = items[k][6][:, 0]
        n_nodes = items[k][6].shape[0]
        ai = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        ai = pad_spatial_pos_unsqueeze(ai, max_node_num).unsqueeze(-1)
        aj = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        aj = pad_spatial_pos_unsqueeze(aj, max_node_num).unsqueeze(-1)
        nae = torch.cat([ai, aj], dim=-1)
        nae = convert_to_single_emb(nae)
        node_type_edges.append(nae.long())
    node_type_edge = torch.cat(node_type_edges)
    return dict(idx=torch.LongTensor(idxs), attn_bias=attn_bias, attn_edge_type=attn_edge_type,
                spatial_pos=spatial_pos, in_degree=in_degree, out_degree=in_degree, x=x,
                edge_input=edge_input, y=y, pos=pos, node_type_edge=node_type_edge)


def custom_collate(items):
    for i, it in enumerate(items):
        it.idx = i
    return collator_3d(items, max_node=512, multi_hop_max_dist=5, spatial_pos_max=1024)


# ============================================================================
# Model -- faithful to downstream/MoleculePrediction/model.py GNN_graphpred, but instantiated
# directly (no torch_geometric / torch_scatter imports).
# ============================================================================
def _tm_args(use_3d=True):
    return SimpleNamespace(
        _name="transformer_m_base", act_dropout=0.3, activation_fn="gelu", add_3d=use_3d,
        apply_init=True, arch="transformer_m_base", attention_dropout=0.3, dropout=0.1,
        droppath_prob=0.1, edge_type="multi_hop", encoder_attention_heads=32,
        encoder_embed_dim=768, encoder_ffn_embed_dim=768, encoder_layers=12,
        encoder_learned_pos=True, encoder_normalize_before=True, max_positions=512,
        multi_hop_max_dist=5, no_2d=False, noise_scale=0.2, num_3d_bias_kernel=128,
        num_atoms=4608, num_classes=1, num_edge_dis=128, num_edges=1536, num_in_degree=512,
        num_out_degree=512, num_segment=2, num_spatial=512, sandwich_ln=False,
        mode_prob="0.2,0.2,0.6", no_token_positional_embeddings=False, remove_head=False,
    )


class MolLMPredictor(nn.Module):
    def __init__(self, num_tasks, use_3d=True):
        super().__init__()
        from Transformer_M.models.transformer_m import TransformerM
        # The 3D-bias GaussianLayer calls a module-level @torch.jit.script `gaussian`; its TorchScript
        # graph gets TE-fused and nvrtc-compiled for the GPU arch at runtime, which torch 1.9.1's
        # bundled nvrtc (cu111) cannot do for the L40S (sm_89). Replace it with an identical EAGER
        # function (resolved as a module global at call time) so no TorchScript/nvrtc is involved.
        import Transformer_M.modules.transformer_m_layers as _tml

        def _eager_gaussian(x, mean, std):
            a = (2 * 3.14159) ** 0.5
            return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
        _tml.gaussian = _eager_gaussian
        self.num_tasks = num_tasks
        self.transformer = TransformerM(args=_tm_args(use_3d=use_3d))
        self.graph_pred_head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.Dropout(p=0.2), nn.ReLU(),
            nn.Linear(512, 300), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(300, num_tasks),
        )
        for layer in self.graph_pred_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def from_pretrained(self, ckpt_path, map_location="cpu"):
        obj = torch.load(ckpt_path, map_location=map_location)
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        enc = {k[len("graph_encoder."):]: v for k, v in sd.items() if k.startswith("graph_encoder.")}
        if not enc:  # fall back: maybe keys are bare encoder keys
            enc = {k: v for k, v in sd.items() if not k.startswith("graph_pred")}
        missing, unexpected = self.transformer.load_state_dict(enc, strict=False)
        print(f"  from_pretrained: loaded {len(enc)} encoder tensors; "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        return len(enc), len(missing), len(unexpected)

    def forward_embed(self, batch):
        inner_states, _atom = self.transformer.molecule_encoder(
            batch, segment_labels=None, perturb=None, last_state_only=True)
        last_state = inner_states[0]
        return last_state.permute(1, 0, 2).mean(dim=1)  # [B, 768]

    def forward(self, batch):
        return self.graph_pred_head(self.forward_embed(batch))
