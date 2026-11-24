"""
Atom/bond featurization — verbatim from snap-stanford/pretrain-gnns `chem/loader.py`
(`allowable_features`, `mol_to_graph_data_obj_simple`). Simplified 2-feature atom representation
(atomic-number index, chirality-tag index) and 2-feature bonds (bond-type index, bond-dir index),
with edges added in both directions. This is what the ported GINConv edge embeddings expect.
"""
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def mol_to_graph_data_obj_simple(mol):
    """Convert an rdkit mol to a PyG Data object with simplified index features (x, edge_index,
    edge_attr). Verbatim from pretrain-gnns."""
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j)); edge_features_list.append(edge_feature)
            edges_list.append((j, i)); edge_features_list.append(edge_feature)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_data(smiles):
    """Parse SMILES and featurize; returns (Data, None) or (None, reason)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "rdkit_parse_failed"
    if mol.GetNumAtoms() == 0:
        return None, "no_atoms"
    try:
        return mol_to_graph_data_obj_simple(mol), None
    except ValueError as e:  # feature index() miss (e.g. exotic element/bond) -> record, skip
        return None, f"featurization_failed: {e}"
