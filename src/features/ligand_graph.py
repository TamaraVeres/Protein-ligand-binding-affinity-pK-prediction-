import os
import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, rdchem, AllChem, ChemicalFeatures
from torch_geometric.data import Data

_FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FEAT_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)
_PHARM_MAP = {"Donor": 0, "Acceptor": 1, "Hydrophobe": 2,
              "Aromatic": 3, "PosIonizable": 4, "NegIonizable": 5}


def load_ligand_mol(path: str):
    if path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
        mol = suppl[0]
    elif path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if mol is None:
        raise ValueError(f"Failed to read ligand from {path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def atom_features(atom: rdchem.Atom, ring_size: float,
                  crippen_logp: float = 0.0, crippen_mr: float = 0.0):
    atomic_num = atom.GetAtomicNum()
    pt = Chem.GetPeriodicTable()
    atomic_num_scaled = atomic_num / 100.0
    mass_scaled = atom.GetMass() / 200.0
    vdw_scaled = pt.GetRvdw(atomic_num) / 3.0
    formal_charge = float(atom.GetFormalCharge())
    num_hs = float(atom.GetTotalNumHs())
    is_donor = 1.0 if (atomic_num in (7, 8) and num_hs > 0) else 0.0
    is_acceptor = 1.0 if atomic_num in (7, 8, 9, 16) else 0.0
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    hyb = atom.GetHybridization()
    hyb_sp = 1.0 if hyb == rdchem.HybridizationType.SP else 0.0
    hyb_sp2 = 1.0 if hyb == rdchem.HybridizationType.SP2 else 0.0
    hyb_sp3 = 1.0 if hyb == rdchem.HybridizationType.SP3 else 0.0
    in_ring = 1.0 if atom.IsInRing() else 0.0
    degree = float(atom.GetDegree())
    total_valence = float(atom.GetTotalValence())
    ring_size_val = float(ring_size)
    return [
        atomic_num_scaled, formal_charge, mass_scaled, vdw_scaled,
        is_donor, is_acceptor, is_aromatic,
        hyb_sp, hyb_sp2, hyb_sp3,
        in_ring, degree, total_valence, num_hs, ring_size_val,
        crippen_logp, crippen_mr,
    ]


def bond_features(bond: rdchem.Bond, conf, i: int, j: int):
    btype = bond.GetBondType()
    pos_i = conf.GetAtomPosition(i)
    pos_j = conf.GetAtomPosition(j)
    dist = pos_i.Distance(pos_j) / 5.0
    has_stereo = 1.0 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 0.0
    is_single = 1.0 if btype == rdchem.BondType.SINGLE else 0.0
    is_double = 1.0 if btype == rdchem.BondType.DOUBLE else 0.0
    is_triple = 1.0 if btype == rdchem.BondType.TRIPLE else 0.0
    is_arom_bt = 1.0 if btype == rdchem.BondType.AROMATIC else 0.0
    is_conj = 1.0 if bond.GetIsConjugated() else 0.0
    in_ring = 1.0 if bond.IsInRing() else 0.0
    return [is_single, is_double, is_triple, is_arom_bt, is_conj, in_ring, dist, has_stereo]


def _pharmacophore_features(mol, conf) -> list:
    feats_by_type = {i: [] for i in range(6)}
    try:
        mol_feats = _FEAT_FACTORY.GetFeaturesForMol(mol)
        for feat in mol_feats:
            family = feat.GetFamily()
            if family in _PHARM_MAP:
                idx = _PHARM_MAP[family]
                pos = conf.GetAtomPosition(feat.GetAtomIds()[0])
                feats_by_type[idx].append(np.array([pos.x, pos.y, pos.z]))
    except Exception:
        pass

    pharm_feats = []
    centroids = {}
    for i in range(6):
        points = feats_by_type[i]
        count = float(len(points))
        if count == 0:
            pharm_feats += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            centroids[i] = np.zeros(3)
        else:
            pts = np.array(points)
            centroid = pts.mean(axis=0)
            centroids[i] = centroid
            if count == 1:
                spread = 0.0
                max_dist = 0.0
            else:
                dists_to_c = np.linalg.norm(pts - centroid, axis=1)
                spread = float(dists_to_c.std())
                max_dist = float(dists_to_c.max())
            pharm_feats += [
                count / 10.0,
                float(centroid[0]) / 10.0,
                float(centroid[1]) / 10.0,
                float(centroid[2]) / 10.0,
                spread / 5.0,
                max_dist / 10.0,
            ]

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in pairs:
        d = float(np.linalg.norm(centroids[i] - centroids[j]))
        pharm_feats.append(d / 15.0)
    return pharm_feats


def ligand_graph_features(mol) -> torch.Tensor:
    conf = mol.GetConformer()
    feats = [
        float(mol.GetNumAtoms()),
        float(mol.GetNumBonds()),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Descriptors.MolWt(mol)),
    ]
    feats += _pharmacophore_features(mol, conf)
    return torch.tensor(feats, dtype=torch.float32)


def build_ligand_graph_from_path(ligand_path: str) -> Data:
    mol = load_ligand_mol(ligand_path)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    num_atoms = mol.GetNumAtoms()
    ring_sizes = [0.0] * num_atoms
    for ring in atom_rings:
        size = len(ring)
        for idx in ring:
            if ring_sizes[idx] == 0.0:
                ring_sizes[idx] = float(size)
            else:
                ring_sizes[idx] = float(min(ring_sizes[idx], size))

    try:
        crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    except Exception:
        crippen = [(0.0, 0.0)] * num_atoms

    atom_feats = []
    positions = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        rs = ring_sizes[idx]
        logp_i, mr_i = crippen[idx] if idx < len(crippen) else (0.0, 0.0)
        if not np.isfinite(logp_i):
            logp_i = 0.0
        if not np.isfinite(mr_i):
            mr_i = 0.0
        atom_feats.append(atom_features(atom, rs, float(logp_i), float(mr_i)))
        pos = conf.GetAtomPosition(idx)
        positions.append([pos.x, pos.y, pos.z])

    x = torch.tensor(atom_feats, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond, conf, i, j)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    try:
        graph_feat = ligand_graph_features(mol)
    except Exception:
        graph_feat = torch.zeros(51, dtype=torch.float32)

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_tensor = torch.tensor(list(fp), dtype=torch.float32)
    except Exception:
        fp_tensor = torch.zeros(1024, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                lig_graph_feat=graph_feat, fingerprint=fp_tensor)
    return data
