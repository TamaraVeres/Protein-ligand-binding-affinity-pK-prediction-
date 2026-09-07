import re
import numpy as np
import torch
from pathlib import Path
from Bio.PDB import PDBParser
from torch_geometric.data import Data

AA_TO_INDEX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
    "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
    "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
NUM_AA_TYPES = 21

AROMATIC = {"PHE", "TYR", "HIS", "TRP"}
CHARGED_NEG = {"GLU", "ASP"}
CHARGED_POS = {"ARG", "HIS", "LYS"}
HYDROPHOBIC = {"PHE", "PRO", "LEU", "MET", "ALA", "VAL", "TRP", "ILE"}
POLAR = {"HIS", "CYS", "TYR", "ASN", "THR", "SER", "GLN"}

DEFAULT_RES_PROPS = {"aromatic": 0, "charge": 0.0, "hba": 0, "hbd": 0,
                     "hydrophobicity": 0.0, "polar": 0, "volume": 120.0}

RESIDUE_PROPERTIES = {
    "ALA": {"hydrophobicity": 1.8, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 88.6},
    "ARG": {"hydrophobicity": -4.5, "charge": 1.0, "polar": 1, "aromatic": 0, "hbd": 1, "hba": 0, "volume": 173.4},
    "ASN": {"hydrophobicity": -3.5, "charge": 0.0, "polar": 1, "aromatic": 0, "hbd": 1, "hba": 1, "volume": 114.1},
    "ASP": {"hydrophobicity": -3.5, "charge": -1.0, "polar": 1, "aromatic": 0, "hbd": 0, "hba": 1, "volume": 111.1},
    "CYS": {"hydrophobicity": 2.5, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 1, "hba": 0, "volume": 108.5},
    "GLN": {"hydrophobicity": -3.5, "charge": 0.0, "polar": 1, "aromatic": 0, "hbd": 1, "hba": 1, "volume": 143.8},
    "GLU": {"hydrophobicity": -3.5, "charge": -1.0, "polar": 1, "aromatic": 0, "hbd": 0, "hba": 1, "volume": 138.4},
    "GLY": {"hydrophobicity": -0.4, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 60.1},
    "HIS": {"hydrophobicity": -3.2, "charge": 0.5, "polar": 1, "aromatic": 1, "hbd": 1, "hba": 1, "volume": 153.2},
    "ILE": {"hydrophobicity": 4.5, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 166.7},
    "LEU": {"hydrophobicity": 3.8, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 166.7},
    "LYS": {"hydrophobicity": -3.9, "charge": 1.0, "polar": 1, "aromatic": 0, "hbd": 1, "hba": 0, "volume": 168.6},
    "MET": {"hydrophobicity": 1.9, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 162.9},
    "PHE": {"hydrophobicity": 2.8, "charge": 0.0, "polar": 0, "aromatic": 1, "hbd": 0, "hba": 0, "volume": 189.9},
    "PRO": {"hydrophobicity": -1.6, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 112.7},
    "SER": {"hydrophobicity": -0.8, "charge": 0.0, "polar": 1, "aromatic": 0, "hbd": 1, "hba": 1, "volume": 89.0},
    "THR": {"hydrophobicity": -0.7, "charge": 0.0, "polar": 1, "aromatic": 0, "hbd": 1, "hba": 1, "volume": 116.1},
    "TRP": {"hydrophobicity": -0.9, "charge": 0.0, "polar": 0, "aromatic": 1, "hbd": 1, "hba": 0, "volume": 227.8},
    "TYR": {"hydrophobicity": -1.3, "charge": 0.0, "polar": 1, "aromatic": 1, "hbd": 1, "hba": 1, "volume": 193.6},
    "VAL": {"hydrophobicity": 4.2, "charge": 0.0, "polar": 0, "aromatic": 0, "hbd": 0, "hba": 0, "volume": 140.0},
}

HOTSPOT_TYPES = {
    "HBD": lambda rn, props: props["hbd"] > 0,
    "HBA": lambda rn, props: props["hba"] > 0,
    "Hydrophobic": lambda rn, props: rn in HYDROPHOBIC,
    "Aromatic": lambda rn, props: rn in AROMATIC,
    "Positive": lambda rn, props: rn in CHARGED_POS,
    "Negative": lambda rn, props: rn in CHARGED_NEG,
}


def _infer_element(atom_name: str, sybyl_type: str) -> str:
    if sybyl_type:
        base = sybyl_type.split(".")[0].upper()
        if base in ("C", "N", "O", "S", "P", "F", "CL", "BR", "I", "H"):
            return base
    name = atom_name.strip().upper()
    if not name:
        return ""
    if name.startswith("CL"):
        return "CL"
    if name.startswith("BR"):
        return "BR"
    first = name[0]
    if first.isalpha():
        return first
    return ""


def _parse_mol2_residue_label(label: str):
    m = re.match(r"^([A-Za-z]+)(\d+)?$", label)
    if not m:
        return (label, "")
    return (m.group(1).upper(), m.group(2) or "")


class _Mol2Atom:
    def __init__(self, name: str, coord, element: str, bfactor: float = 0.0):
        self._name = name
        self._coord = np.array(coord, dtype=float)
        self.element = element
        self._bfactor = bfactor

    def get_coord(self):
        return self._coord

    def get_name(self):
        return self._name

    def get_bfactor(self):
        return self._bfactor


class _Mol2Residue:
    def __init__(self, resname: str):
        self._resname = resname
        self._atoms = []
        self._atom_names = set()

    def add_atom(self, atom: "_Mol2Atom"):
        self._atoms.append(atom)
        self._atom_names.add(atom.get_name())

    def get_resname(self):
        return self._resname

    def get_atoms(self):
        return list(self._atoms)

    def has_id(self, atom_name: str):
        return atom_name in self._atom_names


def load_pocket_residues_mol2(pocket_path: str):
    residues_ordered = []
    residues_map = {}
    with open(pocket_path) as f:
        in_atom_section = False
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("@<TRIPOS>"):
                in_atom_section = line.strip() == "@<TRIPOS>ATOM"
                continue
            if not in_atom_section:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                atom_name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                sybyl = parts[5]
                subst_id = parts[6]
                subst_name = parts[7] if len(parts) > 7 else subst_id
            except (ValueError, IndexError):
                continue
            element = _infer_element(atom_name, sybyl)
            resname, resnum = _parse_mol2_residue_label(subst_name)
            if resname == "HOH":
                continue
            res_key = (subst_id, resname)
            if res_key not in residues_map:
                residue = _Mol2Residue(resname)
                residues_map[res_key] = residue
                residues_ordered.append(residue)
            residues_map[res_key].add_atom(_Mol2Atom(atom_name, (x, y, z), element))
    return [r for r in residues_ordered if r.has_id("CA")]


def load_pocket_residues(pocket_path: str):
    path_str = str(pocket_path).lower()
    if path_str.endswith(".mol2"):
        return load_pocket_residues_mol2(pocket_path)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pocket_path)
    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                resname = res.get_resname().strip()
                if resname == "HOH":
                    continue
                if not res.has_id("CA"):
                    continue
                residues.append(res)
    return residues


def residue_features(residue):
    resname = residue.get_resname().strip().upper()
    if resname in CHARGED_POS:
        charge_sign = 1.0
    elif resname in CHARGED_NEG:
        charge_sign = -1.0
    else:
        charge_sign = 0.0
    is_hydrophobic = 1.0 if resname in HYDROPHOBIC else 0.0
    is_polar = 1.0 if resname in POLAR else 0.0
    is_aromatic = 1.0 if resname in AROMATIC else 0.0

    heavy_atoms = [atom for atom in residue.get_atoms() if atom.element != "H"]
    size = float(len(heavy_atoms))
    props = RESIDUE_PROPERTIES.get(resname, DEFAULT_RES_PROPS)
    hydrophobicity = float(props["hydrophobicity"]) / 5.0
    charge_refined = float(props["charge"])
    polar_flag = float(props["polar"])
    aromatic_flag = float(props["aromatic"])
    hbd = float(props["hbd"])
    hba = float(props["hba"])
    volume = float(props["volume"]) / 250.0

    if len(heavy_atoms) > 1:
        coords = np.array([atom.get_coord() for atom in heavy_atoms], dtype=float)
        spread = float(np.std(coords, axis=0).mean())
    else:
        spread = 0.0

    denom = max(size, 1.0)
    counts = {"C": 0.0, "N": 0.0, "O": 0.0, "S": 0.0}
    for atom in heavy_atoms:
        el = (atom.element or "").strip().upper()
        if not el:
            atom_name = atom.get_name().strip().upper()
            el = atom_name[0] if atom_name else ""
        if el in counts:
            counts[el] += 1.0
    frac_C = counts["C"] / denom
    frac_N = counts["N"] / denom
    frac_O = counts["O"] / denom
    frac_S = counts["S"] / denom

    backbone_names = {"CA", "C", "N", "O"}
    backbone_count = 0.0
    for atom in heavy_atoms:
        atom_name = atom.get_name().strip().upper()
        if atom_name in backbone_names:
            backbone_count += 1.0
    backbone_frac = backbone_count / denom

    return [
        charge_sign, is_hydrophobic, is_polar, is_aromatic, size,
        hydrophobicity, charge_refined, polar_flag, aromatic_flag, hbd, hba, volume,
        spread, frac_C, frac_N, frac_O, frac_S, backbone_frac,
    ]


def residue_centroid(residue):
    coords = []
    for atom in residue.get_atoms():
        if atom.element == "H":
            continue
        pos = atom.get_coord()
        coords.append(pos)
    if not coords:
        for atom in residue.get_atoms():
            coords.append(atom.get_coord())
    coords = np.array(coords, dtype=float)
    return coords.mean(axis=0)


def _hotspot_features(residues, centroids, pocket_center) -> list:
    n = len(residues)
    centroid_arr = np.array(centroids, dtype=float)
    feats = []
    for htype, test_fn in HOTSPOT_TYPES.items():
        indices = []
        for i, res in enumerate(residues):
            rn = res.get_resname().strip().upper()
            props = RESIDUE_PROPERTIES.get(rn, DEFAULT_RES_PROPS)
            if test_fn(rn, props):
                indices.append(i)
        count = float(len(indices))
        if count == 0:
            feats += [0.0] * 9
            continue
        pts = centroid_arr[indices]
        centroid = pts.mean(axis=0)
        dists_to_c = np.linalg.norm(pts - centroid, axis=1)
        spread = float(dists_to_c.std()) if count > 1 else 0.0
        max_dist = float(dists_to_c.max()) if count > 1 else 0.0
        dist_to_center = float(np.linalg.norm(centroid - pocket_center))
        if count > 1:
            from scipy.spatial.distance import cdist
            pw = cdist(pts, pts)
            local_density = float(np.mean(np.sum(pw < 6.0, axis=1) - 1))
        else:
            local_density = 0.0
        fraction = count / n
        feats += [
            count / 10.0,
            float(centroid[0]) / 20.0, float(centroid[1]) / 20.0, float(centroid[2]) / 20.0,
            spread / 5.0, max_dist / 10.0, dist_to_center / 20.0,
            local_density / 5.0, fraction,
        ]
    return feats


def pocket_graph_features(residues, centroids=None) -> torch.Tensor:
    n = len(residues)
    TOTAL_DIM = 79
    if n == 0:
        return torch.zeros(TOTAL_DIM, dtype=torch.float32)

    n_charged_pos = n_charged_neg = n_aromatic = n_hydrophobic = n_polar = 0.0
    total_hbd = total_hba = sum_hydrophobicity = sum_volume = 0.0
    n_cys = n_his = n_gly = n_pro = 0.0
    for res in residues:
        resname = res.get_resname().strip().upper()
        props = RESIDUE_PROPERTIES.get(resname, DEFAULT_RES_PROPS)
        if resname in CHARGED_POS:
            n_charged_pos += 1.0
        if resname in CHARGED_NEG:
            n_charged_neg += 1.0
        if resname in AROMATIC:
            n_aromatic += 1.0
        if resname in HYDROPHOBIC:
            n_hydrophobic += 1.0
        if resname in POLAR:
            n_polar += 1.0
        total_hbd += float(props["hbd"])
        total_hba += float(props["hba"])
        sum_hydrophobicity += float(props["hydrophobicity"])
        sum_volume += float(props["volume"])
        if resname == "CYS":
            n_cys += 1.0
        if resname == "HIS":
            n_his += 1.0
        if resname == "GLY":
            n_gly += 1.0
        if resname == "PRO":
            n_pro += 1.0

    feats = [
        float(n),
        n_charged_pos / n, n_charged_neg / n, n_aromatic / n, n_hydrophobic / n, n_polar / n,
        total_hbd / n, total_hba / n, sum_hydrophobicity / n, sum_volume / n,
    ]
    feats += [n_cys / n, n_his / n, n_gly / n, n_pro / n]

    if centroids is not None and len(centroids) >= 3:
        coords = np.array(centroids, dtype=float)
        center = coords.mean(axis=0)
        centered = coords - center
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = eigvals / (eigvals.sum() + 1e-08)
        dists_to_center = np.linalg.norm(centered, axis=1)
        rg = float(np.sqrt(np.mean(dists_to_center ** 2)))
        from scipy.spatial.distance import pdist
        pw = pdist(coords)
        pw_mean = float(pw.mean())
        pw_std = float(pw.std())
        pw_max = float(pw.max())
        bbox = coords.max(axis=0) - coords.min(axis=0)
        feats += [
            float(eigvals[0]), float(eigvals[1]), float(eigvals[2]),
            rg / 20.0, pw_mean / 20.0, pw_std / 10.0, pw_max / 40.0,
            float(bbox[0]) / 30.0, float(bbox[1]) / 30.0, float(bbox[2]) / 30.0,
            float(np.prod(bbox)) / 27000.0,
        ]
    else:
        feats += [0.0] * 11

    if centroids is not None and len(centroids) >= 1:
        pocket_center = np.array(centroids, dtype=float).mean(axis=0)
        feats += _hotspot_features(residues, centroids, pocket_center)
    else:
        feats += [0.0] * 54

    return torch.tensor(feats, dtype=torch.float32)


def build_pocket_graph_from_path(pocket_path: str, distance_cutoff: float = 12.0) -> Data:
    residues = load_pocket_residues(pocket_path)
    if len(residues) == 0:
        raise ValueError(f"No residues found in pocket: {pocket_path}")

    centroids = []
    base_node_feats = []
    aa_indices = []
    for res in residues:
        base_node_feats.append(residue_features(res))
        centroids.append(residue_centroid(res))
        resname = res.get_resname().strip().upper()
        aa_indices.append(AA_TO_INDEX.get(resname, 20))

    centroids_np = np.array(centroids, dtype=float)
    pocket_center = centroids_np.mean(axis=0)
    centroids_centered = centroids_np - pocket_center
    dist_matrix = np.linalg.norm(
        centroids_centered[:, None, :] - centroids_centered[None, :, :], axis=-1)

    node_feats = []
    for i, _ in enumerate(residues):
        base = base_node_feats[i]
        dists = dist_matrix[i]
        nonzero = dists[dists > 0]
        if len(nonzero) == 0:
            dist_to_center = 0.0
            min_dist = 0.0
            mean_dist = 0.0
            n_close = 0.0
            n_medium = 0.0
        else:
            dist_to_center = float(np.linalg.norm(centroids_centered[i]))
            min_dist = float(nonzero.min())
            mean_dist = float(nonzero.mean())
            n_close = float(np.sum(nonzero < 6.0))
            n_medium = float(np.sum((nonzero >= 6.0) & (nonzero < 10.0)))
        neighbor_feats = [dist_to_center / 20.0, min_dist / 15.0, mean_dist / 20.0,
                          n_close / 5.0, n_medium / 5.0]
        coord_feats = [
            float(centroids_centered[i, 0] / 20.0),
            float(centroids_centered[i, 1] / 20.0),
            float(centroids_centered[i, 2] / 20.0),
        ]
        node_feats.append(base + neighbor_feats + coord_feats)

    x = torch.tensor(node_feats, dtype=torch.float)
    pos = torch.tensor(centroids_np, dtype=torch.float)
    aa_idx = torch.tensor(aa_indices, dtype=torch.long)
    num_nodes = x.size(0)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = np.linalg.norm(centroids_np[i] - centroids_np[j])
            if d < distance_cutoff:
                dist_norm = d / distance_cutoff
                dist_exp = float(np.exp(-d / 5.0))
                res_i = residues[i]
                res_j = residues[j]
                name_i = res_i.get_resname().strip().upper()
                name_j = res_j.get_resname().strip().upper()
                props_i = RESIDUE_PROPERTIES.get(name_i, DEFAULT_RES_PROPS)
                props_j = RESIDUE_PROPERTIES.get(name_j, DEFAULT_RES_PROPS)
                charge_product = float(props_i["charge"] * props_j["charge"])
                both_aromatic = float(props_i["aromatic"] and props_j["aromatic"])
                both_hydrophobic = float(props_i["hydrophobicity"] > 0 and props_j["hydrophobicity"] > 0)
                hbond_possible = float((props_i["hbd"] and props_j["hba"]) or (props_i["hba"] and props_j["hbd"]))
                is_close = 1.0 if d < 6.0 else 0.0
                is_medium = 1.0 if (6.0 <= d < 10.0) else 0.0
                is_far = 1.0 if (10.0 <= d < 12.0) else 0.0
                edge_feat = [dist_norm, dist_exp, charge_product, both_aromatic,
                             both_hydrophobic, hbond_possible, is_close, is_medium, is_far]
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(edge_feat)
                edge_attr.append(edge_feat)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 9), dtype=torch.float)

    poc_graph_feat = pocket_graph_features(residues, centroids=centroids_np)

    VDW_RADII = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8, "F": 1.47, "CL": 1.75, "BR": 1.85}
    atom_positions = []
    atom_res_types = []
    atom_vdw = []
    atom_charges = []
    for res in residues:
        resname = res.get_resname().strip().upper()
        if resname in CHARGED_POS or resname in CHARGED_NEG:
            rtype = 0
        elif resname in AROMATIC:
            rtype = 3
        elif resname in HYDROPHOBIC:
            rtype = 2
        elif resname in POLAR:
            rtype = 1
        else:
            rtype = 4
        props = RESIDUE_PROPERTIES.get(resname, DEFAULT_RES_PROPS)
        res_charge = float(props["charge"])
        for atom in res.get_atoms():
            if atom.element == "H":
                continue
            el = (atom.element or "").strip().upper()
            atom_positions.append(atom.get_coord().tolist())
            atom_res_types.append(rtype)
            atom_vdw.append(VDW_RADII.get(el, 1.7))
            atom_charges.append(res_charge)

    atom_pos = torch.tensor(atom_positions, dtype=torch.float)
    atom_res_type = torch.tensor(atom_res_types, dtype=torch.long)
    atom_vdw_tensor = torch.tensor(atom_vdw, dtype=torch.float)
    atom_charge_tensor = torch.tensor(atom_charges, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, aa_idx=aa_idx,
                poc_graph_feat=poc_graph_feat, atom_pos=atom_pos, atom_res_type=atom_res_type,
                atom_vdw=atom_vdw_tensor, atom_charge=atom_charge_tensor)
    return data


IDX_TO_AA = {v: k for k, v in AA_TO_INDEX.items()}
_EDGE_PROP_KEYS = ("charge", "aromatic", "hydrophobicity", "hbd", "hba")


def _residue_property_arrays(aa_idx):
    """Per-residue property columns, recovered from the amino-acid index."""
    names = [IDX_TO_AA.get(int(i), "UNK") for i in aa_idx]
    props = [RESIDUE_PROPERTIES.get(n, DEFAULT_RES_PROPS) for n in names]
    return {k: np.array([float(p[k]) for p in props], dtype=float) for k in _EDGE_PROP_KEYS}


def rebuild_pocket_edges(pos, aa_idx, distance_cutoff: float = 12.0):
    """Rebuild (edge_index, edge_attr) from residue centroids and identities."""
    c = np.asarray(pos, dtype=float)
    n = c.shape[0]
    if n < 2:
        return (torch.empty((2, 0), dtype=torch.long), torch.empty((0, 9), dtype=torch.float))

    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    iu, ju = np.triu_indices(n, k=1)                 # row-major, matches build order
    keep = d[iu, ju] < distance_cutoff
    i_idx, j_idx = iu[keep], ju[keep]
    if i_idx.size == 0:
        return (torch.empty((2, 0), dtype=torch.long), torch.empty((0, 9), dtype=torch.float))

    dd = d[i_idx, j_idx]
    p = _residue_property_arrays(aa_idx)
    ci, cj = p["charge"][i_idx], p["charge"][j_idx]
    ai, aj = p["aromatic"][i_idx] > 0, p["aromatic"][j_idx] > 0
    hi, hj = p["hydrophobicity"][i_idx] > 0, p["hydrophobicity"][j_idx] > 0
    di, dj = p["hbd"][i_idx] > 0, p["hbd"][j_idx] > 0
    bi, bj = p["hba"][i_idx] > 0, p["hba"][j_idx] > 0

    feats = np.stack([
        dd / distance_cutoff,
        np.exp(-dd / 5.0),
        ci * cj,
        (ai & aj).astype(float),
        (hi & hj).astype(float),
        ((di & bj) | (bi & dj)).astype(float),
        (dd < 6.0).astype(float),
        ((dd >= 6.0) & (dd < 10.0)).astype(float),
        ((dd >= 10.0) & (dd < 12.0)).astype(float),
    ], axis=1)

    # Each undirected pair contributes [i, j] then [j, i].
    edge_index = np.empty((2, 2 * i_idx.size), dtype=np.int64)
    edge_index[0, 0::2], edge_index[1, 0::2] = i_idx, j_idx
    edge_index[0, 1::2], edge_index[1, 1::2] = j_idx, i_idx
    edge_attr = np.repeat(feats, 2, axis=0)
    return torch.from_numpy(edge_index), torch.from_numpy(edge_attr).float()
