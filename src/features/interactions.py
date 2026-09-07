import torch

# Ligand atom type classes
LIG_HBD_HBA = 0
LIG_POS = 2
LIG_NEG = 3
LIG_ARO = 4
LIG_HPHOB = 5
LIG_OTHER = 6
N_LIG_TYPES = 7

# Pocket residue type classes
RES_CHARGED = 0
RES_POLAR = 1
RES_HPHOB = 2
RES_ARO = 3
RES_OTHER = 4
N_RES_TYPES = 5

DIST_BINS = [(0.0, 6.0), (6.0, 10.0), (10.0, 15.0)]
LIG_CTX_DIM = 15   # N_RES_TYPES * len(DIST_BINS)
POC_CTX_DIM = 21   # N_LIG_TYPES * len(DIST_BINS)


def classify_ligand_atom(atom_feat: torch.Tensor):
    atomic_num_scaled = atom_feat[0].item()
    formal_charge = atom_feat[1].item()
    is_aromatic = atom_feat[6].item()
    atomic_num = int(round(atomic_num_scaled * 100))
    if formal_charge > 0.1:
        return LIG_POS
    if formal_charge < -0.1:
        return LIG_NEG
    if atomic_num in (7, 8):
        return LIG_HBD_HBA
    if is_aromatic > 0.5:
        return LIG_ARO
    if atomic_num == 6:
        return LIG_HPHOB
    return LIG_OTHER


def classify_residue(res_feat: torch.Tensor):
    charge = res_feat[0].item()
    is_hphob = res_feat[1].item()
    is_polar = res_feat[2].item()
    is_aromatic = res_feat[3].item()
    if charge > 0.1 or charge < -0.1:
        return RES_CHARGED
    if is_aromatic > 0.5:
        return RES_ARO
    if is_hphob > 0.5:
        return RES_HPHOB
    if is_polar > 0.5:
        return RES_POLAR
    return RES_OTHER


def build_local_cross_context(lig_data, poc_data):
    lig_pos = lig_data.pos
    lig_x = lig_data.x
    poc_pos = poc_data.pos
    poc_x = poc_data.x

    N_lig = lig_pos.size(0)
    N_poc = poc_pos.size(0)

    dist = torch.cdist(lig_pos, poc_pos)

    lig_types = torch.tensor([classify_ligand_atom(lig_x[i]) for i in range(N_lig)], dtype=torch.long)
    poc_types = torch.tensor([classify_residue(poc_x[i]) for i in range(N_poc)], dtype=torch.long)

    lig_ctx = torch.zeros(N_lig, LIG_CTX_DIM, dtype=torch.float)
    poc_ctx = torch.zeros(N_poc, POC_CTX_DIM, dtype=torch.float)

    for b, (lo, hi) in enumerate(DIST_BINS):
        in_band = (dist >= lo) & (dist < hi)
        for r_type in range(N_RES_TYPES):
            r_mask = (poc_types == r_type).unsqueeze(0)
            counts = (in_band & r_mask).sum(dim=1).float()
            lig_ctx[:, r_type * len(DIST_BINS) + b] = counts
        for a_type in range(N_LIG_TYPES):
            a_mask = (lig_types == a_type).unsqueeze(1)
            counts = (in_band & a_mask).sum(dim=0).float()
            poc_ctx[:, a_type * len(DIST_BINS) + b] = counts

    return lig_ctx, poc_ctx
