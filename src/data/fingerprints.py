"""Compact Morgan-fingerprint index extracted from the sample cache (128 bytes per complex)."""
from pathlib import Path
import numpy as np
import torch

N_BITS = 1024
DEFAULT_INDEX_NAME = "fingerprints.npz"


def build_index(cache_index: dict, out_path, progress_every: int = 500):
    """Extract fingerprints from cached samples into a packed-bit index file."""
    out_path = Path(out_path)
    pdb_ids, rows = [], []
    items = sorted(cache_index.items())
    for i, (pdb_id, path) in enumerate(items, 1):
        try:
            sample = torch.load(path, weights_only=False)
            fp = sample["ligand"].fingerprint
        except Exception:
            continue
        if fp is None or fp.numel() != N_BITS:
            continue
        bits = (fp.detach().cpu().numpy() > 0.5).astype(np.uint8)
        if not bits.any():
            continue  
        pdb_ids.append(pdb_id)
        rows.append(np.packbits(bits))
        if progress_every and i % progress_every == 0:
            print(f"  fingerprints: {i}/{len(items)} scanned, {len(pdb_ids)} kept", flush=True)

    if not pdb_ids:
        raise RuntimeError(f"No fingerprints could be extracted from {len(items)} cache entries.")
    packed = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pdb_ids=np.array(pdb_ids), packed=packed)
    print(f"Wrote {len(pdb_ids)} fingerprints to {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB)", flush=True)
    return out_path


def load_index(index_path):
    """Return {pdb_id: np.uint8 packed bits} or None if the index does not exist."""
    index_path = Path(index_path)
    if not index_path.exists():
        return None
    data = np.load(index_path, allow_pickle=False)
    return dict(zip(data["pdb_ids"].tolist(), data["packed"]))


def unpack(packed_rows):
    """(n, 128) uint8 packed bits -> (n, 1024) float32 0/1 matrix."""
    bits = np.unpackbits(np.vstack(packed_rows), axis=1)[:, :N_BITS]
    return bits.astype(np.float32)


def similar_pairs(x, cutoff: float = 0.75, block: int = 512):
    """Row/column indices of every pair with Tanimoto similarity >= cutoff."""
    n = x.shape[0]
    popcount = x.sum(axis=1)
    rows, cols = [], []
    for start in range(0, n, block):
        blk = x[start:start + block]
        inter = blk @ x.T
        union = popcount[start:start + block, None] + popcount[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        r, c = np.nonzero(sim >= cutoff)
        rows.append(r + start)
        cols.append(c)
    return np.concatenate(rows), np.concatenate(cols)


def cluster_by_similarity(x, cutoff: float = 0.75):
    """Single-linkage clusters: connected components of the >= cutoff similarity graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = x.shape[0]
    rows, cols = similar_pairs(x, cutoff)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)
    groups = [[] for _ in range(n_components)]
    for i, label in enumerate(labels):
        groups[label].append(i)
    return groups


if __name__ == "__main__":
    import argparse
    from src.data.dataset import PDBbindDataset

    p = argparse.ArgumentParser(description="Build the Morgan fingerprint index from the cache")
    p.add_argument("--csv", default="data/pl_table_with_paths.csv")
    p.add_argument("--cache_dir", default="data/cache")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ds = PDBbindDataset(a.csv, cache_dir=a.cache_dir)
    out = a.out or (Path(a.csv).parent / DEFAULT_INDEX_NAME)
    build_index(ds.cache_index, out)
