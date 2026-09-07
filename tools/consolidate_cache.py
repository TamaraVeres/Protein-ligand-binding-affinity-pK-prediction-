"""Deduplicate the sample cache, drop fields the model never reads, and write the
fingerprint index. Usage: python tools/consolidate_cache.py --out data/cache_slim
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import PDBbindDataset          # noqa: E402
from src.data.fingerprints import N_BITS             # noqa: E402

DROP_POCKET = ("edge_index", "edge_attr", "atom_pos", "atom_res_type", "atom_vdw", "atom_charge")
DROP_LIGAND = ("fingerprint",)


def slim(sample):
    lig, poc = sample["ligand"], sample["pocket"]
    for k in DROP_LIGAND:
        if k in lig:
            del lig[k]
    for k in DROP_POCKET:
        if k in poc:
            del poc[k]
    return sample


def _process(job):
    """Slim one cached sample; returns its fingerprint bits and byte sizes."""
    pdb_id, path, out_dir = job
    dest = Path(out_dir) / f"{pdb_id}.pt"
    try:
        size_in = Path(path).stat().st_size
        if dest.exists():                      # resume: reuse work from an earlier run
            sample = torch.load(dest, weights_only=False)
            fp = getattr(sample["ligand"], "fingerprint", None)
            packed = None
            if fp is not None and fp.numel() == N_BITS:
                bits = (fp.detach().cpu().numpy() > 0.5).astype(np.uint8)
                packed = np.packbits(bits) if bits.any() else None
            if packed is None:                 # slimmed files no longer carry the fingerprint
                src = torch.load(path, weights_only=False)
                fp = getattr(src["ligand"], "fingerprint", None)
                if fp is not None and fp.numel() == N_BITS:
                    bits = (fp.detach().cpu().numpy() > 0.5).astype(np.uint8)
                    packed = np.packbits(bits) if bits.any() else None
            return pdb_id, packed, size_in, dest.stat().st_size, None

        sample = torch.load(path, weights_only=False)
        fp = getattr(sample["ligand"], "fingerprint", None)
        packed = None
        if fp is not None and fp.numel() == N_BITS:
            bits = (fp.detach().cpu().numpy() > 0.5).astype(np.uint8)
            packed = np.packbits(bits) if bits.any() else None
        torch.save(slim(sample), dest)
        return pdb_id, packed, size_in, dest.stat().st_size, None
    except Exception as e:
        return pdb_id, None, 0, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/pl_table_with_paths.csv")
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--out", default="data/cache_slim")
    ap.add_argument("--fp_out", default="data/fingerprints.npz")
    ap.add_argument("--workers", type=int, default=8, help="Parallel cache readers.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PDBbindDataset(args.csv, cache_dir=args.cache_dir)
    items = sorted(ds.cache_index.items())
    jobs = [(pid, path, str(out_dir)) for pid, path in items]
    print(f"Consolidating {len(jobs)} distinct complexes -> {out_dir} "
          f"with {args.workers} workers", flush=True)

    fp_ids, fp_rows = [], []
    n_done = n_fail = 0
    bytes_in = bytes_out = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, (pdb_id, packed, size_in, size_out, err) in enumerate(
                pool.imap_unordered(_process, jobs, chunksize=16), 1):
            if err:
                n_fail += 1
                print(f"  FAILED {pdb_id}: {err}", flush=True)
            else:
                n_done += 1
                bytes_in += size_in
                bytes_out += size_out
                if packed is not None:
                    fp_ids.append(pdb_id)
                    fp_rows.append(packed)
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el:6.0f}s elapsed, ETA {el/i*(len(jobs)-i):5.0f}s, "
                      f"{bytes_in/1e6:7.0f} MB -> {bytes_out/1e6:6.0f} MB "
                      f"({100*bytes_out/max(bytes_in,1):.1f}%)", flush=True)

    order = np.argsort(fp_ids)
    np.savez_compressed(args.fp_out,
                        pdb_ids=np.array(fp_ids)[order],
                        packed=np.vstack(fp_rows)[order])
    print(f"\nWrote {len(fp_ids)} fingerprints to {args.fp_out} "
          f"({Path(args.fp_out).stat().st_size/1024:.0f} KB)", flush=True)
    print(f"Consolidated {n_done} samples ({n_fail} failed) in {time.time()-t0:.0f}s", flush=True)
    print(f"Cache size: {bytes_in/1e6:.0f} MB -> {bytes_out/1e6:.0f} MB "
          f"({100*bytes_out/max(bytes_in,1):.1f}% of original)", flush=True)


if __name__ == "__main__":
    main()
