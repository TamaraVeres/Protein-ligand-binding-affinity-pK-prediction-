from pathlib import Path
import csv
import os
import re
import torch
from torch.utils.data import Dataset

from src.features import (
    build_ligand_graph_from_path,
    build_pocket_graph_from_path,
    build_local_cross_context,
)
from src.features.pocket_graph import rebuild_pocket_edges

_CACHE_NAME_RE = re.compile(r"^(?P<pdb_id>.+?)(?:_\d+)?\.pt$")


class PDBbindDataset(Dataset):
    MAX_HEAVY_ATOMS = 9999
    MAX_RING_SIZE = 99

    def __init__(self, csv_path: str, cache_dir: str = "cache", assay_filter: str = None):
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []

        assay_filter = assay_filter.lower() if assay_filter else None
        if assay_filter and assay_filter not in ("ki", "kd"):
            raise ValueError(f"assay_filter must be 'ki', 'kd', or None; got {assay_filter}")

        n_dropped_size = 0
        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not (row["ligand_path"] and row["pocket_path"]):
                    continue
                target_col = "log10_nM" if ("log10_nM" in row and row["log10_nM"]) else "p_value"
                if row[target_col] == "" or row[target_col] is None:
                    continue
                if assay_filter:
                    if row.get("assay_type", "").strip().lower() != assay_filter:
                        continue
                heavy_str = row.get("heavy_atoms", "")
                ring_str = row.get("max_ring", "")
                if heavy_str == "" or ring_str == "":
                    n_dropped_size += 1
                    continue
                try:
                    if int(heavy_str) > self.MAX_HEAVY_ATOMS or int(ring_str) > self.MAX_RING_SIZE:
                        n_dropped_size += 1
                        continue
                except ValueError:
                    n_dropped_size += 1
                    continue
                try:
                    value = float(row[target_col])
                except ValueError:
                    continue
                row["target"] = value if target_col == "log10_nM" else 9.0 - value
                self.rows.append(row)

        filter_msg = f" ({assay_filter}-only)" if assay_filter else ""
        print(f"Loaded {len(self.rows)} samples{filter_msg} from {self.csv_path}")

        self.cache_index = self._build_cache_index()
        n_cached = sum(1 for r in self.rows if r["pdb_id"] in self.cache_index)
        print(f"Cache: {n_cached}/{len(self.rows)} samples found in {self.cache_dir}")
        if n_cached < len(self.rows):
            print(f"  WARNING: {len(self.rows) - n_cached} sample(s) are not cached and will be "
                  f"rebuilt from raw structure files, which are not distributed with this repo.")

    def _build_cache_index(self):
        """Map pdb_id -> cache file path. Keyed by id, not row index, which shifts
        with the assay filter."""
        index = {}
        if not self.cache_dir.is_dir():
            return index
        # Sorted so the choice among duplicate copies is reproducible.
        for entry in sorted(os.scandir(self.cache_dir), key=lambda e: e.name):
            if not entry.is_file():
                continue
            m = _CACHE_NAME_RE.match(entry.name)
            if m:
                index.setdefault(m.group("pdb_id"), entry.path)
        return index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pdb_id = row["pdb_id"]
        cache_path = self.cache_index.get(pdb_id)
        if cache_path is not None:
            sample = torch.load(cache_path, weights_only=False)
            poc = sample["pocket"]
            
            if getattr(poc, "edge_index", None) is None:
                poc.edge_index, poc.edge_attr = rebuild_pocket_edges(poc.pos, poc.aa_idx)
            return sample

        ligand_path = self.csv_path.parent / row["ligand_path"]
        pocket_path = self.csv_path.parent / row["pocket_path"]
        missing = [str(p) for p in (ligand_path, pocket_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{pdb_id}: not in the cache at {self.cache_dir}, and the raw structure "
                f"file(s) needed to rebuild it are missing: {', '.join(missing)}.\n"
                f"Download the preprocessed cache and unpack it into {self.cache_dir} "
                f"(see the Data section of the README). The raw PDBbind/scPDB structures "
                f"(~33 GB) are only needed to featurise complexes that are not in the cache.")
        ligand_path, pocket_path = str(ligand_path), str(pocket_path)
        y = torch.tensor(row["target"], dtype=torch.float32)
        lig_data = build_ligand_graph_from_path(ligand_path)
        poc_data = build_pocket_graph_from_path(pocket_path)
        lig_ctx, poc_ctx = build_local_cross_context(lig_data, poc_data)
        lig_data.x = torch.cat([lig_data.x, lig_ctx], dim=-1)
        poc_data.x = torch.cat([poc_data.x, poc_ctx], dim=-1)
        sample = {"ligand": lig_data, "pocket": poc_data, "y": y, "pdb_id": pdb_id}

        cache_path = self.cache_dir / f"{pdb_id}.pt"
        torch.save(sample, cache_path)
        self.cache_index[pdb_id] = str(cache_path)
        return sample
