"""Smoke tests: feature dimensions, checkpoint loading, split behaviour."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (build_ligand_graph_from_path, build_local_cross_context,
                          build_pocket_graph_from_path)
from src.features.pocket_graph import rebuild_pocket_edges
from src.data.fingerprints import N_BITS, similar_pairs
from src.model import AffinityModel
from src.training.trainer import build_pdbbind_batch, evaluate

LIGAND = ROOT / "examples" / "ligand.sdf"
POCKET = ROOT / "examples" / "pocket.pdb"
CHECKPOINT = ROOT / "checkpoints" / "random_run2.pt"


# Changing any of these invalidates every shipped checkpoint.
# Changing any of these invalidates every shipped checkpoint.
LIGAND_NODE_DIM, LIGAND_EDGE_DIM, LIGAND_GRAPH_DIM = 32, 8, 51
POCKET_NODE_DIM, POCKET_EDGE_DIM, POCKET_GRAPH_DIM = 47, 9, 79


@pytest.fixture(scope="module")
def graphs():
    lig = build_ligand_graph_from_path(str(LIGAND))
    poc = build_pocket_graph_from_path(str(POCKET))
    lig_ctx, poc_ctx = build_local_cross_context(lig, poc)
    lig.x = torch.cat([lig.x, lig_ctx], dim=-1)
    poc.x = torch.cat([poc.x, poc_ctx], dim=-1)
    return lig, poc


def test_feature_dimensions(graphs):
    lig, poc = graphs
    assert lig.x.shape[1] == LIGAND_NODE_DIM
    assert lig.edge_attr.shape[1] == LIGAND_EDGE_DIM
    assert lig.lig_graph_feat.numel() == LIGAND_GRAPH_DIM
    assert poc.x.shape[1] == POCKET_NODE_DIM
    assert poc.edge_attr.shape[1] == POCKET_EDGE_DIM
    assert poc.poc_graph_feat.numel() == POCKET_GRAPH_DIM


def test_features_are_finite(graphs):
    for g in graphs:
        for key in ("x", "edge_attr"):
            assert torch.isfinite(getattr(g, key)).all(), f"non-finite values in {key}"


def test_pocket_edges_rebuild_exactly(graphs):
    _, poc = graphs
    edge_index, edge_attr = rebuild_pocket_edges(poc.pos, poc.aa_idx)
    assert torch.equal(edge_index, poc.edge_index)
    assert torch.allclose(edge_attr, poc.edge_attr, atol=1e-5)


def test_similar_pairs_matches_rdkit():
    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect

    rng = np.random.default_rng(0)
    n, cutoff = 40, 0.75
    x = (rng.random((n, N_BITS)) < 0.06).astype(np.float32)
    x[3] = x[7]

    fps = []
    for row in x:
        bv = ExplicitBitVect(N_BITS)
        bv.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
        fps.append(bv)
    expected = set()
    for i in range(n):
        for j, sim in enumerate(DataStructs.BulkTanimotoSimilarity(fps[i], fps)):
            if sim >= cutoff:
                expected.add((i, j))

    rows, cols = similar_pairs(x, cutoff)
    assert set(zip(rows.tolist(), cols.tolist())) == expected
    assert (3, 7) in expected


def test_clusters_contain_all_similar_pairs():
    from src.data.fingerprints import cluster_by_similarity

    rng = np.random.default_rng(1)
    x = (rng.random((60, N_BITS)) < 0.05).astype(np.float32)
    for i in range(0, 60, 6):
        x[i + 1] = x[i].copy()
        flip = rng.choice(N_BITS, 2, replace=False)
        x[i + 1][flip] = 1.0 - x[i + 1][flip]

    groups = cluster_by_similarity(x, 0.75)
    label = {i: g for g, members in enumerate(groups) for i in members}
    rows, cols = similar_pairs(x, 0.75)
    for i, j in zip(rows.tolist(), cols.tolist()):
        assert label[i] == label[j], f"similar pair ({i},{j}) split across clusters"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
def test_checkpoint_loads_and_predicts(graphs):
    from torch_geometric.data import Batch

    lig, poc = graphs
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = AffinityModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(Batch.from_data_list([lig]), Batch.from_data_list([poc]))
    log10_nm = out.item() * ckpt["y_std"].item() + ckpt["y_mean"].item()
    assert torch.isfinite(out).all()
    assert -5.0 < log10_nm < 15.0, f"implausible prediction: {log10_nm}"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
@pytest.mark.parametrize("n", [1, 33, 65])
def test_evaluate_handles_trailing_batch_of_one(graphs, n):
    lig, poc = graphs
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = AffinityModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    class Repeat(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"ligand": lig, "pocket": poc,
                    "y": torch.tensor(3.0 + 0.37 * i, dtype=torch.float32),
                    "pdb_id": f"x{i}"}

    loader = DataLoader(Repeat(), 32, False, collate_fn=build_pdbbind_batch)
    metrics = evaluate(model, loader, torch.device("cpu"), ckpt["y_mean"], ckpt["y_std"])
    assert np.isfinite(metrics["rmse"])


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")
def test_predict_cli_reports_every_row(tmp_path):
    csv_in = tmp_path / "pairs.csv"
    csv_in.write_text(
        "pdb_id,ligand_path,pocket_path\n"
        f"good,{LIGAND},{POCKET}\n"
        f"bad,{tmp_path / 'nope.sdf'},{POCKET}\n")
    out = tmp_path / "out.csv"
    r = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--csv", str(csv_in),
                        "--model", str(CHECKPOINT), "--output", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    assert lines[1].endswith("ok")
    assert "error:" in lines[2]


def _synthetic_dataset(tmp_path, n=120, n_families=12):
    """n ligands from n_families chemotypes; family members share most bits."""
    import csv as _csv

    from src.data.dataset import PDBbindDataset
    from src.data.fingerprints import DEFAULT_INDEX_NAME

    rng = np.random.default_rng(4)
    families = (rng.random((n_families, N_BITS)) < 0.05).astype(np.uint8)
    ids, bits = [], []
    for i in range(n):
        base = families[i % n_families].copy()
        flip = rng.choice(N_BITS, size=3, replace=False)
        base[flip] ^= 1
        ids.append(f"x{i:03d}")
        bits.append(np.packbits(base))
    np.savez_compressed(tmp_path / DEFAULT_INDEX_NAME,
                        pdb_ids=np.array(ids), packed=np.vstack(bits))

    csv_path = tmp_path / "table.csv"
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["pdb_id", "ligand_path", "pocket_path", "log10_nM", "p_value",
                    "assay_type", "heavy_atoms", "max_ring"])
        for i, pid in enumerate(ids):
            w.writerow([pid, "l.sdf", "p.pdb", f"{3.0 + 0.01 * i}", "", "Ki", "20", "6"])
    return PDBbindDataset(str(csv_path), cache_dir=str(tmp_path / "cache")), ids, bits


def test_similarity_split_is_not_sequential_and_has_no_leakage(tmp_path):
    from src.training.trainer import split_dataset_by_similarity
    from src.data.fingerprints import unpack

    ds, ids, bits = _synthetic_dataset(tmp_path)
    train, val, test = split_dataset_by_similarity(ds, seed=42)
    n = len(ds.rows)

    assert sorted(train + val + test) == list(range(n))
    assert not (set(train) & set(test)) and not (set(train) & set(val))
    assert train != list(range(len(train))), "split collapsed back to sequential order"
    assert test, "test split is empty"

    by_id = dict(zip(ids, bits))
    a = unpack([by_id[ds.rows[i]["pdb_id"]] for i in train])
    b = unpack([by_id[ds.rows[i]["pdb_id"]] for i in test])
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    assert sim.max() < 0.75, f"train/test leakage: max Tanimoto {sim.max():.3f}"


def test_similarity_split_raises_when_fingerprints_missing(tmp_path):
    from src.training.trainer import split_dataset_by_similarity
    from src.data.fingerprints import DEFAULT_INDEX_NAME

    ds, _, _ = _synthetic_dataset(tmp_path)
    (tmp_path / DEFAULT_INDEX_NAME).unlink()
    with pytest.raises(Exception):
        split_dataset_by_similarity(ds, seed=42)
