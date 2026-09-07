import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Batch
from src.data import PDBbindDataset
from src.model import AffinityModel
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pdbbind_batch(batch):
    return {
        "ligand": Batch.from_data_list([s["ligand"] for s in batch]),
        "pocket": Batch.from_data_list([s["pocket"] for s in batch]),
        "y": torch.stack([s["y"] for s in batch]),
        "pdb_id": [s["pdb_id"] for s in batch],
    }


def split_dataset_random(dataset, train=0.8, val=0.1, test=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(train * n)
    n_val = int(val * n)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def split_dataset_by_similarity(dataset, train=0.8, val=0.1, test=0.1,
                              similarity_cutoff=0.75, seed=42):
    """Split so that Tanimoto-similar ligands never straddle train and test."""
    from src.data.fingerprints import (DEFAULT_INDEX_NAME, build_index, cluster_by_similarity,
                                       load_index, unpack)

    if hasattr(dataset, "rows"):
        base = dataset
        indices = list(range(len(dataset.rows)))
    else:
        base = dataset.dataset
        indices = list(dataset.indices)
    rows = base.rows

    index_path = Path(base.csv_path.parent) / DEFAULT_INDEX_NAME
    fps_by_id = load_index(index_path)
    if fps_by_id is None:
        print(f"Fingerprint index not found at {index_path}; building it once from "
              f"{len(base.cache_index)} cached samples (this takes a while)...")
        build_index(base.cache_index, index_path)
        fps_by_id = load_index(index_path)

    packed, valid_local_idx = [], []
    for local_idx, orig_idx in enumerate(indices):
        fp = fps_by_id.get(rows[orig_idx]["pdb_id"])
        if fp is None:
            continue
        packed.append(fp)
        valid_local_idx.append(local_idx)

    n_missing = len(indices) - len(valid_local_idx)
    if not valid_local_idx:
        raise RuntimeError(
            f"No ligand fingerprints available for any of the {len(indices)} samples, so a "
            f"similarity split cannot be built. Check that {index_path} matches the dataset, "
            f"or rebuild it with: python -m src.data.fingerprints")
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(indices)} samples have no fingerprint and will be "
              f"placed in singleton clusters.")

    print(f"Clustering {len(valid_local_idx)} ligands at Tanimoto >= {similarity_cutoff}...")
    clusters = cluster_by_similarity(unpack(packed), similarity_cutoff)
    groups = [[valid_local_idx[i] for i in cluster] for cluster in clusters]
    groups.extend([lidx] for lidx in sorted(set(range(len(indices))) - set(valid_local_idx)))
    print(f"  {len(groups)} clusters; largest holds {max(len(g) for g in groups)} ligands")

    # Largest cluster first, into whichever split is furthest below its target.
    groups.sort(key=lambda g: (-len(g), g[0]))
    n_total = len(indices)
    targets = [train * n_total, val * n_total, test * n_total]
    buckets = [[], [], []]
    for group in groups:
        deficits = [t - len(b) for t, b in zip(targets, buckets)]
        buckets[deficits.index(max(deficits))].extend(group)
    train_idx, val_idx, test_idx = buckets
    if not val_idx or not test_idx:
        print(f"  WARNING: only {len(groups)} cluster(s) for {n_total} samples; the split is "
              f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}.")
    return train_idx, val_idx, test_idx


def train(model, loader, optimizer, device, y_mean, y_std):
    model.train()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    n_samples = 0
    y_mean_dev = y_mean.to(device)
    y_std_dev = y_std.to(device)
    for batch in loader:
        ligand_data = batch["ligand"].to(device)
        pocket_data = batch["pocket"].to(device)
        y_true = batch["y"].to(device).view(-1)
        y_standardized = (y_true - y_mean_dev) / y_std_dev

        # Coordinate-noise augmentation.
        ligand_data.pos = ligand_data.pos + torch.randn_like(ligand_data.pos) * 0.05
        pocket_data.pos = pocket_data.pos + torch.randn_like(pocket_data.pos) * 0.05

        optimizer.zero_grad()
        y_pred = model(ligand_data, pocket_data)
        loss = mse_loss(y_pred, y_standardized)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_samples += 1
    return total_loss / max(n_samples, 1)


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    all_preds_std, all_trues_std, all_trues_orig = [], [], []
    with torch.no_grad():
        for batch in loader:
            ligand_data = batch["ligand"].to(device)
            pocket_data = batch["pocket"].to(device)
            y_true = batch["y"].to(device).view(-1)
            y_standardized = (y_true - y_mean.to(device)) / y_std.to(device)
            y_pred = model(ligand_data, pocket_data)
            all_preds_std.append(y_pred.detach().cpu())
            all_trues_std.append(y_standardized.detach().cpu())
            all_trues_orig.append(y_true.detach().cpu())

    keys = ["mse", "mae", "rmse", "r2", "rp", "mae_um", "rmse_um", "r2_um", "rp_um",
            "median_fold", "within_2x", "within_5x", "within_10x"]
    if not all_preds_std:
        return {k: float("nan") for k in keys}

    y_pred_std = torch.cat(all_preds_std)
    y_true_std = torch.cat(all_trues_std)
    y_true_orig = torch.cat(all_trues_orig)
    y_mean_cpu = y_mean.cpu()
    y_std_cpu = y_std.cpu()
    y_pred_orig = y_pred_std * y_std_cpu + y_mean_cpu

    mae = torch.mean(torch.abs(y_pred_orig - y_true_orig)).item()
    mse_val = torch.mean((y_pred_orig - y_true_orig) ** 2).item()
    rmse_val = mse_val ** 0.5
    var_y = torch.var(y_true_orig, unbiased=False).item()
    r2 = (1 - mse_val / var_y) if var_y > 0 else float("nan")

    mae_z = torch.mean(torch.abs(y_pred_std - y_true_std)).item()
    mse_z = torch.mean((y_pred_std - y_true_std) ** 2).item()
    rmse_z = mse_z ** 0.5

    pred_c = y_pred_orig - y_pred_orig.mean()
    true_c = y_true_orig - y_true_orig.mean()
    rp = (torch.sum(pred_c * true_c) /
          (torch.sqrt(torch.sum(pred_c ** 2)) * torch.sqrt(torch.sum(true_c ** 2)) + 1e-8)).item()

    y_pred_um = torch.pow(10.0, y_pred_orig) / 1000.0
    y_true_um = torch.pow(10.0, y_true_orig) / 1000.0
    mae_um = torch.mean(torch.abs(y_pred_um - y_true_um)).item()
    mse_um = torch.mean((y_pred_um - y_true_um) ** 2).item()
    rmse_um = mse_um ** 0.5
    var_y_um = torch.var(y_true_um, unbiased=False).item()
    r2_um = (1 - mse_um / var_y_um) if var_y_um > 0 else float("nan")
    pred_um_c = y_pred_um - y_pred_um.mean()
    true_um_c = y_true_um - y_true_um.mean()
    rp_um = (torch.sum(pred_um_c * true_um_c) /
             (torch.sqrt(torch.sum(pred_um_c ** 2)) * torch.sqrt(torch.sum(true_um_c ** 2)) + 1e-8)).item()

    abs_err_log = torch.abs(y_pred_orig - y_true_orig)
    fold_err = torch.pow(10.0, abs_err_log)
    median_fold = torch.median(fold_err).item()
    within_2x = (fold_err < 2.0).float().mean().item()
    within_5x = (fold_err < 5.0).float().mean().item()
    within_10x = (fold_err < 10.0).float().mean().item()

    return {
        "mse": mse_val, "mae": mae, "rmse": rmse_val, "r2": r2, "rp": rp,
        "mse_z": mse_z, "mae_z": mae_z, "rmse_z": rmse_z,
        "mae_um": mae_um, "rmse_um": rmse_um, "r2_um": r2_um, "rp_um": rp_um,
        "median_fold": median_fold,
        "within_2x": within_2x, "within_5x": within_5x, "within_10x": within_10x,
    }


def compute_feature_stats(train_subset):
    buckets = {"lig_x": [], "lig_edge": [], "lig_graph": [],
               "poc_x": [], "poc_edge": [], "poc_graph": []}
    for i in range(len(train_subset)):
        s = train_subset[i]
        buckets["lig_x"].append(s["ligand"].x)
        if s["ligand"].edge_attr.size(0) > 0:
            buckets["lig_edge"].append(s["ligand"].edge_attr)
        buckets["lig_graph"].append(s["ligand"].lig_graph_feat.view(1, -1))
        buckets["poc_x"].append(s["pocket"].x)
        if s["pocket"].edge_attr.size(0) > 0:
            buckets["poc_edge"].append(s["pocket"].edge_attr)
        buckets["poc_graph"].append(s["pocket"].poc_graph_feat.view(1, -1))
    stats = {}
    for name, tensors in buckets.items():
        if not tensors:
            continue
        cat = torch.cat(tensors, dim=0)
        mean = cat.mean(dim=0)
        std = cat.std(dim=0, unbiased=False).clamp(min=1e-6)
        stats[name] = (mean, std)
    return stats


def main(csv_path="pl_table_with_paths.csv", num_epochs=50, batch_size=32,
         hidden_dim=64, seed=42, split="random", assay_filter=None, save_path=None,
         cache_dir="cache", num_workers=0):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PDBbindDataset(csv_path, cache_dir=cache_dir, assay_filter=assay_filter)

    if split == "similarity":
        train_idx, val_idx, test_idx = split_dataset_by_similarity(dataset, seed=seed)
        print(f"Similarity split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        train_idx, val_idx, test_idx = split_dataset_random(dataset, seed=seed)
        print(f"Random split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    ys = [train_set[i]["y"].item() for i in range(len(train_set))]
    ys_tensor = torch.tensor(ys, dtype=torch.float32)
    y_mean = ys_tensor.mean()
    y_std = ys_tensor.std(unbiased=False)

    print("Computing feature statistics on training set...")
    feature_stats = compute_feature_stats(train_set)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size, True, collate_fn=build_pdbbind_batch,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size, False, collate_fn=build_pdbbind_batch,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size, False, collate_fn=build_pdbbind_batch,
                             num_workers=num_workers, pin_memory=pin)

    sample = dataset[0]
    ligand_in_dim = sample["ligand"].x.size(1)
    pocket_in_dim = sample["pocket"].x.size(1)
    ligand_edge_dim = sample["ligand"].edge_attr.size(1)
    pocket_edge_dim = sample["pocket"].edge_attr.size(1)
    lig_graph_feat_in_dim = sample["ligand"].lig_graph_feat.numel()
    poc_graph_feat_in_dim = sample["pocket"].poc_graph_feat.numel()

    model = AffinityModel(
        ligand_in_dim, pocket_in_dim, lig_graph_feat_in_dim, poc_graph_feat_in_dim,
        hidden_dim, ligand_edge_dim, pocket_edge_dim, feature_stats=feature_stats,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)

    best_val_mse = float("inf")
    best_state = None

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/train_{timestamp}_{split}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_mae", "val_rmse", "val_r2", "val_rp",
                         "val_mae_um", "val_rmse_um", "val_r2_um", "val_rp_um",
                         "val_median_fold", "val_within_2x", "val_within_5x", "val_within_10x", "lr"])

    print("Training started...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, y_mean, y_std)
        m = evaluate(model, val_loader, device, y_mean, y_std)
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        if m["mse"] < best_val_mse:
            best_val_mse = m["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            marker = ""
        print(f"Epoch {epoch:03d} | Train loss={train_loss:.4f} | Val MAE={m['mae_z']:.4f} "
              f"| Val RMSE={m['rmse_z']:.4f} | Val R2={m['r2']:.3f} | LR={lr:.1e}{marker}")
        log_writer.writerow([epoch, f"{train_loss:.4f}", f"{m['mae']:.4f}", f"{m['rmse']:.4f}",
                             f"{m['r2']:.3f}", f"{m['rp']:.3f}", f"{m['mae_um']:.4f}",
                             f"{m['rmse_um']:.4f}", f"{m['r2_um']:.3f}", f"{m['rp_um']:.3f}",
                             f"{m['median_fold']:.3f}", f"{m['within_2x']:.4f}",
                             f"{m['within_5x']:.4f}", f"{m['within_10x']:.4f}", lr])
        log_file.flush()

    model.load_state_dict(best_state)
    t = evaluate(model, test_loader, device, y_mean, y_std)
    print(f"Test (z-scored): MSE={t['mse_z']:.4f}, MAE={t['mae_z']:.4f}, RMSE={t['rmse_z']:.4f}, R2={t['r2']:.3f}")
    print(f"Test (uM):       MAE={t['mae_um']:.4f} uM, RMSE={t['rmse_um']:.4f} uM, R2={t['r2_um']:.3f}, Rp={t['rp_um']:.3f}")
    log_file.close()
    print(f"Training log saved to {log_path}")

    checkpoint = {
        "model_state": best_state,
        "y_mean": y_mean,
        "y_std": y_std,
        "model_config": {
            "ligand_in_dim": ligand_in_dim,
            "pocket_in_dim": pocket_in_dim,
            "lig_graph_feat_in_dim": lig_graph_feat_in_dim,
            "poc_graph_feat_in_dim": poc_graph_feat_in_dim,
            "hidden_dim": hidden_dim,
            "ligand_edge_dim": ligand_edge_dim,
            "pocket_edge_dim": pocket_edge_dim,
            "aa_embed_dim": 8,
        },
    }
    ckpt_name = save_path or f"best_model_seed{seed}.pt"
    if os.path.dirname(ckpt_name):
        os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)
    torch.save(checkpoint, ckpt_name)
    print(f"Saved checkpoint to {ckpt_name}")


if __name__ == "__main__":
    main()
