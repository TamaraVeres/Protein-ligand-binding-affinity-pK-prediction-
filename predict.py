"""Predict binding affinity for a CSV of ligand/pocket pairs. See README for usage."""
import argparse
import csv
import os

import torch
from torch_geometric.data import Batch

from src.model import AffinityModel
from src.features import (
    build_ligand_graph_from_path,
    build_pocket_graph_from_path,
    build_local_cross_context,
)


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AffinityModel(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt["y_mean"].to(device), ckpt["y_std"].to(device), device


def predict_one(model, y_mean, y_std, device, ligand_path, pocket_path):
    lig = build_ligand_graph_from_path(ligand_path)
    poc = build_pocket_graph_from_path(pocket_path)
    lig_ctx, poc_ctx = build_local_cross_context(lig, poc)
    lig.x = torch.cat([lig.x, lig_ctx], dim=1)
    poc.x = torch.cat([poc.x, poc_ctx], dim=1)
    lb = Batch.from_data_list([lig]).to(device)
    pb = Batch.from_data_list([poc]).to(device)
    with torch.no_grad():
        # The model regresses standardized log10(Ki in nM); convert back to pK.
        log10_nM = (model(lb, pb) * y_std + y_mean).item()
    pK = 9.0 - log10_nM
    ki_nM = 10.0 ** log10_nM
    return pK, ki_nM


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein-ligand binding affinity from 3D structures.")
    parser.add_argument("--csv", required=True,
                        help="Input CSV with columns: ligand_path, pocket_path (optional: pdb_id)")
    parser.add_argument("--model", default="best_model.pt",
                        help="Path to a trained model checkpoint (.pt)")
    parser.add_argument("--output", default="predictions.csv",
                        help="Path to write predictions CSV")
    args = parser.parse_args()

    model, y_mean, y_std, device = load_model(args.model)
    print(f"Loaded model from {args.model} (device: {device})")

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} input pairs")

    results = []
    n_ok = n_failed = 0
    for i, row in enumerate(rows):
        ligand_path = row.get("ligand_path", "")
        pocket_path = row.get("pocket_path", "")
        record = {
            "pdb_id": row.get("pdb_id", ""),
            "ligand_path": ligand_path,
            "pocket_path": pocket_path,
            "predicted_pK": "",
            "predicted_Ki_nM": "",
            "status": "ok",
        }
        if not ligand_path or not pocket_path:
            record["status"] = "skipped: missing ligand_path or pocket_path"
            n_failed += 1
        else:
            try:
                pK, ki_nM = predict_one(model, y_mean, y_std, device, ligand_path, pocket_path)
                record["predicted_pK"] = f"{pK:.3f}"
                record["predicted_Ki_nM"] = f"{ki_nM:.2f}"
                n_ok += 1
            except Exception as e:
                record["status"] = f"error: {type(e).__name__}: {e}"
                n_failed += 1
                print(f"  Error on {row.get('pdb_id', i)}: {e}")
        results.append(record)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(rows)}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pdb_id", "ligand_path", "pocket_path", "predicted_pK", "predicted_Ki_nM",
            "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {args.output} "
          f"({n_ok} predicted, {n_failed} failed)")
    if n_failed:
        print("  Failed rows are kept in the output with a non-'ok' status column.")


if __name__ == "__main__":
    main()
