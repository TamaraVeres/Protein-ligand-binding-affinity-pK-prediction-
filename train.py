"""Train the binding-affinity model. See README for usage."""
import argparse
from src.training import train_main


def main():
    p = argparse.ArgumentParser(description="Train protein-ligand affinity model (v2)")
    p.add_argument("--csv", default="data/pl_table_with_paths.csv")
    p.add_argument("--cache_dir", default="data/cache")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="random", choices=["random", "similarity"],
                   help="random, or similarity (Tanimoto-clustered, leakage-controlled)")
    p.add_argument("--assay_filter", default=None, choices=["ki", "kd"],
                   help="Restrict training data to a single assay type")
    p.add_argument("--save", default=None)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. The cache is latency-bound, so >0 "
                        "greatly speeds up the first pass over the data.")
    a = p.parse_args()
    train_main(csv_path=a.csv, num_epochs=a.epochs, batch_size=a.batch_size,
               hidden_dim=a.hidden_dim, seed=a.seed, split=a.split,
               assay_filter=a.assay_filter, save_path=a.save, cache_dir=a.cache_dir,
               num_workers=a.num_workers)


if __name__ == "__main__":
    main()
