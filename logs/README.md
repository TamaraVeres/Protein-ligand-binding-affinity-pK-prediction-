# Training logs

The runs behind the Results table in the top-level README. Each file records per-epoch
validation metrics and a final `test` row.

| Log | Deck row | Checkpoint | Seed | Test R2 | Rp |
|---|---|---|---|---|---|
| `random_run1.csv` | Random — run 1 | `checkpoints/random_run1.pt` | 42 | 0.581 | 0.766 |
| `random_run2.csv` | Random — run 2 | `checkpoints/random_run2.pt` | 0 | 0.658 | 0.817 |
| `random_run3.csv` | Random — run 3 | `checkpoints/random_run3.pt` | 1 | 0.652 | 0.809 |
| `similarity_run1.csv` | Similarity split | `checkpoints/similarity_run1.pt` | 42 | 0.377 | 0.622 |

Columns: `epoch, train_loss, val_mae, val_rmse, val_r2, val_rp, val_mae_um, val_rmse_um,
val_r2_um, val_rp_um, val_median_fold, val_within_2x, val_within_5x, val_within_10x, lr`.

MAE and RMSE in these files are in log10(Ki in nM); the Results table reports them
standardized (divided by `y_std`).
