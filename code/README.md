# Construction code

Python used to clean the checkpoint grid, fit the continuous surrogates, and
run the five Syne Tune optimizers. Cluster scripts are omitted.

| file | role |
|---|---|
| `build_blackbox.py` | drop incomplete (lr, β) trajectories, fill rare gaps, z-score composites including Z-Macro |
| `rebuild_blackbox_tables.py` | pandas-only rebuild of `data/*_blackbox.csv` from `data/*_raw.csv` |
| `build_blackbox_surrogate.py` | GP / kNN surrogate over log10(lr) × log10(β) |
| `run_simulations_surrogate.py` | RandomSearch, TPE, CQR, ASHA, BOHB × 25 seeds |
| `schedulers.py` | ASHA/BOHB rungs and seeded RandomSearch/TPE |
| `select_incumbent.py` | finished-trial incumbent (not argmax over every checkpoint) |
| `common.py` | protocol constants; paths point at this artifact |

Released **traces** are already the completed searches; re-running the
simulators needs `syne-tune`, scikit-learn, and a local `work/` directory.
