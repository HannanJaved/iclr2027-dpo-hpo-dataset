# DPO-AO offline HPO blackbox (checkpoint-level)

Anonymized tables of **evaluated DPO hyperparameters**, **intermediate training fidelities**, **measured training cost**, and **downstream evaluation scores**. These tables are the source used to construct the offline HPO blackboxes in the paper.

No author names, institution names, cluster paths, or training-run identifiers are included.

## Contents

```
data/
  {qwen3,llama}_{size}_raw.csv         collected checkpoint grid (missing evals left as empty)
  {qwen3,llama}_{size}_blackbox.csv    same grid after the blackbox cleaning pipeline + Z-scores
  all_checkpoints_raw.csv              concatenation of the raw tables
  all_checkpoints_blackbox.csv         concatenation of the blackbox tables
manifest.json                          per-size config / fidelity counts
```

**Qwen3:** 0.6B, 1.7B, 4B, 8B, 14B.
**Llama:** 1B, 3B, 8B.

Each (lr, β) configuration is evaluated at DPO steps 200, 400, …, 2000 and the final checkpoint (~2031). Cost is `elapsed_time_sec`, derived from each run’s own trainer runtime (seconds per step × step).

## Columns

| column | meaning |
|---|---|
| `family`, `size` | model family and parameter count |
| `lr`, `beta` | DPO learning rate and β |
| `dpo_step` | fidelity (training step) |
| `elapsed_time_sec` | cumulative training cost to that checkpoint |
| `ARC-C` … `ELO` | 11 downstream scores (higher is better) |
| `Z-Static`, `Z-Dynamic`, `Z-All` | **blackbox tables only** — per-fidelity z-score averages (static / preference / all benchmarks) |

`*_raw.csv` keeps empty cells where an evaluation was missing. `*_blackbox.csv` applies the same cleaning used to build the searchable blackbox: drop (lr, β) pairs that lack a full fidelity trajectory; forward/back-fill rare intra-config gaps; z-normalize each benchmark **within a size and fidelity**, then average.

## Intended use

Offline multi-fidelity HPO / surrogate simulation (e.g. Syne Tune `BlackboxTabular` or a regression surrogate over `log10(lr)` × `log10(beta)`). Not a drop-in Hugging Face model release.

## License

CC BY 4.0 (see `LICENSE`). Cite the paper after acceptance.
