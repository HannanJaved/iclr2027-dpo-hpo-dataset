# DPO offline HPO blackbox

**Checkpoint grid**, **blackbox construction code**, and **HPO simulation
traces** for DPO hyperparameter selection (learning rate and β).

## Layout

```
data/                         checkpoint tables (core artifact)
  {qwen3,llama}_{size}_raw.csv
  {qwen3,llama}_{size}_blackbox.csv
  all_checkpoints_raw.csv              Qwen3 AO + Llama AO
  all_checkpoints_blackbox.csv
  llama_all_checkpoints_{raw,blackbox}.csv   Llama-only concat (1B/3B/8B)
  qwen3_{0.6b,1.7b,4b,8b,14b}_ultrafeedback_{raw,blackbox}.csv
                                             Qwen3 trained on UltraFeedback
                                             (not AO self-play; short schedule)
traces/                       Qwen3 informed-protocol search logs (gzipped)
  qwen3_{size}/{gp,knn1,knn3,knn5}/{generous,tight}/
    simulation_raw.csv.gz     every optimizer report (config × checkpoint)
    best_found.csv.gz         one incumbent per (objective, optimizer, seed)
code/                         construction + simulation
protocol.json                 HPO settings (ASHA rungs, budgets, seeds, …)
manifest.json                 per-size config / fidelity counts
traces_manifest.json          per-trace row counts and objectives
```

**Qwen3 (tables + traces):** 0.6B, 1.7B, 4B, 8B, 14B.
**Llama (tables only):** 1B, 3B, 8B.

Each (lr, β) configuration is scored at DPO steps 200, 400, …, 2000 and the
final checkpoint (~2031). Cost is `elapsed_time_sec` from that run’s trainer
runtime. **UltraFeedback** Qwen3-0.6B / 1.7B / 4B / 8B / 14B use a shorter schedule (steps 48, 96, …,
432 and final ~478) and add a `dataset=ultrafeedback_binarized` column so they
are not mixed into the AO blackbox.

## Core table schema

`all_checkpoints_*.csv` is indexed by `(family, size, lr, beta, dpo_step)`.

| column | meaning |
|---|---|
| `family`, `size` | model family and parameter count |
| `lr`, `beta` | DPO learning rate and β |
| `dpo_step` | fidelity (training step) |
| `elapsed_time_sec` | cumulative training cost to that checkpoint |
| `ARC-C` … `ELO` | 11 downstream scores (higher is better) |
| `Z-Static`, `Z-Dynamic`, `Z-All`, `Z-Macro` | **blackbox tables only** — per-fidelity z-scores (static / preference / 11-bench mean / ½(dynamic+static)) |

`*_raw.csv` keeps empty cells where an evaluation was missing. `*_blackbox.csv`
applies the paper pipeline: drop (lr, β) pairs without a full fidelity
trajectory; forward/back-fill rare intra-config gaps; z-normalize each
benchmark **within a size and fidelity**, then average. Reproduce with
`python code/rebuild_blackbox_tables.py`.

## Simulation traces

One row per blackbox query. Optimizers: RandomSearch, TPE, CQR, ASHA, BOHB.
Search objectives: Z-Dynamic, Z-All, Z-Macro. 25 seeds. Surrogates: GP
(primary) and kNN-1/3/5. Budgets: generous = 8× one full DPO run, tight = 1×.
ASHA/BOHB grace period 400, reduction factor 2 (rungs 400 / 800 / 1600). See
`protocol.json`.

| column | meaning |
|---|---|
| `search_objective`, `optimizer`, `seed`, `trial_id` | which run |
| `dpo_step`, `st_tuner_time` | fidelity and simulated wall-clock |
| `lr`, `beta` | proposed hyperparameters |
| benchmark / Z-* columns | surrogate-evaluated scores at that checkpoint |

`best_found.csv.gz` stores the incumbent under the finished-trial rule in
`code/select_incumbent.py`.

## License

CC BY 4.0 (see `LICENSE`). Please cite the accompanying paper.
