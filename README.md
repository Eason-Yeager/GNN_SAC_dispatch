# Topology-Aware Safe Deep Reinforcement Learning for Economic Dispatch across Microgrid Configurations

Complete, self-contained code for the paper. Trains a message-passing GNN
dispatch policy (SAC / DDPG / PPO backbones) on random microgrid feeders and
evaluates zero-shot transfer across feeder sizes, topologies, adversarial
conditions and a panel of SOTA baselines. No external datasets or power-system
packages are required: feeders are generated procedurally with seeded RNGs, and
power flow uses the linearised DistFlow (LinDistFlow) model.

Two study tracks live here. **The current main paper is the topology-aware GNN
track** (`paper_gnn.md`); the single-microgrid safety+CVaR track (`paper.md`) is
kept as the foundational baseline study.

## Contents

```
grid_env.py          Radial/meshed feeder environment + LinDistFlow power flow
gnn_policy.py        GNN encoders, actors/critics, GraphSAC / GraphDDPG / GraphPPO
run_gnn.py           Main experiment: train GNN/MLP on 8-bus feeders, zero-shot
                     size sweep (6-14 buses), rule-based baseline
run_gnn_mixed.py     Mixed-size training (6-10 buses, padded batching), zero-shot
                     evaluation up to 16 buses
run_gnn_verify.py    Robustness (sigma=0.2/0.4 + sensor noise), large-scale
                     32/64-bus, inference latency measurements
run_gnn_advanced.py  N-1 contingency, constraint ablation (none/soft/projection),
                     DDPG/PPO/SAC backbone comparison, PCA of node embeddings
run_gnn_sota.py      SOTA panel: GAT-SAC, PPO-Lagrangian (CMDP), padded MLP-SAC,
                     Indep-DDPG, Graph-MADDPG; weakly meshed topology migration;
                     FDI + load-step adversarial stress test
run_gnn_dropout.py   Bernoulli telemetry-dropout robustness
run_gnn_sens.py      Hyperparameter sensitivity: depth L in {2,3,4} x width H
                     in {32,64,128}
env.py               Single-microgrid environment (PV, BESS, diesel, grid tie)
safety_layer.py      Feasibility projection layer (merit-order restoration)
cvar_sac.py          Flat MLP-SAC with CVaR (Rockafellar-Uryasev) shaping
baselines.py         Rule-based and greedy/MPC baselines (single microgrid)
run_experiments.py   Train 4 SAC variants + 2 baselines, evaluate (baseline track)
lambda_sweep.py      CVaR lambda sensitivity sweep (baseline track)
make_figures.py      All paper figures from saved results/*.json / *.npz
build_gnn_docx.py    Builds the MDPI-format manuscript (gnn_paper_mdpi.docx)
                     and its markdown mirror (paper_gnn.md)
results/             All measured JSON/npz outputs backing every table/figure
figures/             Generated figures (png + pdf)
paper_gnn.md         Markdown mirror of the manuscript
paper.md, OUTLINE.md Artifacts of the earlier single-microgrid track
```

## Requirements

Python 3.9+ (tested on 3.9.6, macOS). CPU-only is sufficient; no GPU needed.

```
numpy>=1.23
torch>=1.12
matplotlib>=3.5
pandas>=1.4
python-docx>=1.1    # only needed for build_gnn_docx.py
```

```bash
pip install -r requirements.txt
```

## Reproducing every number in the paper

All scripts are seeded; re-running them reproduces the JSONs on the same
hardware/software stack. Approximate wall-clock on one modern CPU core:

| Script | Approx. time | Produces |
|---|---|---|
| `run_gnn.py --episodes 600 --eval-days 40` | ~10 min | `gnn_results.json` (Tables 3-5) |
| `run_gnn_mixed.py --episodes 700 --eval-days 40` | ~10 min | `gnn_mixed_results.json` (Table 5) |
| `run_gnn_verify.py` | ~15 min | `gnn_verify.json` (Tables 6, 7, 10) |
| `run_gnn_advanced.py --ep8 500 --epmix 400 --days 20` | ~25 min | `gnn_advanced.json` (Tables 8, 9, 11) |
| `run_gnn_sota.py --eps 500 --days 20` | ~26 min | `gnn_sota.json` (Tables 13-16) |
| `run_gnn_dropout.py` | ~5 min | `gnn_dropout.json` (Table 12) |
| `run_gnn_sens.py --eps 500 --days 20` | ~40 min | `gnn_sens.json` (Table 17) |
| `run_experiments.py --episodes 300 --eval-days 100` | ~20 min | baseline-track results |

Quick smoke test of the main pipeline (~3 min total):

```bash
python run_gnn.py --episodes 40 --eval-days 2
python run_gnn_sota.py --quick
python run_gnn_sens.py --quick
```

Regenerate all figures from the saved results:

```bash
python make_figures.py
```

Rebuild the manuscript (docx + markdown) — requires the MDPI template docx at
the path configured at the top of `build_gnn_docx.py`:

```bash
python build_gnn_docx.py
```

## Environment model (grid_env.py)

* `make_feeder(n, seed)` — random radial tree: uniform parent attachment,
  impedances in [0.015, 0.05] pu, ~60% of non-slack buses have PV, ~40% are
  controllable (BESS + diesel DG).
* `make_meshed_feeder(n, seed, loop_frac)` — radial tree closed with extra
  tie-lines at 10%/30% of tree-edge count (for the topology-migration test).
* `FeederEnv` — 24-step daily dispatch; TOU tariff + marginal demand charge on
  the daily peak import; SOC dynamics with eta = 0.95; constraint modes
  `projection` (hard, zero violations) / `soft` / `none` (counted violations).
* LinDistFlow: radial feeders use the O(n) subtree-sum closed form; meshed
  feeders assemble one linear system over branch flows (KCL rows + one
  voltage-difference row per tie). With zero ties the system reproduces the
  radial closed form to machine precision (verified in-code).

## Method summary

* Policy: L-layer message-passing GCN (default L = 3, H = 64), per-node
  Gaussian actor head with tanh squashing; twin critics mean-pool over
  controllable buses; SAC with automatic entropy tuning (target -2|C|).
* Mixed-size training pads all feeders to N_MAX = 10 with isolated self-looped
  dummy nodes, masked out of pooling and the entropy target.
* Safety: closed-form projection clamps each battery command to its
  SOC-feasible window and rating, each DG to [0, P_g,max], and the substation
  to the tie-line rating -> zero box/balance violations by construction.
  Voltages are encouraged by a soft penalty and reported as a metric.

## Key results (all in results/, measured)

* GNN trained only on 8-bus feeders dispatches 6-14-bus feeders zero-shot
  (35% cheaper than rule-based at 10 buses); mixed 6-10 training reaches
  16 buses at 61% lower cost than rule-based; the MLP cannot leave its size.
* SOTA panel (10-bus): GNN-SAC+projection 675.2 $ (lowest, only exactly
  feasible); GAT +5.5%, PPO-Lagrangian +6.6%, padded MLP +4.4% but with 54%
  box violations, Indep-DDPG +10.9%, Graph-MADDPG +31.4%.
* Zero-shot migration to weakly meshed feeders (10%/30% loop edges) with
  feasibility intact; 32-bus FDI+load-step attack costs +11.8% with zero hard
  violations; N-1 reconfiguration absorbed without incident.
* Inference latency 0.064-0.078 ms from 8 to 64 buses (linear scaling).

## Data

The environment reproduces the *statistical* structure of public datasets
(diurnal PV from clear-sky models, twin-peak load shape, time-of-use tariff)
via a seeded stochastic generator, so the whole study runs end-to-end with no
downloads. To use real traces, replace `_pv_profile` / `_load_profile` /
`_tou_price` in `env.py` with CSV loaders.
