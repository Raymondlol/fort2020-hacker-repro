# Fort et al. 2020 — Hacker-role reproduction (DS 598, BU CDS)

> **Course context.** Built for the *Hacker* role in **CDS DS 598 — Special Topics in ML: Science of Deep Learning**
> (Boston University, Fall 2026, instructor Naomi Saphra), a role-playing paper-reading seminar. The Hacker's job is to
> implement a simplified version of the assigned paper on a toy problem and present a live demo. Paper discussed
> Thu 2026-09-10, week 2 ("Lazy vs. rich learning"), paired with Lee et al. 2019, *Wide neural networks of any depth
> evolve as linear models*. Team 4. Code and analysis here are the author's deliverable; slides are separate.

Small-scale reproduction of *Deep learning versus kernel learning* (Fort, Dziugaite et al., NeurIPS 2020),
§6–§7 / Fig. 8–9: linearised training from onset t̃, the paper's low-LR nonlinear control, a linear-probe
null model, and kernel velocity / distance. Full write-up: `README.md` → "Results, v2". Design and
pre-registered criteria: `PLAN-linearized.md`.

## Core code (v2)

| file | role |
|---|---|
| `linearized.py` | linearised model f_t̃(x) + J_t̃(x)·dw by explicit tangent propagation; SGD training, evaluation, linear probe; `python linearized.py` runs the exactness self-test |
| `fig8.py` | parent (constant LR) → checkpoints → kernel geometry → per-onset green / blue / probe. Stages `parent` / `grid` / `onsets` / `all` |
| `fig8_merge.py` | merge per-onset JSONs into one file per seed |
| `fig8_plot.py` | 5-seed table, pre-registered criteria PASS/FAIL, figure |
| `fig_deck_v2.py` | the two slide figures |
| `core.py`, `measure.py` | data, `SmallCNN`, SGD loop; NTK Jacobian / Gram / kernel distance |
| `modal/fig8_modal.py` | one L4 container per (seed, onset) on Modal |

`v1/` holds the earlier error-barrier / LR-sweep code (Frankle-style spawning). Its §6 estimator
(`measure.ntk_kernel_regression`, kernel ridge on 192 examples) is **not** the paper's method and its
numbers are superseded; kept for provenance only.

## Data included

`results/fig8/fig8_full_seed{0..4}_merged.json` — every curve, every epoch, 5 seeds (parent trajectory,
kernel geometry, green/blue/probe per onset). `fig8_lrgrid_*.json` — the lin_lr grid. Figures in `results/`.

## Reproduce

```bash
pip install torch torchvision numpy matplotlib     # PyTorch >= 2.1 for torch.func
python linearized.py                               # self-test, seconds
python fig8.py --preset smoke --stage all          # ~2 min on an Apple-silicon Mac (MPS) or any GPU
python fig8.py --preset full --stage all --lin-lr 0.003 --seed 0   # ~70 min on MPS, ~15 min on an L4
python fig8_plot.py results/fig8_full_seed0.json
```

CIFAR-10 is expected under `./data` (torchvision layout, `download=False`; set `download=True` in
`core.load_cifar10` or pre-download). The full 5-seed run used Modal (`modal/fig8_modal.py`, ~3 GPU-h).

## Setup summary

SmallCNN(w=32), 188,810 params, no BatchNorm, no augmentation; full CIFAR-10; parent SGD+momentum 0.9,
constant lr 0.02, batch 128, T = 40 epochs; onsets t̃ ∈ {0, 0.01, 0.04, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4,
12.8, 25.6, 40} epochs; green/blue lr 0.003 (validation-chosen once), 40 epochs, early-stopped on a
2,000-image validation split of the test set, errors on the remaining 8,000; 5 seeds.
