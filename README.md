# Hacker role — Fort et al. (2020), *Deep learning versus kernel learning*

arXiv:[2010.15110](https://arxiv.org/abs/2010.15110) · NeurIPS 2020 · discussed Thu 2026-09-10

A from-scratch, small-scale reproduction of the paper's central claim.

## The claim being tested

Fort et al. argue that deep network training has a **chaotic transient** lasting
~2–3 epochs, and that several apparently unrelated quantities all change together
during it and then stop. Their Fig. 7 is the whole paper in one plot: the child
**error barrier** and the **NTK velocity** fall off together and stabilise at the
same time. The consequence is that the NTK-at-initialisation picture — a fixed,
random, data-independent kernel — is the wrong description of exactly the phase
of training where everything interesting happens.

This directory reproduces that, plus their Fig. 8 result that the *data-dependent*
NTK rapidly learns useful features.

## What is measured (all four are transcribed from the paper's §2)

| Quantity | Definition | Where |
|---|---|---|
| Kernel distance | `S(w,w') = 1 − Tr(κ_w κ_w'ᵀ) / √(Tr(κ_w κ_wᵀ)·Tr(κ_w' κ_w'ᵀ))` | `measure.kernel_distance` |
| Kernel velocity | `v(t) = S(w_t, w_{t+dt}) / dt`, dt = 0.4 epochs (paper's value) | `measure.kernel_velocity` |
| Error barrier | `max_α R(αw + (1−α)w') − ½(R(w)+R(w'))`, R = 0–1 error | `measure.error_barrier` |
| ReLU pattern distance | normalised Hamming distance of on/off patterns | `measure.relu_distance` |

The empirical NTK `κ_t(x,x') = J_t(x)J_t(x')ᵀ` (their Eq. 4) is computed exactly,
as a full `mK × mK` Gram matrix over a fixed probe set, via `torch.func`
`vmap(jacrev(·))` — no kernel approximation, no random features.

## Design choices, and why

- **CIFAR-10 at lr = 0.1.** The dataset matters far less than the step size, which
  was not what I expected going in. Measured directly (`mnist_check.py`,
  `modal/sweep.py`): MNIST *does* show a barrier (0.070 at spawn 0, collapsing to
  0.0002), and CIFAR-10 at lr = 0.03 shows essentially none (0.0015) despite
  reaching zero training error. What selects the regime is the learning rate —
  see "The phenomenon has a step-size threshold" below. CIFAR-10 at lr = 0.1 is
  chosen because it is a standard setup and sits in the regime the paper studies,
  not because MNIST would show nothing.
- **No BatchNorm.** Linearly interpolating two networks is ill-defined with BN:
  the running statistics of the interpolated weights do not match the
  interpolated network's actual activation statistics, so part of the midpoint
  error is a normalisation artifact rather than landscape structure. The standard
  fix (Frankle et al.; Git Re-Basin) is to reset and re-estimate those buffers at
  every α — correct, but ~11 extra passes over the training set per pair and a
  quiet correctness trap in code that has to be readable on a screen.
  The paper does run no-BN ResNet20 controls (Appendix C.3, Figs. 16–21) and sees
  the same phenomena, though C.3's stated subject is the §7 nonlinear-advantage
  experiments rather than the error barrier specifically.
  Note also that the main text never describes any handling of BN statistics
  during interpolation. If none was done, their headline barriers (~0.8 on
  CIFAR-10, where chance is 0.9) include that artifact, and the ~0.46 measured
  here without BN is the cleaner number. This is a question for the paper's
  authors, not a demonstrated error.
  The cost of the choice: BN materially changes optimisation dynamics, so this is
  not their exact setup, and C.3 mitigates that gap without closing it.
- **No data augmentation.** Removes a second source of child-to-child randomness,
  so that children differ *only* in SGD minibatch order — which is what the
  parent/child design is meant to isolate.
- **Children inherit the parent's momentum buffer**, not just its weights. Without
  this, spawning injects an optimiser transient that has nothing to do with the
  phenomenon.
- **~48k-parameter CNN.** The NTK Jacobian is `[N, K, d]`, so cost is linear in the
  parameter count; this size keeps the exact kernel affordable on a laptop.

## Section 6 — linearised training, by the paper's method (v2)

An earlier version of this directory answered §6 with closed-form kernel ridge regression on
192 training examples (`measure.ntk_kernel_regression`). That is **not** the paper's procedure
and is retired: it dropped the f₀ term, used a trace kernel instead of the mK×mK block kernel,
fit 192 examples instead of 50,000, and used an untuned λ with a squared loss. Its numbers below
are marked superseded. Details in `PLAN-linearized.md` §1.

`linearized.py` now trains the linearised model exactly as §6 / App. D.4 describe:
`f_lin(x) = f_t̃(x) + J_t̃(x)·dw`, cross-entropy, SGD + momentum on `dw` from 0, early stopping.
`J·dw` is propagated explicitly through the ReLU / max-pool masks (PyTorch has no forward-AD
kernel for max_pool2d); the self-test checks it against autograd's double-VJP to 1e-8.
`fig8.py` runs, per onset t̃: the linearised model (green), the paper's low-LR nonlinear control
from the same weights (blue, §7), and a linear probe on the penultimate features (a null model
the paper does not have). `fig8_plot.py` evaluates criteria pre-registered in the plan.

## Files

```
core.py          data, model, SGD loop, flat-weight get/set
measure.py       kernel distance / velocity / ReLU distance / error barrier (+ retired ntk_kernel_regression)
run.py           v1: parent -> checkpoints -> children -> error barriers -> JSON
linearized.py    v2: linearised training along the tangent plane (paper §6), self-test in __main__
fig8.py          v2: parent (constant LR) -> kernel geometry -> per-onset green / blue / probe
fig8_merge.py    merge per-onset JSONs;  fig8_plot.py  5-seed table, criteria, figure
modal/fig8_modal.py   one L4 container per (seed, onset); ~3 GPU-hours, ~$3
```

## Results

Four runs, 2 data scales x 2 LR schedules (`results/results_*.json`, `python compare.py`).

|                        | 10k / cosine | 10k / step | 50k / cosine | 50k / step |
|------------------------|-------------:|-----------:|-------------:|-----------:|
| params                 |      106,666 |    106,666 |      188,810 |    188,810 |
| final test acc         |        0.573 |      0.548 |        0.718 |      0.671 |
| final train err        |       0.0005 |     0.0881 |       0.0103 |     0.1451 |
| barrier @ spawn 0      |       0.3204 |     0.2574 |       0.4608 |     0.4669 |
| barrier @ spawn 3.2    |       0.0043 |     0.0363 |       0.0029 |     0.0445 |
| collapse epoch (<10%)  |          1.6 |        4.8 |          1.6 |        3.2 |
| NTK acc @ init         |        0.230 |      0.230 |        0.250 |      0.250 |
| NTK acc best           |        0.574 |      0.531 |        0.714 |      0.599 |
| NTK best / full net    |        1.002 |      0.969 |        0.994 |      0.892 |
| v/LR decay 0.4 -> 4ep  |        0.81x |      0.57x |        0.49x |      0.74x |

**Reproduced (v1).** The error barrier between children collapses within a few epochs
in all four runs. (The v1 sentence about the kernel "reaching 0.714 vs 0.718" rested on a
192-image evaluation and flipped sign on re-measurement; see "Results, v2".)

**Reproduced only after fixing the measurement.** Raw kernel velocity disagreed
across the four runs and rose again late in training, which looked like a failed
reproduction. Velocity is close to proportional to the learning rate, so the raw
comparison across schedules was mostly comparing schedules. Dividing the LR out
(`compare.vels_norm`) makes all four runs agree: a decay over the first ~3 epochs
followed by a flat band.

**Two things this reproduction does not support.**

1. *The specific "2-3 epochs".* Collapse lands anywhere from 1.6 to 4.8 epochs
   depending on the run, tracking how converged that run is rather than anything
   intrinsic. The qualitative claim (early, and then irreversible) is robust; the
   number is not.
2. *A clean schedule comparison.* Both step-schedule runs are underfit (train
   error 8.8% and 14.5%) because the drops sit at 60%/85% of a short budget, so
   the cosine-vs-step axis is confounded with convergence. Reading any difference
   between those columns as a schedule effect would be wrong.

The barrier measurement needs a converged network: an earlier constant-LR
configuration stopped at 32% train error and produced barriers of 0.019 that were
indistinguishable from noise. Nothing about the phenomenon changed when the LR
schedule was fixed -- only whether the network had finished choosing a basin.

## Results, v2 — §6/§7 by the paper's method (5 seeds)

SmallCNN(32), 188,810 params, no BN, full CIFAR-10, **constant** parent LR 0.02 (the paper's 0.1
is a ResNet+BN number; without BN this net sits at 47 % train error at 0.1), T = 40 epochs.
Linearised LR 0.003 chosen once on validation (t̃ = 3.2, seed 0). 2,000 validation / 8,000 test.
Errors are test error at the validation-optimal epoch, mean ± SE over 5 seeds. Parent at T
("purple"): 0.277 ± 0.004. Full table: `python fig8_plot.py results/_modal/fig8/fig8_full_seed*_merged.json`.

| t̃ (ep) | NN at t̃ | green: linearised at t̃ | blue: nonlinear from t̃, lr 0.003 | linear probe | green − blue |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.906 | 0.617 ± 0.007 | 0.249 ± 0.002 | 0.597 | +0.369 |
| 0.4 | 0.585 | 0.413 ± 0.004 | 0.249 ± 0.002 | 0.501 | +0.164 |
| 1.6 | 0.384 | 0.283 ± 0.002 | 0.248 ± 0.002 | 0.340 | +0.035 |
| 3.2 | 0.302 | 0.247 ± 0.002 | 0.242 ± 0.001 | 0.273 | +0.005 |
| 6.4 | 0.268 | 0.237 ± 0.001 | 0.230 ± 0.002 | 0.247 | +0.007 |
| 12.8 | 0.260 | 0.240 ± 0.001 | 0.232 ± 0.001 | 0.245 | +0.008 |
| 40 | 0.277 | 0.253 ± 0.001 | 0.243 ± 0.003 | 0.264 | +0.010 |

Kernel geometry (128-image probe, mK×mK Gram, dt = 0.4 ep): velocity 1.23 in the first 0.4 epoch
vs 0.34 mean over 4–20 epochs (3.6×); then it **rises** again to 0.45 by epoch 40 under the
constant LR. Distance from the initial kernel reaches 0.49 by 0.4 epoch, dips to 0.47 at 1.6,
then climbs to 0.71.

Pre-registered criteria (PLAN-linearized.md §4):

| | result |
|---|---|
| A  literal: err(3.2) ≤ ⅓ err(0) | FAIL — 2.5×, not 3× (0.247 vs 0.206) |
| A′ scale-free: ≥ ⅔ of the gap to N(T) closed by 3.2 ep | PASS — 109 % |
| B  green reaches purple at t̃*/T ≤ 0.5 | PASS — t̃* = 1.6 ep (4 % of T); paper: 15–45 % |
| B′ nonlinear advantage vanishes after a few epochs | FAIL — falls from +0.37 to +0.005 by 3.2 ep, then a persistent +0.007…+0.010 (3–4 SE) |
| K  early velocity ≥ 2× plateau, plateau > 0 | PASS — 3.6× |
| P  green beats the linear probe (mid t̃) | PASS — by 6 pp at 1.6 ep, 0.5 pp at 12.8 ep |

What this says:

1. **The Fig. 8 shape reproduces, but "matches full training" is an annealing artifact.** The
   blue control is a flat line: low-LR nonlinear training from *any* t̃, including from
   initialisation, reaches 0.24–0.25, which beats the constant-LR parent at T (0.277). Green
   reaching purple therefore only says purple is a noisy endpoint. The paper's purple is also a
   constant-LR network (App. D.3), so the same confound is in the original.
2. **The fair comparison (green vs blue) reproduces §7 in shape**: the advantage of nonlinear
   training collapses from 37 pp to 0.5 pp within ~3 epochs, on the same timescale as the
   kernel-velocity drop (0.4–0.8 ep). It does **not** reach zero: a ~1 pp gap persists at every
   later t̃, including t̃ = T, at 3–4 SE. The paper's Fig. 9 shows it vanishing; n = 1 there.
3. **A random NTK is no better than a linear read-out of random features** (0.617 vs 0.597).
   The tangent space earns its keep only between ~0.4 and ~6 epochs; after that, linearised
   training ≈ linear probe within 1 pp, i.e. "kernel learning" ≈ representation learning.
4. **Kernel velocity is not constant after the transient** under a constant LR: it rises
   from 0.31 (6 ep) to 0.45 (40 ep), and the kernel linearised at T is *worse* than the one at
   6.4 ep (0.253 vs 0.237). Consistent across all 5 seeds.

Caveats: early-onset green runs (t̃ ≤ 1.6) were still improving at the 40-epoch budget, so those
errors are upper bounds; blue and green use the same LR and budget rather than the paper's
"train to convergence"; test-stopped vs validation-stopped numbers differ by ≤ 0.4 pp.

## The phenomenon has a step-size threshold

Ten learning rates, everything else held fixed (10k CIFAR-10, same net, same
cosine schedule, 3 children at each of 3 spawn epochs). Run on Modal as ten
parallel containers; `modal/sweep.py` submits, `modal/fetch.py` collects.

|    lr | test acc | train err | barrier@0 | barrier@1.6 | barrier@6 |
|------:|---------:|----------:|----------:|------------:|----------:|
| 0.003 |    0.623 |    0.2350 |    0.0016 |      0.0017 |    0.0020 |
| 0.010 |    0.625 |    0.0000 |    0.0021 |      0.0022 |    0.0004 |
| 0.020 |    0.631 |    0.0000 |    0.0010 |      0.0009 |    0.0003 |
| 0.030 |    0.637 |    0.0000 |    0.0015 |      0.0008 |    0.0016 |
| 0.045 |    0.631 |    0.0000 |    0.0124 |      0.0008 |    0.0020 |
| 0.060 |    0.609 |    0.0000 |    0.0287 |      0.0010 |    0.0023 |
| 0.080 |    0.598 |    0.0001 |    0.0931 |      0.0047 |    0.0022 |
| 0.100 |    0.586 |    0.0007 |    0.2634 |      0.0240 |    0.0033 |
| 0.150 |    0.501 |    0.0028 |    0.2737 |      0.2220 |    0.0692 |
| 0.200 |    0.461 |    0.0181 |    0.2263 |      0.2452 |    0.1759 |

Below lr ≈ 0.04 the barrier is indistinguishable from zero; above it, it rises by
two orders of magnitude within a factor of two in step size, then saturates. So
the chaotic transient this paper is about is not a property of "deep network
training" as such — it needs a large enough step.

Three things make this hard to dismiss as an artifact:

1. **It is not underfitting.** lr = 0.01 through 0.03 all reach exactly zero
   training error and still show no barrier. (lr = 0.003 does not converge in 60
   epochs, so that row is excluded from the argument.)
2. **The no-barrier regime holds the best models.** Test accuracy peaks at
   lr = 0.03 (0.637) — inside the flat region — and falls monotonically as the
   step size enters the chaotic regime (0.586 at lr = 0.1, 0.461 at 0.2).
3. **The transient gets longer with step size, not just taller.** At lr = 0.1 the
   barrier is gone by spawn epoch 6 (0.0033); at lr = 0.2 it is still 0.1759
   there. The paper's "2 to 3 epochs" is a statement about a particular step size.

Caveat on (2): this setup has no data augmentation and a small network, which
favours smaller steps. In the paper's setting — ResNet-20, full CIFAR-10, standard
augmentation — large-LR-with-drops is the accuracy-optimal recipe, so "the best
models have no barrier" should not be carried over there without checking.

## Running

```bash
python run.py --preset fast   # ~1 min,  sanity check / live demo
python run.py --preset full   # ~15 min, the real figures
python figures.py --preset full
```

## Credit

Written from the paper's equations; no existing implementation of this paper was
downloaded or run. Backbone dependencies: PyTorch, and `torch.func` for
per-example Jacobians. The parent/child spawning methodology is originally from
Frankle, Dziugaite, Roy & Carbin, *Linear Mode Connectivity and the Lottery Ticket
Hypothesis* (ICML 2020) — reference [7] in Fort et al., and itself a paper we read
on Nov 30.

## What is *not* implemented, and why

**§7 — the low-learning-rate nonlinear advantage.** This is arguably the paper's
most surprising result: even at very low learning rates, where NTK theory ought to
hold, full nonlinear training still beats training linearised at time *t̃* — but
only during the chaotic phase. Reproducing it needs a second training loop that
optimises along the tangent plane,

```
f_lin(x; w) = f_{w0}(x) + J_{w0}(x)(w − w0)
```

which is ~15 lines with `torch.func.jvp`, but a separate linearised run for every
t̃ on top of the runs already here. Left out to keep the demo to four claims that
can each be defended, rather than five that can't.

**Their scale.** ResNet-20 / WideResNet on full CIFAR-10 and CIFAR-100 for 200
epochs with LR drops. Ours is a 48k-parameter CNN with a constant learning rate.
If the qualitative shape survives that gap, that is evidence the phenomenon is
about the optimisation process rather than about their specific architecture —
which is the Practical Impact team's question, and worth handing to them.
