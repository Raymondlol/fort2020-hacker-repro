# Plan v2 — small-scale reproduction of Fort et al. §5–§7 (kernel change + "matches full training")

**Status 2026-09-10: executed.** 5 seeds × 13 onsets on Modal (~3 GPU-h). Results and criteria
outcomes are in `README.md` → "Results, v2"; raw JSON in `results/_modal/fig8/`, figure
`results/fig8_replica.png`. Constant parent LR chosen by the rule in §2 → 0.02; lin_lr grid → 0.003.

**Scope, unchanged:** two claims only. (1) how fast the tangent kernel changes early in
training; (2) whether the kernel learned by time *t̃* matches full nonlinear training.
Error barrier / basins / spawning stay out of scope.

**What changed from v1** (v1 archived in the session scratchpad):

| | v1 | v2 | why |
|---|---|---|---|
| parent LR schedule | cosine | **constant 0.1** | App. D.3: the Fig. 8/9 parent is trained "at a constant learning rate of 0.1". Cosine also forces kernel velocity → 0 for a trivial reason (Fig. 4 is the paper's own constant-LR control). |
| baselines | purple only (`N(T)`) | purple **and blue** (low-LR nonlinear from *t̃*) | Blue is the paper's own fair comparison (§7, Fig. 9). Without it, "matches full training" is confounded with the fact that a low-LR linearised run is effectively annealed while the constant-LR parent is not. Same machinery, +30 % compute. |
| claim A criterion | ≥3× error reduction | ≥3× **and** a scale-free version | 3× is a ResNet-20 number (0.6 → 0.2 with `N(T)` ≈ 0.1). A 189k-param CNN with `N(T)` ≈ 0.3 cannot show 3× regardless of kernel learning. |
| linearisation | `torch.func.jvp` | reverse-mode double-VJP | forward AD has no `max_pool2d` kernel (checked, PyTorch 2.14). |
| seeds | 5 | **5** (Modal fan-out, one container per seed × onset) | the paper is n = 1. |
| claim C | tested | dropped to a footnote | green below red early is trivially true (the linear model trained 40 more epochs). |
| kernel geometry | distance from init | distance from init **and to final**, + velocity at the paper's dt | "distance to final kernel" is the direct measure of "how much kernel change is still to come at *t̃*". |

---

## 1. What the paper actually does (Fig. 8/9, §6–§7, App. D.3–D.4)

- Parent: ResNet-20, CIFAR-10/100, **constant LR 0.1**, SGD + momentum, T = 200 epochs.
- Green (`Linearized`): at onset *t̃*, first-order Taylor expand *f_w* around *w_t̃*; train the linear
  model with SGD + momentum 0.9, **LR 0.001** (grid-searched once over 1e-4 … 1e-1), **200 epochs**,
  report the error at the **optimal early-stopping point on the test set**.
- Blue (`Low LR nonlin`): from the same *w_t̃*, ordinary nonlinear training at LR 0.001 (the smallest
  LR that converges within 1000 epochs), independent SGD noise, until convergence; average of 2 runs.
- Purple: the constant-LR parent's error at T.
- Red: the parent's error at *t̃*.

Claims, restated so each has a number attached:

| | claim | paper's number |
|---|---|---|
| **A** | data-dependent NTK at *t̃* = 3–4 epochs beats the random NTK (*t̃* = 0) | error ≥ 3× lower |
| **B** | green reaches purple | at *t̃* = 30–90 epochs = 15–45 % of T |
| **B′** | green vs blue: nonlinear advantage exists only early | gone after a few epochs (Fig. 9), tracks the error barrier |
| **K** | kernel velocity is high early, then constant nonzero | drops within ~5 epochs (Fig. 7 middle), plateau ≠ 0 (Fig. 4D) |

Nowhere: a kernel Gram matrix for training, a squared loss, a closed-form solve, a train subset.
The "kernel ridge on 192 examples" route in `measure.ntk_kernel_regression` is therefore not the
paper's method and is retired for this experiment (it stays in the file; nothing new calls it).

---

## 2. Design

**Model / data:** `core.SmallCNN(w=32)`, 188,810 params, no BN, no augmentation. Full CIFAR-10.
Batch 128, momentum 0.9, **constant LR**, **T = 40 epochs** (390 steps/epoch, 15,600 steps).

**Constant-LR value (decided before the main run, rule fixed here):** the paper's 0.1 is a
ResNet-20 + BatchNorm number. Without BN this CNN at constant 0.1 sits at 47 % train error after
40 epochs (measured, seed 0), i.e. the parent never becomes a "fully trained network" and claim B
would be trivially true. Rule: probe constant LR ∈ {0.01, 0.02, 0.03, 0.05} for 40 epochs, seed 0,
and take the **largest LR whose train error at T is ≤ 0.05** (a converged parent, as large a step
as that allows — the paper's regime). If none reaches 0.05, take the LR with the lowest test error
at T. The probe logs are `logs/constlr_*.log`.
Test set split once, by a fixed seed: **2,000 validation / 8,000 test**.

**Parent checkpoints** (weights + momentum), union of two grids:
- log grid in steps for the transient: `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 390, 780, 1560, 3120, 6240, 12480, 15600`
- uniform grid every 0.4 epoch (156 steps) for the paper's velocity definition.

**Onset grid for *t̃*** (13 points), log-spaced, in epochs:
`0, 0.01, 0.04, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 40`

**Per onset *t̃*, three runs, same budget (40 epochs), same batch size:**
1. **green** — linearised model `f_lin(x) = f_t̃(x) + J_t̃(x)·dw`, cross-entropy on the logits,
   SGD + momentum 0.9, LR `lr_lin` (§3), `dw` starts at 0. Validation error every epoch; keep the best.
2. **blue** — nonlinear network from `w_t̃` (momentum buffer restored), LR = `lr_lin`, fresh minibatch
   seed. Same early stopping. Deviation from the paper: matched 40-epoch budget rather than
   "until convergence in 1000 epochs"; recorded per run whether the validation curve was still
   improving at the end.
3. **linear probe** — logistic regression on the 128-d penultimate activations at `w_t̃`
   (full-batch L-BFGS, seconds). Null model: if green ≈ probe, "kernel learning" is just
   representation learning and the tangent-space structure adds nothing.

**Reported per onset:** error on the 8,000 test images at the validation-optimal epoch
(primary), **and** at the test-optimal epoch (the paper's protocol, for direct comparison),
the stopping epoch, total epochs consumed (*t̃* + stop).

**Kernel geometry from the parent checkpoints**, probe set of 128 fixed training images
(Gram 1280 × 1280, mK × mK exactly as Eq. 4):
- `S(w_0, w_t)` distance from init, `S(w_t, w_T)` distance to final — on the log grid
- velocity `S(w_t, w_{t+0.4ep}) / 0.4` — on the uniform grid, dt = 0.4 epochs as in the paper.

**Seeds:** 5 (parent init + minibatch order + child seeds all derived from it). The 2k/8k split
is fixed across seeds.

---

## 3. Fixed once, before the main run

- `lr_lin`: grid `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}` on seed 0, *t̃* = 3.2 epochs, 40 epochs each,
  pick by validation error. **Used for green and blue at every *t̃*, never re-tuned.** The paper
  used a single value (0.001) for the same reason. Record the grid result in the results JSON.
- Sanity checks that must pass before anything is reported:
  - at `dw = 0`, `f_lin` equals the network's logits at `w_t̃` to float precision, and the
    linearised gradient equals the ordinary gradient (checked: 0 and 5e-7 at width 32);
  - `J·dw` vs finite difference at ‖dw‖ = 1e-3 · ‖randn‖: relative error ≪ 1 (checked: 3e-3);
  - the green run at *t̃* = 0 must start at chance and the blue run must start at exactly the
    parent's error at *t̃*.

---

## 4. Pre-registered criteria (written before the main run)

SE = across-seed standard error (n = 5), not binomial.

| | confirmed if | falsified if | notes |
|---|---|---|---|
| **A** (literal) | `err_lin(3.2) ≤ ⅓ · err_lin(0)` | otherwise | expected to fail on scale grounds alone; report anyway |
| **A′** (scale-free) | fraction of the gap `err_lin(0) − N(T)` closed by *t̃* = 3.2 is ≥ ⅔ | < ⅓ | ⅔ is what the paper's 0.6 → 0.2 with N(T) ≈ 0.1 works out to (0.8); between ⅓ and ⅔ = partial |
| **B** | ∃ *t̃*\* with `err_lin(t̃*) ≤ N(T) + 2 SE` and `t̃*/T ≤ 0.5` | never, or only at *t̃*\* ≈ T | report *t̃*\*/T; paper: 0.15–0.45 |
| **B′** | nonlinear advantage `err_lin(t̃) − err_blue(t̃)` is > 2 SE for *t̃* ≲ 1–3 epochs and ≤ 2 SE after | advantage persists to *t̃* ≫ 5 epochs, or never exists | the paper's §7 claim |
| **K** | velocity in the first 0.4 epoch is ≥ 2× the 4–20 epoch mean, and that mean is > 0 with the constant LR | velocity flat from the start, or plateau ≈ 0 | earlier cosine run: 2.2× — but with the LR decaying |
| **P** (probe null) | `err_lin(t̃) < err_probe(t̃) − 2 SE` for mid-range *t̃* | `err_lin ≈ err_probe` throughout | if falsified, the NTK framing adds nothing over "features" |
| **timing** | the *t̃* at which B′ vanishes and the epoch at which velocity plateaus agree to within a factor of 2 | they differ by ≫ 2× | the paper's "one story" claim, Fig. 7 / Fig. 9 |

Compute accounting: every green point also carries `t̃ + stop_epoch`; when that exceeds T the
comparison with purple is flagged. The paper does not do this.

---

## 5. Cost

MPS on this laptop, width 32, 50k: 2.2 s per parent epoch, linearised step ≈ 2–3× a normal step.

| item | per seed |
|---|---|
| parent, 40 epochs + checkpoints | ~2 min |
| kernel geometry, ~115 checkpoints × Jacobian on 128 images | ~5 min |
| green, 13 onsets × 40 epochs | ~40 min |
| blue, 13 onsets × 40 epochs | ~20 min |
| probes | < 1 min |
| **total** | **~70 min** local; ~10 min on an L4 |

Smoke test first: 10k subset, width 24, T = 8, 4 onsets, 1 seed — ~5 min. Then either 3 seeds
sequentially in the background locally (~3.5 h) or one Modal container per seed (~15 min wall,
`modal/` pattern already exists). LR grid for `lr_lin` runs before the main run (~25 min local).

---

## 6. Implementation

New file `linearized.py`:

```python
# J dw by the double-VJP trick (forward AD lacks max_pool2d):
out = functional_call(model, w0, (x,))                     # f_w0(x), graph kept
v   = torch.zeros_like(out, requires_grad=True)
u   = autograd.grad(out, w0, grad_outputs=v, create_graph=True)   # J^T v, linear in v
jdw = autograd.grad(sum((u_i * dw_i).sum()), v)            # d/dv <J^T v, dw> = J dw
f_lin = out.detach() + jdw.detach()
g   = d CE(f_lin, y) / d f_lin
grad_dw = autograd.grad(out, w0, grad_outputs=g)           # J^T g, same graph
```

`dw` are ordinary leaf tensors handed to `torch.optim.SGD(momentum=0.9)` with `.grad` set by
hand, so the optimiser is literally the same object as the nonlinear run's. Evaluation of the
linear model on val/test is the first three lines in batches of 1,000 under no-grad.

New driver `fig8.py`: parent → checkpoints → kernel geometry → per-onset {green, blue, probe} →
`results/fig8_seed{N}.json`. `fig8_plot.py` makes the three panels: (i) Fig. 8 replica with all
four curves + probe, (ii) nonlinear advantage vs *t̃* with kernel velocity overlaid (Fig. 9 without
the barrier axis), (iii) kernel distance from init / to final / velocity vs log epoch.

Reused unchanged: `core.py` (data, model, `train_steps`, flat get/set), `measure.ntk_jacobian`,
`measure.ntk_gram`, `measure.kernel_distance`.

Environment: `/Users/raymond/WorkSpace/DS598/.venv/bin/python`, PyTorch 2.14, MPS (no float64
on device — keep the 1280² Gram cosine in float64 on CPU, as `measure.kernel_distance` does).

---

## 7. What the earlier (v1-era) runs established, kept for reference

`results_full2k_cosine.json`, seed 0, cosine LR, ridge on 192 examples, matched 2,000-image eval:
kernel-ridge accuracy 0.25 at init → 0.68 at epoch 40 vs network 0.72 (−4 pp); velocity in the
first 0.4 epoch 2.2× the 4-epoch-onward level. Neither number is comparable to Fig. 8: different
estimator, different LR schedule. They are superseded by whatever `fig8.py` produces.

The LR sweep (`results_lr*_cosine.json`, error barrier vs LR) is out of scope here and stands alone.

## 8. Out of scope

Error barrier, spawning, basins, kernel-distance heatmaps, ResNet/BN, CIFAR-100, augmentation.
Slides/poster prose (course AI policy: code and analysis are in bounds, presentation text is not).
