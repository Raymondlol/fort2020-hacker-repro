"""Why is our kernel velocity so noisy?

Two candidate causes, and they call for different fixes:
  (a) the probe set is only 64 images, so each Gram matrix is a noisy estimate;
  (b) dt = 0.4 epochs is short, so consecutive checkpoints differ mostly by
      minibatch noise rather than by systematic drift of the kernel.

Same checkpoints, velocity recomputed at several probe sizes and several dt.
If (a) dominates, noise falls with probe size. If (b) dominates, noise falls
with dt and probe size barely matters.
"""
import statistics as st
from pathlib import Path
import torch
import core, measure

HERE = Path(__file__).parent
EPOCHS, CKPT = 12, 0.2
PROBES, DTS = [64, 256], [0.2, 0.4, 0.8, 1.6]

dev = core.get_device()
Xtr, Ytr, Xte, Yte = core.load_cifar10(HERE / "data", 10_000, 2_000, dev)
spe = len(Xtr) // 128
ckpt_steps, total = round(CKPT * spe), round(EPOCHS * spe)

model = core.make_model(dev, width=24, seed=0)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
gen = torch.Generator(device=dev); gen.manual_seed(0)
lr_fn = core.make_lr_fn("cosine", total, 0.1)

flats = []
for start in range(0, total, ckpt_steps):
    if start: core.train_steps(model, opt, Xtr, Ytr, ckpt_steps, 128, gen, lr_fn=lr_fn, step0=start-ckpt_steps)
    flats.append(core.get_flat(model).cpu())
core.train_steps(model, opt, Xtr, Ytr, ckpt_steps, 128, gen, lr_fn=lr_fn, step0=total-ckpt_steps)
print(f"{len(flats)} checkpoints, {CKPT} epochs apart, final test err "
      f"{core.evaluate(model, Xte, Yte)[1]:.3f}")

cpu_model = core.SmallCNN(w=24)
grams = {}
for p in PROBES:
    probe = Xtr[:p].cpu()
    grams[p] = []
    for f in flats:
        core.set_flat(cpu_model, f)
        grams[p].append(measure.ntk_gram(measure.ntk_jacobian(cpu_model, probe, chunk=16)))
    print(f"  probe {p}: {len(grams[p])} Grams computed")

print(f"\n{'probe':>7}{'dt (ep)':>9}{'peak':>8}{'@4ep':>8}{'decay':>8}{'step noise sd':>15}{'SNR':>7}")
for p in PROBES:
    for dt in DTS:
        k = max(1, round(dt / CKPT))
        vs = [(i * CKPT, measure.kernel_distance(grams[p][i], grams[p][i + k]) / dt)
              for i in range(0, len(grams[p]) - k)]
        norm = [(e, v / core.cosine_lr(round(e * spe), total, 0.1)) for e, v in vs
                if core.cosine_lr(round(e * spe), total, 0.1) > 1e-3]
        peak = max(v for e, v in norm if e <= 1.0)
        at4 = min(norm, key=lambda q: abs(q[0] - 4.0))[1]
        seg = [v for e, v in norm if 4 <= e <= 9]
        sd = st.stdev([seg[i+1] - seg[i] for i in range(len(seg) - 1)]) if len(seg) > 2 else float("nan")
        print(f"{p:>7}{dt:>9.1f}{peak:>8.1f}{at4:>8.1f}{at4/peak:>8.2f}{sd:>15.2f}{(peak-at4)/sd:>7.1f}")
