"""Does the chaotic transient depend on the step size?

Found by accident while building a MNIST control: at lr=0.03 the CIFAR-10 error
barrier at spawn 0 is 0.0036 -- the phenomenon is simply absent -- while at
lr=0.1 it is 0.3204. Since the lr=0.03 run is also the *better* model (63.1% vs
57.3% test), this is not a case of low LR failing to train.

That is a boundary condition on the paper's central claim, so it deserves a
measurement rather than an anecdote. Everything else is held fixed.
"""
import json
from pathlib import Path
import core, run

HERE = Path(__file__).parent
LRS = [0.003, 0.01, 0.03, 0.06, 0.1, 0.2]

for lr in LRS:
    tag = f"lr{lr:g}".replace(".", "")
    if (HERE / f"results/results_{tag}_cosine.json").exists():
        print(f"skip {tag} (exists)"); continue
    run.main(dict(device="auto", momentum=0.9, batch_size=128, kr_reg=1e-2, seed=0,
                  schedule="cosine", preset=tag, n_train=10_000, n_test=10_000,
                  width=24, epochs=60, lr=lr, ckpt_every=0.4, n_children=3,
                  spawn_epochs=[0.0, 1.6, 6.0], ntk_probe=64, kr_train=256,
                  kr_test=256, kr_every=100))

rows = []
for lr in LRS:
    tag = f"lr{lr:g}".replace(".", "")
    r = json.loads((HERE / f"results/results_{tag}_cosine.json").read_text())
    rows.append((lr, r["final_test_acc"], r["parent"][-1]["train_err"],
                 *[b["mean_test_barrier"] for b in r["barriers"]]))

print("\n" + "=" * 74)
print(f"{'lr':>7}{'test acc':>11}{'train err':>11}{'bar@0':>11}{'bar@1.6':>11}{'bar@6':>11}")
for lr, acc, tre, b0, b16, b6 in rows:
    print(f"{lr:>7.3f}{acc:>11.3f}{tre:>11.4f}{b0:>11.4f}{b16:>11.4f}{b6:>11.4f}")
print("=" * 74)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot([r[0] for r in rows], [r[3] for r in rows], "o-", color="#c1121f", lw=2.2,
        ms=7, label="barrier, children split at epoch 0")
ax.plot([r[0] for r in rows], [r[5] for r in rows], "s--", color="#8d99ae", lw=1.6,
        ms=5, label="barrier, children split at epoch 6")
ax2 = ax.twinx()
ax2.plot([r[0] for r in rows], [r[1] for r in rows], "^:", color="#0353a4", lw=1.6,
         ms=6, label="final test accuracy")
ax2.set_ylabel("Final test accuracy", color="#0353a4")
ax2.tick_params(axis="y", colors="#0353a4")
ax.set_xscale("log")
ax.set(xlabel="Learning rate (cosine-decayed from this value)",
       ylabel="Test error barrier",
       title="The chaotic transient is a large-step phenomenon\n"
             "CIFAR-10 10k, 106,666-param CNN, everything else held fixed")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
ax.grid(alpha=.25, lw=.6); ax.spines[["top"]].set_visible(False)
fig.tight_layout()
out = HERE / "results/fig_lr_sweep.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); print("wrote", out)
