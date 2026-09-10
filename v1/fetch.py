"""Pull the sweep results off the Modal Volume and rebuild table + figure.

The summary table printed by sweep.py's local_entrypoint runs on the *client*,
so it is lost if the laptop sleeps mid-run. Nothing of value goes with it: the
table is only a rendering of the per-arm JSON, and those live on the Volume,
committed one arm at a time. This script is the durable path.

    python modal/fetch.py            # fetch from the volume, then rebuild
    python modal/fetch.py --local    # rebuild from whatever is already local
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent.parent
RES = HERE / "results"
MODAL = "/Users/raymond/Downloads/SubPY/.modalenv/bin/modal"
VOLUME = "ds598-fort2020"

if "--local" not in sys.argv:
    RES.mkdir(exist_ok=True)
    # The destination MUST already exist as a directory: given a directory source
    # and a missing destination, `modal volume get` treats the destination as a
    # file and concatenates the arms into it.
    dest = RES / "_modal"
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([MODAL, "volume", "get", "--force", VOLUME, "results", str(dest)],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    for f in sorted((RES / "_modal").rglob("results_lr*.json")):
        (RES / f.name).write_bytes(f.read_bytes())
        print("pulled", f.name)

rows = []
for f in sorted(RES.glob("results_lr*.json")):
    r = json.loads(f.read_text())
    rows.append((r["config"]["lr"], r["final_test_acc"], r["parent"][-1]["train_err"],
                 [b["mean_test_barrier"] for b in r["barriers"]],
                 [b["spawn_epoch"] for b in r["barriers"]]))
rows.sort()
if not rows:
    sys.exit("no results_lr*.json found")

print("\n" + "=" * 74)
print(f"{'lr':>7}{'test acc':>11}{'train err':>11}{'bar@0':>11}{'bar@1.6':>11}{'bar@6':>11}")
for lr, acc, tre, bars, _ in rows:
    b = bars + [float('nan')] * 3
    print(f"{lr:>7.3f}{acc:>11.3f}{tre:>11.4f}{b[0]:>11.4f}{b[1]:>11.4f}{b[2]:>11.4f}")
print("=" * 74)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.8, 4.7))
lrs = [r[0] for r in rows]
ax.plot(lrs, [r[3][0] for r in rows], "o-", color="#c1121f", lw=2.3, ms=7,
        label="barrier, children split at epoch 0")
if all(len(r[3]) > 2 for r in rows):
    ax.plot(lrs, [r[3][2] for r in rows], "s--", color="#8d99ae", lw=1.6, ms=5,
            label="barrier, children split at epoch 6")
ax2 = ax.twinx()
ax2.plot(lrs, [r[1] for r in rows], "^:", color="#0353a4", lw=1.7, ms=6,
         label="final test accuracy")
ax2.set_ylabel("Final test accuracy", color="#0353a4")
ax2.tick_params(axis="y", colors="#0353a4")
ax.set_xscale("log")
ax.axvline(0.1, color="#c1121f", lw=1, ls=":", alpha=.55)
ax.annotate("the main runs\nin this project", xy=(0.1, 0.20), xytext=(0.105, 0.155),
            fontsize=8.5, color="#c1121f")
ax.axvspan(min(lrs) * 0.9, 0.038, color="#8d99ae", alpha=.10, lw=0)
ax.annotate("no barrier at all,\nand the best models", xy=(0.0037, 0.045), fontsize=9,
            color="#4a5560")
ax.set(xlabel="Learning rate (cosine-decayed from this value)", ylabel="Test error barrier",
       title="The chaotic transient only exists above a step-size threshold\n"
             "CIFAR-10 10k, 106,666-param CNN, everything else held fixed")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center left", framealpha=.95,
          handlelength=3.4)
ax.grid(alpha=.25, lw=.6); ax.spines[["top"]].set_visible(False)
fig.tight_layout()
out = RES / "fig_lr_sweep.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); print("wrote", out)
