"""Deck figures for the v2 (§6/§7) reproduction, from the 5-seed merged JSONs.

    python fig_deck_v2.py results/_modal/fig8/fig8_full_seed*_merged.json
Writes results/fig_deck_v2_fig8.png and results/fig_deck_v2_kernel.png (7.45 x 4.3 in at 200 dpi).
"""
import json, math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fig8_plot as fp

HERE = Path(__file__).parent
runs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
rows, purple, T = fp.collect(runs); kern = fp.kernel_agg(runs)
x = [max(r["onset"], 0.007) for r in rows]
plr = runs[0]["config"]["lr"]; llr = runs[0]["onsets"][0]["lin_lr"]
RED, GREEN, BLUE, GREY, PURPLE, ORANGE = "#c1121f", "#2a9d3f", "#0353a4", "#6e6e6e", "#7b2cbf", "#e07a1f"
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5})

def band(a, key, c, lb, ls="-", mk="o"):
    m = [r[key][0] for r in rows]; s = [0 if math.isnan(r[key][1]) else r[key][1] for r in rows]
    a.plot(x, m, ls, color=c, marker=mk, ms=3.5, lw=1.8, label=lb)
    a.fill_between(x, [u - v for u, v in zip(m, s)], [u + v for u, v in zip(m, s)], color=c, alpha=.18, lw=0)

# ------------------------------------------------------------- figure 1: Fig 8 replica
fig, ax = plt.subplots(figsize=(7.45, 4.3))
band(ax, "net", RED, "parent network at t̃  (no extra training)")
band(ax, "green", GREEN, f"linearised at t̃, then trained at lr {llr:g} for 40 ep")
band(ax, "blue", BLUE, f"network, continued from t̃ at lr {llr:g} for 40 ep  (paper §7 control)")
band(ax, "probe", GREY, "linear probe on penultimate features at t̃", mk="s")
ax.axhline(purple[0], color=PURPLE, ls=":", lw=1.8, label=f"network at T = {T} ep, constant lr {plr:g}")
ax.set(xscale="log", xlabel="onset t̃ of linearised training (epochs, log scale)", ylabel="test error (8,000 held-out images)",
       ylim=(0.18, 0.95))
ax.legend(fontsize=8.2, loc="upper right", frameon=False)
ax.grid(alpha=.25, lw=.6); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "results/fig_deck_v2_fig8.png", dpi=200); print("wrote fig_deck_v2_fig8.png")

# ------------------------------------------------------------- figure 2: advantage + kernel geometry
fig, ax = plt.subplots(1, 2, figsize=(7.45, 4.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]
band(a, "adv", RED, "linearised − network (green − blue)")
gp = [r["green"][0] - r["probe"][0] for r in rows]
a.plot(x, gp, "-", color=GREY, marker="s", ms=3.5, lw=1.6, label="linearised − linear probe")
a.axhline(0, color="k", lw=.8)
a.set(xscale="log", xlabel="onset t̃ (epochs, log)", ylabel="test-error difference", title="A. What the tangent plane buys, vs t̃")
a.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
a.set_ylim(-0.12, 0.40)
b = ax[1]
ke = [k["epoch"] for k in kern if not math.isnan(k["v"][0])]; kv = [k["v"][0] for k in kern if not math.isnan(k["v"][0])]
b.plot(ke, kv, color=ORANGE, lw=1.6, label="kernel velocity  S(wₜ, wₜ₊₀.₄)/0.4")
b2 = b.twinx()
ke2 = [max(k["epoch"], 0.002) for k in kern]
b2.plot(ke2, [k["d0"][0] for k in kern], color=BLUE, lw=1.6, label="distance from initial kernel")
b2.plot(ke2, [k["dT"][0] for k in kern], color=RED, lw=1.6, label="distance to final kernel")
b.set(xscale="log", xlabel="training epoch (log)", ylabel="velocity", title="B. Kernel geometry, 128-image probe")
b2.set_ylabel("kernel distance S")
b.set_ylim(0, 1.4); b2.set_ylim(0, 0.8)
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
for a_ in (ax[0], b): a_.grid(alpha=.25, lw=.6); a_.spines[["top"]].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "results/fig_deck_v2_kernel.png", dpi=200); print("wrote fig_deck_v2_kernel.png")
