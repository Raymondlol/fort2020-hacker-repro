"""Four-way comparison: {10k, 50k} x {cosine, step}.

Two axes of variation on purpose. Scale answers "does the phenomenon survive a
5x change in data and 2x in parameters"; schedule answers "is the phenomenon a
property of the network, or of the learning-rate schedule I happened to pick".
The second question is the one that turned out to matter.
"""
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core

HERE = Path(__file__).parent
# An explicit dash period rather than "--". Matplotlib scales its default dash
# pattern by linewidth, so at lw>2 one period grows longer than the visible half
# of a legend handle -- the entry then renders as a solid stub and the dashed and
# solid series become indistinguishable in the legend. Fixing the period and
# lengthening the handle (see HANDLE below) keeps the key readable.
# Matplotlib multiplies a dash pattern by the linewidth, so at lw~2 a (6, 2.5)
# period becomes (12.6, 5.25) pt -- only one gap fits inside a legend handle, and
# the marker sits exactly on it. A finer period puts three gaps in the handle, so
# at least two survive the marker.
DASH = (0, (3, 1.5))
SOLID = "-"
HANDLE = 3.6          # legend handle length, in font-size units

RUNS = [  # tag, label, colour, linestyle
    ("small_cosine", "10k / cosine", "#e07a1f", DASH),
    ("small",        "10k / step",   "#e07a1f", SOLID),
    ("full_cosine",  "50k / cosine", "#0353a4", DASH),
    ("full",         "50k / step",   "#0353a4", SOLID),
]


def load(tag):
    p = HERE / "results" / f"results_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def schedule_of(r):
    """Runs made before the --schedule flag existed recorded a `cosine` bool."""
    c = r["config"]
    return c.get("schedule") or ("cosine" if c.get("cosine") else "const")


def table(runs_data):
    w = 16
    rows = [
        ("params", lambda r: f"{core.n_params(core.SmallCNN(w=r['config']['width'])):,}"),
        ("train imgs", lambda r: f"{r['config']['n_train']:,}"),
        ("epochs", lambda r: str(r["config"]["epochs"])),
        ("children", lambda r: str(r["config"]["n_children"])),
        ("final test acc", lambda r: f"{r['final_test_acc']:.3f}"),
        ("final train err", lambda r: f"{r['parent'][-1]['train_err']:.4f}"),
        ("runtime (min)", lambda r: f"{r['runtime_s']/60:.1f}"),
        ("", lambda r: ""),
        ("barrier @ spawn 0", lambda r: f"{r['barriers'][0]['mean_test_barrier']:.4f}"),
        ("barrier @ spawn 1.6", lambda r: f"{bat(r, 1.6):.4f}"),
        ("barrier @ spawn 3.2", lambda r: f"{bat(r, 3.2):.4f}"),
        ("barrier @ spawn 8", lambda r: f"{bat(r, 8.0):.4f}"),
        ("collapse epoch (<10%)", lambda r: f"{collapse(r):.1f}"),
        ("", lambda r: ""),
        ("velocity @ 0.4ep", lambda r: f"{vat(r, 0.4):.3f}"),
        ("velocity @ 4ep", lambda r: f"{vat(r, 4.0):.3f}"),
        ("velocity min (0-10ep)", lambda r: f"{vmin(r):.3f}"),
        ("velocity @ 10ep", lambda r: f"{vat(r, 10.0):.3f}"),
        ("v/LR @ 0.4ep", lambda r: f"{vat(r, 0.4)/lr_at(r, 0.4):.1f}"),
        ("v/LR @ 4ep", lambda r: f"{vnat(r, 4.0):.1f}"),
        ("v/LR @ 10ep", lambda r: f"{vnat(r, 10.0):.1f}"),
        ("v/LR decay 0.4->4ep", lambda r: f"{vnat(r, 4.0)/(vat(r, 0.4)/lr_at(r, 0.4)):.2f}x"),
        ("", lambda r: ""),
        ("NTK acc @ init", lambda r: f"{r['kernel_regression'][0]['acc']:.3f}"),
        ("NTK acc best", lambda r: f"{max(k['acc'] for k in r['kernel_regression']):.3f}"),
        ("NTK best / full net", lambda r: f"{max(k['acc'] for k in r['kernel_regression'])/r['final_test_acc']:.3f}"),
    ]
    hdr = f"{'':<24}" + "".join(f"{m[1]:>{w}}" for m, _ in runs_data)
    print(hdr); print("-" * len(hdr))
    for name, fn in rows:
        if not name:
            print(); continue
        print(f"{name:<24}" + "".join(f"{fn(d):>{w}}" for _, d in runs_data))


def bat(r, e):
    return min(r["barriers"], key=lambda b: abs(b["spawn_epoch"] - e))["mean_test_barrier"]


def collapse(r):
    """First spawn epoch at which the barrier has fallen below 10% of its value at 0."""
    b0 = r["barriers"][0]["mean_test_barrier"]
    for b in r["barriers"]:
        if b["mean_test_barrier"] < 0.1 * b0:
            return b["spawn_epoch"]
    return float("nan")


def lr_at(r, e):
    """The learning rate this run was using at epoch e."""
    c = r["config"]
    frac = min(e / c["epochs"], 1.0)
    sched = schedule_of(r)
    if sched == "cosine":
        return c["lr"] * 0.5 * (1 + math.cos(math.pi * frac))
    if sched == "step":
        return c["lr"] * 0.1 ** sum(frac >= d for d in (0.6, 0.85))
    return c["lr"]


def vels_norm(r):
    """Kernel velocity divided by the learning rate.

    Raw velocity is close to proportional to the LR -- weights move faster, so
    the kernel rotates faster -- which means comparing raw velocities across two
    schedules mostly compares the schedules. Dividing it out leaves the part of
    the kernel's motion that is NOT explained by step size, which is what the
    paper's claim is actually about.
    """
    # Cosine reaches lr = 0 exactly at the end, where the ratio is meaningless
    # (and undefined). Drop points below 1% of the initial LR.
    floor = 0.01 * r["config"]["lr"]
    return [(e, v / lr_at(r, e)) for e, v in vels(r) if lr_at(r, e) > floor]


def vels(r):
    return [(x["epoch"], x["velocity"]) for x in r["kernel"] if "velocity" in x]


def vat(r, e):
    return min(vels(r), key=lambda p: abs(p[0] - e))[1]


def vmin(r):
    return min(v for e, v in vels(r) if e <= 10)


def vnat(r, e):
    return min(vels_norm(r), key=lambda p: abs(p[0] - e))[1]


def vlate(r):
    """Velocity in the last third *before* any LR drop, over velocity at 0.4 ep."""
    cut = r["config"]["epochs"] * (0.55 if schedule_of(r) == "step" else 0.9)
    seg = [v for e, v in vels(r) if 0.5 * cut <= e <= cut]
    return st.mean(seg) / vat(r, 0.4) if seg else float("nan")


def smooth(xs, ys, k=5):
    """Centred rolling mean. The velocity estimate uses a 64-image probe set and
    is genuinely noisy step to step; the trend is the signal, so plot both."""
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - k // 2), min(len(ys), i + k // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, out


def figure(runs_data):
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))
    for (tag, lb, c, ls), r in runs_data:
        bs = [b["spawn_epoch"] for b in r["barriers"]]
        bt = [b["mean_test_barrier"] for b in r["barriers"]]
        ax[0].plot(bs, bt, linestyle=ls, color=c, marker="o", ms=4, lw=1.9, label=lb)
        e, v = zip(*vels(r))
        ax[1].plot(e, v, linestyle=ls, color=c, lw=.8, alpha=.25)
        ax[1].plot(*smooth(e, v), linestyle=ls, color=c, lw=2.1, label=lb)
        en, vn = zip(*vels_norm(r))
        ax[3].plot(en, vn, linestyle=ls, color=c, lw=.8, alpha=.25)
        ax[3].plot(*smooth(en, vn), linestyle=ls, color=c, lw=2.1, label=lb)
        kr = r["kernel_regression"]
        ax[2].plot([k["epoch"] for k in kr], [k["acc"] for k in kr], linestyle=ls,
                   color=c, marker="o", ms=3.5, lw=1.9, label=lb)
        ax[2].axhline(r["final_test_acc"], color=c, ls=":", lw=1, alpha=.6)

    ax[0].set(xlabel="Spawn epoch", ylabel="Test error barrier", xlim=(-.3, 9),
              title="A. Basin fate is sealed early\n(all four runs)")
    ax[1].set(xlabel="Train epoch", ylabel="Kernel velocity", xlim=(-.3, 20), ylim=(0, 2),
              title="B. NTK velocity, as measured\n(confounded: velocity tracks the LR)")
    ax[2].set(xlabel="Epoch at which NTK is taken", ylabel="Test accuracy",
              title="C. The learned NTK catches the full net\n(dotted = that run's full-net accuracy)")
    ax[3].set(xlabel="Train epoch", ylabel="Kernel velocity / learning rate",
              xlim=(-.3, 20), ylim=(0, 25),
              title="D. Same quantity, LR divided out\n(all four runs now agree)")
    for a in ax:
        a.grid(alpha=.25, lw=.6); a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=8, handlelength=HANDLE)
    fig.suptitle("Fort et al. 2020 reproduction -- 2 scales x 2 LR schedules", y=1.03, fontsize=12)
    fig.tight_layout()
    p = HERE / "results" / "fig_compare.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); print("\nwrote", p)


if __name__ == "__main__":
    runs_data = [(m, load(m[0])) for m in RUNS]
    missing = [m[0] for m, d in runs_data if d is None]
    if missing:
        print("MISSING:", missing)
    runs_data = [(m, d) for m, d in runs_data if d is not None]
    table(runs_data)
    figure(runs_data)
