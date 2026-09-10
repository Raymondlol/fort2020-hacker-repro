"""Figures. Panel layout mirrors Fort et al. Fig. 7 (+ their Fig. 8)."""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core

HERE = Path(__file__).parent
C = dict(barrier="#c1121f", kernel="#0353a4", extra="#2a9d8f", grey="#8d99ae")


def make(res, tag):
    kern = res["kernel"]
    ke = [k["epoch"] for k in kern if "velocity" in k]
    kv = [k["velocity"] for k in kern if "velocity" in k]
    bs = [b["spawn_epoch"] for b in res["barriers"]]
    bt = [b["mean_test_barrier"] for b in res["barriers"]]
    blo = [min(p["test_barrier"] for p in b["pairs"]) for b in res["barriers"]]
    bhi = [max(p["test_barrier"] for p in b["pairs"]) for b in res["barriers"]]
    zoom = max(6.0, min(bs[-1], 8.0))

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.1))

    # (A) error barrier vs spawn epoch  -- "when is basin fate sealed?"
    ax[0].fill_between(bs, blo, bhi, color=C["barrier"], alpha=.18, lw=0)
    ax[0].plot(bs, bt, "o-", color=C["barrier"], lw=2, ms=5)
    ax[0].axhline(0, color=C["grey"], lw=.8, ls=":")
    ax[0].set(xlabel="Spawn epoch", ylabel="Test error barrier",
              title="A. Child error barrier\n(basin fate)", xlim=(-.3, zoom))

    # (B) kernel velocity vs epoch -- "how fast is the NTK rotating?"
    ax[1].plot(ke, kv, "o-", color=C["kernel"], lw=2, ms=4)
    ax[1].set(xlabel="Train epoch", ylabel=f"Kernel velocity  (dt={res['config']['ckpt_every']} ep)",
              title="B. NTK velocity\n(chaotic - stable)", xlim=(-.3, zoom))

    # (C) the correlation -- this is the paper's actual claim
    common = [(b, next(k["velocity"] for k in kern
                       if "velocity" in k and abs(k["epoch"] - s) < 1e-6))
              for s, b in zip(bs, bt)
              if any("velocity" in k and abs(k["epoch"] - s) < 1e-6 for k in kern)]
    if common:
        bb, vv = zip(*common)
        sc = ax[2].scatter(bb, vv, c=[s for s, b in zip(bs, bt)
                                      if any("velocity" in k and abs(k["epoch"] - s) < 1e-6
                                             for k in kern)],
                           cmap="viridis", s=70, zorder=3, edgecolor="w", lw=.8)
        plt.colorbar(sc, ax=ax[2], label="Spawn epoch")
    ax[2].set(xlabel="Test error barrier", ylabel="Kernel velocity",
              title="C. They fall together\n(paper Fig. 7 right)")

    # (D) what the data-dependent NTK has learned
    kr = res["kernel_regression"]
    e = [k["epoch"] for k in kr]; a = [k["acc"] for k in kr]
    ax[3].plot(e, a, "o-", color=C["extra"], lw=2, ms=4, label="NTK kernel regression")
    ax[3].axhline(a[0], color=C["grey"], ls="--", lw=1.2,
                  label=f"NTK at init ({a[0]:.3f})")
    ax[3].axhline(res["final_test_acc"], color="k", ls=":", lw=1.2,
                  label=f"full net, end ({res['final_test_acc']:.3f})")
    ax[3].set(xlabel="Epoch at which NTK is taken", ylabel="Test accuracy",
              title="D. The NTK learns features\n(paper Fig. 8)")
    ax[3].legend(fontsize=8, loc="lower right")

    for a_ in ax:
        a_.grid(alpha=.25, lw=.6); a_.spines[["top", "right"]].set_visible(False)
    d = core.n_params(core.SmallCNN(w=res["config"]["width"]))
    fig.suptitle("Fort et al. 2020 (arXiv:2010.15110), small-scale reproduction  |  "
                 f"CIFAR-10 ({res['config']['n_train']:,} imgs), {d:,}-param CNN, no BN, "
                 f"{res['config']['epochs']} ep", y=1.04, fontsize=11)
    fig.tight_layout()
    p = HERE / "results" / f"fig_main_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); print("wrote", p)

    # ---- supporting figure: the raw interpolation curves the barrier comes from
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.cm.plasma
    for b in res["barriers"]:
        pr = b["pairs"][0]
        ax2[0].plot(pr["alphas"], pr["test_curve"], "-o", ms=3,
                    color=cmap(min(b["spawn_epoch"] / zoom, 1.0)),
                    label=f"spawn {b['spawn_epoch']:.1f}")
    ax2[0].set(xlabel=r"interpolation $\alpha$", ylabel="Test error",
               title="Linear path between two children")
    ax2[0].legend(fontsize=7, ncol=2)

    pe = [p["epoch"] for p in res["parent"]]
    ax2[1].plot(pe, [p["train_err"] for p in res["parent"]], label="train error")
    ax2[1].plot(pe, [p["test_err"] for p in res["parent"]], label="test error")
    ax2[1].plot([k["epoch"] for k in kern], [k["relu_dist_from_init"] for k in kern],
                color=C["extra"], ls="--", label="ReLU pattern dist. from init")
    ax2[1].plot([k["epoch"] for k in kern], [k["dist_from_init"] for k in kern],
                color=C["kernel"], ls="--", label="kernel dist. from init")
    ax2[1].set(xlabel="Train epoch", title="Parent trajectory")
    ax2[1].legend(fontsize=8)
    for a_ in ax2:
        a_.grid(alpha=.25, lw=.6); a_.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    p2 = HERE / "results" / f"fig_support_{tag}.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight"); print("wrote", p2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preset", default="full")
    t = ap.parse_args().preset
    make(json.loads((HERE / "results" / f"results_{t}.json").read_text()), t)
