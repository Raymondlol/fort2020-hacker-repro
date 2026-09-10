"""Analyse + plot merged fig8 results across seeds; evaluate the pre-registered criteria (PLAN §4).

    python fig8_plot.py results/fig8_full_seed{0,1,2,3,4}_merged.json
"""
import json, math, statistics as st, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def mean_se(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs: return float("nan"), float("nan")
    return st.mean(xs), (st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def collect(runs):
    """Per-onset aggregates across seeds. Keyed by onset epoch (rounded)."""
    onsets = sorted({round(o["onset_epoch"], 3) for r in runs for o in r["onsets"]})
    T = runs[0]["config"]["epochs"]
    rows = []
    for e in onsets:
        os_ = [o for r in runs for o in r["onsets"] if round(o["onset_epoch"], 3) == e]
        g = lambda f: mean_se([f(o) for o in os_])
        rows.append(dict(
            onset=e, n=len(os_),
            net=g(lambda o: o["net_test_err"]),
            green=g(lambda o: o["green"].get("test_err_at_best_val")),
            green_teststop=g(lambda o: o["green"].get("test_err_at_best_test")),
            green_stop=g(lambda o: o["green"].get("best_val_epoch")),
            green_improving=sum(bool(o["green"].get("still_improving")) for o in os_),
            green_div=sum(bool(o["green"].get("diverged")) for o in os_),
            blue=g(lambda o: o["blue"].get("test_err_at_best_val")),
            blue_stop=g(lambda o: o["blue"].get("best_val_epoch")),
            blue_improving=sum(bool(o["blue"].get("still_improving")) for o in os_),
            probe=g(lambda o: o["probe"]["test_err"]),
            adv=g(lambda o: (o["green"].get("test_err_at_best_val") or float("nan")) - o["blue"]["test_err_at_best_val"]),
        ))
    purple = mean_se([r["final"]["test_err"] for r in runs])
    return rows, purple, T


def kernel_agg(runs):
    keys = sorted({r_["step"] for r in runs for r_ in r["kernel"]})
    out = []
    for s in keys:
        rs = [k for r in runs for k in r["kernel"] if k["step"] == s]
        out.append(dict(step=s, epoch=rs[0]["epoch"],
                        d0=mean_se([k["dist_from_init"] for k in rs]),
                        dT=mean_se([k["dist_to_final"] for k in rs]),
                        v=mean_se([k["velocity"] for k in rs if "velocity" in k])))
    return out


def criteria(rows, purple, kern, T):
    by = {r["onset"]: r for r in rows}
    near = lambda e: by[min(by, key=lambda k: abs(k - e))]
    r0, r3 = near(0), near(3.2)
    out = []
    # A literal / A' scale-free
    out.append(("A  err_lin(3.2) <= 1/3 err_lin(0)", f"{r3['green'][0]:.3f} vs {r0['green'][0]/3:.3f}", r3["green"][0] <= r0["green"][0] / 3))
    gap0 = r0["green"][0] - purple[0]
    frac = (r0["green"][0] - r3["green"][0]) / gap0 if gap0 > 0 else float("nan")
    out.append(("A' gap closed by t~=3.2 >= 2/3", f"{frac:.2f}", frac >= 2 / 3))
    # B
    thr = purple[0] + 2 * (purple[1] if not math.isnan(purple[1]) else 0)
    hits = [r["onset"] for r in rows if r["green"][0] <= thr]
    tstar = min(hits) if hits else None
    out.append(("B  first t~ with green <= purple+2SE, t~/T<=0.5",
                f"t*={tstar} (t*/T={tstar/T:.2f})" if tstar is not None else "never", tstar is not None and tstar / T <= 0.5))
    # B'
    early = [r for r in rows if r["onset"] <= 1.0]
    late = [r for r in rows if r["onset"] >= 5.0 and r["onset"] < T]
    se = lambda r: r["adv"][1] if not math.isnan(r["adv"][1]) else 0
    early_ok = any(r["adv"][0] > 2 * se(r) for r in early)
    late_ok = bool(late) and all(r["adv"][0] <= 2 * se(r) for r in late)
    mx = lambda rs: f"{max(r['adv'][0] for r in rs):+.3f}" if rs else "n/a"
    out.append(("B' nonlinear advantage early only", f"early max {mx(early)}; late max {mx(late)}", early_ok and late_ok))
    # K
    vs = [(k["epoch"], k["v"][0]) for k in kern if not math.isnan(k["v"][0])]
    v_first = vs[0][1]
    mid = [v for e, v in vs if 4 <= e <= 20]
    vmid = st.mean(mid) if mid else float("nan")
    out.append(("K  v(first 0.4ep) >= 2x mean v(4-20ep) > 0", f"{v_first:.3f} vs {vmid:.3f} ({v_first/vmid:.2f}x)", v_first >= 2 * vmid and vmid > 0.01))
    # P
    midr = [r for r in rows if 1.0 <= r["onset"] <= T / 2]
    pse = lambda r: r["green"][1] if not math.isnan(r["green"][1]) else 0
    out.append(("P  green < probe - 2SE (mid t~)", "; ".join(f"{r['onset']}: {r['green'][0]:.3f} vs {r['probe'][0]:.3f}" for r in midr),
                bool(midr) and all(r["green"][0] < r["probe"][0] - 2 * pse(r) for r in midr)))
    return out


def main(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    rows, purple, T = collect(runs)
    kern = kernel_agg(runs)
    lin_lr = runs[0]["onsets"][0]["lin_lr"]; plr = runs[0]["config"]["lr"]
    print(f"{len(runs)} seeds, T={T}, parent lr={plr}, lin_lr={lin_lr}, purple N(T)={purple[0]:.4f}±{purple[1]:.4f}\n")
    print(f"{'t~':>6} {'net(t~)':>8} {'green':>14} {'stop':>5} {'blue':>14} {'stop':>5} {'probe':>8} {'adv':>8}  {'green(test-stop)':>16}")
    for r in rows:
        f = lambda m: f"{m[0]:.3f}±{m[1]:.3f}" if not math.isnan(m[1]) else f"{m[0]:.3f}"
        flag = ("+" if r["green_improving"] else "") + ("D" * r["green_div"])
        print(f"{r['onset']:>6} {r['net'][0]:>8.3f} {f(r['green']):>14} {r['green_stop'][0]:>4.0f}{flag:<1} {f(r['blue']):>14} "
              f"{r['blue_stop'][0]:>4.0f}{'+' if r['blue_improving'] else ' '} {r['probe'][0]:>8.3f} {r['adv'][0]:>+8.3f}  {f(r['green_teststop']):>16}")
    print("\nkernel geometry (mean over seeds):")
    for k in kern:
        if k["step"] in (0, 1, 4, 16, 39, 78, 156, 312, 624, 1248, 2496, 4992, 9984) or k["epoch"] == T:
            print(f"  {k['epoch']:>7.3f} ep  d(init)={k['d0'][0]:.3f}  d(final)={k['dT'][0]:.3f}  v={k['v'][0]:.3f}")
    print("\npre-registered criteria:")
    for name, detail, ok in criteria(rows, purple, kern, T):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<45} {detail}")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    x = [max(r["onset"], 0.007) for r in rows]
    def band(a, key, c, lb, ls="-"):
        m = [r[key][0] for r in rows]; s = [0 if math.isnan(r[key][1]) else r[key][1] for r in rows]
        a.plot(x, m, ls, color=c, marker="o", ms=4, lw=2, label=lb)
        a.fill_between(x, [u - v for u, v in zip(m, s)], [u + v for u, v in zip(m, s)], color=c, alpha=.15, lw=0)
    band(ax[0], "net", "#c1121f", "NN at t~ (red)")
    band(ax[0], "green", "#2a9d3f", "linearised at t~, val-stopped (green)")
    band(ax[0], "green_teststop", "#2a9d3f", "linearised, test-stopped (paper protocol)", ls="--")
    band(ax[0], "blue", "#0353a4", f"network continued from t~ at lr={lin_lr:g} (blue)")
    band(ax[0], "probe", "#7a7a7a", "linear probe on features at t~")
    ax[0].axhline(purple[0], color="#7b2cbf", ls=":", lw=1.8, label=f"NN at T={T} (purple)")
    ax[0].set(xscale="log", xlabel="onset t~ (epochs, log)", ylabel="test error (8k)", title="Fig. 8 replica")
    band(ax[1], "adv", "#c1121f", "green − blue (linearised minus network)")
    ax[1].axhline(0, color="k", lw=.8)
    ax1b = ax[1].twinx()
    ke = [k["epoch"] for k in kern if not math.isnan(k["v"][0])]; kv = [k["v"][0] for k in kern if not math.isnan(k["v"][0])]
    ax1b.plot(ke, kv, color="#e07a1f", lw=1.6, alpha=.8, label="kernel velocity (dt=0.4 ep)")
    ax1b.set_ylabel("kernel velocity", color="#e07a1f")
    ax[1].set(xscale="log", xlabel="onset t~ / epoch (log)", ylabel="error difference", title="Nonlinear advantage vs kernel velocity")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=8)
    ke2 = [max(k["epoch"], 0.002) for k in kern]
    ax[2].plot(ke2, [k["d0"][0] for k in kern], color="#0353a4", lw=2, label="S(w0, wt): distance from init")
    ax[2].plot(ke2, [k["dT"][0] for k in kern], color="#c1121f", lw=2, label="S(wt, wT): distance to final")
    ax[2].set(xscale="log", xlabel="epoch (log)", ylabel="kernel distance", title="Kernel geometry")
    for a in (ax[0], ax[2]): a.legend(fontsize=8)
    for a in ax: a.grid(alpha=.25, lw=.6); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Fort et al. 2020 §6–7 at small scale — SmallCNN(32) CIFAR-10, constant lr={plr}, {len(runs)} seeds", y=1.02)
    fig.tight_layout()
    out = HERE / "results" / "fig8_replica.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
