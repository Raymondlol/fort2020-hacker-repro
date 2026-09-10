"""Replicate Fort et al. 2020 Fig. 8 / Fig. 9 (Sec. 6-7) at small scale, by the paper's method.

For each onset t~ on a log grid:
  green  linearise the parent at w_t~ and train the linear model (linearized.py)
  blue   ordinary nonlinear training from w_t~ at the same (low) learning rate
  probe  logistic regression on the penultimate features at w_t~   (control, not in the paper)
Baselines: red = parent error at t~ (its own training curve), purple = parent error at T.
Plus kernel geometry from the same checkpoints: distance from init, distance to final,
and velocity at the paper's dt = 0.4 epochs.

Parent uses a CONSTANT learning rate (App. D.3), T = 40 epochs. Test set is split once,
by a fixed seed, into 2,000 validation / 8,000 test; early stopping is on validation and
the test-stopped number (the paper's protocol) is reported alongside.

Stages (so the onsets can fan out across machines):
    --stage parent   train parent, save checkpoints (.pt) + kernel geometry  -> fig8_<tag>.json
    --stage grid     choose lin_lr on validation at grid_onset (needs --load-ckpts)
    --stage onsets   green/blue/probe for --onsets (needs --load-ckpts)      -> fig8_<tag>_onset<step>.json
    --stage all      everything in one process (local use)
Merge per-onset files with fig8_merge.py.

    python fig8.py --preset smoke --stage all
"""
import argparse, json, sys, time
from pathlib import Path

import torch

import core
import measure
import linearized as lin

HERE = Path(__file__).parent
ONSETS = [0, 0.01, 0.04, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 40]
LOG_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

PRESETS = {
    "smoke": dict(n_train=10_000, width=24, epochs=8, lr=0.1, onsets=[0, 0.4, 1.6, 8],
                  lin_epochs=8, probe_n=64, grid_onset=1.6),
    "full": dict(n_train=50_000, width=32, epochs=40, lr=0.1, onsets=ONSETS,
                 lin_epochs=40, probe_n=128, grid_onset=3.2),
}


def summarize(curve):
    """Early-stopping bookkeeping for one run's per-epoch curve."""
    ok = [r for r in curve if not r.get("diverged")]
    if not ok:
        return dict(diverged=True)
    bv = min(ok, key=lambda r: r["val_err"])
    bt = min(ok, key=lambda r: r["test_err"])
    return dict(diverged=any(r.get("diverged") for r in curve),
                epochs_run=len(ok),
                best_val_epoch=bv["epoch"], val_err_at_best_val=bv["val_err"],
                test_err_at_best_val=bv["test_err"],
                best_test_epoch=bt["epoch"], test_err_at_best_test=bt["test_err"],
                final_test_err=ok[-1]["test_err"], final_train_err=ok[-1].get("train_err"),
                still_improving=bv["epoch"] == ok[-1]["epoch"])


def grids(cfg, spe):
    T = cfg["epochs"] * spe
    dt_steps = round(cfg["dt"] * spe)
    onset_steps = sorted({min(round(e * spe), T) for e in cfg["onsets"]})
    uniform = set(range(0, T + 1, dt_steps)) | {T}
    ckpt_steps = sorted(set(onset_steps) | uniform | {s for s in LOG_STEPS if s <= T})
    return T, dt_steps, onset_steps, uniform, ckpt_steps


def main(cfg):
    sys.stdout.reconfigure(line_buffering=True)
    dev = core.get_device(cfg["device"])
    t0 = time.time()
    data_dir = Path(cfg.get("data_dir") or HERE / "data")
    out_dir = Path(cfg.get("out_dir") or HERE / "results"); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg['preset']}_seed{cfg['seed']}" + (f"_{cfg['tag']}" if cfg.get("tag") else "")
    stage = cfg["stage"]

    Xtr, Ytr, Xall, Yall = core.load_cifar10(data_dir, cfg["n_train"], None, dev, seed=cfg["seed"])
    g = torch.Generator().manual_seed(cfg["split_seed"])
    perm = torch.randperm(len(Xall), generator=g).to(dev)
    Xval, Yval = Xall[perm[:cfg["n_val"]]], Yall[perm[:cfg["n_val"]]]
    Xte, Yte = Xall[perm[cfg["n_val"]:]], Yall[perm[cfg["n_val"]:]]
    evalsets = {"train": (Xtr[:5000], Ytr[:5000]), "val": (Xval, Yval), "test": (Xte, Yte)}

    B, spe = cfg["batch_size"], cfg["n_train"] // cfg["batch_size"]
    T, dt_steps, onset_steps, uniform, ckpt_steps = grids(cfg, spe)
    ep = lambda s: s / spe
    ckpt_path = out_dir / f"fig8_ckpt_{tag}.pt"

    # ------------------------------------------------------------------ parent
    if cfg.get("load_ckpts"):
        blob = torch.load(cfg["load_ckpts"], map_location="cpu")
        assert blob["spe"] == spe and blob["config"]["width"] == cfg["width"], "checkpoint/config mismatch"
        ckpts, parent = blob["ckpts"], blob["parent"]
        print(f"loaded {len(ckpts)} checkpoints from {cfg['load_ckpts']}")
    else:
        model = core.make_model(dev, width=cfg["width"], seed=cfg["seed"])
        print(f"device={dev} params={core.n_params(model)} steps/epoch={spe} T={T} steps "
              f"ckpts={len(ckpt_steps)} onsets(steps)={onset_steps}")
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
        gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"])
        ckpts, parent = {}, []

        def snapshot(step):
            mom = torch.cat([opt.state[p]["momentum_buffer"].flatten()
                             if p in opt.state and opt.state[p].get("momentum_buffer") is not None
                             else torch.zeros(p.numel(), device=dev) for p in model.parameters()])
            ckpts[step] = (core.get_flat(model).cpu(), mom.cpu())
            parent.append(dict(step=step, epoch=ep(step),
                               **{k + "_err": core.evaluate(model, X, Y)[1] for k, (X, Y) in evalsets.items()}))

        snapshot(0)
        for a, b in zip(ckpt_steps[:-1], ckpt_steps[1:]):
            core.train_steps(model, opt, Xtr, Ytr, b - a, B, gen, lr_fn=None, step0=a)
            snapshot(b)
        f = parent[-1]
        print(f"[parent] T={cfg['epochs']} ep  train {f['train_err']:.3f}  val {f['val_err']:.3f}  "
              f"test {f['test_err']:.3f}   ({time.time()-t0:.0f}s)")
        if stage in ("parent", "all") and cfg["save_ckpts"]:
            torch.save(dict(config=cfg, spe=spe, ckpts=ckpts, parent=parent), ckpt_path)
            print(f"wrote {ckpt_path} ({ckpt_path.stat().st_size/1e6:.0f} MB)")
    final = parent[-1]

    meas = core.SmallCNN(w=cfg["width"]).to(dev)
    def load(step):
        core.set_flat(meas, ckpts[step][0].to(dev)); return meas

    # ------------------------------------------------------- kernel geometry
    if stage in ("parent", "all"):
        res = dict(config=cfg, spe=spe, parent=parent, final=final, kernel=[], onsets=[])
        path = out_dir / f"fig8_{tag}.json"
        probe = Xtr[:cfg["probe_n"]]
        gram = lambda s: measure.ntk_gram(measure.ntk_jacobian(load(s), probe, chunk=cfg["jac_chunk"]))
        K0, KT = gram(0), gram(T)
        K_prev_u, s_prev_u = None, None
        for s in ckpt_steps:
            K = gram(s)
            row = dict(step=s, epoch=ep(s), dist_from_init=measure.kernel_distance(K0, K),
                       dist_to_final=measure.kernel_distance(K, KT))
            if s in uniform:
                if K_prev_u is not None and s - s_prev_u == dt_steps:
                    row["velocity"] = measure.kernel_distance(K_prev_u, K) / cfg["dt"]
                K_prev_u, s_prev_u = K, s
            res["kernel"].append(row)
        del K0, KT, K_prev_u
        vr = [r for r in res["kernel"] if "velocity" in r]
        mid = [r["velocity"] for r in vr if 4 <= r["epoch"] <= 20]
        print(f"[kernel] {len(res['kernel'])} ckpts; velocity first={vr[0]['velocity']:.3f} "
              f"mean(4-20ep)={sum(mid)/max(1,len(mid)):.3f}  ({time.time()-t0:.0f}s)")
        path.write_text(json.dumps(res, indent=1))
        print(f"wrote {path}")
        if stage == "parent":
            return

    # ------------------------------------------------------------ lr grid
    if stage == "grid":
        s = min(onset_steps, key=lambda q: abs(ep(q) - cfg["grid_onset"]))
        p = lin.params_of(load(s))
        lrs = cfg["grid_lrs"] or LR_GRID
        rows = []
        path = out_dir / (f"fig8_lrgrid_{tag}" + ("" if len(lrs) > 1 else f"_lr{lrs[0]:g}") + ".json")
        for lr in lrs:
            gg = torch.Generator(device=dev); gg.manual_seed(cfg["seed"] * 1000 + 7)
            curve, _ = lin.train_linearized(p, Xtr, Ytr, evalsets, cfg["lin_epochs"], lr, B,
                                            cfg["momentum"], gg)
            sm = summarize(curve)
            rows.append(dict(lr=lr, onset_epoch=ep(s), curve=curve, **sm))
            print(f"  [grid] lr={lr:g}  best val {sm.get('val_err_at_best_val', float('nan')):.4f} "
                  f"@ep{sm.get('best_val_epoch')}  test {sm.get('test_err_at_best_val', float('nan')):.4f}  "
                  f"diverged={sm['diverged']}  ({time.time()-t0:.0f}s)")
            path.write_text(json.dumps(dict(config=cfg, lr_grid=rows), indent=1))
        print(f"wrote {path}")
        return

    # -------------------------------------------------------- per onset runs
    lr = cfg["lin_lr"]
    child = core.SmallCNN(w=cfg["width"]).to(dev)
    entries = []
    for s in onset_steps:
        w0, m0 = ckpts[s]
        p = lin.params_of(load(s))
        prow = next(r for r in parent if r["step"] == s)
        entry = dict(onset_step=s, onset_epoch=ep(s), lin_lr=lr,
                     net_val_err=prow["val_err"], net_test_err=prow["test_err"])

        gg = torch.Generator(device=dev); gg.manual_seed(cfg["seed"] * 1000 + s + 1)
        curve, _ = lin.train_linearized(p, Xtr, Ytr, evalsets, cfg["lin_epochs"], lr, B, cfg["momentum"], gg)
        entry["green"] = dict(curve=curve, **summarize(curve))

        gb = torch.Generator(device=dev); gb.manual_seed(cfg["seed"] * 1000 + s + 2)
        curve = lin.train_nonlinear(child, w0.to(dev), m0.to(dev), Xtr, Ytr, evalsets,
                                    cfg["lin_epochs"], lr, B, cfg["momentum"], gb)
        entry["blue"] = dict(curve=curve, **summarize(curve))

        entry["probe"] = lin.linear_probe(p, Xtr, Ytr, {"val": evalsets["val"], "test": evalsets["test"]})
        entries.append(entry)

        gr, bl = entry["green"], entry["blue"]
        print(f"  [onset {ep(s):6.2f} ep] net {prow['test_err']:.3f} | green {gr.get('test_err_at_best_val', float('nan')):.3f}"
              f" (ep{gr.get('best_val_epoch')}{'+' if gr.get('still_improving') else ''}"
              f"{' DIVERGED' if gr['diverged'] else ''}) | blue {bl['test_err_at_best_val']:.3f}"
              f" (ep{bl['best_val_epoch']}{'+' if bl['still_improving'] else ''}) | probe {entry['probe']['test_err']:.3f}"
              f" | purple {final['test_err']:.3f}   ({time.time()-t0:.0f}s)")
        if stage == "onsets":
            opath = out_dir / f"fig8_{tag}_onset{s}.json"
            opath.write_text(json.dumps(dict(config=cfg, final=final, onsets=[entry]), indent=1))
        else:
            res["onsets"] = entries; res["runtime_s"] = time.time() - t0
            path.write_text(json.dumps(res, indent=1))
    print(f"done ({(time.time()-t0)/60:.1f} min)")


def build_cfg(**kw):
    cfg = dict(PRESETS[kw["preset"]], batch_size=128, momentum=0.9, dt=0.4, n_val=2000, split_seed=1234,
               seed=0, device="auto", lin_lr=1e-3, stage="all", tag=None, save_ckpts=True, jac_chunk=16,
               load_ckpts=None, grid_lrs=None, data_dir=None, out_dir=None)
    cfg.update({k: v for k, v in kw.items() if v is not None})
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--stage", default="all", choices=["all", "parent", "grid", "onsets"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lin-lr", type=float, default=None)
    ap.add_argument("--grid-lrs", type=str, default=None, help="comma list; default LR_GRID")
    ap.add_argument("--onsets", type=str, default=None, help="comma list of onset epochs (overrides preset)")
    ap.add_argument("--load-ckpts", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--no-save-ckpts", dest="save_ckpts", action="store_false")
    ap.add_argument("--jac-chunk", type=int, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    a = ap.parse_args()
    kw = vars(a)
    if a.onsets:
        kw["onsets"] = [float(x) for x in a.onsets.split(",")]
    if a.grid_lrs:
        kw["grid_lrs"] = [float(x) for x in a.grid_lrs.split(",")]
    main(build_cfg(**kw))
