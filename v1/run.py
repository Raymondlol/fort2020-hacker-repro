"""Reproduce the core claim of Fort et al. 2020 (arXiv:2010.15110) at small scale.

The claim (their Fig. 7): during the first ~2-3 epochs of training there is a
"chaotic transient". Two things happen together and then both stop:

  (1) the *error barrier* between children spawned at time t collapses to zero
      -- i.e. the final basin fate of the network is sealed;
  (2) the *kernel velocity* -- how fast the empirical NTK rotates -- falls from
      very high to a low constant.

And (their Fig. 8) the NTK is not a fixed random kernel: within a few epochs it
has learned features and vastly outperforms the NTK at initialisation.

Usage:
    python run.py --preset fast   # ~1 min,  sanity check / live demo
    python run.py --preset full   # ~15 min, the real figures
"""
import argparse, itertools, json, sys, time
from pathlib import Path

import torch

import core
import measure

HERE = Path(__file__).parent

# Validated 2026-09-08: both configs reach ~0 train error (which the barrier
# measurement requires -- see README "Why the learning-rate schedule matters"),
# and config `small` shows a 118x barrier ratio between spawn epoch 0 and 6.
SPAWN = [0.0, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2, 4.8, 8.0, 16.0]

PRESETS = {
    # ~1 min. Sanity check / live demo only -- too short to show the effect.
    "fast": dict(n_train=10_000, n_test=5_000, width=16, epochs=8, lr=0.1,
                 ckpt_every=0.4, n_children=2, spawn_epochs=[0.0, 0.4, 1.2, 4.0],
                 ntk_probe=64, kr_train=256, kr_test=256, kr_every=4),
    # ~20 min. CIFAR-10 subset the net can fully fit. 4 children -> 6 pairs.
    "small": dict(n_train=10_000, n_test=10_000, width=24, epochs=60, lr=0.1,
                  ckpt_every=0.4, n_children=4, spawn_epochs=SPAWN + [32.0],
                  ntk_probe=64, kr_train=256, kr_test=256, kr_every=10),
    # ~45 min. Full CIFAR-10, ~72% test accuracy, still converged.
    "full": dict(n_train=50_000, n_test=10_000, width=32, epochs=40, lr=0.1,
                 ckpt_every=0.4, n_children=3, spawn_epochs=SPAWN + [28.0],
                 ntk_probe=64, kr_train=192, kr_test=192, kr_every=10),
}


def main(cfg):
    # Line-buffer stdout: these runs take tens of minutes and are watched from a
    # log file, where Python's default block buffering hides all progress until
    # the process exits.
    sys.stdout.reconfigure(line_buffering=True)
    dev = core.get_device(cfg["device"])
    t0 = time.time()

    data_dir = Path(cfg.get("data_dir") or HERE / "data")
    out_dir = Path(cfg.get("out_dir") or HERE / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    Xtr, Ytr, Xte, Yte = core.load_cifar10(data_dir, cfg["n_train"], cfg["n_test"], dev)
    steps_per_epoch = len(Xtr) // cfg["batch_size"]
    ckpt_steps = max(1, round(cfg["ckpt_every"] * steps_per_epoch))
    total_steps = round(cfg["epochs"] * steps_per_epoch)

    model = core.make_model(dev, width=cfg["width"], seed=cfg["seed"])
    print(f"device={dev}  params={core.n_params(model)}  "
          f"steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    # Fixed probe set S for the NTK and ReLU-pattern measurements. Fixed so that
    # every kernel we compare is evaluated on exactly the same inputs.
    probe = Xtr[: cfg["ntk_probe"]]

    # ---------------------------------------------------------------- parent
    # One parent run. Snapshot (weights, momentum) every ckpt_every epochs; the
    # spawn points are a subset of these snapshots, and consecutive snapshots
    # are exactly dt = ckpt_every apart, which is what kernel velocity needs.
    lr_fn = core.make_lr_fn(cfg["schedule"], total_steps, cfg["lr"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"])
    ckpts, traj = {}, []

    def snapshot(step):
        mom = torch.cat([opt.state[p]["momentum_buffer"].flatten()
                         if p in opt.state and opt.state[p].get("momentum_buffer") is not None
                         else torch.zeros(p.numel(), device=dev)
                         for p in model.parameters()])
        ckpts[step] = (core.get_flat(model).cpu(), mom.cpu())

    snapshot(0)
    for start in range(0, total_steps, ckpt_steps):
        n = min(ckpt_steps, total_steps - start)
        core.train_steps(model, opt, Xtr, Ytr, n, cfg["batch_size"], gen, lr_fn=lr_fn, step0=start)
        snapshot(start + n)
        tr, te = core.evaluate(model, Xtr[:10_000], Ytr[:10_000]), core.evaluate(model, Xte, Yte)
        traj.append(dict(epoch=(start + n) / steps_per_epoch, train_err=tr[1], test_err=te[1]))
    print(f"[parent] done  final test err={traj[-1]['test_err']:.3f}  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------- NTK velocity + ReLU
    # Kernel distance is scale-invariant, so this measures how much the tangent
    # kernel has *rotated* -- Fig. 1D in the paper.
    # The measurement used to be pinned to the CPU, which made it ~85% of the
    # wall clock on a GPU box: the Gram matrices are large GEMMs and the CPU is
    # roughly 300x slower at them. Running it on the training device is the
    # single biggest speedup available here.
    meas_model = core.SmallCNN(w=cfg["width"]).to(dev)
    # Jacobian chunk: [chunk, K, d] floats resident. 16 is a laptop default; on
    # a GPU it makes the measurement launch-bound, so raise it there.
    JC = cfg.get("jac_chunk", 16)

    def _load(step):
        core.set_flat(meas_model, ckpts[step][0].to(dev))
        return meas_model

    def kernel_at(step):
        return measure.ntk_gram(measure.ntk_jacobian(_load(step), probe, chunk=JC))

    def relu_at(step):
        return measure.relu_pattern(_load(step), probe)

    steps_sorted = sorted(ckpts)
    K_prev, K_init, B_init = None, kernel_at(0), relu_at(0)
    kern = []
    for s in steps_sorted:
        K = kernel_at(s)
        row = dict(epoch=s / steps_per_epoch,
                   dist_from_init=measure.kernel_distance(K_init, K),
                   relu_dist_from_init=measure.relu_distance(B_init, relu_at(s)))
        if K_prev is not None:
            row["velocity"] = measure.kernel_distance(K_prev, K) / cfg["ckpt_every"]
        kern.append(row); K_prev = K
    print(f"[kernel] {len(kern)} checkpoints measured  ({time.time()-t0:.0f}s)")

    # ------------------------------------------------ NTK kernel regression
    # "How good is the kernel this network has learned so far?"  (paper Sec. 6)
    kr_steps = [steps_sorted[i] for i in range(0, len(steps_sorted), cfg["kr_every"])]
    krtr_X, krtr_Y = Xtr[: cfg["kr_train"]], Ytr[: cfg["kr_train"]]
    krte_X, krte_Y = Xte[: cfg["kr_test"]], Yte[: cfg["kr_test"]]
    kreg = []
    for s in kr_steps:
        _load(s)
        acc = measure.ntk_kernel_regression(meas_model, krtr_X, krtr_Y, krte_X, krte_Y,
                                            reg=cfg["kr_reg"], chunk=JC,
                                            te_block=cfg.get("kr_te_block", 256))
        # The kernel and the network must be scored on the SAME images, or the
        # difference between them is confounded with how hard that particular
        # test subset happens to be. `net_acc_full` is kept only as a reference
        # point; every kernel-vs-network claim uses net_acc_same.
        net_same = 1 - core.evaluate(meas_model, krte_X, krte_Y)[1]
        net_full = 1 - core.evaluate(meas_model, Xte, Yte)[1]
        kreg.append(dict(epoch=s / steps_per_epoch, acc=acc,
                         net_acc_same=net_same, net_acc_full=net_full,
                         n_eval=len(krte_X)))
        print(f"  [ntk-kr] epoch {s/steps_per_epoch:5.1f}  kernel={acc:.4f}  "
              f"net(same {len(krte_X)} imgs)={net_same:.4f}  gap={acc-net_same:+.4f}  "
              f"net(all)={net_full:.4f}")

    # ------------------------------------------------------------- children
    # Children share the parent's weights AND momentum at the spawn point, and
    # differ *only* in the seed driving minibatch order. That isolation is the
    # whole point of the parent/child design (paper Sec. 2, Fig. 1B).
    barriers = []
    spawns = [] if cfg.get("no_children") else cfg["spawn_epochs"]
    if cfg.get("no_children"):
        print("[children] skipped (--no-children): nothing here needs the error barrier")
    for se in spawns:
        s_step = min(steps_sorted, key=lambda s: abs(s / steps_per_epoch - se))
        w0, m0 = ckpts[s_step]
        finals = []
        for c in range(cfg["n_children"]):
            core.set_flat(model, w0.to(dev))
            opt_c = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
            i = 0
            for p in model.parameters():          # restore momentum state
                opt_c.state[p] = dict(momentum_buffer=m0[i:i + p.numel()].view_as(p).to(dev).clone())
                i += p.numel()
            g_c = torch.Generator(device=dev); g_c.manual_seed(10_000 + 97 * c + s_step)
            core.train_steps(model, opt_c, Xtr, Ytr, total_steps - s_step, cfg["batch_size"], g_c,
                             lr_fn=lr_fn, step0=s_step)
            finals.append(core.get_flat(model))

        pairs = []
        for a, b in itertools.combinations(range(cfg["n_children"]), 2):
            bt, _, curve_t = measure.error_barrier(model, finals[a], finals[b], Xte, Yte)
            btr, alphas, curve_tr = measure.error_barrier(model, finals[a], finals[b],
                                                          Xtr[:10_000], Ytr[:10_000])
            pairs.append(dict(test_barrier=bt, train_barrier=btr,
                              test_curve=curve_t, train_curve=curve_tr, alphas=alphas))
        mean_t = sum(p["test_barrier"] for p in pairs) / len(pairs)
        barriers.append(dict(spawn_epoch=s_step / steps_per_epoch, pairs=pairs,
                             mean_test_barrier=mean_t,
                             mean_train_barrier=sum(p["train_barrier"] for p in pairs) / len(pairs)))
        print(f"  [barrier] spawn epoch {s_step/steps_per_epoch:5.2f}  "
              f"mean test barrier={mean_t:.4f}  ({time.time()-t0:.0f}s)")

    out = dict(config=cfg, parent=traj, kernel=kern, kernel_regression=kreg,
               barriers=barriers, final_test_acc=1 - traj[-1]["test_err"],
               runtime_s=time.time() - t0)
    tag = cfg["preset"] + ("" if cfg["schedule"] == "step" else "_" + cfg["schedule"])

    # Save the parent trajectory. Every measurement change so far has forced a
    # full retrain because the weights only ever lived in memory; 101 x 0.75 MB
    # is a cheap way to never pay that again.
    if cfg.get("save_ckpts", True):
        cpath = out_dir / f"checkpoints_{tag}.pt"
        torch.save(dict(config=cfg, steps_per_epoch=steps_per_epoch,
                        ckpts={k: (w, m) for k, (w, m) in ckpts.items()}), cpath)
        print(f"wrote {cpath}  ({cpath.stat().st_size/1e6:.0f} MB, {len(ckpts)} checkpoints)")
    path = out_dir / f"results_{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--kr-reg", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", default="step", choices=["step", "cosine", "const"])
    ap.add_argument("--no-children", action="store_true",
                    help="parent trajectory only -- skips the error-barrier runs (29x cheaper)")
    # These shadow keys in PRESETS. Default None so that an unset flag leaves the
    # preset alone; the old code did cfg.update(PRESETS[...]) *after* vars(a),
    # so the preset silently won and `--lr 0.03` did nothing.
    for flag, typ in (("--lr", float), ("--width", int), ("--epochs", int),
                      ("--jac-chunk", int),
                      ("--kr-train", int), ("--kr-test", int), ("--kr-every", int),
                      ("--n-train", int), ("--n-test", int)):
        ap.add_argument(flag, type=typ, default=None)
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset])
    cfg.update({k: v for k, v in vars(a).items() if v is not None})
    main(cfg)
