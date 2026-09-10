"""Fan the Fig. 8 replication out on Modal: parents -> lin_lr grid -> one container per (seed, onset).

Orchestration runs server-side (`orchestrate`), so the laptop can disconnect after submitting.

    MODAL=/Users/raymond/Downloads/SubPY/.modalenv/bin/modal
    cd Implementations/03-fort-dl-vs-kernel && nohup $MODAL run --detach modal/fig8_modal.py \
        > logs/modal_fig8.log 2>&1 &
    #   ... modal/fig8_modal.py --parent-lr 0.03 [--lin-lr 0.003]
Collect:  modal volume get --force ds598-fort2020 fig8 results/_modal/fig8
"""
import modal

CIFAR_DIR = "/opt/cifar"
OUT = "/data/fig8"
SEEDS = [0, 1, 2, 3, 4]
GPU = "L4"

app = modal.App("ds598-fort2020-fig8")
vol = modal.Volume.from_name("ds598-fort2020", create_if_missing=True)


def _bake_cifar():
    import torchvision
    torchvision.datasets.CIFAR10(CIFAR_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(CIFAR_DIR, train=False, download=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "numpy")
    .run_function(_bake_cifar)
    .add_local_python_source("core", "measure", "linearized", "fig8")
)
common = dict(image=image, gpu=GPU, volumes={"/data": vol}, timeout=60 * 60)


def _ckpt(seed):
    return f"{OUT}/fig8_ckpt_full_seed{seed}.pt"


@app.function(**common)
def parent_stage(seed: int, parent_lr: float):
    import fig8
    fig8.main(fig8.build_cfg(preset="full", stage="parent", seed=seed, jac_chunk=64, lr=parent_lr,
                             data_dir=CIFAR_DIR, out_dir=OUT))
    vol.commit()
    return seed


@app.function(**common)
def grid_stage(seed: int, lr: float, parent_lr: float):
    import json, fig8
    vol.reload()
    fig8.main(fig8.build_cfg(preset="full", stage="grid", seed=seed, grid_lrs=[lr], lr=parent_lr,
                             load_ckpts=_ckpt(seed), data_dir=CIFAR_DIR, out_dir=OUT))
    vol.commit()
    r = json.load(open(f"{OUT}/fig8_lrgrid_full_seed{seed}_lr{lr:g}.json"))["lr_grid"][0]
    return dict(lr=lr, diverged=r["diverged"], val=r.get("val_err_at_best_val"),
                test=r.get("test_err_at_best_val"), ep=r.get("best_val_epoch"))


@app.function(**common)
def onset_stage(seed: int, onset_epoch: float, lin_lr: float, parent_lr: float):
    import fig8
    vol.reload()
    fig8.main(fig8.build_cfg(preset="full", stage="onsets", seed=seed, onsets=[onset_epoch], lr=parent_lr,
                             lin_lr=lin_lr, load_ckpts=_ckpt(seed), data_dir=CIFAR_DIR, out_dir=OUT))
    vol.commit()
    return (seed, onset_epoch)


@app.function(image=image, volumes={"/data": vol}, timeout=4 * 60 * 60)
def orchestrate(parent_lr: float, lin_lr: float = None):
    import time, fig8
    t0 = time.time()
    print("parents:", [h.get() for h in [parent_stage.spawn(s, parent_lr) for s in SEEDS]], f"{time.time()-t0:.0f}s")
    if lin_lr is None:
        rows = [h.get() for h in [grid_stage.spawn(0, lr, parent_lr) for lr in fig8.LR_GRID]]
        for r in rows:
            print(f"  grid lr={r['lr']:g} val={r['val']} test={r['test']} ep={r['ep']} diverged={r['diverged']}")
        ok = [r for r in rows if not r["diverged"] and r["val"] is not None]
        lin_lr = min(ok, key=lambda r: r["val"])["lr"]
        print(f"chosen lin_lr={lin_lr:g}  {time.time()-t0:.0f}s")
    done = [h.get() for h in [onset_stage.spawn(s, e, lin_lr, parent_lr) for s in SEEDS for e in fig8.ONSETS]]
    print(f"onsets done: {len(done)}  lin_lr={lin_lr:g}  total {time.time()-t0:.0f}s")
    return dict(lin_lr=lin_lr, parent_lr=parent_lr, n=len(done), seconds=time.time() - t0)


@app.local_entrypoint()
def main(parent_lr: float, lin_lr: float = None):
    call = orchestrate.spawn(parent_lr, lin_lr)
    print(f"orchestrator spawned: {call.object_id}; client may exit. parent_lr={parent_lr} lin_lr={'grid' if lin_lr is None else lin_lr}")
