"""Learning-rate sweep of the child error barrier, fanned out on Modal.

One learning rate = one @app.function call = one fresh container, all in
parallel. The sweep is embarrassingly parallel and each arm is small, so the
wall clock is one arm, not ten.

Launch (from the repo root, per MODAL_PLAYBOOK golden pattern):

    MODAL=/Users/raymond/Downloads/SubPY/.modalenv/bin/modal
    cd Implementations/03-fort-dl-vs-kernel && nohup \
      $MODAL run --detach modal/sweep.py > /tmp/modal_lrsweep01.log 2>&1 &
"""
import modal

LRS = [0.003, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.15, 0.2]
CIFAR_DIR = "/opt/cifar"          # baked into the image, so no GPU time spent downloading
RESULTS = "/data/results"

app = modal.App("ds598-fort2020-lrsweep")
vol = modal.Volume.from_name("ds598-fort2020", create_if_missing=True)


def _bake_cifar():
    import torchvision
    torchvision.datasets.CIFAR10(CIFAR_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(CIFAR_DIR, train=False, download=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "numpy")
    .run_function(_bake_cifar)                       # runs on a CPU builder, ~free
    .add_local_python_source("core", "measure", "run")
)


@app.function(image=image, gpu="L4", volumes={"/data": vol}, timeout=60 * 45)
def sweep_one(lr: float):
    """One learning rate. Runtime knobs arrive as arguments, never module globals."""
    import json, time
    from pathlib import Path
    import run

    t0 = time.time()
    tag = f"lr{lr:g}".replace(".", "")
    run.main(dict(
        device="auto", momentum=0.9, batch_size=128, kr_reg=1e-2, seed=0,
        schedule="cosine", preset=tag, n_train=10_000, n_test=10_000,
        width=24, epochs=60, lr=lr, ckpt_every=0.4, n_children=3,
        spawn_epochs=[0.0, 1.6, 6.0], ntk_probe=64, kr_train=256, kr_test=256,
        kr_every=100, data_dir=CIFAR_DIR, out_dir=RESULTS,
    ))
    vol.commit()                                     # commit per arm, not at exit

    r = json.loads((Path(RESULTS) / f"results_{tag}.json").read_text())
    return dict(lr=lr, test_acc=r["final_test_acc"],
                train_err=r["parent"][-1]["train_err"],
                barriers=[b["mean_test_barrier"] for b in r["barriers"]],
                seconds=time.time() - t0)


@app.local_entrypoint()
def main():
    """Submit and leave.

    Deliberately does NOT wait for results. `--detach` protects work that has
    already been dispatched, not work the client has yet to submit -- an earlier
    attempt lost the whole run when the laptop's network dropped between the
    image build finishing and `.map()` dispatching, leaving a stopped app with
    zero tasks. `spawn` hands all ten calls to Modal, then the client exits
    gracefully and the arms run to completion server-side regardless of what
    happens to the laptop.

    Nothing is lost by not collecting here: each arm commits its own JSON to the
    Volume, and `modal/fetch.py` rebuilds the table and figure from those.
    """
    handles = [sweep_one.spawn(lr) for lr in LRS]
    print(f"spawned {len(handles)} arms: {LRS}")
    for lr, h in zip(LRS, handles):
        print(f"  lr={lr:<6g} call={h.object_id}")
    print("\nclient exiting; arms continue server-side.")
    print("collect with:  python modal/fetch.py")
