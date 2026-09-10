"""Re-run the parent trajectory with a matched, larger evaluation set.

Why this run exists
-------------------
In the original runs the kernel was scored on `Xte[:192]` while the network was
scored on all 10,000 test images. Two problems followed:

  1. Different yardsticks. The gap between the two curves is confounded with
     how hard that particular 192-image subset happens to be.
  2. Worse, the *same* 192 images were used at every checkpoint, so one unlucky
     subset draw biases every checkpoint the same way. That invalidates any
     "the sign is consistent across checkpoints" argument -- the checkpoints are
     not independent draws.

This run scores BOTH models on the same images (`net_acc_same` in the output)
and raises that set from 192 to 2000, which drops the binomial standard error
from +/-3.3pp to +/-1.0pp.

Three things make it cheap:
  * `no_children=True` -- the error-barrier runs are 29x the training cost and
    nothing here needs them.
  * the measurement now runs on the GPU instead of being pinned to the CPU,
    where it was ~85% of the wall clock.
  * `Kte` is accumulated in blocks, so the evaluation set can grow without the
    test-side Jacobian ever being resident in full.

Checkpoints are written to the Volume, so the next measurement change does not
mean another retrain -- which is what forced this one.

Launch (per MODAL_PLAYBOOK golden pattern):

    MODAL=/Users/raymond/Downloads/SubPY/.modalenv/bin/modal
    cd Implementations/03-fort-dl-vs-kernel && nohup \
      $MODAL run --detach modal/rerun.py > /tmp/modal_rerun01.log 2>&1 &
"""
import modal

CIFAR_DIR = "/opt/cifar"
RESULTS = "/data/results"
TAG = "full2k"          # distinct from "full" so the original run is preserved

app = modal.App("ds598-fort2020-rerun")
vol = modal.Volume.from_name("ds598-fort2020", create_if_missing=True)


def _bake_cifar():
    import torchvision
    torchvision.datasets.CIFAR10(CIFAR_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(CIFAR_DIR, train=False, download=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "numpy")
    .run_function(_bake_cifar)
    .add_local_python_source("core", "measure", "run")
)


@app.function(image=image, gpu="L4", volumes={"/data": vol}, timeout=60 * 60)
def rerun():
    import json, time
    from pathlib import Path
    import run

    t0 = time.time()
    run.main(dict(
        device="auto", momentum=0.9, batch_size=128, seed=0, schedule="cosine",
        preset=TAG, n_train=50_000, n_test=10_000, width=32, epochs=40, lr=0.1,
        ckpt_every=0.4,
        # parent only: the barrier runs are 29x the training cost and are not
        # what this run is fixing.
        no_children=True, n_children=0, spawn_epochs=[],
        ntk_probe=64,
        # the fix: same images for both models, and 10x more of them.
        kr_train=192, kr_test=2000, kr_every=5, kr_reg=1e-2, kr_te_block=250,
        # bigger Jacobian chunk: on a GPU the measurement is launch-bound, not
        # FLOP-bound, so 16 leaves most of the card idle.
        jac_chunk=64,
        data_dir=CIFAR_DIR, out_dir=RESULTS,
    ))
    vol.commit()

    r = json.loads((Path(RESULTS) / f"results_{TAG}_cosine.json").read_text())
    ck = Path(RESULTS) / f"checkpoints_{TAG}_cosine.pt"
    return dict(results=r, seconds=time.time() - t0,
                ckpt_mb=ck.stat().st_size / 1e6 if ck.exists() else None)


@app.local_entrypoint()
def main():
    out = rerun.remote()
    r, kr = out["results"], out["results"]["kernel_regression"]
    print(f"\nwall clock {out['seconds']/60:.1f} min   checkpoints {out['ckpt_mb']:.0f} MB")
    print(f"final test acc (all 10k): {r['final_test_acc']:.4f}\n")
    n = kr[0]["n_eval"]
    print(f"{'epoch':>7}{'kernel':>10}{f'net(same {n})':>16}{'gap':>10}{'net(all 10k)':>14}")
    for k in kr:
        print(f"{k['epoch']:>7.1f}{k['acc']:>10.4f}{k['net_acc_same']:>16.4f}"
              f"{k['acc']-k['net_acc_same']:>+10.4f}{k['net_acc_full']:>14.4f}")
    gaps = [k["acc"] - k["net_acc_same"] for k in kr]
    print(f"\nmatched-set gap: mean {sum(gaps)/len(gaps):+.4f}, "
          f"{sum(g < 0 for g in gaps)}/{len(gaps)} negative")
    se = (0.7 * 0.3 / n) ** 0.5
    print(f"binomial SE at n={n}: +/-{se*100:.2f}pp "
          f"(was +/-3.3pp at n=192)")
