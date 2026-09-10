"""Why CIFAR-10 and not MNIST: check it, don't cite it.

Frankle et al. (2020) report that MNIST-scale problems are already *stable at
initialisation* -- two children split at step 0 still land in the same linearly
connected basin -- while CIFAR-10 ones are not. If that holds here, an MNIST
version of this demo would produce a flat line and no phenomenon to show.

Controlled: same architecture, same optimiser, same schedule, same 10k images,
same learning rate. Only the pixels change.

LR note: the 0.1 used for the main CIFAR-10 runs diverges on MNIST (loss pins at
ln 10 = 2.303, i.e. the network collapses to a uniform predictor), so both arms
here run at 0.03, which trains both datasets. Matching the LR across arms matters
more than matching the main runs -- otherwise a difference in barrier could be
attributed to the step size rather than to the dataset.
"""
import json, sys
from pathlib import Path
import torch, torchvision
import core, run

HERE = Path(__file__).parent
LR = 0.03


def load_mnist(root, n_train, n_test, device, seed=0):
    """Pad 28->32 and repeat to 3 channels so SmallCNN is byte-identical."""
    out = []
    for train, n in ((True, n_train), (False, n_test)):
        ds = torchvision.datasets.MNIST(root, train=train, download=True)
        X = ds.data.float().div_(255.0).unsqueeze(1)
        X = torch.nn.functional.pad(X, (2, 2, 2, 2)).repeat(1, 3, 1, 1)
        X = (X - 0.1307) / 0.3081
        Y = ds.targets.clone()
        if n and n < len(Y):
            X, Y = X[:n], Y[:n]
        out += [X.to(device), Y.to(device)]
    return out


CIFAR_LOADER = core.load_cifar10
ARMS = {"mnist": load_mnist, "cifar_lr003": CIFAR_LOADER}

for tag, loader in ARMS.items():
    core.load_cifar10 = loader
    run.main(dict(device="auto", momentum=0.9, batch_size=128, kr_reg=1e-2, seed=0,
                  schedule="cosine", preset=tag, n_train=10_000, n_test=10_000,
                  width=24, epochs=60, lr=LR, ckpt_every=0.4, n_children=2,
                  spawn_epochs=[0.0, 6.0], ntk_probe=64, kr_train=256, kr_test=256,
                  kr_every=100))

R = {t: json.loads((HERE / f"results/results_{t}_cosine.json").read_text()) for t in ARMS}
print("\n" + "=" * 66)
print(f"{'':<24}{'MNIST':>14}{'CIFAR-10':>14}    same net/opt/lr={LR}")
print(f"{'final test acc':<24}{R['mnist']['final_test_acc']:>14.3f}{R['cifar_lr003']['final_test_acc']:>14.3f}")
print(f"{'final train err':<24}{R['mnist']['parent'][-1]['train_err']:>14.4f}"
      f"{R['cifar_lr003']['parent'][-1]['train_err']:>14.4f}")
for i in range(2):
    e = R['mnist']['barriers'][i]['spawn_epoch']
    print(f"{'barrier @ spawn ' + format(e, '.1f'):<24}"
          f"{R['mnist']['barriers'][i]['mean_test_barrier']:>14.4f}"
          f"{R['cifar_lr003']['barriers'][i]['mean_test_barrier']:>14.4f}")
print("=" * 66)
