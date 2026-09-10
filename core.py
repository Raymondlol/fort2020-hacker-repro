"""Data, model, and training loop.

Deliberately minimal so the whole pipeline fits on a screen during the demo.
Everything is expressed in *steps*; epochs are a derived unit (steps_per_epoch),
because the paper's interesting timescale (the chaotic transient) is a fraction
of an epoch wide and we need to spawn children at t = 0.4, 0.8, ... epochs.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cifar10(root, n_train, n_test, device, seed=0):
    """Whole subset lives on-device as one tensor: no DataLoader, no augmentation.

    No augmentation is a deliberate simplification -- it removes a second source
    of child-to-child randomness so that children differ *only* in SGD minibatch
    order, which is what the parent/child spawning experiment is meant to isolate.
    """
    out = []
    for train, n in ((True, n_train), (False, n_test)):
        ds = torchvision.datasets.CIFAR10(root, train=train, download=False)
        X = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2)
        X = (X - torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        Y = torch.tensor(ds.targets, dtype=torch.long)
        if n is not None and n < len(Y):
            idx = torch.from_numpy(np.random.default_rng(seed).permutation(len(Y))[:n])
            X, Y = X[idx], Y[idx]
        out += [X.to(device), Y.to(device)]
    return out  # Xtr, Ytr, Xte, Yte


class SmallCNN(nn.Module):
    """~48k params, no BatchNorm.

    No BN is on purpose. Linearly interpolating two networks (the error-barrier
    measurement) is ill-defined with BN because the running statistics of the
    interpolated weights are wrong; the usual fix is to reset and re-estimate
    them at every interpolation point. Fort et al. run exactly this no-BN control
    in their Appendix C.3 and see the same phenomenology, so dropping BN keeps
    the measurement honest and the code readable.

    Small width also keeps the NTK affordable: the Jacobian is [N, K, d], so
    cost is linear in the parameter count d.
    """

    def __init__(self, w=16, n_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, 2 * w, 3, padding=1)
        self.c3 = nn.Conv2d(2 * w, 2 * w, 3, padding=1)
        self.f1 = nn.Linear(2 * w * 4 * 4, 4 * w)
        self.f2 = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)   # 32 -> 16
        x = F.max_pool2d(F.relu(self.c2(x)), 2)   # 16 -> 8
        x = F.max_pool2d(F.relu(self.c3(x)), 2)   # 8  -> 4
        x = F.relu(self.f1(x.flatten(1)))
        return self.f2(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def make_model(device, width=16, seed=0):
    torch.manual_seed(seed)
    return SmallCNN(w=width).to(device)


@torch.no_grad()
def evaluate(model, X, Y, batch=1000):
    """Returns (mean cross-entropy, 0-1 error). The paper's barrier uses 0-1 error."""
    model.eval()
    loss, wrong = 0.0, 0
    for i in range(0, len(X), batch):
        out = model(X[i:i + batch])
        loss += F.cross_entropy(out, Y[i:i + batch], reduction="sum").item()
        wrong += (out.argmax(1) != Y[i:i + batch]).sum().item()
    return loss / len(X), wrong / len(X)


def cosine_lr(step, total_steps, lr0):
    """Cosine decay from lr0 to 0 over the whole run.

    A decaying learning rate is not cosmetic here. The error barrier asks which
    *basin* two children ended up in, which only means something once they have
    stopped moving; at a constant LR they are still bouncing around at the end of
    training and the barrier measures SGD noise rather than basin structure.
    """
    return lr0 * 0.5 * (1.0 + math.cos(math.pi * min(step / total_steps, 1.0)))


def step_lr(step, total_steps, lr0, drops=(0.6, 0.85), gamma=0.1):
    """Piecewise-constant LR with multiplicative drops -- what Fort et al. use.

    Preferred over cosine for this experiment. Cosine decays the LR to exactly 0,
    which forces the kernel velocity to 0 at the end of training for a trivial
    reason (the weights stop moving). That destroys the paper's actual claim,
    which is that after the chaotic transient the NTK keeps rotating at a
    constant *nonzero* speed. A piecewise-constant schedule leaves long stretches
    of fixed LR on which "constant nonzero velocity" is a statement that can
    actually be checked.
    """
    frac = min(step / total_steps, 1.0)
    return lr0 * gamma ** sum(frac >= d for d in drops)


def make_lr_fn(schedule, total_steps, lr0):
    if schedule == "cosine":
        return lambda s: cosine_lr(s, total_steps, lr0)
    if schedule == "step":
        return lambda s: step_lr(s, total_steps, lr0)
    if schedule == "const":
        return None
    raise ValueError(f"unknown schedule {schedule!r}")


def train_steps(model, opt, X, Y, n_steps, batch_size, gen, lr_fn=None, step0=0, on_step=None):
    """Run n_steps of SGD. `gen` is a torch.Generator that drives minibatch order.

    Two children spawned from the same parent weights get *different* generators
    and nothing else different -- that is the entire source of their divergence.

    `lr_fn(global_step)` is indexed by *global* step, not step-within-this-call,
    so a child spawned at step 500 continues the parent's schedule rather than
    restarting it.
    """
    model.train()
    n = len(X)
    perm, cursor = torch.randperm(n, generator=gen, device=X.device), 0
    for s in range(n_steps):
        if lr_fn is not None:
            lr = lr_fn(step0 + s)
            for g in opt.param_groups:
                g["lr"] = lr
        if cursor + batch_size > n:
            perm, cursor = torch.randperm(n, generator=gen, device=X.device), 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(X[idx]), Y[idx]).backward()
        opt.step()
        if on_step is not None:
            on_step(s + 1)
    return model


def get_flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()


def set_flat(model, flat):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(flat[i:i + p.numel()].view_as(p))
            i += p.numel()
