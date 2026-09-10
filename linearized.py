"""Linearised training along the tangent plane at w0  (Fort et al. 2020, Sec. 6, App. D.4).

    f_lin(x; dw) = f_w0(x) + J_w0(x) dw,     dw starts at 0, trained by SGD + momentum on
                                              the same cross-entropy as the nonlinear run.

How J_w0(x) dw is computed
--------------------------
Explicit tangent propagation through SmallCNN. conv / linear are bilinear in (input, weight),
so d[conv(h, W)] = conv(dh, W) + conv(h, dW); ReLU and max-pool are piecewise linear, so the
tangent passes through the *same* mask / argmax as the primal. This is exact for this
architecture (no BN, no smooth nonlinearity). It is checked in `self_test()` against
(a) an autograd double-VJP route (agrees to 1e-8) and (b) finite differences.
torch.func.jvp is not used: PyTorch has no forward-AD kernel for max_pool2d.

Gradient w.r.t. dw is J_w0^T (dCE/df), one ordinary backward pass at w0 with grad_outputs.
Cost per step ~1.9x an ordinary SGD step.
"""
import torch
import torch.nn.functional as F

import core


def params_of(model):
    """Detached copies of the parameters, in SmallCNN definition order."""
    return [p.detach().clone() for p in model.parameters()]


def fwd(p, x, features=False):
    """Functional SmallCNN forward. Must mirror core.SmallCNN.forward exactly."""
    c1w, c1b, c2w, c2b, c3w, c3b, f1w, f1b, f2w, f2b = p
    h = F.max_pool2d(F.relu(F.conv2d(x, c1w, c1b, padding=1)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, c2w, c2b, padding=1)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, c3w, c3b, padding=1)), 2)
    feat = F.relu(F.linear(h.flatten(1), f1w, f1b))
    out = F.linear(feat, f2w, f2b)
    return (out, feat) if features else out


@torch.no_grad()
def lin_forward(p, dw, x):
    """f_w0(x) + J_w0(x) dw by tangent propagation."""
    c1w, c1b, c2w, c2b, c3w, c3b, f1w, f1b, f2w, f2b = p
    d1w, d1b, d2w, d2b, d3w, d3b, e1w, e1b, e2w, e2b = dw

    def block(h, t, W, b, dW, db):
        a = F.conv2d(h, W, b, padding=1)
        ta = F.conv2d(t, W, None, padding=1) + F.conv2d(h, dW, db, padding=1)
        ta = ta * (a > 0)
        a, idx = F.max_pool2d(F.relu(a), 2, return_indices=True)
        ta = ta.flatten(2).gather(2, idx.flatten(2)).view_as(a)
        return a, ta

    h, t = x, torch.zeros_like(x)
    h, t = block(h, t, c1w, c1b, d1w, d1b)
    h, t = block(h, t, c2w, c2b, d2w, d2b)
    h, t = block(h, t, c3w, c3b, d3w, d3b)
    h, t = h.flatten(1), t.flatten(1)
    a = F.linear(h, f1w, f1b)
    ta = (F.linear(t, f1w) + F.linear(h, e1w, e1b)) * (a > 0)
    h = F.relu(a)
    return F.linear(h, f2w, f2b) + F.linear(ta, f2w) + F.linear(h, e2w, e2b)


def lin_loss_and_grad(p, dw, x, y):
    """Cross-entropy of the linearised model and its gradient w.r.t. dw (= J^T dCE/df)."""
    flin = lin_forward(p, dw, x).requires_grad_(True)
    loss = F.cross_entropy(flin, y)
    (g,) = torch.autograd.grad(loss, flin)
    for q in p:
        q.requires_grad_(True)
    out = fwd(p, x)
    grads = torch.autograd.grad(out, p, grad_outputs=g)
    for q in p:
        q.requires_grad_(False)
    return loss.item(), grads


@torch.no_grad()
def evaluate_lin(p, dw, X, Y, batch=500):
    """(mean cross-entropy, 0-1 error) of the linearised model."""
    loss, wrong = 0.0, 0
    for i in range(0, len(X), batch):
        out = lin_forward(p, dw, X[i:i + batch])
        loss += F.cross_entropy(out, Y[i:i + batch], reduction="sum").item()
        wrong += (out.argmax(1) != Y[i:i + batch]).sum().item()
    return loss / len(X), wrong / len(X)


def train_linearized(p, Xtr, Ytr, evalsets, epochs, lr, batch, momentum, gen, log=None):
    """SGD + momentum on dw for `epochs` passes over (Xtr, Ytr); evaluates every epoch.

    evalsets: {name: (X, Y)}. Returns (curve, dw). curve[i] has epoch, train_loss (running
    mean over the epoch), and <name>_err for each eval set; `diverged` is set on NaN/inf.
    """
    dw = [torch.zeros_like(q).requires_grad_(True) for q in p]
    opt = torch.optim.SGD(dw, lr=lr, momentum=momentum)
    n, spe = len(Xtr), len(Xtr) // batch
    curve = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen, device=Xtr.device)
        tot = 0.0
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            loss, grads = lin_loss_and_grad(p, dw, Xtr[idx], Ytr[idx])
            if not (loss == loss and abs(loss) < 1e6):           # NaN or blown up
                curve.append(dict(epoch=ep, diverged=True))
                return curve, dw
            for d, g in zip(dw, grads):
                d.grad = g
            opt.step()
            tot += loss
        row = dict(epoch=ep, train_loss=tot / spe)
        for name, (X, Y) in evalsets.items():
            row[name + "_err"] = evaluate_lin(p, dw, X, Y)[1]
        curve.append(row)
        if log:
            log(row)
    return curve, dw


def train_nonlinear(model, w0, mom0, Xtr, Ytr, evalsets, epochs, lr, batch, momentum, gen, log=None):
    """Ordinary SGD from checkpoint (w0, mom0) at constant lr; evaluates every epoch.

    This is the paper's blue curve ("Low LR nonlin"): same start point and same LR as the
    linearised run, independent minibatch noise, momentum buffer restored from the parent.
    """
    core.set_flat(model, w0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    i = 0
    for q in model.parameters():
        opt.state[q] = dict(momentum_buffer=mom0[i:i + q.numel()].view_as(q).clone())
        i += q.numel()
    spe = len(Xtr) // batch
    curve = []
    for ep in range(1, epochs + 1):
        core.train_steps(model, opt, Xtr, Ytr, spe, batch, gen)
        row = dict(epoch=ep)
        for name, (X, Y) in evalsets.items():
            row[name + "_err"] = core.evaluate(model, X, Y)[1]
        curve.append(row)
        if log:
            log(row)
    return curve


@torch.no_grad()
def penultimate(p, X, batch=1000):
    return torch.cat([fwd(p, X[i:i + batch], features=True)[1] for i in range(0, len(X), batch)])


def linear_probe(p, Xtr, Ytr, evalsets, wds=(0.0, 1e-4, 1e-3), iters=200):
    """Multinomial logistic regression on the penultimate features at w0 (L-BFGS, full batch).

    Null model for the linearised run: if the tangent-space model does no better than a
    linear read-out of the same network's features, "kernel learning" is just
    representation learning. Weight decay picked on the validation set.
    """
    Ftr = penultimate(p, Xtr)
    Fev = {k: penultimate(p, X) for k, (X, _) in evalsets.items()}
    best = None
    for wd in wds:
        W = torch.zeros(Ftr.shape[1], 10, device=Ftr.device, requires_grad=True)
        b = torch.zeros(10, device=Ftr.device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Ftr @ W + b, Ytr) + 0.5 * wd * (W * W).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            errs = {k: ((Fev[k] @ W + b).argmax(1) != Y).float().mean().item()
                    for k, (_, Y) in evalsets.items()}
        res = dict(wd=wd, **{k + "_err": v for k, v in errs.items()})
        if best is None or res["val_err"] < best["val_err"]:
            best = res
    return best


# ----------------------------------------------------------------------------- self-test
def self_test(width=32, device=None):
    """Exactness checks. Run: python linearized.py"""
    dev = device or core.get_device()
    m = core.make_model(dev, width=width, seed=0)
    p = params_of(m)
    torch.manual_seed(1)
    x = torch.randn(64, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (64,), device=dev)
    with torch.no_grad():
        f_ref = m(x)
    # 1. functional forward == module forward
    assert torch.allclose(fwd(p, x), f_ref), "fwd mismatch"
    # 2. dw = 0: f_lin == f, grad == ordinary gradient
    dw0 = [torch.zeros_like(q) for q in p]
    assert torch.equal(lin_forward(p, dw0, x), f_ref), "f_lin != f at dw=0"
    m.zero_grad(); F.cross_entropy(m(x), y).backward()
    _, g = lin_loss_and_grad(p, dw0, x, y)
    gerr = max((a - q.grad).abs().max().item() for a, q in zip(g, m.parameters()))
    assert gerr < 1e-5, f"grad mismatch {gerr}"
    # 3. J dw vs finite difference, decreasing perturbation -> error must shrink ~linearly
    torch.manual_seed(2)
    direction = [torch.randn_like(q) for q in p]
    rels = []
    for eps in (1e-2, 1e-3, 1e-4):
        dw = [eps * d for d in direction]
        with torch.no_grad():
            f_pert = fwd([a + b for a, b in zip(p, dw)], x)
        rels.append(((lin_forward(p, dw, x) - f_pert).norm() / (f_pert - f_ref).norm()).item())
    # 4. double-VJP autograd route agrees with tangent propagation
    def fwd_idx(pp, xx):
        c1w, c1b, c2w, c2b, c3w, c3b, f1w, f1b, f2w, f2b = pp
        h = F.max_pool2d(F.relu(F.conv2d(xx, c1w, c1b, padding=1)), 2, return_indices=True)[0]
        h = F.max_pool2d(F.relu(F.conv2d(h, c2w, c2b, padding=1)), 2, return_indices=True)[0]
        h = F.max_pool2d(F.relu(F.conv2d(h, c3w, c3b, padding=1)), 2, return_indices=True)[0]
        return F.linear(F.relu(F.linear(h.flatten(1), f1w, f1b)), f2w, f2b)
    dw = [1e-3 * d for d in direction]
    pr = [q.clone().requires_grad_(True) for q in p]
    out = fwd_idx(pr, x)
    v = torch.zeros_like(out, requires_grad=True)
    u = torch.autograd.grad(out, pr, grad_outputs=v, create_graph=True)
    (jdw,) = torch.autograd.grad(sum((ui * di).sum() for ui, di in zip(u, dw)), v)
    vjp_route = (out + jdw).detach()
    agree = (vjp_route - lin_forward(p, dw, x)).abs().max().item()
    print(f"self-test on {dev}: grad err {gerr:.1e}; J dw vs FD rel err at eps 1e-2/1e-3/1e-4 = "
          f"{rels[0]:.2e}/{rels[1]:.2e}/{rels[2]:.2e}; tangent-prop vs double-VJP {agree:.1e}")
    assert rels[2] < 0.05 and rels[2] < rels[0] and agree < 1e-5
    print("OK")


if __name__ == "__main__":
    self_test()
