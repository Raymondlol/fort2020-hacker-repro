"""The four measurements from Fort et al. (2020), Section 2.

Each function below is a direct transcription of one boxed definition in the
paper. Notation follows the paper: w = weights, S = a fixed probe set of inputs,
J_w(x) in R^{K x d} = Jacobian of the K logits w.r.t. the d parameters.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap

import core


# --------------------------------------------------------------------------
# Empirical NTK:  kappa_t(x, x') = J_t(x) J_t(x')^T          (paper Eq. 4)
# --------------------------------------------------------------------------
def ntk_jacobian(model, X, chunk=16, store_device=None):
    """Per-example Jacobian of all K logits w.r.t. all d params -> [N, K, d].

    Chunked over examples because the tensor is N*K*d floats: at N=192, K=10,
    d=189k that is 1.35 GB in one shot.

    store_device=None keeps the result on the same device as X. That is the
    whole point of the GPU path -- the Gram matrices below are large GEMMs
    (640x189k @ 189kx640), which are ~300x faster on an L4 than on the CPU the
    measurement used to be pinned to.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def f_single(p, x):
        return functional_call(model, (p, buffers), (x.unsqueeze(0),)).squeeze(0)

    store = X.device if store_device is None else store_device
    out = []
    for i in range(0, len(X), chunk):
        jac = vmap(jacrev(f_single), (None, 0))(params, X[i:i + chunk])
        out.append(torch.cat([j.flatten(2) for j in jac.values()], dim=2).to(store))
    return torch.cat(out, 0)


def ntk_gram(J):
    """Full mK x mK Gram matrix kappa_t(S) from a [m, K, d] Jacobian."""
    m, K, d = J.shape
    return (J.reshape(m * K, d) @ J.reshape(m * K, d).T)


def ntk_trace_gram(J):
    """m x m scalar kernel k(x,x') = Tr(J(x)J(x')^T), used for kernel regression."""
    return torch.einsum("ikd,jkd->ij", J, J)


def kernel_distance(K1, K2):
    """S(w, w') = 1 - Tr(K1 K2^T) / sqrt(Tr(K1 K1^T) Tr(K2 K2^T))   (paper Sec. 2)

    Scale-invariant, so it measures how much the kernel has *rotated*, not how
    much its overall magnitude grew.
    """
    # float64 on the CPU: the Gram is only (mK)^2 = 640x640 here, so the copy is
    # free, it keeps the normalised inner product well conditioned, and MPS has
    # no float64 at all.
    K1, K2 = K1.cpu().double(), K2.cpu().double()
    num = (K1 * K2).sum()
    den = torch.sqrt((K1 * K1).sum() * (K2 * K2).sum())
    return (1.0 - num / den).clamp_min(0.0).item()


def kernel_velocity(K_t, K_t_plus_dt, dt_epochs):
    """v(t) = S(w_t, w_{t+dt}) / dt. Paper uses dt = 0.4 epochs."""
    return kernel_distance(K_t, K_t_plus_dt) / dt_epochs


# --------------------------------------------------------------------------
# ReLU activation-pattern distance                            (paper Sec. 2)
# --------------------------------------------------------------------------
@torch.no_grad()
def relu_pattern(model, X):
    """Boolean tensor of which ReLU units are on, flattened over units+layers."""
    acts = []
    h = X
    h = F.max_pool2d(F.relu(model.c1(h)), 2); acts.append(h.flatten(1) > 0)
    h = F.max_pool2d(F.relu(model.c2(h)), 2); acts.append(h.flatten(1) > 0)
    h = F.max_pool2d(F.relu(model.c3(h)), 2); acts.append(h.flatten(1) > 0)
    h = F.relu(model.f1(h.flatten(1)));       acts.append(h > 0)
    return torch.cat(acts, dim=1)


def relu_distance(B1, B2):
    """Normalised Hamming distance between two activation-pattern tensors."""
    return (B1 != B2).float().mean().item()


# --------------------------------------------------------------------------
# Error barrier between children                              (paper Sec. 2)
#   max_alpha R(alpha w + (1-alpha) w') - 0.5 (R(w) + R(w'))
# --------------------------------------------------------------------------
@torch.no_grad()
def error_barrier(model, w1, w2, X, Y, n_alpha=11):
    """Returns (barrier, alphas, error curve). R is the 0-1 error, as in the paper."""
    alphas = torch.linspace(0, 1, n_alpha).tolist()
    errs = []
    for a in alphas:
        core.set_flat(model, a * w1 + (1 - a) * w2)
        errs.append(core.evaluate(model, X, Y)[1])
    endpoints = 0.5 * (errs[-1] + errs[0])   # alpha=1 -> w1, alpha=0 -> w2
    return max(errs) - endpoints, alphas, errs


# --------------------------------------------------------------------------
# What has the data-dependent NTK learned?                    (paper Sec. 6)
# --------------------------------------------------------------------------
def ntk_kernel_regression(model, Xtr, Ytr, Xte, Yte, n_classes=10, reg=1e-2,
                          chunk=16, te_block=256):
    """Kernel ridge regression with the *current* empirical NTK.

    SIMPLIFICATION: the paper linearises the network at time t and then runs
    gradient descent along that tangent plane. For squared loss that procedure
    converges to exactly this closed-form kernel ridge solution -- BUT ONLY IF
    both fit the same training set. The paper fits all 50k examples; we fit
    `len(Xtr)`. Those are not the same estimator when the sizes differ, and the
    gap is the most likely reason our kernel underperforms where theirs
    overperforms. Sweeping len(Xtr) is the experiment that settles it.

    K_te is accumulated one block of test rows at a time, so the test-side
    Jacobian is never resident in full. Memory is set by `te_block`, not by
    len(Xte). This matters because the two sample sizes have very different
    costs: len(Xtr) drives an N^2 Gram *and* a resident [N,K,d] tensor, while
    len(Xte) only adds rows to K_te. Evaluating on 2000 test points instead of
    192 therefore costs almost nothing and shrinks the binomial standard error
    of the reported accuracy from +/-3.3pp to +/-1.0pp.
    """
    # Split by size: the Jacobians and the two einsums are large and stay on the
    # training device in float32; the Gram matrices they produce are small
    # (n_tr x n_tr and te_block x n_tr) and the solve wants float64, which MPS
    # does not have. Moving just those to the CPU costs a few hundred KB.
    Jtr = ntk_jacobian(model, Xtr, chunk=chunk)
    Ktr = ntk_trace_gram(Jtr).cpu().double()

    # One scale for both Gram matrices. The old code rescaled K_te by a
    # different factor; that was invisible because argmax over classes is
    # invariant to a positive scale, but it made the regression values
    # meaningless. Using the same scale keeps them interpretable.
    scale = Ktr.diagonal().mean().clone()
    Ktr /= scale

    Yoh = F.one_hot(Ytr.cpu(), n_classes).double() - 1.0 / n_classes
    eye = torch.eye(len(Ktr), dtype=torch.float64)
    A = torch.linalg.solve(Ktr + reg * len(Ktr) * eye, Yoh)

    correct = 0
    for i in range(0, len(Xte), te_block):
        Jte = ntk_jacobian(model, Xte[i:i + te_block], chunk=chunk)
        Kte = torch.einsum("ikd,jkd->ij", Jte, Jtr).cpu().double() / scale
        del Jte
        correct += ((Kte @ A).argmax(1) == Yte[i:i + te_block].cpu()).sum().item()
        del Kte
    del Jtr
    return correct / len(Xte)
