"""solvers.py — structure-preserving splitting integrators for GREAD-AC (shared by consist.py)."""
import math

import torch


def _diffuse_exact(f, z, lam, tol=1e-7):
    """Exact diffusion flow exp(lam (A - I)) z for row-stochastic A, lam = dt*alpha (scalar or per channel):
    sum_k Poisson(k; lam) A^k z  (heat kernel). Truncated where the Poisson tail < tol; the leftover
    probability mass is put on the last term, so the update stays a convex combination (max principle kept)."""
    lam = lam * torch.ones(z.shape[1], device=z.device, dtype=z.dtype)
    lmax = float(lam.max())
    K = int(math.ceil(lmax + 6 * math.sqrt(lmax + 1e-12) + 8))
    w = torch.exp(-lam)
    acc, term, cum = w * z, z, w.clone()
    for k in range(1, K + 1):
        term = f.sparse_multiply(term)
        w = w * lam / k
        acc = acc + w * term
        cum = cum + w
        if float((1 - cum).max()) < tol:
            break
    acc = acc + (1 - cum).clamp_min(0) * term
    f.nfe += k
    return acc


def _make_split(exact_diffusion):
    def split_odeint(func, x, t, method=None, options=None, **kw):
        """Splitting: reaction dt/2 (exact) -> diffusion dt -> reaction dt/2 (exact).
        exact_diffusion=False: diffusion by explicit Euler (first order, needs dt*alpha <= 1).
        exact_diffusion=True : diffusion by its exact flow (Strang with exact subflows: second order, bounded for any dt)."""
        f = getattr(func, "odefunc", func)
        T = float(t[-1] - t[0])
        n = max(1, math.ceil(T / float(options["step_size"]) - 1e-9))
        dt = T / n
        alpha = f.alpha_train if f.opt["no_alpha_sigmoid"] else torch.sigmoid(f.alpha_train)
        b = torch.diagonal(f.Beta)

        def react(z, tau):
            e = torch.exp(b * tau)
            return z * e / torch.sqrt(1 + z * z * (e * e - 1))

        z = x
        for _ in range(n):
            z = react(z, dt / 2)
            if exact_diffusion:
                z = _diffuse_exact(f, z, dt * alpha)
            else:
                z = z + dt * alpha * (f.sparse_multiply(z) - z)
                f.nfe += 1
            z = react(z, dt / 2)
        return torch.stack([x, z])
    return split_odeint


split_odeint = _make_split(False)
strang_odeint = _make_split(True)
SOLVERS = {"split": split_odeint, "strang": strang_odeint}
