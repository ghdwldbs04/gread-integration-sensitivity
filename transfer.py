"""
transfer.py — solver-transfer matrix + stiffness index. Run from GREAD/src.

Train GREAD-AC with one integrator (--train split|strang|euler|rk4, --train_h), then freeze the weights and
re-evaluate the SAME model with every integrator in EVALS (eval mode). For each solver it records test
accuracy, prediction disagreement and hidden-state relative error against the converged reference, and the
fraction of test nodes whose prediction is certified unchanged by the margin bound (l_y - l_k) > ||w_y - w_k|| eps. Also records the stiffness index
    kappa = train_h * max_c b_c * (3 * max_i x0_ic^2 - 1)
for the eval-mode and train-mode (dropout) initial states. Explicit Euler on the reaction is
linearly stable only if h*|lambda| <= 2, lambda = b(1 - 3x^2), so kappa > 2 predicts a solver gap.
Usage: python transfer.py --dataset texas --train euler --train_h 1.0 --splits 0 1 2
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_sweep as R  # noqa: E402
from torchdiffeq import odeint  # noqa: E402


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


p = argparse.ArgumentParser()
p.add_argument("--dataset", default="Cora")
p.add_argument("--train", default="euler", choices=["split", "strang", "euler", "rk4"])
p.add_argument("--train_h", type=float, default=1.0)
p.add_argument("--no_beta_pos", action="store_true", help="unconstrained b (default: b = softplus(w) > 0)")
p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
p.add_argument("--out", default="../results/transfer.jsonl")
a = p.parse_args()

EVALS = [("euler", 1.0), ("euler", 0.5), ("euler", 0.1), ("rk4", 0.25), ("rk4", 0.01),
         ("split", 1.0), ("split", 0.5), ("split", 0.25), ("split", 0.1),
         ("strang", 2.0), ("strang", 1.0), ("strang", 0.5), ("strang", 0.25), ("split", 0.001)]
REF = "split_0.001"  # converged reference: for b > 0 the fine-step split solution is always finite

ns = argparse.Namespace(dataset=a.dataset, reaction="allen-cahn", method="euler" if a.train in SOLVERS else a.train,
                        step_size=a.train_h, tol_scale=1.0, max_nfe=20000, epoch=None, time=None,
                        beta_pos=not a.no_beta_pos)
opt = R.build_opt(ns)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt["device"] = dev
ds = R.get_dataset(opt, os.path.join(R.ROOT_DIR, "data"), opt["not_lcc"])
if opt.get("hetero_SL"):
    from torch_geometric.utils import add_remaining_self_loops
    ds.data.edge_index, _ = add_remaining_self_loops(ds.data.edge_index)
if opt.get("hetero_undir"):
    from torch_geometric.utils import to_undirected
    ds.data.edge_index = to_undirected(ds.data.edge_index)

holder = {}
_GNN = R.GNN


def GNN_hook(*args, **kw):
    m = _GNN(*args, **kw)
    if a.train in SOLVERS:
        m.odeblock.train_integrator = m.odeblock.test_integrator = SOLVERS[a.train]
    holder["m"] = m
    return m


R.GNN = GNN_hook

for s in a.splits:
    t0 = time.time()
    rec = R.run_one(dict(opt), ds, s, dev)
    out = dict(dataset=a.dataset, train=a.train, train_h=a.train_h, beta_pos=not a.no_beta_pos, split=s,
               status=rec.get("status"), train_test_acc=rec.get("test_acc"), evals={})
    if rec.get("test_acc") is None:
        with open(a.out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(out) + "\n")
        print(f"[{a.dataset} {a.train} h={a.train_h}] split={s} training failed: {rec.get('status')}", flush=True)
        continue
    m = holder["m"]
    data = ds.data.to(dev)
    f = m.odeblock.odefunc

    # stiffness index from the initial state, eval mode and train (dropout) mode
    with torch.no_grad():
        for mode in ("eval", "train"):
            torch.manual_seed(1234)
            m.train(mode == "train")
            x0 = m.encoder(data.x)
            f.Beta = f.set_Beta()
            b = torch.diagonal(f.Beta)
            x2 = (x0 * x0).max(dim=0).values
            out[f"max_x0_{mode}"] = float(x0.abs().max())
            out[f"kappa_{mode}"] = float((a.train_h * b * (3 * x2 - 1)).max())
        out["bmax"] = float(b.max())
        out["bmin"] = float(b.min())
    m.eval()

    # Jacobian risk q_h = (h^2/4) ||J_f(x0) f(x0)|| (leading term of Euler step-doubling mismatch), via one JVP
    try:
        with torch.no_grad():
            x0 = m.encoder(data.x)
            m.odeblock.set_x0(x0)
            f.Beta = f.set_Beta()
            f.attention_weights = m.odeblock.get_attention_weights(x0)
        t0_ = torch.tensor(0.0, device=dev)
        vf = lambda z: f(t0_, z)
        with torch.no_grad():
            fx = vf(x0)
        _, jf = torch.autograd.functional.jvp(vf, x0, fx)
        q = (a.train_h ** 2 / 4) * jf.norm(dim=1)                 # per node
        out["q_h_mean"], out["q_h_max"] = float(q.mean()), float(q.max())
        out["q_h_total"] = float((a.train_h ** 2 / 4) * jf.norm())
        out["f_norm"] = float(fx.norm())
    except Exception as e:  # noqa: BLE001
        out["q_h_error"] = str(e)[:120]

    # same weights, every evaluation integrator (eval mode: no dropout, identical input)
    def set_solver(meth, h):
        if meth in SOLVERS:
            m.odeblock.test_integrator = SOLVERS[meth]
        else:
            m.odeblock.test_integrator = odeint
            m.odeblock.opt["method"] = meth
        m.odeblock.opt["step_size"] = h

    tm = data.test_mask
    W = getattr(getattr(m, "m2", None), "weight", None)  # linear decoder, used for the margin certificate
    res = {}
    with torch.no_grad():
        for meth, h in EVALS:
            key = f"{meth}_{h}"
            set_solver(meth, h)
            try:
                xN = m.forward_XN(data.x)
                lg = m(data.x)
                fin = bool(torch.isfinite(lg).all())
                res[key] = (xN, lg) if fin else None
                out["evals"][key] = float(lg[tm].argmax(1).eq(data.y[tm]).float().mean()) if fin else None
            except Exception as e:  # noqa: BLE001
                res[key] = None
                out["evals"][key] = None
                out.setdefault("errors", {})[key] = str(e)[:80]
        m.resetNFE()
        out["disagree"], out["relerr"], out["certified"] = {}, {}, {}
        if res.get(REF) is not None:
            xr, lr = res[REF]
            pr = lr[tm].argmax(1)
            # logits are W relu(x) + c, relu is 1-Lipschitz: a node's prediction cannot flip if for every class k
            # (l_y - l_k) > ||w_y - w_k|| * ||x - x_ref||  (sufficient condition, per node)
            if W is not None and W.shape[1] != xr.shape[1]:
                W = None
            if W is not None:
                # certificate only valid if the decoder really is logits = m2(relu(x_N)); check it
                chk = m.m2(torch.relu(xr))
                out["decoder_check"] = float((chk - lr).abs().max())
                if out["decoder_check"] > 1e-3:
                    W = None
            if W is not None:
                y = pr
                gap_l = lr[tm].gather(1, y[:, None]) - lr[tm]                      # (n, C)
                wd = (W[y][:, None, :] - W[None, :, :]).norm(dim=2)                  # (n, C)
            for key, v in res.items():
                if v is None:
                    continue
                xN, lg = v
                out["disagree"][key] = float((lg[tm].argmax(1) != pr).float().mean())
                out["relerr"][key] = float((xN - xr).norm() / xr.norm())
                if W is not None:
                    eps = (xN[tm] - xr[tm]).norm(dim=1)[:, None]
                    ok = (gap_l > wd * eps) | (wd == 0)
                    out["certified"][key] = float(ok.all(1).float().mean())
    ref = out["evals"].get(REF)
    out["ref_acc"] = ref
    out["gap"] = None if ref is None else rec["test_acc"] - ref
    out["wall_sec"] = time.time() - t0
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(out) + "\n")
    g = "-" if out["gap"] is None else f"{out['gap']*100:+.2f}"
    print(f"[{a.dataset} {a.train} h={a.train_h}] split={s} acc={rec['test_acc']*100:.2f} ref={'-' if ref is None else f'{ref*100:.2f}'} "
          f"gap={g} kappa_eval={out['kappa_eval']:.2f} kappa_train={out['kappa_train']:.2f} "
          + " ".join(f"{k}={'-' if v is None else f'{v*100:.1f}'}" for k, v in out["evals"].items())
          + f" | disagree(euler_1.0)={out['disagree'].get('euler_1.0')} cert(euler_1.0)={out['certified'].get('euler_1.0')}", flush=True)
