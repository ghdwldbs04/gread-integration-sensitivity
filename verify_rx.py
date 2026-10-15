"""
verify_rx.py - checks raised in review of the BS / Fisher / Zeldovich run. Run from GREAD/src.

For each split: train exactly as bs_transfer.py (same seeds, published hyperparameters, Euler h=1), then on the
SAME weights (eval mode, same x0, same T, same graph):
  (1) training check: epochs run, training wall time, and the training-solver accuracy re-evaluated on the frozen
      weights (must equal the accuracy logged during training -> same weights).
  (2) is a non-finite reference a real blow-up or an RK4 stability failure?
      Own fixed-step RK4 loop with N = 1000, 4000, 16000 (and 64000 on graphs < 1000 nodes) steps over [0, T].
      Records whether it stays finite, the first time max|H| > 1e3 (t_blow), the final accuracy, and the relative
      state difference between successive N. A real finite-time blow-up gives a t_blow that converges as N grows;
      a stability failure of RK4 disappears once the step is small enough.
  (3) reaction-only analytic blow-up: for every element, the closed-form blow-up time t* of the scalar reaction
      ODE x' = b r(x) (Allen-Cahn, Fisher, Zeldovich); reports the fraction of elements with t* <= T and min t*.
  (4) risk indices on the same checkpoint, so they can be compared directly:
      kappa_R   = dt * max |b_c r'(x0_ic)|        (reaction Jacobian, diagonal, exact)
      kappa_gen = dt * |lambda_max(J)|            (full Jacobian, power iteration with JVPs, estimate)
      rho_euler = |lambda_max(I + dt J)|          (power iteration; > 1 means the linearised Euler step amplifies)
Usage: python verify_rx.py --dataset texas --reaction fisher --splits 0 1 2 --out ../results/verify_X.jsonl --run_id X
       python verify_rx.py --dataset texas --reaction allen-cahn --beta_pos --splits 0 1 2 --out ... --run_id X
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

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="texas")
p.add_argument("--reaction", default="fisher", choices=["fisher", "zeldovich", "allen-cahn"])
p.add_argument("--beta_pos", action="store_true")
p.add_argument("--h", type=float, default=1.0)
p.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--steps", type=int, nargs="+", default=[1000, 4000, 16000, 64000])
p.add_argument("--out", required=True)
p.add_argument("--run_id", required=True)
a = p.parse_args()

ns = argparse.Namespace(dataset=a.dataset, reaction=a.reaction, method="euler", step_size=a.h, tol_scale=1.0,
                        max_nfe=10 ** 7, epoch=None, time=None, beta_pos=a.beta_pos)
opt = R.build_opt(ns)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt["device"] = dev
T = float(opt["time"])
dt = min(a.h, T)
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
    holder["m"] = m
    return m


R.GNN = GNN_hook

# scalar reaction r(x) and r'(x) as implemented in function_gread.py (f = ... + r(H) @ diag(b))
RX = {
    "allen-cahn": (lambda x: -(x ** 2 - 1) * x, lambda x: 1 - 3 * x ** 2),
    "fisher": (lambda x: -(x - 1) * x, lambda x: 1 - 2 * x),
    "zeldovich": (lambda x: -(x ** 2 - x) * x, lambda x: 2 * x - 3 * x ** 2),
}


def blowup_time(x, b):
    """Closed-form blow-up time of x' = b r(x) for each element (inf if no blow-up). x: (n, d), b: (d,)."""
    b = b[None, :].expand_as(x)
    inf = torch.full_like(x, float("inf"))
    if a.reaction == "allen-cahn":      # x' = b x (1 - x^2): blows up iff b < 0 and |x| > 1
        m = (b < 0) & (x.abs() > 1)
        t = torch.log(x ** 2 / (x ** 2 - 1)) / (2 * b.abs())
    elif a.reaction == "fisher":        # logistic x' = b x (1 - x): x(t) = x e^{bt} / (1 - x + x e^{bt})
        ratio = (x - 1) / x                       # blow-up when e^{bt} = (x - 1) / x
        m = (ratio > 0) & (torch.log(ratio.clamp_min(1e-30)) / b > 0)
        t = torch.log(ratio.clamp_min(1e-30)) / b
    else:                               # x' = b x^2 (1 - x): t* = -F(x)/b, F(x) = -1/x + ln|x/(1-x)|, F(+-inf) = 0
        m = (b < 0) & ((x > 1) | (x < 0))
        F = -1 / x + torch.log((x / (1 - x)).abs())
        t = -F / b
    return torch.where(m & (t > 0), t, inf)


def setup(m, data):
    f = m.odeblock.odefunc
    x0 = m.encoder(data.x)
    m.odeblock.set_x0(x0)
    if opt.get("beta_diag"):
        f.Beta = f.set_Beta()
    if hasattr(m.odeblock, "get_attention_weights"):
        f.attention_weights = m.odeblock.get_attention_weights(x0)
    return f, x0


def power(op, x0, iters=40):
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(x0.shape, generator=g).to(x0)
    v = v / v.norm()
    lams = []
    for _ in range(iters):
        w = op(v)
        lam = float(w.norm())
        lams.append(lam)
        if not lam > 0 or not math.isfinite(lam):
            break
        v = w / lam
    return max(lams[-10:])


def rk4_track(f, x0, N):
    """Fixed-step RK4 over [0, T] with N steps; stops at the first non-finite state. Returns final state or None."""
    hh = T / N
    z = x0
    t_blow, t_nan = None, None
    t0 = torch.tensor(0.0, device=dev)
    for k in range(N):
        k1 = f(t0, z)
        k2 = f(t0, z + hh / 2 * k1)
        k3 = f(t0, z + hh / 2 * k2)
        k4 = f(t0, z + hh * k3)
        z = z + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        mx = float(z.abs().max())
        if t_blow is None and mx > 1e3:
            t_blow = (k + 1) * hh
        if not math.isfinite(mx):
            t_nan = (k + 1) * hh
            break
    f.nfe = 0
    return (None if t_nan is not None else z), t_blow, t_nan


for s in a.splits:
    t_start = time.time()
    rec = R.run_one(dict(opt), ds, s, dev)
    out = dict(run_id=a.run_id, dataset=a.dataset, reaction=a.reaction, beta_pos=a.beta_pos, h=a.h, T=T, dt=dt, split=s,
               status=rec.get("status"), epochs_run=rec.get("epochs_run"), best_epoch=rec.get("best_epoch"),
               train_sec=rec.get("total_train_sec"), train_test_acc=rec.get("test_acc"), ref={})
    if rec.get("test_acc") is None:
        with open(a.out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(out) + "\n")
        print(f"[{a.dataset} {a.reaction}] split={s} training failed: {rec.get('status')}", flush=True)
        continue
    m = holder["m"]
    data = ds.data.to(dev)
    m.eval()
    tm = data.test_mask
    acc = lambda z: float(m.m2(m.GNN_postXN(z))[tm].argmax(1).eq(data.y[tm]).float().mean())

    # (1) same weights: training solver re-evaluated on the frozen model
    with torch.no_grad():
        m.odeblock.test_integrator = odeint
        m.odeblock.opt["method"], m.odeblock.opt["step_size"] = "euler", a.h
        out["reeval_train_solver_acc"] = acc(m.forward_XN(data.x))
        m.resetNFE()

    # (3) + (4) on the same x0
    with torch.no_grad():
        f, x0 = setup(m, data)
    b = torch.diagonal(f.Beta).detach()
    r, dr = RX[a.reaction]
    with torch.no_grad():
        tstar = blowup_time(x0, b)
        out["frac_scalar_blowup"] = float((tstar <= T).float().mean())
        out["min_tstar"] = float(tstar.min()) if torch.isfinite(tstar.min()) else None
        out["kappa_R"] = float(dt * (b[None, :] * dr(x0)).abs().max())
        out["euler_amp_R"] = float((1 + dt * b[None, :] * dr(x0)).abs().max())
        out["bmin"], out["bmax"], out["max_x0"] = float(b.min()), float(b.max()), float(x0.abs().max())
    t0 = torch.tensor(0.0, device=dev)
    vf = lambda z: f(t0, z)
    Jv = lambda v: torch.autograd.functional.jvp(vf, x0, v)[1]
    try:
        lam = power(Jv, x0)
        out["kappa_gen"] = dt * lam
        out["rho_euler"] = power(lambda v: v + dt * Jv(v), x0)
    except Exception as e:  # noqa: BLE001
        out["risk_error"] = str(e)[:150]
    f.nfe = 0

    # (2) reference ladder
    prev = None
    with torch.no_grad():
        for N in a.steps:
            if N > 16000 and data.num_nodes > 1000:
                continue
            tt = time.time()
            z, t_blow, t_nan = rk4_track(f, x0, N)
            e = dict(finite=z is not None, t_blow=t_blow, t_nan=t_nan, sec=round(time.time() - tt, 2))
            if z is not None:
                e["acc"] = acc(z)
                if prev is not None:
                    e["relerr_vs_prev"] = float((z - prev).norm() / z.norm())
                prev = z
            out["ref"][str(N)] = e
    out["wall_sec"] = time.time() - t_start
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(out) + "\n")
    lad = " ".join(f"N{k}:{'ok' if v['finite'] else 'NaN'}(tb={v['t_blow'] if v['t_blow'] is None else round(v['t_blow'], 4)},"
                   f"acc={v.get('acc', float('nan'))*100:.1f},dz={v.get('relerr_vs_prev', float('nan')):.1e})" for k, v in out["ref"].items())
    print(f"[{a.dataset} {a.reaction}{'+b>0' if a.beta_pos else ''}] split={s} ep={out['epochs_run']} acc={out['train_test_acc']*100:.1f} "
          f"reeval={out['reeval_train_solver_acc']*100:.1f} kR={out['kappa_R']:.3g} kgen={out.get('kappa_gen', float('nan')):.3g} "
          f"rhoE={out.get('rho_euler', float('nan')):.3g} scalar_blowup={out['frac_scalar_blowup']:.2e} min_t*={out['min_tstar']} T={T:.3f} | {lad}",
          flush=True)
