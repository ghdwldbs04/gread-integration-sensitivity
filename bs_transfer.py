"""
bs_transfer.py - does the solver-transfer gap also appear in the PUBLISHED GREAD model (GREAD-BS, published
hyperparameters)? Run from GREAD/src.

Training (same seeds and splits as before, published best hyperparameters for each dataset):
  --train published   the published integrator and step size of that dataset (e.g. Cornell rk4 0.2, Texas euler 1)
  --train euler       explicit Euler, step --h (default 1.0)
After training, the SAME weights (eval mode) are re-evaluated with: the training solver, Euler 1.0, Euler 0.5,
and RK4 with step T/1000 (reference) and T/2000 (reference convergence check).
GREAD-BS has no closed-form reaction flow, so the reference is fine-step RK4 (the BS vector field is linear in H).

Model-agnostic risk indices at the encoder output x0 (eval mode), with dt = min(training step, T):
  kappa_gen = dt * |lambda_max(J)|, |lambda_max| estimated by power iteration with Jacobian-vector products
  q_h       = (dt^2 / 4) * ||J f||  (leading term of the Euler step-doubling mismatch)
These work for any vector field, including the constant-adjacency block used by Texas.
Every line carries run_id; each run writes to a new time-stamped file.
Usage: python bs_transfer.py --dataset texas --train published --out ../results/bs_X.jsonl --run_id X
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_sweep as R  # noqa: E402
from gread_params import best_params_dict  # noqa: E402
from torchdiffeq import odeint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="texas")
p.add_argument("--train", default="published", choices=["published", "euler"])
p.add_argument("--h", type=float, default=1.0)
p.add_argument("--reaction", default="bspm")
p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
p.add_argument("--out", required=True)
p.add_argument("--run_id", required=True)
a = p.parse_args()

bp = best_params_dict[a.dataset]
if a.train == "published":
    method, h = bp["method"], float(bp["step_size"])
else:
    method, h = "euler", a.h
ns = argparse.Namespace(dataset=a.dataset, reaction=a.reaction, method=method, step_size=h, tol_scale=1.0,
                        max_nfe=200000, epoch=None, time=None, beta_pos=False)
opt = R.build_opt(ns)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt["device"] = dev
T = float(opt["time"])
dt = min(h, T)
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

EVALS = [(method, h), ("euler", 1.0), ("euler", 0.5), ("rk4", T / 1000), ("rk4", T / 2000)]
REF, REF2 = f"rk4_{T/1000:.6g}", f"rk4_{T/2000:.6g}"


def risk(m, data, iters=40):
    """kappa_gen and q_h at x0 for any GREAD vector field (attention or constant adjacency)."""
    m.eval()
    f = m.odeblock.odefunc
    with torch.no_grad():
        x0 = m.encoder(data.x)
        m.odeblock.set_x0(x0)
        if opt.get("beta_diag"):
            f.Beta = f.set_Beta()
        if hasattr(m.odeblock, "get_attention_weights"):
            f.attention_weights = m.odeblock.get_attention_weights(x0)
    t0 = torch.tensor(0.0, device=dev)
    vf = lambda z: f(t0, z)
    with torch.no_grad():
        fx = vf(x0)
    _, jf = torch.autograd.functional.jvp(vf, x0, fx)
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(x0.shape, generator=g).to(x0)
    v = v / v.norm()
    lams = []
    for _ in range(iters):
        _, jv = torch.autograd.functional.jvp(vf, x0, v)
        lam = float(jv.norm())
        lams.append(lam)
        if not lam > 0:
            break
        v = jv / lam
    f.nfe = 0
    return dict(kappa_gen=dt * max(lams[-10:]), lam_est=max(lams[-10:]),
                q_h_mean=float((dt ** 2 / 4) * jf.norm(dim=1).mean()), q_h_total=float((dt ** 2 / 4) * jf.norm()),
                f_norm=float(fx.norm()), max_x0=float(x0.abs().max()))


for s in a.splits:
    t_start = time.time()
    rec = R.run_one(dict(opt), ds, s, dev)
    out = dict(run_id=a.run_id, dataset=a.dataset, reaction=a.reaction, train=a.train, method=method, h=h, T=T, dt=dt,
               split=s, status=rec.get("status"), train_test_acc=rec.get("test_acc"),
               sec_per_epoch=rec.get("sec_per_epoch"), fwd_nfe_per_epoch=rec.get("fwd_nfe_per_epoch"), evals={})
    if rec.get("test_acc") is None:
        out["wall_sec"] = time.time() - t_start
        with open(a.out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(out) + "\n")
        print(f"[{a.dataset} {a.train} {method} {h}] split={s} training failed: {rec.get('status')}", flush=True)
        continue
    m = holder["m"]
    data = ds.data.to(dev)
    try:
        out.update(risk(m, data))
    except Exception as e:  # noqa: BLE001
        out["risk_error"] = str(e)[:150]
    m.eval()
    tm = data.test_mask
    preds, states = {}, {}
    with torch.no_grad():
        for meth, hh in EVALS:
            key = f"{meth}_{hh:.6g}"
            m.odeblock.test_integrator = odeint
            m.odeblock.opt["method"] = meth
            m.odeblock.opt["step_size"] = hh
            try:
                xN = m.forward_XN(data.x)
                lg = m(data.x)
                fin = bool(torch.isfinite(lg).all())
                preds[key] = lg[tm].argmax(1) if fin else None
                states[key] = xN if fin else None
                out["evals"][key] = float(preds[key].eq(data.y[tm]).float().mean()) if fin else None
            except Exception as e:  # noqa: BLE001
                preds[key], states[key], out["evals"][key] = None, None, None
                out.setdefault("errors", {})[key] = str(e)[:100]
            m.resetNFE()
    ref, ref2 = out["evals"].get(REF), out["evals"].get(REF2)
    out["ref_acc"] = ref
    out["gap"] = None if ref is None else rec["test_acc"] - ref
    out["ref_converged_diff"] = None if ref is None or ref2 is None else abs(ref - ref2)
    if states.get(REF) is not None and states.get(REF2) is not None:
        out["ref_state_relerr"] = float((states[REF] - states[REF2]).norm() / states[REF2].norm())
    if preds.get(REF) is not None:
        out["disagree"] = {k: float((v != preds[REF]).float().mean()) for k, v in preds.items() if v is not None}
    out["wall_sec"] = time.time() - t_start
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(out) + "\n")
    g = "-" if out["gap"] is None else f"{out['gap']*100:+.1f}"
    print(f"[{a.dataset} {a.train} {method} {h}] split={s} acc={rec['test_acc']*100:.1f} "
          f"ref={'-' if ref is None else f'{ref*100:.1f}'} gap={g} refconv={out['ref_converged_diff']} "
          f"kappa_gen={out.get('kappa_gen', float('nan')):.3g} q_h={out.get('q_h_mean', float('nan')):.3g} "
          + " ".join(f"{k}={'-' if v is None else f'{v*100:.1f}'}" for k, v in out["evals"].items()), flush=True)
