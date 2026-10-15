"""
consist.py — RQ3: can a generic fix (step-doubling consistency regularization) match the structure-preserving
integrator? Also checks that the reference solution has converged (RQ1/RQ2 reliability). Run from GREAD/src.

Training modes (GREAD-AC, b = softplus(w) > 0, same BS-tuned hyperparameters as before):
  --train euler --lam 0      plain Euler h (baseline)
  --train euler --lam L      Euler h + L * mean ||z_h - z_{h/2}||^2 on the logits of ALL nodes,
                             where z_h and z_{h/2} are computed with the same weights, the same input and the
                             same dropout mask (step-doubling consistency, a generic regularizer)
  --train strang             structure-preserving splitting (reaction exact / heat-kernel diffusion / reaction exact)
After training, the same weights (eval mode) are re-evaluated with: Euler h, Euler h/2, split 0.001 (reference),
split 0.0005 (reference check), Strang 0.01. Records accuracy, gap to the reference, prediction disagreement,
reference convergence |acc(0.001) - acc(0.0005)|, seconds per epoch and NFE.
Every line carries run_id; write each run to a new, time-stamped file.
Usage: python consist.py --dataset texas --train euler --lam 1.0 --out ../results/consist_X.jsonl --run_id X
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_sweep as R  # noqa: E402
from solvers import split_odeint, strang_odeint  # noqa: E402
from torchdiffeq import odeint  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="texas")
p.add_argument("--train", default="euler", choices=["euler", "strang"])
p.add_argument("--h", type=float, default=1.0)
p.add_argument("--lam", type=float, default=0.0)
p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
p.add_argument("--out", required=True)
p.add_argument("--run_id", required=True)
a = p.parse_args()

ns = argparse.Namespace(dataset=a.dataset, reaction="allen-cahn", method="euler", step_size=a.h, tol_scale=1.0,
                        max_nfe=20000, epoch=None, time=None, beta_pos=True)
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
    if a.train == "strang":
        m.odeblock.train_integrator = m.odeblock.test_integrator = strang_odeint
    holder["m"] = m
    return m


R.GNN = GNN_hook

_train_step = R.train_step


def consistency_train_step(model, optimizer, data):
    """CE on the Euler-h logits + lam * ||z_h - z_{h/2}||^2 (same weights, input and dropout mask)."""
    model.train()
    optimizer.zero_grad()
    model.resetNFE()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    z_c = model(data.x)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    model.odeblock.opt["step_size"] = a.h / 2
    try:
        z_f = model(data.x)
    finally:
        model.odeblock.opt["step_size"] = a.h
    y = data.y.squeeze()
    loss = F.cross_entropy(z_c[data.train_mask], y[data.train_mask]) + a.lam * ((z_c - z_f) ** 2).sum(1).mean()
    fnfe = model.getNFE()
    model.resetNFE()
    if not torch.isfinite(loss):
        return float("nan"), fnfe, 0
    loss.backward()
    optimizer.step()
    bnfe = model.getNFE()
    model.resetNFE()
    return loss.item(), fnfe, bnfe


if a.train == "euler" and a.lam > 0:
    R.train_step = consistency_train_step

EVALS = [("euler", a.h), ("euler", a.h / 2), ("split", 0.001), ("split", 0.0005), ("strang", 0.01)]
SOLVERS = {"split": split_odeint, "strang": strang_odeint}

for s in a.splits:
    t0 = time.time()
    rec = R.run_one(dict(opt), ds, s, dev)
    out = dict(run_id=a.run_id, dataset=a.dataset, train=a.train, h=a.h, lam=a.lam, split=s, status=rec.get("status"),
               train_test_acc=rec.get("test_acc"), sec_per_epoch=rec.get("sec_per_epoch"),
               fwd_nfe_per_epoch=rec.get("fwd_nfe_per_epoch"), epochs_run=rec.get("epochs_run"), evals={})
    if rec.get("test_acc") is None:
        out["wall_sec"] = time.time() - t0
        with open(a.out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(out) + "\n")
        print(f"[{a.dataset} {a.train} lam={a.lam}] split={s} training failed: {rec.get('status')}", flush=True)
        continue
    m = holder["m"]
    data = ds.data.to(dev)
    m.eval()
    tm = data.test_mask
    preds = {}
    with torch.no_grad():
        for meth, h in EVALS:
            key = f"{meth}_{h:g}"
            if meth in SOLVERS:
                m.odeblock.test_integrator = SOLVERS[meth]
            else:
                m.odeblock.test_integrator = odeint
                m.odeblock.opt["method"] = meth
            m.odeblock.opt["step_size"] = h
            try:
                lg = m(data.x)
                fin = bool(torch.isfinite(lg).all())
                preds[key] = lg[tm].argmax(1) if fin else None
                out["evals"][key] = float(preds[key].eq(data.y[tm]).float().mean()) if fin else None
            except Exception as e:  # noqa: BLE001
                preds[key], out["evals"][key] = None, None
                out.setdefault("errors", {})[key] = str(e)[:100]
        m.resetNFE()
    ref = out["evals"].get("split_0.001")
    ref2 = out["evals"].get("split_0.0005")
    out["ref_acc"] = ref
    out["gap"] = None if ref is None else rec["test_acc"] - ref
    out["ref_converged_diff"] = None if ref is None or ref2 is None else abs(ref - ref2)
    if preds.get("split_0.001") is not None:
        out["disagree"] = {k: float((v != preds["split_0.001"]).float().mean()) for k, v in preds.items() if v is not None}
    out["wall_sec"] = time.time() - t0
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(out) + "\n")
    g = "-" if out["gap"] is None else f"{out['gap']*100:+.1f}"
    print(f"[{a.dataset} {a.train} lam={a.lam}] split={s} acc={rec['test_acc']*100:.1f} ref={'-' if ref is None else f'{ref*100:.1f}'} "
          f"gap={g} refconv={out['ref_converged_diff']} s/ep={rec.get('sec_per_epoch')}", flush=True)
