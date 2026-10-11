"""같은 체크포인트(AC, RK4 h=0.5로 8 epoch)에서 step을 줄이며 폭주 관측이 어떻게 변하는지. GREAD/src 에서 실행."""
import argparse, copy, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_sweep as R
from torchdiffeq import odeint
from GNN import GNN
from data import get_dataset
from heterophilic import get_fixed_splits
from utils import ROOT_DIR

ns = argparse.Namespace(dataset="Cora", reaction="allen-cahn", method="rk4", step_size=0.5, tol_scale=1.0,
                        max_nfe=10 ** 8, epoch=None, time=None)
opt = R.build_opt(ns); dev = torch.device("cuda:0"); opt["device"] = dev
ds = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"]); R.set_seed(0)
ds.data = get_fixed_splits(ds.data, "Cora", 0); data = ds.data.to(dev)
model = GNN(copy.deepcopy(opt), ds, dev).to(dev)
optim = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=opt["lr"], weight_decay=opt["decay"])
for ep in range(8):
    R.train_step(model, optim, data); R.evaluate(model, data)
T = float(opt["time"])
with torch.no_grad():
    for mode in ("train", "eval"):
        torch.manual_seed(1234); model.train(mode == "train")
        f = model.odeblock.odefunc; x0 = model.encoder(data.x); model.odeblock.set_x0(x0)
        f.Beta = f.set_Beta(); f.attention_weights = model.odeblock.get_attention_weights(x0)
        print(f"== {mode} mode (T={T:.2f}, max|x0|={float(x0.abs().max()):.2f})")
        for m, h in [("euler", 1.0), ("euler", 0.5), ("euler", 0.25), ("euler", 0.1), ("euler", 0.05), ("euler", 0.01),
                     ("rk4", 1.0), ("rk4", 0.5), ("rk4", 0.25), ("rk4", 0.1), ("rk4", 0.01)]:
            g = list(np.arange(0.0, T, h)) + [T]
            t = torch.tensor(sorted(set(round(x, 6) for x in g)), dtype=x0.dtype, device=dev)
            traj = odeint(f, x0, t, method=m, options={"step_size": h})
            mx = traj.abs().flatten(1).amax(1)
            bad = [float(tt) for tt, v in zip(t.cpu(), mx.cpu()) if not np.isfinite(float(v)) or float(v) > 10]
            fin = float(mx[-1]) if torch.isfinite(mx[-1]) else float("inf")
            print(f"  {m:5s} h={h:<5g} steps={len(t)-1:4d}  first |H|>10 at t={bad[0] if bad else '-':<6}  max|H(T)|={fin:.3g}")
