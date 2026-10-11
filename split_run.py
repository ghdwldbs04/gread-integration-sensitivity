"""
split_run.py — 제안법: 부호 제약 AC 반응항(b = softplus(w) > 0) + 정확해 Strang 분할 적분기. GREAD/src 에서 실행.

한 step(dt):  반응 dt/2 (정확해) -> 확산 dt (명시적 Euler) -> 반응 dt/2 (정확해)
  AC 반응 dx/dt = b(x - x^3) 의 정확해: x(t) = x e^{bt} / sqrt(1 + x^2 (e^{2bt} - 1))
  b > 0 이고 dt*alpha <= 1 이면 ||H_n||_inf <= max(||H_0||_inf, 1)  (step 크기와 무관한 이산 최대값 원리)
  b < 0 이면 정확해가 연속 해의 폭주 시각에 정확히 무한대가 된다 (거짓 안정을 만들지 않음).
학습 후 같은 가중치를 기준해(RK4 h=--ref_h)로 다시 풀어 정확도·유한성도 기록한다.
사용: python split_run.py --beta_pos --step_size 1.0
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


def split_odeint(func, x, t, method=None, options=None, **kw):
    f = getattr(func, "odefunc", func)
    h = float(options["step_size"])
    T = float(t[-1] - t[0])
    n = max(1, math.ceil(T / h - 1e-9))
    dt = T / n
    alpha = f.alpha_train if f.opt["no_alpha_sigmoid"] else torch.sigmoid(f.alpha_train)
    b = torch.diagonal(f.Beta)

    def react(z, tau):
        e = torch.exp(b * tau)
        return z * e / torch.sqrt(1 + z * z * (e * e - 1))

    z = x
    for _ in range(n):
        z = react(z, dt / 2)
        z = z + dt * alpha * (f.sparse_multiply(z) - z)
        z = react(z, dt / 2)
    f.nfe += n
    return torch.stack([x, z])


p = argparse.ArgumentParser()
p.add_argument("--dataset", default="Cora")
p.add_argument("--beta_pos", action="store_true")
p.add_argument("--step_size", type=float, default=1.0)
p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
p.add_argument("--ref_h", type=float, default=0.05)
p.add_argument("--out", default="../results/split.jsonl")
a = p.parse_args()

ns = argparse.Namespace(dataset=a.dataset, reaction="allen-cahn", method="euler", step_size=a.step_size,
                        tol_scale=1.0, max_nfe=5000, epoch=None, time=None, beta_pos=a.beta_pos)
opt = R.build_opt(ns)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt["device"] = dev
ds = R.get_dataset(opt, os.path.join(R.ROOT_DIR, "data"), opt["not_lcc"])

holder = {}
_GNN = R.GNN


def GNN_split(*args, **kw):
    m = _GNN(*args, **kw)
    m.odeblock.train_integrator = m.odeblock.test_integrator = split_odeint
    holder["m"] = m
    return m


R.GNN = GNN_split
for s in a.splits:
    t0 = time.time()
    rec = R.run_one(dict(opt), ds, s, dev)
    m = holder["m"]
    if rec.get("test_acc") is not None:
        m.odeblock.test_integrator = odeint
        m.odeblock.opt["method"], m.odeblock.opt["step_size"] = "rk4", a.ref_h
        (_, _, te), _, finite = R.evaluate(m, ds.data.to(dev))
        rec.update(ref_finite=finite, ref_test_acc=te if finite else None)
    rec.update(dataset=a.dataset, reaction="allen-cahn" + ("+bpos" if a.beta_pos else ""), method="split",
               step_size=a.step_size, ref_h=a.ref_h, wall_sec=time.time() - t0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    acc = f"{rec['test_acc']*100:.2f}" if rec.get("test_acc") is not None else "-"
    ref = rec.get("ref_test_acc")
    print(f"[split] bpos={a.beta_pos} h={a.step_size} split={s} status={rec['status']} test={acc} "
          f"ref={'-' if ref is None else f'{ref*100:.2f}'} fNFE/ep={rec['fwd_nfe_per_epoch']} "
          f"s/ep={rec['sec_per_epoch']}", flush=True)
