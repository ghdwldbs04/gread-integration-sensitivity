"""
diag_train.py — GREAD-AC 학습 중 연속 동역학의 폭주 여부를 epoch마다 추적. GREAD/src 에서 실행.

학습은 지정한 적분기로 진행하고, 매 epoch 학습 스텝 후 (RNG 상태를 보존한 채로) 다음을 잰다.
  - eval 모드 / train 모드(dropout 활성) 각각의 초기 상태 x0 = encoder(X) 의 max|x0|
  - 반응 계수 b (Beta 대각) 의 최솟값, b<0 채널에서의 스칼라 폭주시간 근사 t* = 1/(2|b|x0^2) 최솟값
  - 기준 적분(RK4 h=0.05)으로 [0,T]를 풀었을 때 max|H(t)| (eval 모드 x0, train 모드 x0 각각)
  - 학습 적분기로 푼 train 모드 max|H(t)|
사용: python diag_train.py --method rk4 --step_size 0.5 --epochs 15
      python diag_train.py --method euler --step_size 1.0 --epochs 60
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rd_sweep as R  # noqa: E402
from torchdiffeq import odeint  # noqa: E402
from GNN import GNN  # noqa: E402
from data import get_dataset  # noqa: E402
from heterophilic import get_fixed_splits  # noqa: E402
from utils import ROOT_DIR, MaxNFEException  # noqa: E402


def rng_get():
    return torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def rng_set(st):
    torch.set_rng_state(st[0])
    if st[1] is not None:
        torch.cuda.set_rng_state_all(st[1])


@torch.no_grad()
def probe(model, data, T, train_mode, method, h):
    model.train(train_mode)
    f = model.odeblock.odefunc
    x0 = model.encoder(data.x)
    model.odeblock.set_x0(x0)
    f.Beta = f.set_Beta()
    f.attention_weights = model.odeblock.get_attention_weights(x0)
    b = torch.diagonal(f.Beta)
    neg = b < 0
    tstar = float("inf")
    if neg.any():
        tstar = float((1.0 / (2.0 * b[neg].abs().unsqueeze(0) * x0[:, neg].abs().clamp_min(1e-8) ** 2)).min())
    t = torch.linspace(0, T, 16, dtype=x0.dtype, device=x0.device)
    out = {}
    for name, (m, hh) in {"ref": ("rk4", 0.05), "own": (method, h)}.items():
        f.nfe = 0
        try:
            traj = odeint(f, x0, t, method=m, options={"step_size": hh})
            v = traj.abs().amax()
            out[name] = float(v) if torch.isfinite(v) else float("inf")
        except (MaxNFEException, AssertionError, RuntimeError):
            out[name] = float("inf")
    f.nfe = 0
    return dict(x0max=float(x0.abs().max()), bmin=float(b.min()), tstar=tstar, maxH_ref=out["ref"], maxH_own=out["own"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="rk4")
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "diag_train.jsonl"))
    a = p.parse_args()
    ns = argparse.Namespace(dataset="Cora", reaction="allen-cahn", method=a.method, step_size=a.step_size,
                            tol_scale=1.0, max_nfe=20000, epoch=None, time=None)
    opt = R.build_opt(ns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt["device"] = device
    dataset = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"])
    R.set_seed(a.split)
    dataset.data = get_fixed_splits(dataset.data, "Cora", a.split)
    data = dataset.data.to(device)
    model = GNN(copy.deepcopy(opt), dataset, device).to(device)
    optim = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=opt["lr"], weight_decay=opt["decay"])
    T = float(opt["time"])
    print(f"[{a.method} h={a.step_size}] T={T:.2f}")
    print(f"{'ep':>3} {'loss':>7} {'val':>6} | {'x0max(ev)':>9} {'x0max(tr)':>9} {'bmin':>6} {'t*(ev)':>8} {'t*(tr)':>8} | "
          f"{'maxH ref(ev)':>12} {'maxH ref(tr)':>12} {'maxH own(tr)':>12}")
    for ep in range(0, a.epochs + 1):
        status = ""
        if ep > 0:
            try:
                loss, _, _ = R.train_step(model, optim, data)
                (tr, va, te), _, finite = R.evaluate(model, data)
            except (MaxNFEException, AssertionError, RuntimeError) as e:
                loss, va, finite, status = float("nan"), float("nan"), False, type(e).__name__
        else:
            loss, va, finite = float("nan"), float("nan"), True
        st = rng_get()
        pe = probe(model, data, T, False, a.method, a.step_size)
        pt = probe(model, data, T, True, a.method, a.step_size)
        rng_set(st)
        model.train()
        rec = dict(method=a.method, h=a.step_size, epoch=ep, loss=loss, val=va, eval=pe, train=pt, status=status)
        with open(a.out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"{ep:>3} {loss:>7.3f} {va:>6.3f} | {pe['x0max']:>9.3f} {pt['x0max']:>9.3f} {pt['bmin']:>6.2f} "
              f"{pe['tstar']:>8.3f} {pt['tstar']:>8.3f} | {pe['maxH_ref']:>12.3g} {pt['maxH_ref']:>12.3g} {pt['maxH_own']:>12.3g} {status}",
              flush=True)
        if ep > 0 and (not np.isfinite(loss) or not finite):
            print("-> 학습 발산, 중단")
            break


if __name__ == "__main__":
    main()
