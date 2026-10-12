"""
fixed_ckpt.py — 같은 가중치에서 적분기만 바꿔 '수치적 불안정성'을 직접 확인. GREAD/src 에서 실행.

AC(b>0) 모델을 안정한 적분기로 학습(--train split|rk4, --train_h)한 뒤 가중치를 고정하고,
추론(eval 모드)과 학습 모드(dropout) 입력 각각에 대해 RK4 h = 1, 0.5, 0.25, 0.05, 0.01, dopri5, 제안 적분기로
H(T)를 구해 유한 여부, max|H(T)|, 기준해(RK4 0.01) 대비 상대오차, 테스트 정확도를 기록한다.
사용: python fixed_ckpt.py --dataset Cora --train rk4 --train_h 0.25 --splits 0 1 2
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_sweep as R  # noqa: E402
from torchdiffeq import odeint  # noqa: E402

def split_odeint(func, x, t, method=None, options=None, **kw):
    """제안 적분기 (split_run.py 와 동일): 반응 dt/2 정확해 -> 확산 Euler dt -> 반응 dt/2 정확해."""
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
        z = z + dt * alpha * (f.sparse_multiply(z) - z)
        z = react(z, dt / 2)
    f.nfe += n
    return torch.stack([x, z])


p = argparse.ArgumentParser()
p.add_argument("--dataset", default="Cora")
p.add_argument("--train", default="rk4", choices=["rk4", "split"])
p.add_argument("--train_h", type=float, default=0.25)
p.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--out", default="../results/fixed_ckpt.jsonl")
a = p.parse_args()

ns = argparse.Namespace(dataset=a.dataset, reaction="allen-cahn", method="rk4" if a.train == "rk4" else "euler",
                        step_size=a.train_h, tol_scale=1.0, max_nfe=20000, epoch=None, time=None, beta_pos=True)
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
    if a.train == "split":
        m.odeblock.train_integrator = m.odeblock.test_integrator = split_odeint
    holder["m"] = m
    return m


R.GNN = GNN_hook
EVALS = [("rk4", 1.0), ("rk4", 0.5), ("rk4", 0.25), ("rk4", 0.05), ("rk4", 0.01), ("dopri5", None), ("split", 1.0), ("split", 0.01)]

for s in a.splits:
    rec = R.run_one(dict(opt), ds, s, dev)
    if rec.get("test_acc") is None:
        print(f"split={s} training failed: {rec['status']}"); continue
    m = holder["m"]
    data = ds.data.to(dev)
    f = m.odeblock.odefunc
    T = float(opt["time"])
    out = dict(dataset=a.dataset, train=a.train, train_h=a.train_h, split=s, train_test_acc=rec["test_acc"], evals={})
    with torch.no_grad():
        for mode in ("eval", "train"):
            torch.manual_seed(1234)
            m.train(mode == "train")
            x0 = m.encoder(data.x)
            m.odeblock.set_x0(x0)
            f.Beta = f.set_Beta()
            f.attention_weights = m.odeblock.get_attention_weights(x0)
            t = torch.tensor([0.0, T], device=dev)
            res = {}
            for meth, h in EVALS:
                f.nfe = 0
                try:
                    if meth == "split":
                        hT = split_odeint(f, x0, t, options={"step_size": h})[-1]
                    elif meth == "dopri5":
                        hT = odeint(f, x0, t, method="dopri5", rtol=1e-6, atol=1e-8)[-1]
                    else:
                        hT = odeint(f, x0, t, method=meth, options={"step_size": h})[-1]
                    res[f"{meth}_{h}"] = hT
                except Exception as e:  # noqa: BLE001
                    res[f"{meth}_{h}"] = str(e)[:80]
            ref = res["rk4_0.01"]
            for k, hT in res.items():
                if isinstance(hT, str):
                    out["evals"][f"{mode}:{k}"] = dict(finite=False, error=hT); continue
                fin = bool(torch.isfinite(hT).all())
                row = dict(finite=fin, maxH=float(hT.abs().max()) if fin else None)
                if fin and not isinstance(ref, str) and torch.isfinite(ref).all():
                    row["relerr"] = float((hT - ref).norm() / ref.norm())
                out["evals"][f"{mode}:{k}"] = row
            out[f"max_x0_{mode}"] = float(x0.abs().max())
            out[f"bmax"] = float(torch.diagonal(f.Beta).max())
    # 추론 정확도: 같은 가중치로 적분기만 바꿔 모델 전체 순전파
    for meth, h in EVALS:
        key = f"eval:{meth}_{h}"
        if not out["evals"].get(key, {}).get("finite"):
            continue
        if meth == "split":
            m.odeblock.test_integrator = split_odeint
        else:
            m.odeblock.test_integrator = odeint
            m.odeblock.opt["method"] = meth
        m.odeblock.opt["step_size"] = h if h is not None else 1.0
        try:
            (_, _, te), _, fin = R.evaluate(m, data)
            out["evals"][key]["test_acc"] = te if fin else None
        except Exception:  # noqa: BLE001
            pass
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"== {a.dataset} train={a.train} h={a.train_h} split={s} acc={rec['test_acc']*100:.2f} "
          f"max|x0| eval={out['max_x0_eval']:.2f} train={out['max_x0_train']:.2f} bmax={out['bmax']:.2f}")
    for k, v in out["evals"].items():
        acc = f"{v['test_acc']*100:.2f}" if v.get("test_acc") is not None else "-"
        print(f"   {k:16s} finite={v['finite']!s:5s} maxH={v.get('maxH')} relerr={v.get('relerr')} acc={acc}")
