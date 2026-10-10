"""
diag_ac.py — GREAD-AC(Allen-Cahn) 발산 원인 진단. GREAD/src 에서 실행.

A. 스윕 결과에서 AC 설정별 상태 / 발산 epoch / 오류 메시지 요약
B. 학습 전(초기화 직후) 같은 초기 상태 H(0)를 여러 적분기로 풀면서 max|H(t)| 궤적 기록
   + 연속 동역학의 폭주 시간 추정: 반응항 기여가 -b_c (x^2-1) x 이므로 b_c<0 채널은 +|b_c| x^3 로 자람
     -> 스칼라 근사 dx/dt = |b| x^3 의 폭주 시간 t* = 1 / (2 |b| x0^2)
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rd_sweep as R  # noqa: E402  (torch.load 호환 패치 포함)
from torchdiffeq import odeint  # noqa: E402
from GNN import GNN  # noqa: E402
from data import get_dataset  # noqa: E402
from heterophilic import get_fixed_splits  # noqa: E402
from utils import ROOT_DIR  # noqa: E402

# ---------------- A ----------------
rows = [json.loads(l) for l in open(os.path.join(ROOT_DIR, "results", "sweep.jsonl"), encoding="utf-8") if l.strip()]
groups = defaultdict(list)
for r in rows:
    if r["reaction"] != "allen-cahn":
        continue
    v = r["tol_scale"] if r["method"] == "dopri5" else r["step_size"]
    groups[(r["method"], float(v))].append(r)
print("== A. GREAD-AC 설정별 상태 (10 splits)")
for k in sorted(groups):
    rs = groups[k]
    st = dict(Counter(r["status"] for r in rs))
    ep = [r["nan_epoch"] for r in rs if r.get("nan_epoch")]
    errs = dict(Counter((r.get("error") or "")[:70] for r in rs if r.get("error")))
    print(f"{k[0]:6s} {k[1]:>8g}  {st}  diverge_epoch={'-' if not ep else (min(ep), int(np.median(ep)), max(ep))}  {errs}")

# ---------------- B ----------------
print("\n== B. 초기화 직후 H(0)에서의 궤적 max|H(t)| (split 0, seed 0)")
a = argparse.Namespace(dataset="Cora", reaction="allen-cahn", method="rk4", step_size=0.5, tol_scale=1.0,
                       max_nfe=10 ** 8, epoch=None, time=None)
opt = R.build_opt(a)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt["device"] = device
dataset = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"])
R.set_seed(0)
dataset.data = get_fixed_splits(dataset.data, "Cora", 0)
data = dataset.data.to(device)
model = GNN(opt, dataset, device).to(device)
model.eval()
T = float(opt["time"])

with torch.no_grad():
    x0 = model.encoder(data.x)
    model.odeblock.set_x0(x0)
    f = model.odeblock.odefunc
    f.Beta = f.set_Beta()
    f.attention_weights = model.odeblock.get_attention_weights(x0)
    b = torch.diagonal(f.Beta)
    neg = b < 0
    print(f"T={T:.3f}  hidden={b.numel()}  b<0 채널 {int(neg.sum())}개  |x0| max={x0.abs().max():.3f} "
          f"mean={x0.abs().mean():.3f}")
    if neg.any():
        xs = x0[:, neg].abs()
        tstar = 1.0 / (2.0 * b[neg].abs().unsqueeze(0) * xs.clamp_min(1e-8) ** 2)
        print(f"스칼라 근사 폭주시간 t* = 1/(2|b|x0^2): min={tstar.min():.4f}  "
              f"하위1%={torch.quantile(tstar.flatten().float().cpu(), 0.01):.4f}  "
              f"t*<T 인 (노드,채널) 비율={(tstar < T).float().mean():.4f}")

    def report(name, method, t, **kw):
        f.nfe = 0
        try:
            traj = odeint(f, x0, t, method=method, **kw)
            m = traj.abs().flatten(1).amax(dim=1)
            m = torch.where(torch.isfinite(m), m, torch.full_like(m, float("inf")))
            pts = "  ".join(f"{float(tt):.2f}:{float(v):.1e}" for tt, v in zip(t.cpu(), m.cpu()))
            bad = [float(tt) for tt, v in zip(t.cpu(), m.cpu()) if not np.isfinite(float(v)) or float(v) > 1e6]
            print(f"[{name}] NFE={f.nfe}  첫 폭주(>1e6 또는 inf) t={bad[0] if bad else '-'}\n    {pts}")
        except Exception as e:  # dopri5 underflow 등
            print(f"[{name}] 실패 NFE={f.nfe}: {type(e).__name__}: {str(e)[:120]}")

    def fixed_grid(h):
        g = list(np.arange(0.0, T, h))
        if T - g[-1] > 1e-9:
            g.append(T)
        return torch.tensor(g, dtype=x0.dtype, device=device)

    dense = torch.linspace(0, T, 16, dtype=x0.dtype, device=device)
    report("rk4 h=0.05 (기준해)", "rk4", dense, options={"step_size": 0.05})
    report("rk4 h=0.5", "rk4", fixed_grid(0.5), options={"step_size": 0.5})
    report("euler h=0.1", "euler", fixed_grid(0.1), options={"step_size": 0.1})
    report("euler h=0.5", "euler", fixed_grid(0.5), options={"step_size": 0.5})
    report("euler h=1", "euler", fixed_grid(1.0), options={"step_size": 1.0})
    report("euler h=2", "euler", fixed_grid(2.0), options={"step_size": 2.0})
    report("dopri5 tol x1e5", "dopri5", dense, atol=1e-7 * 1e5, rtol=1e-9 * 1e5)
