"""
toy_node.py — 그래프 없는 일반 Neural ODE에서도 같은 현상이 나는지 확인 (일반화 실험). 아무 폴더에서나 실행 가능.

모델: x(2차원, two-moons) -> 인코더 Linear(2,d) -> dh/dt = MLP_tanh(h) [+ b ⊙ (h - h^3)] -> 디코더 Linear(d,2)
  - mlp     : 3차항 없음 (전역 립시츠 → 이론상 유한시간 폭주 불가)
  - cubic   : AC형 3차항, b 부호 자유 (b<0 채널에서 ±1 불안정 → 폭주 가능)
  - cubic+  : 3차항, b = softplus(w) > 0 (불안정 평형 제거)
각 모델 x 적분기(Euler h=1, Euler h=0.25, RK4 h=0.5) x 시드 10개 학습 → 발산율, 정확도,
학습 후 가중치 고정 기준해(RK4 h=0.01) 추론 시 유한 여부·정확도.
사용: python toy_node.py --out ../results/toy_node.jsonl
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torchdiffeq import odeint

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--iters", type=int, default=400)
p.add_argument("--T", type=float, default=4.0)
p.add_argument("--d", type=int, default=32)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--out", default="toy_node.jsonl")
p.add_argument("--solvers", default="euler:1.0,euler:0.25,rk4:0.5")
p.add_argument("--kinds", default="mlp,cubic,cubic+")
a = p.parse_args()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X, y = make_moons(2000, noise=0.2, random_state=0)
X = (X - X.mean(0)) / X.std(0)
X = torch.tensor(X, dtype=torch.float32, device=dev)
y = torch.tensor(y, device=dev)
tr, te = slice(0, 1000), slice(1000, 2000)


class Field(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))
        self.w = nn.Parameter(torch.empty(d).uniform_(-1, 1))

    def forward(self, t, h):
        out = self.net(h)
        if self.kind != "mlp":
            b = F.softplus(self.w) if self.kind == "cubic+" else self.w
            out = out + b * (h - h ** 3)
        return out


class Model(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.enc, self.f, self.dec = nn.Linear(2, d), Field(d, kind), nn.Linear(d, 2)

    def forward(self, x, method, h):
        t = torch.tensor([0.0, a.T], device=x.device)
        if method == "dopri5":
            hT = odeint(self.f, self.enc(x), t, method="dopri5", rtol=1e-5, atol=1e-6)[-1]
        else:
            hT = odeint(self.f, self.enc(x), t, method=method, options={"step_size": h})[-1]
        return hT, self.dec(hT)


SOLVERS = [(x.split(":")[0], float(x.split(":")[1])) for x in a.solvers.split(",")]
KINDS = a.kinds.split(",")
rows = []
for kind in KINDS:
    for method, h in SOLVERS:
        for s in range(a.seeds):
            torch.manual_seed(s)
            m = Model(a.d, kind).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=a.lr)
            status, fail_it = "ok", None
            for it in range(1, a.iters + 1):
                try:
                    _, lg = m(X[tr], method, h)
                    loss = F.cross_entropy(lg, y[tr])
                except (AssertionError, RuntimeError):
                    status, fail_it = "solver_error", it; break
                if not torch.isfinite(loss):
                    status, fail_it = "nan", it; break
                opt.zero_grad(); loss.backward(); opt.step()
            rec = dict(kind=kind, method=method, h=h, seed=s, status=status, fail_iter=fail_it)
            if status == "ok":
                with torch.no_grad():
                    _, lg = m(X[te], method, h)
                    rec["acc"] = float((lg.argmax(1) == y[te]).float().mean())
                    hr, lr_ = m(X[te], "rk4", 0.01)
                    rec["ref_finite"] = bool(torch.isfinite(hr).all())
                    rec["ref_acc"] = float((lr_.argmax(1) == y[te]).float().mean()) if rec["ref_finite"] else None
                    rec["x0max"] = float(m.enc(X[te]).abs().max())
                    rec["bmin"] = float((F.softplus(m.f.w) if kind == "cubic+" else m.f.w).min())
            rows.append(rec)
            with open(a.out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")

print(f"T={a.T}, d={a.d}, iters={a.iters}, seeds={a.seeds}")
print(f"{'model':8s} {'solver':12s} {'diverged':>9s} {'fail_iter':>12s} {'acc(own)':>9s} {'REF finite':>10s} {'acc(REF)':>9s}")
for kind in KINDS:
    for method, h in SOLVERS:
        g = [r for r in rows if r["kind"] == kind and r["method"] == method and r["h"] == h]
        ok = [r for r in g if r["status"] == "ok"]
        fi = sorted(r["fail_iter"] for r in g if r["fail_iter"])
        acc = f"{np.mean([r['acc'] for r in ok])*100:.1f}" if ok else "-"
        rf = sum(1 for r in ok if r.get("ref_finite"))
        racc = [r["ref_acc"] for r in ok if r.get("ref_acc") is not None]
        print(f"{kind:8s} {method+' h='+str(h):12s} {len(g)-len(ok):>4d}/{len(g):<4d} "
              f"{(str(fi[0])+'-'+str(fi[-1])) if fi else '-':>12s} {acc:>9s} {str(rf)+'/'+str(len(ok)):>10s} "
              f"{(f'{np.mean(racc)*100:.1f}' if racc else '-'):>9s}")
