"""
diag_channel.py — 발산 직전 가중치에서 '어느 (노드, 채널)이 폭주하는가'를 직접 확인. GREAD/src 에서 실행.

1) AC를 RK4 h=0.5로 E epoch 학습(split 0; 스윕에서 9 epoch에 NaN) → 가중치 고정
2) train 모드(dropout) 초기 상태 x0로 기준해(RK4 h=0.01)를 풀어 (노드,채널)별 첫 폭주 시각(|H|>10 또는 inf) 기록
3) 폭주한 원소들의 반응 계수 b 부호, |x0|>1 비율
4) 확산항을 뺀 반응-only 모델 dx/dt=b(x-x^3)의 폭주 시각, 해석해 t*=ln(x0^2/(x0^2-1))/(2|b|)와 비교
5) 같은 상태를 Euler h=1 / RK4 h=0.5로 풀었을 때의 max|H|
사용: python diag_channel.py --epochs 8
"""
import argparse
import copy
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
from utils import ROOT_DIR  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--split", type=int, default=0)
p.add_argument("--thr", type=float, default=10.0)
a = p.parse_args()

ns = argparse.Namespace(dataset="Cora", reaction="allen-cahn", method="rk4", step_size=0.5, tol_scale=1.0,
                        max_nfe=10 ** 8, epoch=None, time=None)
opt = R.build_opt(ns)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt["device"] = device
ds = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"])
R.set_seed(a.split)
ds.data = get_fixed_splits(ds.data, "Cora", a.split)
data = ds.data.to(device)
model = GNN(copy.deepcopy(opt), ds, device).to(device)
optim = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=opt["lr"], weight_decay=opt["decay"])
for ep in range(1, a.epochs + 1):
    loss, _, _ = R.train_step(model, optim, data)
    R.evaluate(model, data)
    print(f"epoch {ep} loss {loss:.3f}")
T = float(opt["time"])

with torch.no_grad():
    torch.manual_seed(1234)
    model.train()
    f = model.odeblock.odefunc
    x0 = model.encoder(data.x)
    model.odeblock.set_x0(x0)
    f.Beta = f.set_Beta()
    f.attention_weights = model.odeblock.get_attention_weights(x0)
    b = torch.diagonal(f.Beta)
    t = torch.linspace(0, T, 380, dtype=x0.dtype, device=device)

    def first_cross(traj):
        bad = (~torch.isfinite(traj)) | (traj.abs() > a.thr)        # [time, node, ch]
        hit = bad.any(0)
        idx = torch.where(hit, bad.float().argmax(0), torch.full_like(hit, -1, dtype=torch.long))
        tt = torch.where(hit, t[idx.clamp_min(0)], torch.full_like(x0, float("inf")))
        return hit, tt

    f.nfe = 0
    full = odeint(f, x0, t, method="rk4", options={"step_size": 0.01})
    hit_f, t_f = first_cross(full)
    # 원소별 곱 (행렬곱 @diag(b)는 inf*0=NaN으로 같은 노드의 다른 채널을 오염시킴)
    react_only = odeint(lambda _t, x: (x - x ** 3) * b, x0, t, method="rk4", options={"step_size": 0.01})
    hit_r, t_r = first_cross(react_only)

    bb = b.unsqueeze(0).expand_as(x0)
    ax0 = x0.abs()
    pred = torch.where((bb < 0) & (ax0 > 1),
                       torch.log(ax0 ** 2 / (ax0 ** 2 - 1).clamp_min(1e-12)) / (2 * bb.abs()),
                       torch.full_like(x0, float("inf")))

    n = x0.numel()
    print(f"\nT={T:.2f}, 원소 수={n}, b<0 채널 {int((b < 0).sum())}/{b.numel()}, |x0|>1 원소 {int((ax0 > 1).sum())}")
    for name, hit, tt in (("전체 GREAD(기준해)", hit_f, t_f), ("반응-only", hit_r, t_r)):
        k = int(hit.sum())
        if k == 0:
            print(f"[{name}] 폭주 원소 없음"); continue
        print(f"[{name}] 폭주 원소 {k}개 | 그중 b<0 {float((bb[hit] < 0).float().mean())*100:.1f}% | "
              f"|x0|>1 {float((ax0[hit] > 1).float().mean())*100:.1f}% | 첫 폭주 t={float(tt.min()):.3f}")
    t0 = float(t_f.min())
    early = hit_f & (t_f <= t0 + 1e-6)
    print(f"[전체 GREAD 최초 폭주 원소(t={t0:.2f})] {int(early.sum())}개 | b<0 {float((bb[early] < 0).float().mean())*100:.1f}% | "
          f"|x0|>1 {float((ax0[early] > 1).float().mean())*100:.1f}% | 해석해 t*<T 원소 포함 {int((early & torch.isfinite(pred) & (pred < T)).sum())}개")
    first_nodes = early.any(1); seed_nodes = (torch.isfinite(pred) & (pred < T)).any(1)
    print(f"[노드 단위] 최초 폭주 노드 {int(first_nodes.sum())}개 중 t*<T 원소를 가진 노드 {int((first_nodes & seed_nodes).sum())}개 / 그런 노드 전체 {int(seed_nodes.sum())}개")
    m = torch.isfinite(pred) & hit_r
    if m.any():
        err = (t_r[m] - pred[m]).abs()
        print(f"[해석해 vs 반응-only] 예측 t* 최솟값 {float(pred.min()):.3f}, 관측 {float(t_r.min()):.3f}, "
              f"원소별 |오차| 중앙값 {float(err.median()):.3f} (시간 격자 {float(t[1]-t[0]):.3f})")
    pc = int((torch.isfinite(pred) & (pred < T)).sum())
    print(f"[해석해] t*<T 로 예측된 원소 {pc}개, 그중 전체 모델에서도 폭주 {int((hit_f & torch.isfinite(pred) & (pred < T)).sum())}개")
    for m_, h_ in (("rk4", 0.5), ("euler", 1.0)):
        tr = odeint(f, x0, torch.tensor([0.0, T], device=device), method=m_, options={"step_size": h_})
        v = tr.abs().amax()
        print(f"[{m_} h={h_}] max|H| = {float(v) if torch.isfinite(v) else 'inf'}")
