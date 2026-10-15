"""
analyze_more.py - follow-up analyses on the existing logs (no training). Run from GREAD/src.
  (0) effective step: GREAD integrates over [0, T] with T from the published hyperparameters, so the largest Euler
      step is dt = min(h, T) (Cornell T = 0.116, Film T = 0.307). kappa is rescaled to kappa_eff = kappa * dt / h
      and the Spearman correlations are recomputed.
  (1) reverse transfer: weights trained with split / Strang, re-evaluated with Euler h = 1. If they collapse, the
      structure-preserving models are stiff too, and the integrator (not the stiffness) decides the outcome.
  (2) order ablation: first-order split (diffusion by Euler) vs Strang (exact diffusion, 2nd order).
  (3) step robustness of Strang-trained weights: Strang h = 2, 1, 0.5, 0.25 vs the reference.
  (4) paired tests over the 10 splits (consistency run): Strang precise accuracy vs Euler precise accuracy, and vs
      the accuracy Euler itself reports (its training-solver accuracy).
Usage: python analyze_more.py --transfer ../results/transfer.jsonl --consist ../results/consist_1015_1731.jsonl
"""
import argparse
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gread_params import best_params_dict  # noqa: E402


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = math.sqrt(sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry))
    return num / den if den else float("nan")


p = argparse.ArgumentParser()
p.add_argument("--transfer", default="../results/transfer.jsonl")
p.add_argument("--consist", default="../results/consist_1015_1731.jsonl")
a = p.parse_args()

DS = ["cornell", "wisconsin", "texas", "chameleon", "film", "Cora"]
TT = {d: float(best_params_dict[d]["time"]) for d in DS}
P = lambda v: "   -" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:5.1f}"
mean = lambda v: st.mean(v) if v else float("nan")

T = [json.loads(l) for l in open(a.transfer) if l.strip()]
T = [r for r in T if r.get("train_test_acc") is not None and r.get("gap") is not None]
for r in T:
    r["kappa_eff"] = r["kappa_eval"] * min(r["train_h"], TT[r["dataset"]]) / r["train_h"]

print("== (0) integration horizon T and effective step ==")
for d in DS:
    print(f"  {d:10s} T={TT[d]:.3f}  Euler h=1 -> largest step {min(1.0, TT[d]):.3f}")
print("\n   split-level Spearman with kappa_eff (dt = min(h, T))")
for name, rs in {"euler h=1": [r for r in T if r["train"] == "euler" and r["train_h"] == 1.0],
                 "euler h=1,0.5": [r for r in T if r["train"] == "euler"],
                 "split+strang": [r for r in T if r["train"] in ("split", "strang")]}.items():
    print(f"  {name:14s} n={len(rs):3d} rho(kappa,gap)={spearman([r['kappa_eval'] for r in rs], [r['gap'] for r in rs]):+.3f} "
          f"rho(kappa_eff,gap)={spearman([r['kappa_eff'] for r in rs], [r['gap'] for r in rs]):+.3f}")
print("   per dataset kappa_eff (mean): " + ", ".join(
    f"{d} E={mean([r['kappa_eff'] for r in T if r['dataset']==d and r['train']=='euler' and r['train_h']==1.0]):.4g}"
    f"/S={mean([r['kappa_eff'] for r in T if r['dataset']==d and r['train']=='strang']):.4g}" for d in DS))

print("\n== (1) reverse transfer: trained with X, evaluated with Euler h=1 (same weights) ==")
print(f"  {'dataset':10s} {'train':7s} n  kappa_eff  acc(train) acc(ref) acc(Euler1) ref-Euler1 Euler1 finite")
for d in DS:
    for tr in ["euler", "split", "strang"]:
        rs = [r for r in T if r["dataset"] == d and r["train"] == tr and r["train_h"] == 1.0]
        if not rs:
            continue
        e1 = [r["evals"].get("euler_1.0") for r in rs]
        fin = [v for v in e1 if v is not None]
        drop = [r["ref_acc"] - r["evals"]["euler_1.0"] for r in rs if r["evals"].get("euler_1.0") is not None]
        print(f"  {d:10s} {tr:7s} {len(rs):2d} {mean([r['kappa_eff'] for r in rs]):9.3g} "
              f"{P(mean([r['train_test_acc'] for r in rs]))}      {P(mean([r['ref_acc'] for r in rs]))}    "
              f"{P(mean(fin))}       {P(mean(drop))}     {len(fin)}/{len(rs)}")

print("\n== (2) order ablation: split (1st order, Euler diffusion) vs Strang (2nd order, exact diffusion) ==")
print(f"  {'dataset':10s} split: ref acc / gap      strang: ref acc / gap")
for d in DS:
    row = []
    for tr in ["split", "strang"]:
        rs = [r for r in T if r["dataset"] == d and r["train"] == tr]
        row.append(f"{P(mean([r['ref_acc'] for r in rs]))} / {P(mean([r['gap'] for r in rs]))} (n={len(rs)})")
    print(f"  {d:10s} " + "     ".join(row))

print("\n== (3) Strang-trained weights, other Strang steps (acc - ref, %p) ==")
keys = ["strang_2.0", "strang_1.0", "strang_0.5", "strang_0.25", "split_1.0", "rk4_0.25", "euler_0.1"]
print(f"  {'dataset':10s} " + " ".join(f"{k:>11s}" for k in keys))
for d in DS:
    rs = [r for r in T if r["dataset"] == d and r["train"] == "strang"]
    cells = []
    for k in keys:
        v = [r["evals"][k] - r["ref_acc"] for r in rs if r["evals"].get(k) is not None]
        n_fail = sum(r["evals"].get(k) is None for r in rs)
        cells.append(f"{mean(v)*100:+6.1f}" + (f"({n_fail}x)" if n_fail else "    "))
    print(f"  {d:10s} " + " ".join(f"{c:>11s}" for c in cells))

print("\n== (4) paired tests over splits (consistency run) ==")
C = [json.loads(l) for l in open(a.consist) if l.strip()]
C = [r for r in C if r.get("ref_acc") is not None]
try:
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001
    wilcoxon = None
print(f"  {'dataset':10s} Strang-Euler(ref) wins  p      | Strang(ref)-Euler(reported) wins  p")
for d in DS:
    E = {r["split"]: r for r in C if r["dataset"] == d and r["train"] == "euler" and r["lam"] == 0}
    S = {r["split"]: r for r in C if r["dataset"] == d and r["train"] == "strang"}
    sp = sorted(set(E) & set(S))
    d1 = [S[s]["ref_acc"] - E[s]["ref_acc"] for s in sp]
    d2 = [S[s]["ref_acc"] - E[s]["train_test_acc"] for s in sp]
    def pv(x):
        if wilcoxon is None or all(v == 0 for v in x):
            return float("nan")
        return wilcoxon(x).pvalue
    print(f"  {d:10s} {mean(d1)*100:+6.1f} {sum(v > 0 for v in d1):2d}/{len(sp)} {pv(d1):.4f} | "
          f"{mean(d2)*100:+6.1f} {sum(v > 0 for v in d2):2d}/{len(sp)} {pv(d2):.4f}")
