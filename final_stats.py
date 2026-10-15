"""
final_stats.py - statistics requested in the 10/16 review (no training). Run from GREAD/src.
  (1) kappa-gap correlation WITHIN each dataset (10 splits each), so the pooled rho is not just a between-dataset
      effect; stratified permutation test (gap shuffled within dataset) for the mean within-dataset rho.
  (2) q_h with the effective step dt = min(h, T): q_h_eff = q_h * (dt/h)^2.
  (3) Holm correction of the paired Wilcoxon tests (Strang vs Euler, 6 datasets x 2 comparisons).
  (4) cost table: forward NFE per epoch, seconds per epoch, epochs, total training seconds (consistency run).
Usage: python final_stats.py --transfer ../results/transfer.jsonl --consist ../results/consist_1015_1731.jsonl
"""
import argparse
import json
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gread_params import best_params_dict  # noqa: E402


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y); mx, my = st.mean(rx), st.mean(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = math.sqrt(sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry))
    return num / den if den else float("nan")


p = argparse.ArgumentParser()
p.add_argument("--transfer", default="../results/transfer.jsonl")
p.add_argument("--consist", default="../results/consist_1015_1731.jsonl")
p.add_argument("--perm", type=int, default=10000)
a = p.parse_args()

DS = ["cornell", "wisconsin", "texas", "chameleon", "film", "Cora"]
TT = {d: float(best_params_dict[d]["time"]) for d in DS}
T = [json.loads(l) for l in open(a.transfer) if l.strip()]
T = [r for r in T if r.get("train_test_acc") is not None and r.get("gap") is not None]
for r in T:
    s = min(r["train_h"], TT[r["dataset"]]) / r["train_h"]
    r["kappa_eff"] = r["kappa_eval"] * s
    r["q_eff"] = r["q_h_mean"] * s * s if r.get("q_h_mean") is not None else None
nan = float("nan")


def within(rs, key):
    out = {}
    for d in DS:
        g = [r for r in rs if r["dataset"] == d and r.get(key) is not None]
        if len(g) >= 3:
            out[d] = (spearman([r[key] for r in g], [r["gap"] for r in g]), len(g),
                      st.mean(r["gap"] for r in g) * 100, st.pstdev([r["gap"] for r in g]) * 100)
    return out


def perm_test(rs, key, n):
    """mean within-dataset rho; null: gaps permuted within each dataset."""
    groups = [[r for r in rs if r["dataset"] == d and r.get(key) is not None] for d in DS]
    groups = [g for g in groups if len(g) >= 3]
    xs = [[r[key] for r in g] for g in groups]
    ys = [[r["gap"] for r in g] for g in groups]
    def stat(Y):
        v = [q for q in (spearman(x, y) for x, y in zip(xs, Y)) if not math.isnan(q)]
        return st.mean(v) if v else float("nan")
    obs = stat(ys)
    if math.isnan(obs):
        return obs, float("nan")
    rng = random.Random(0)
    cnt = 0
    for _ in range(n):
        Y = [rng.sample(y, len(y)) for y in ys]
        if stat(Y) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


print("== (1)+(2) kappa / q_h vs gap: pooled and WITHIN dataset ==")
for name, sel in [("Euler h=1", lambda r: r["train"] == "euler" and r["train_h"] == 1.0),
                  ("Euler h=1 and 0.5", lambda r: r["train"] == "euler"),
                  ("split + Strang", lambda r: r["train"] in ("split", "strang"))]:
    rs = [r for r in T if sel(r)]
    for key in ["kappa_eff", "q_eff"]:
        g = [r for r in rs if r.get(key) is not None]
        if len(g) < 3:
            print(f"\n  [{name}] {key}: not available")
            continue
        pooled = spearman([r[key] for r in g], [r["gap"] for r in g])
        w = within(rs, key)
        obs, pv = perm_test(rs, key, a.perm)
        print(f"\n  [{name}] {key}: pooled rho = {pooled:+.3f} (n={len(g)}); mean within-dataset rho = {obs:+.3f}, "
              f"stratified permutation p = {pv:.4f}; positive in {sum(v[0] > 0 for v in w.values() if not math.isnan(v[0]))}/{len(w)} datasets")
        for d, (rho, n, mg, sg) in w.items():
            print(f"     {d:10s} n={n:2d} rho={rho:+.3f}   gap mean {mg:6.1f} sd {sg:5.1f} %p")
    # dataset-level (6 points)
    agg = [(st.mean(r["kappa_eff"] for r in rs if r["dataset"] == d), st.mean(r["gap"] for r in rs if r["dataset"] == d))
           for d in DS if any(r["dataset"] == d for r in rs)]
    print(f"  [{name}] dataset-level rho(mean kappa_eff, mean gap) = {spearman([x for x, _ in agg], [y for _, y in agg]):+.3f} (n={len(agg)})")

print("\n== (3) paired Wilcoxon, Holm-corrected (consistency run) ==")
C = [json.loads(l) for l in open(a.consist) if l.strip()]
C = [r for r in C if r.get("ref_acc") is not None]
try:
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001
    wilcoxon = None
tests = []
for d in DS:
    E = {r["split"]: r for r in C if r["dataset"] == d and r["train"] == "euler" and r["lam"] == 0}
    S = {r["split"]: r for r in C if r["dataset"] == d and r["train"] == "strang"}
    sp = sorted(set(E) & set(S))
    for lab, diff in [("Strang(ref) - Euler(ref)", [S[s]["ref_acc"] - E[s]["ref_acc"] for s in sp]),
                      ("Strang(ref) - Euler(reported)", [S[s]["ref_acc"] - E[s]["train_test_acc"] for s in sp])]:
        pv = nan if wilcoxon is None or all(v == 0 for v in diff) else float(wilcoxon(diff).pvalue)
        tests.append([d, lab, st.mean(diff) * 100, sum(v > 0 for v in diff), len(sp), pv])
valid = sorted([t for t in tests if not math.isnan(t[5])], key=lambda t: t[5])
m = len(valid)
run = 0.0
for i, t in enumerate(valid):
    run = max(run, min(1.0, (m - i) * t[5]))
    t.append(run)
for t in tests:
    if len(t) == 6:
        t.append(nan)
for lab in ["Strang(ref) - Euler(ref)", "Strang(ref) - Euler(reported)"]:
    print(f"  {lab}")
    for t in tests:
        if t[1] == lab:
            print(f"     {t[0]:10s} diff {t[2]:+6.1f}%p  wins {t[3]}/{t[4]}  p {t[5]:.4f}  Holm p {t[6]:.4f} (m={m})")

print("\n== (4) cost (consistency run, means over splits) ==")
print(f"  {'dataset':10s} {'method':16s} fwdNFE/ep  sec/ep   epochs  train sec")
for d in DS:
    for lab, sel in [("Euler", lambda r: r["train"] == "euler" and r["lam"] == 0),
                     ("Euler+cons 0.1", lambda r: r["train"] == "euler" and r["lam"] == 0.1),
                     ("Euler+cons 1", lambda r: r["train"] == "euler" and r["lam"] == 1.0),
                     ("Strang", lambda r: r["train"] == "strang")]:
        rs = [r for r in C if r["dataset"] == d and sel(r)]
        if not rs:
            continue
        nfe = [r["fwd_nfe_per_epoch"] for r in rs if r.get("fwd_nfe_per_epoch") is not None]
        sec = [r["sec_per_epoch"] for r in rs if r.get("sec_per_epoch") is not None]
        ep = [r["epochs_run"] for r in rs if r.get("epochs_run") is not None]
        tot = [r["sec_per_epoch"] * r["epochs_run"] for r in rs if r.get("sec_per_epoch") and r.get("epochs_run")]
        print(f"  {d:10s} {lab:16s} {st.mean(nfe) if nfe else nan:8.1f}  {st.mean(sec) if sec else nan:.4f}  "
              f"{st.mean(ep) if ep else nan:6.0f}  {st.mean(tot) if tot else nan:8.2f}")
