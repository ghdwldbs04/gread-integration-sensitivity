"""
summarize_bs.py - summary of the GREAD-BS / Fisher / Zeldovich solver-transfer runs (bs_transfer.py logs).
Per reaction x dataset x training solver: trained / failed, accuracy with the training solver, reference accuracy,
gap, reference convergence, model-agnostic risk indices kappa_gen and q_h, Euler h=1 accuracy of the same weights.
Then: Spearman rho(kappa_gen, gap), rho(q_h, gap) over all models, and the kappa_gen > 2 rule as a classifier of
"gap > 2 %p". A missing reference (non-finite fine-step RK4) means the CONTINUOUS model itself diverges.
Usage: python summarize_bs.py ../results/bs_XXXX.jsonl ../results/rx_XXXX.jsonl
"""
import json
import math
import statistics as st
import sys


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


R = []
for path in sys.argv[1:]:
    R += [json.loads(l) for l in open(path) if l.strip()]
print(f"lines {len(R)} from {sys.argv[1:]}")
mean = lambda v: st.mean(v) if v else float("nan")
P = lambda v: "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:5.1f}"

DS = ["cornell", "wisconsin", "texas", "chameleon", "film", "Cora"]
print(f"\n{'reaction':9s} {'dataset':10s} {'train':16s} ok/fail refNaN  acc(train) acc(ref)  gap  refconv  kappa_gen   q_h    acc(Euler1)")
for rx in ["bspm", "fisher", "zeldovich"]:
    for d in DS:
        for tr in sorted(set((r["train"], r["method"], r["h"]) for r in R if r["reaction"] == rx and r["dataset"] == d)):
            rs = [r for r in R if r["reaction"] == rx and r["dataset"] == d and (r["train"], r["method"], r["h"]) == tr]
            ok = [r for r in rs if r.get("train_test_acc") is not None]
            refnan = [r for r in ok if r.get("ref_acc") is None]
            g = [r for r in ok if r.get("gap") is not None]
            rc = [r["ref_converged_diff"] for r in g if r.get("ref_converged_diff") is not None]
            e1 = [r["evals"].get("euler_1") for r in ok if r["evals"].get("euler_1") is not None]
            print(f"{rx:9s} {d:10s} {tr[1]+' '+format(tr[2], 'g')+' ('+tr[0][:3]+')':16s} {len(ok):2d}/{len(rs)-len(ok):<2d}  {len(refnan):2d}    "
                  f"{P(mean([r['train_test_acc'] for r in ok]))}     {P(mean([r['ref_acc'] for r in g]))}  {P(mean([r['gap'] for r in g]))}  "
                  f"{(max(rc)*100 if rc else float('nan')):5.1f}  {mean([r['kappa_gen'] for r in ok if 'kappa_gen' in r]):9.3g} "
                  f"{mean([r['q_h_mean'] for r in ok if 'q_h_mean' in r]):8.3g}   {P(mean(e1))} ({len(e1)}/{len(ok)} finite)")

print("\n== rank correlation over all trained models with a finite reference ==")
for name, sel in {"all": lambda r: True, "euler-trained": lambda r: r["method"] == "euler",
                  "nonlinear (fisher, zeldovich)": lambda r: r["reaction"] != "bspm", "bspm": lambda r: r["reaction"] == "bspm"}.items():
    g = [r for r in R if r.get("gap") is not None and "kappa_gen" in r and sel(r)]
    print(f"  {name:30s} n={len(g):3d} rho(kappa_gen,gap)={spearman([r['kappa_gen'] for r in g], [r['gap'] for r in g]):+.3f} "
          f"rho(q_h,gap)={spearman([r['q_h_mean'] for r in g], [r['gap'] for r in g]):+.3f}")

print("\n== rule 'kappa_gen > 2' vs 'gap > 2 %p' (Euler-trained, finite reference) ==")
g = [r for r in R if r.get("gap") is not None and "kappa_gen" in r and r["method"] == "euler"]
tp = sum(r["kappa_gen"] > 2 and r["gap"] > 0.02 for r in g)
fp = sum(r["kappa_gen"] > 2 and r["gap"] <= 0.02 for r in g)
fn = sum(r["kappa_gen"] <= 2 and r["gap"] > 0.02 for r in g)
tn = sum(r["kappa_gen"] <= 2 and r["gap"] <= 0.02 for r in g)
print(f"  kappa>2 & gap>2: {tp}   kappa>2 & gap<=2: {fp}   kappa<=2 & gap>2: {fn}   kappa<=2 & gap<=2: {tn}")
errs = [r for r in R if "risk_error" in r]
if errs:
    print(f"\nrisk errors: {len(errs)}; first: {errs[0]['risk_error']}")
