"""
summarize_verify.py - summary of verify_rx.py (review checks). Usage: python summarize_verify.py ../results/verify_X.jsonl
Per model the reference ladder (RK4 with N = 1000 ... 64000 steps) is classified as
  finite      : finite at every N
  blowup      : non-finite at every N and the blow-up time t_blow changes < 5% between the two largest N
  numerical   : non-finite at small N but finite at the largest N (RK4 stability failure, not a real blow-up)
  unclear     : anything else
"""
import json
import math
import statistics as st
import sys


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


def classify(r):
    lad = [r["ref"][k] for k in sorted(r["ref"], key=int)]
    fin = [e["finite"] for e in lad]
    if all(fin):
        return "finite"
    if fin[-1] and not fin[0]:
        return "numerical"
    if not any(fin):
        tb = [e["t_blow"] for e in lad if e["t_blow"] is not None]
        if len(tb) >= 2 and abs(tb[-1] - tb[-2]) / tb[-1] < 0.05:
            return "blowup"
    return "unclear"


R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
R = [r for r in R if r.get("train_test_acc") is not None]
mean = lambda v: st.mean(v) if v else float("nan")
print(f"models {len(R)}; same-weights check (re-evaluated training-solver acc == logged acc): "
      f"{sum(abs(r['reeval_train_solver_acc'] - r['train_test_acc']) < 1e-9 for r in R)}/{len(R)}; "
      f"epochs run: {sorted(set(r['epochs_run'] for r in R))}")
print(f"\n{'reaction':12s} {'dataset':10s} class(finite/blowup/numerical/unclear)  acc(Euler)  acc(ref,largest N)  "
      f"t_blow(largest N)  analytic min t*  T      scalar%   kappa_R  kappa_gen  rho_E   dz(largest N)")
for rx in ["fisher", "zeldovich", "allen-cahn"]:
    for d in ["cornell", "wisconsin", "texas", "chameleon", "film", "Cora"]:
        rs = [r for r in R if r["reaction"] == rx and r["dataset"] == d]
        if not rs:
            continue
        cl = [classify(r) for r in rs]
        last = [r["ref"][max(r["ref"], key=int)] for r in rs]
        tb = [e["t_blow"] for e in last if e["t_blow"] is not None]
        ts = [r["min_tstar"] for r in rs if r["min_tstar"] is not None]
        accs = [e["acc"] for e in last if e.get("acc") is not None]
        dz = [e["relerr_vs_prev"] for e in last if e.get("relerr_vs_prev") is not None]
        print(f"{rx:12s} {d:10s} {cl.count('finite')}/{cl.count('blowup')}/{cl.count('numerical')}/{cl.count('unclear')}"
              f"{'':27s} {mean([r['train_test_acc'] for r in rs])*100:5.1f}       "
              f"{(mean(accs)*100 if accs else float('nan')):5.1f}              {mean(tb):9.4f}          {mean(ts):9.4f}   "
              f"{rs[0]['T']:.3f}  {mean([r['frac_scalar_blowup'] for r in rs])*100:6.2f}  {mean([r['kappa_R'] for r in rs]):8.3g} "
              f"{mean([r.get('kappa_gen', float('nan')) for r in rs]):9.3g} {mean([r.get('rho_euler', float('nan')) for r in rs]):7.3g}  "
              f"{(max(dz) if dz else float('nan')):.1e}")

print("\n== blow-up time: measured (largest N) vs analytic reaction-only min t* (models classified 'blowup') ==")
b = [r for r in R if classify(r) == "blowup" and r["min_tstar"] is not None]
rel = [abs(r["ref"][max(r["ref"], key=int)]["t_blow"] - r["min_tstar"]) / r["min_tstar"] for r in b]
print(f"  n={len(b)}  median |t_blow - t*| / t* = {st.median(rel) if rel else float('nan'):.3f}   max = {max(rel) if rel else float('nan'):.3f}")
print(f"  blow-up before T/10: {sum(r['ref'][max(r['ref'], key=int)]['t_blow'] < r['T']/10 for r in b)}/{len(b)}")

print("\n== index agreement on the same checkpoint ==")
g = [r for r in R if "kappa_gen" in r]
print(f"  rho_spearman(kappa_R, kappa_gen) = {spearman([r['kappa_R'] for r in g], [r['kappa_gen'] for r in g]):+.3f} (n={len(g)}); "
      f"median kappa_gen/kappa_R = {st.median([r['kappa_gen']/r['kappa_R'] for r in g if r['kappa_R'] > 0]):.3f}")
print(f"  rho_spearman(kappa_R, rho_euler) = {spearman([r['kappa_R'] for r in g], [r['rho_euler'] for r in g]):+.3f}")
print("\n== class counts by reaction ==")
for rx in ["fisher", "zeldovich", "allen-cahn"]:
    cl = [classify(r) for r in R if r["reaction"] == rx]
    print(f"  {rx:12s} finite {cl.count('finite')}  blowup {cl.count('blowup')}  numerical {cl.count('numerical')}  unclear {cl.count('unclear')}  (n={len(cl)})")
