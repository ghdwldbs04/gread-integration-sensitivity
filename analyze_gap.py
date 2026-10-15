"""
analyze_gap.py — RQ1: does the stiffness of the learned dynamics track the solver-transfer gap?
Reads the solver-transfer log (one line per trained model, all splits) and reports, per training solver,
the split-level Spearman rank correlation between the risk indices (kappa, q_h) and the gap
(test accuracy with the training solver - test accuracy of the same weights under the converged reference).
Also writes a scatter plot (kappa vs gap, log x axis).
Usage: python analyze_gap.py --log ../results/transfer.jsonl --fig ../results/fig_kappa_gap.png
"""
import argparse
import json
import math
import statistics as st


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
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


p = argparse.ArgumentParser()
p.add_argument("--log", default="../results/transfer.jsonl")
p.add_argument("--fig", default="../results/fig_kappa_gap.png")
a = p.parse_args()

R = [json.loads(l) for l in open(a.log)]
ok = [r for r in R if r.get("train_test_acc") is not None and r.get("gap") is not None]
print(f"lines {len(R)}, trained-and-evaluated {len(ok)}, failed {sum(r.get('train_test_acc') is None for r in R)}")

errs = [r for r in R if "q_h_error" in r]
if errs:
    print(f"q_h errors: {len(errs)} (datasets: {sorted(set(r['dataset'] for r in errs))}); first message: {errs[0]['q_h_error']}")

print("\n== split-level Spearman rho (risk index vs gap) ==")
groups = {"euler (h=1 and 0.5)": [r for r in ok if r["train"] == "euler"],
          "euler h=1": [r for r in ok if r["train"] == "euler" and r["train_h"] == 1.0],
          "structure-preserving (split, strang)": [r for r in ok if r["train"] in ("split", "strang")],
          "all solvers": ok}
for name, rs in groups.items():
    k = [r["kappa_eval"] for r in rs]
    g = [r["gap"] for r in rs]
    q = [(r["q_h_mean"], r["gap"]) for r in rs if r.get("q_h_mean") is not None and math.isfinite(r["q_h_mean"])]
    rho_q = spearman([x for x, _ in q], [y for _, y in q]) if q else float("nan")
    print(f"  {name:38s} n={len(rs):3d}  rho(kappa,gap)={spearman(k, g):+.3f}  rho(q_h,gap)={rho_q:+.3f} (n={len(q)})")

print("\n== per dataset x training solver (means) ==")
print(f"  {'dataset':10s} {'train':10s} n  kappa      gap(%p)  disagree(%)")
for ds in ["cornell", "wisconsin", "texas", "chameleon", "film", "Cora"]:
    for tr, h in [("euler", 1.0), ("euler", 0.5), ("rk4", 0.25), ("split", 1.0), ("strang", 1.0)]:
        rs = [r for r in ok if r["dataset"] == ds and r["train"] == tr and r["train_h"] == h]
        if not rs:
            continue
        dis = [r["disagree"].get(f"{tr}_{h}") for r in rs if r.get("disagree", {}).get(f"{tr}_{h}") is not None]
        print(f"  {ds:10s} {tr+' '+str(h):10s} {len(rs):2d} {st.mean(r['kappa_eval'] for r in rs):9.1f} "
              f"{st.mean(r['gap'] for r in rs)*100:8.1f} {st.mean(dis)*100 if dis else float('nan'):10.1f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style = {"euler": ("#d1495b", "o", "Euler"), "rk4": ("#7a7a7a", "s", "RK4"),
             "split": ("#2e6fb2", "^", "Split (1st order)"), "strang": ("#2e8b57", "D", "Strang (2nd order)")}
    for tr, (c, mk, lab) in style.items():
        rs = [r for r in ok if r["train"] == tr]
        if rs:
            ax.scatter([max(r["kappa_eval"], 1e-2) for r in rs], [r["gap"] * 100 for r in rs],
                       s=14, c=c, marker=mk, alpha=0.75, label=lab, edgecolors="none")
    ax.set_xscale("log")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("stiffness index  kappa (learned weights)")
    ax.set_ylabel("solver-transfer gap (%p)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=200)
    print(f"\nfigure: {a.fig}")
except Exception as e:  # noqa: BLE001
    print(f"\n(figure skipped: {e})")
