"""
paper_figs.py - export the paper tables (CSV) and figures (PNG/PDF) from the raw logs. Run from GREAD/src.
  Fig. 1: stiffness index kappa vs solver-transfer gap (one point per trained model, transfer.jsonl)
  Fig. 2: precise (reference) accuracy and gap per training method (consist_*.jsonl)
Usage: python paper_figs.py --transfer ../results/transfer.jsonl --consist ../results/consist_1015_1731.jsonl --out ../results/paper
"""
import argparse
import csv
import json
import math
import os
import statistics as st

p = argparse.ArgumentParser()
p.add_argument("--transfer", default="../results/transfer.jsonl")
p.add_argument("--consist", default="../results/consist_1015_1731.jsonl")
p.add_argument("--out", default="../results/paper")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

DS = ["cornell", "wisconsin", "texas", "chameleon", "film", "Cora"]
NAME = {"cornell": "Cornell", "wisconsin": "Wisconsin", "texas": "Texas", "chameleon": "Chameleon", "film": "Film", "Cora": "Cora"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


# ---------- table 1: one row per trained model (transfer.jsonl) ----------
T = [r for r in load(a.transfer) if r.get("train_test_acc") is not None and r.get("gap") is not None]
with open(os.path.join(a.out, "table_transfer_splits.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "train", "train_h", "split", "kappa", "q_h", "train_acc", "ref_acc", "gap"])
    for r in T:
        w.writerow([r["dataset"], r["train"], r["train_h"], r["split"], f"{r['kappa_eval']:.4g}",
                    "" if r.get("q_h_mean") is None else f"{r['q_h_mean']:.4g}",
                    f"{r['train_test_acc']*100:.2f}", f"{r['ref_acc']*100:.2f}", f"{r['gap']*100:.2f}"])

# ---------- table 2: method summary (consist.jsonl) ----------
C = [r for r in load(a.consist) if r.get("ref_acc") is not None]
METHODS = [("Euler", lambda r: r["train"] == "euler" and r["lam"] == 0),
           ("Euler + consistency (λ=0.1)", lambda r: r["train"] == "euler" and r["lam"] == 0.1),
           ("Euler + consistency (λ=1)", lambda r: r["train"] == "euler" and r["lam"] == 1.0),
           ("Strang (ours)", lambda r: r["train"] == "strang")]
summ = {}
with open(os.path.join(a.out, "table_methods.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "method", "n", "train_acc", "ref_acc", "ref_acc_sd", "gap", "gap_sd", "sec_per_epoch", "ref_converged_diff_max"])
    for ds in DS:
        for mname, cond in METHODS:
            rs = [r for r in C if r["dataset"] == ds and cond(r)]
            if not rs:
                continue
            ref = [r["ref_acc"] * 100 for r in rs]
            gap = [r["gap"] * 100 for r in rs]
            sec = [r["sec_per_epoch"] for r in rs if r.get("sec_per_epoch") is not None]
            rc = [r["ref_converged_diff"] for r in rs if r.get("ref_converged_diff") is not None]
            summ[(ds, mname)] = (st.mean(ref), sd(ref), st.mean(gap), sd(gap))
            w.writerow([ds, mname, len(rs), f"{st.mean(r['train_test_acc'] for r in rs)*100:.1f}", f"{st.mean(ref):.1f}",
                        f"{sd(ref):.1f}", f"{st.mean(gap):.1f}", f"{sd(gap):.1f}",
                        f"{st.mean(sec):.4f}" if sec else "", f"{max(rc):.4f}" if rc else ""])

print("\n== precise accuracy (gap) ==")
print(f"{'dataset':10s} " + " | ".join(f"{m:28s}" for m, _ in METHODS))
for ds in DS:
    cells = []
    for m, _ in METHODS:
        v = summ.get((ds, m))
        cells.append(f"{v[0]:5.1f}±{v[1]:4.1f} ({v[2]:+5.1f})".ljust(28) if v else "-".ljust(28))
    print(f"{ds:10s} " + " | ".join(cells))

# ---------- figures ----------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# Fig. 1
fig, ax = plt.subplots(figsize=(3.4, 2.4))
style = [("euler", "#d1495b", "o", "Euler (h=1, 0.5)"), ("rk4", "#8a8a8a", "s", "RK4 (h=0.25)"),
         ("split", "#2e6fb2", "^", "Split (1st order)"), ("strang", "#2e8b57", "D", "Strang (2nd order)")]
for tr, c, mk, lab in style:
    rs = [r for r in T if r["train"] == tr]
    if rs:
        ax.scatter([max(r["kappa_eval"], 1e-2) for r in rs], [r["gap"] * 100 for r in rs],
                   s=11, c=c, marker=mk, alpha=0.75, label=lab, edgecolors="none")
ax.axvline(2, color="#555", ls=":", lw=0.8, zorder=0)
ax.axhline(0, color="#aaa", lw=0.6, zorder=0)
ax.set_xscale("log")
lo = min(r["gap"] for r in T) * 100
ax.set_ylim(lo - 6, None)
ax.text(2.4, lo - 4, "κ = 2 (Euler stability limit)", fontsize=6, color="#555", va="center")
ax.set_xlabel("stiffness index κ of the learned dynamics")
ax.set_ylabel("solver-transfer gap (%p)")
ax.legend(fontsize=6, loc="upper left", frameon=True, framealpha=1, edgecolor="none", borderpad=0.3,
          handletextpad=0.2, labelspacing=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig1_kappa_gap.{ext}"), dpi=300)
plt.close(fig)

# Fig. 2
cols = ["#d1495b", "#f0a35e", "#e8c547", "#2e8b57"]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
x = list(range(len(DS)))
wd = 0.2
handles = []
for k, ((m, _), c) in enumerate(zip(METHODS, cols)):
    xs = [i + (k - 1.5) * wd for i in x]
    acc = [summ.get((ds, m), (math.nan,) * 4) for ds in DS]
    mean = [v[0] for v in acc]
    lower = [min(v[1], v[0]) for v in acc]                 # error bar never goes below 0 %
    upper = [min(v[1], 100 - v[0]) for v in acc]
    hb = axes[0].bar(xs, mean, wd, yerr=[lower, upper], color=c, label=m, error_kw=dict(lw=0.5, capsize=1))
    axes[1].bar(xs, [v[2] for v in acc], wd, color=c)
    handles.append(hb)
axes[0].set_ylim(0, 100)
for ax, yl, t in zip(axes, ["precise test accuracy (%)", "solver-transfer gap (%p)"], ["(a)", "(b)"]):
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[d] for d in DS], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(yl)
    ax.set_title(t, loc="left", fontsize=8)
axes[1].axhline(0, color="#888", lw=0.6)
fig.legend(handles, [m for m, _ in METHODS], loc="upper center", ncol=4, fontsize=7, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig2_methods.{ext}"), dpi=300)
plt.close(fig)

print(f"\nsaved to {a.out}: " + ", ".join(sorted(os.listdir(a.out))))
