"""
make_figs.py — 논문용 그림 2개 생성. GREAD/src 에서 실행.
  fig_conv_div.png : (a) 가중치 고정 후 표현오차 vs step size (log-log, 기울기 1/4)
                     (b) 반응항 x 적분 설정별 학습 발산 비율
  fig_mechanism.png: GREAD-AC 학습 중 폭주 추적 (split 0)
사용: python make_figs.py ../results ../results/figs
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = sys.argv[1] if len(sys.argv) > 1 else "../results"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(res, "figs")
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.bbox": "tight"})
C = {"bspm": "#1f5fbf", "allen-cahn": "#c8481c", "diffusion": "#6b6b6b", "euler": "#1f5fbf", "rk4": "#c8481c",
     "ref": "#222222"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------- Figure 1 ----------------
ce = load(os.path.join(res, "crosseval.jsonl"))
sw = load(os.path.join(res, "sweep.jsonl"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})

for m, mk in (("euler", "o"), ("rk4", "s")):
    for tr, ls, lab in (("bspm|rk4|0.5", "-", "BS"), ("diffusion|rk4|0.5", ":", "Diffusion")):
        rows = [r for r in ce if r["train"] == tr and r["eval_method"] == m and not r["is_ref"] and r.get("rel_err") is not None]
        hs = sorted({r["eval_h"] for r in rows})
        ys = [np.mean([r["rel_err"] for r in rows if r["eval_h"] == h]) for h in hs]
        ax1.plot(hs, ys, ls, marker=mk, ms=3.5, color=C[m], lw=1.1, label=f"{lab}, {m.upper() if m == 'rk4' else 'Euler'}")
hs = np.array([0.1, 2.0])
ax1.plot(hs, 0.05 * hs, color="#999", lw=0.8, ls="--")
ax1.text(0.13, 0.05 * 0.13 * 2.2, "slope 1", color="#777", fontsize=6.5)
h4 = np.array([0.25, 2.0])
ax1.plot(h4, 2.5e-5 * (h4 / 0.5) ** 4, color="#999", lw=0.8, ls="--")
ax1.text(0.3, 2.5e-5 * (0.3 / 0.5) ** 4 * 0.12, "slope 4", color="#777", fontsize=6.5)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("inference step size $h$ (weights frozen)")
ax1.set_ylabel(r"$\|H_h(T)-H_{ref}(T)\|_F / \|H_{ref}(T)\|_F$")
ax1.set_title("(a) Representation error vs. step size")
ax1.legend(frameon=False, loc="lower right")
ax1.grid(alpha=0.25, which="both", lw=0.4)

cfgs = [("euler", 0.1), ("euler", 0.25), ("euler", 0.5), ("euler", 1.0), ("euler", 1.5), ("euler", 2.0),
        ("rk4", 0.25), ("rk4", 0.5), ("rk4", 1.0), ("rk4", 2.0), ("dopri5", 1e5), ("dopri5", 1e3), ("dopri5", 1e1)]
rx = [("bspm", "BS"), ("allen-cahn", "AC"), ("diffusion", "Diff.")]
M = np.full((len(rx), len(cfgs)), np.nan)
for i, (r, _) in enumerate(rx):
    for j, (m, v) in enumerate(cfgs):
        g = [x for x in sw if x["reaction"] == r and x["method"] == m and x.get("tag", "") == ""
             and (float(x["tol_scale"]) == v if m == "dopri5" else float(x["step_size"]) == v)]
        if g:
            M[i, j] = np.mean([x["status"] != "ok" for x in g])
im = ax2.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
labels = [f"E {v:g}" if m == "euler" else (f"R {v:g}" if m == "rk4" else f"D {v:.0e}".replace("+0", "")) for m, v in cfgs]
ax2.set_xticks(range(len(cfgs))); ax2.set_xticklabels(labels, rotation=60, ha="right")
ax2.set_yticks(range(len(rx))); ax2.set_yticklabels([b for _, b in rx])
for i in range(len(rx)):
    for j in range(len(cfgs)):
        if not np.isnan(M[i, j]):
            ax2.text(j, i, f"{int(round(M[i, j] * 10))}", ha="center", va="center", fontsize=6.5,
                     color="white" if M[i, j] > 0.5 else "#333")
ax2.set_title("(b) Diverged runs out of 10 (E: Euler h, R: RK4 h, D: dopri5 tol×)")
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_conv_div.png"))
fig.savefig(os.path.join(out, "fig_conv_div.pdf"))
plt.close(fig)

# ---------------- Figure 2 ----------------
dt = load(os.path.join(res, "diag_train.jsonl"))
CAP = 1e9
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for meth, lab in (("rk4", "RK4 $h$=0.5"), ("euler", "Euler $h$=1")):
    rs = [r for r in dt if r["method"] == meth and r["epoch"] > 0]
    ep = [r["epoch"] for r in rs]
    own = [min(r["train"]["maxH_own"], CAP) if np.isfinite(r["train"]["maxH_own"]) else CAP for r in rs]
    a1.plot(ep, own, marker="o" if meth == "rk4" else None, ms=3, lw=1.1, color=C[meth], label=f"own integrator: {lab}")
ref_rs = [r for r in dt if r["method"] == "euler" and r["epoch"] > 0]
a1.plot([r["epoch"] for r in ref_rs],
        [min(r["train"]["maxH_ref"], CAP) if np.isfinite(r["train"]["maxH_ref"]) else CAP for r in ref_rs],
        color=C["ref"], lw=0.8, ls="--", label="reference RK4 $h$=0.05 (same weights)")
a1.set_yscale("log"); a1.set_ylim(0.5, CAP * 60)
a1.axhline(CAP, color="#bbb", lw=0.6); a1.text(13, CAP * 2.5, "inf (plotted at 1e9)", fontsize=6, color="#888", ha="left")
rk = [r for r in dt if r["method"] == "rk4" and r["status"] == "" and not np.isfinite(r["loss"]) and r["epoch"] > 0]
a1.set_xlabel("epoch"); a1.set_ylabel(r"max$_{t\in[0,T]}|H(t)|$ (train mode)")
a1.set_title("(a) GREAD-AC: blow-up of $H(t)$ during training")
a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6)
a1.grid(alpha=0.25, lw=0.4)

ev = [r for r in dt if r["method"] == "euler"]
e = [r["epoch"] for r in ev]
a2.plot(e, [r["train"]["x0max"] for r in ev], color=C["allen-cahn"], lw=1.1, label=r"max$|x_0|$ (train mode)")
a2.plot(e, [r["eval"]["x0max"] for r in ev], color=C["allen-cahn"], lw=1.1, ls=":", label=r"max$|x_0|$ (eval mode)")
a2.plot(e, [-r["train"]["bmin"] for r in ev], color="#555", lw=1.1, ls="-.", label=r"$-\min_c b_c$")
a2.axhline(1.0, color="#999", lw=0.8, ls="--")
a2.text(60, 0.96, "unstable equilibria $\\pm 1$ (for $b_c<0$)", fontsize=6, color="#777", ha="right", va="top")
a2.set_xlabel("epoch (Euler $h$=1 training)"); a2.set_title("(b) State magnitude and reaction coefficient")
a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6)
a2.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
fig.savefig(os.path.join(out, "fig_mechanism.png"))
fig.savefig(os.path.join(out, "fig_mechanism.pdf"))
plt.close(fig)
print("saved:", sorted(os.listdir(out)))
