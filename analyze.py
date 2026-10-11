"""
analyze.py — sweep.jsonl -> 표(CSV/Markdown) + 그림(PNG)
사용: python analyze.py ../results/sweep.jsonl ../results/report
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "../results/sweep.jsonl"
outdir = sys.argv[2] if len(sys.argv) > 2 else "../results/report"
os.makedirs(outdir, exist_ok=True)

rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
df = pd.DataFrame(rows)
df = df[df["tag"].fillna("") == ""] if "tag" in df else df
import warnings; warnings.filterwarnings("ignore")
df["setting"] = np.where(df["method"] == "dopri5",
                         "tol×" + df["tol_scale"].map(lambda v: f"{v:g}" if pd.notna(v) else ""),
                         "h=" + df["step_size"].map(lambda v: f"{v:g}" if pd.notna(v) else ""))
df["sval"] = np.where(df["method"] == "dopri5", df["tol_scale"], df["step_size"])
df["ok"] = df["status"] == "ok"
df["test_pct"] = df["test_acc"] * 100

RX = {"bspm": "BS", "allen-cahn": "AC", "fisher": "F", "zeldovich": "Z", "diffusion": "Diffusion",
      "allen-cahn+bpos": "AC (b>0)", "fisher+bpos": "F (b>0)", "zeldovich+bpos": "Z (b>0)", "bspm+bpos": "BS (b>0)"}
MO = ["euler", "rk4", "dopri5"]

# 중복 실행(같은 설정·split)이 있으면 마지막 것만 사용
df = df.drop_duplicates(subset=["dataset", "reaction", "method", "sval", "split"], keep="last")


def agg(g):
    okg = g[g["ok"]]
    return pd.Series({
        "n": len(g),
        "n_fail": int((~g["ok"]).sum()),
        "fail_types": ",".join(sorted(set(g.loc[~g["ok"], "status"]))) or "",
        "acc_mean": okg["test_pct"].mean(),
        "acc_std": okg["test_pct"].std(ddof=1) if len(okg) > 1 else np.nan,
        "fwd_nfe": okg["fwd_nfe_per_epoch"].mean(),
        "sec_ep": okg["sec_per_epoch"].mean(),
        "de_xN": okg["de_xN_norm"].mean() if "de_xN_norm" in okg else np.nan,
    })


tab = (df.groupby(["dataset", "reaction", "method", "sval", "setting"]).apply(agg).reset_index()
         .sort_values(["dataset", "reaction", "method", "sval"]))
tab.to_csv(f"{outdir}/table_full.csv", index=False)


def fmt(r):
    if r["n_fail"] == r["n"]:
        return f"발산({r['fail_types']})"
    s = f"{r['acc_mean']:.2f}±{r['acc_std']:.2f}" if pd.notna(r["acc_std"]) else f"{r['acc_mean']:.2f}"
    return s + (f" [{int(r['n_fail'])}/{int(r['n'])}✗]" if r["n_fail"] else "")


tab["cell"] = tab.apply(fmt, axis=1)
lines = []
for ds, t in tab.groupby("dataset"):
    piv = t.pivot_table(index=["method", "sval", "setting"], columns="reaction", values="cell", aggfunc="first")
    piv = piv[[c for c in RX if c in piv.columns] + [c for c in piv.columns if c not in RX]].rename(columns=RX)
    piv = piv.reset_index().drop(columns="sval")
    piv["method"] = pd.Categorical(piv["method"], MO, ordered=True)
    lines.append(f"## {ds} — 테스트 정확도(%) mean±std over splits\n")
    lines.append(piv.sort_values("method").to_markdown(index=False))
    nfe = t.pivot_table(index=["method", "setting"], columns="reaction", values="fwd_nfe", aggfunc="first")
    lines.append(f"\n\n## {ds} — 학습 forward NFE / epoch\n")
    lines.append(nfe.rename(columns=RX).round(1).to_markdown())
    if t["de_xN"].notna().any():
        de = t.pivot_table(index=["method", "setting"], columns="reaction", values="de_xN", aggfunc="first")
        lines.append(f"\n\n## {ds} — 정규화 Dirichlet energy E(X_T/‖X_T‖)\n")
        lines.append(de.rename(columns=RX).round(4).to_markdown())
    # RQ1 요약: 반응항별 정확도 변화 폭 (성공한 설정만)
    rng = (t[t["n_fail"] < t["n"]].groupby(["reaction", "method"])["acc_mean"]
             .agg(lambda s: s.max() - s.min()).unstack())
    lines.append(f"\n\n## {ds} — RQ1: 설정에 따른 정확도 변화 폭(max−min, %p)\n")
    lines.append(rng.rename(index=RX).round(2).to_markdown())
    # RQ2 요약: 발산 비율
    fr = t.groupby(["reaction", "method"]).apply(lambda g: f"{int(g['n_fail'].sum())}/{int(g['n'].sum())}").unstack()
    lines.append(f"\n\n## {ds} — RQ2: 실패(NaN/NFE한도/솔버오류) run 수\n")
    lines.append(fr.rename(index=RX).to_markdown())
    lines.append("\n\n")
open(f"{outdir}/tables.md", "w", encoding="utf-8").write("\n".join(lines))

# ---------- 그림 ----------
colors = {"bspm": "#2a6fdb", "allen-cahn": "#d9480f", "diffusion": "#6c757d", "fisher": "#2b8a3e", "zeldovich": "#862e9c",
          "allen-cahn+bpos": "#f08c00", "fisher+bpos": "#69db7c", "zeldovich+bpos": "#da77f2"}
for ds, t in tab.groupby("dataset"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, m in zip(axes, MO):
        tm = t[t["method"] == m]
        for r, g in tm.groupby("reaction"):
            g = g.sort_values("sval")
            okm = g["n_fail"] < g["n"]
            ax.errorbar(g.loc[okm, "sval"], g.loc[okm, "acc_mean"], yerr=g.loc[okm, "acc_std"].fillna(0),
                        marker="o", ms=4, capsize=2, color=colors.get(r), label=RX.get(r, r))
            for x in g.loc[~okm, "sval"]:
                ax.scatter([x], [ax.get_ylim()[0] if ax.has_data() else 0], marker="x", color=colors.get(r), s=50)
        ax.set_xscale("log")
        ax.set_title({"euler": "Euler (step size h)", "rk4": "RK4 (step size h)", "dopri5": "dopri5 (tol_scale)"}[m])
        ax.set_xlabel("tol_scale" if m == "dopri5" else "h")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Test acc (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig1_acc_vs_setting_{ds}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for (r, m), g in t[t["n_fail"] < t["n"]].groupby(["reaction", "method"]):
        ax.scatter(g["fwd_nfe"], g["acc_mean"], color=colors.get(r),
                   marker={"euler": "o", "rk4": "s", "dopri5": "^"}[m], s=28,
                   label=f"{RX.get(r, r)}·{m}")
    ax.set_xscale("log")
    ax.set_xlabel("forward NFE / epoch")
    ax.set_ylabel("Test acc (%)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_acc_vs_nfe_{ds}.png", dpi=200)
    plt.close(fig)

print(open(f"{outdir}/tables.md", encoding="utf-8").read())
print("saved ->", outdir)
