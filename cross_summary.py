"""cross_summary.py — crosseval.jsonl 요약 표. 사용: python cross_summary.py ../results/crosseval.jsonl"""
import json
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "../results/crosseval.jsonl"
df = pd.DataFrame([json.loads(l) for l in open(src, encoding="utf-8") if l.strip()])
df["eval"] = np.where(df["eval_method"] == "dopri5",
                      "dopri5 tol×" + df["eval_tol"].map(lambda v: f"{v:g}" if pd.notna(v) else ""),
                      df["eval_method"] + " h=" + df["eval_h"].map(lambda v: f"{v:g}" if pd.notna(v) else ""))
df.loc[df["is_ref"], "eval"] = "REF rk4 h=0.05"
order = list(dict.fromkeys(df["eval"]))
for tr, g in df.groupby("train", sort=False):
    rows = []
    for ev in order:
        e = g[g["eval"] == ev]
        if e.empty:
            continue
        ok = e[e["eval_error"].isna()]
        rows.append({
            "eval": ev + (" (self)" if e["is_self"].any() else ""),
            "finite": f"{len(ok)}/{len(e)}",
            "acc": f"{ok['test_acc'].mean()*100:.2f}±{ok['test_acc'].std(ddof=1)*100:.2f}" if len(ok) > 1 else "-",
            "flip%(vs REF)": f"{ok['flip_test'].mean()*100:.2f}" if "flip_test" in ok and ok["flip_test"].notna().any() else "-",
            "flip%(vs train)": f"{ok['flip_vs_train'].mean()*100:.2f}" if "flip_vs_train" in ok and ok["flip_vs_train"].notna().any() else "-",
            "relErr": f"{ok['rel_err'].mean():.3g}" if "rel_err" in ok and ok["rel_err"].notna().any() else "-",
            "NFE": f"{ok['eval_nfe'].mean():.0f}" if len(ok) else "-",
            "errors": ",".join(sorted(set(e["eval_error"].dropna().str.split(":").str[0]))),
        })
    n_tr = g["split"].nunique()
    print(f"\n## train = {tr}  (학습 성공 split {n_tr}개)\n")
    print(pd.DataFrame(rows).to_markdown(index=False))
