"""
train_switch.py — 적분기 전환(coarse-to-fine) 학습. GREAD/src 에서 실행.

처음 K epoch는 거친 Euler(h=1.0)로 학습하고, 그 뒤로는 정밀한 적분기(RK4 h=0.5)로 바꿔 학습한다.
K=0 이면 처음부터 RK4 (스윕에서 GREAD-AC는 전부 발산했던 설정).
모델 선택(best val)은 전환 이후 epoch에서만 하므로 최종 모델은 RK4로 평가된 성능이다.

사용: python train_switch.py --reaction allen-cahn --switch 0 5 20 50
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rd_sweep as R  # noqa: E402
from cross_eval import set_integrator, infer, REF  # noqa: E402
from GNN import GNN  # noqa: E402
from data import get_dataset  # noqa: E402
from heterophilic import get_fixed_splits  # noqa: E402
from utils import ROOT_DIR, MaxNFEException  # noqa: E402

COARSE = ("euler", 1.0, None)
FINE = ("rk4", 0.5, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora")
    p.add_argument("--reaction", default="allen-cahn", choices=["bspm", "allen-cahn", "diffusion"])
    p.add_argument("--switch", type=int, nargs="+", default=[0, 5, 20, 50])
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "switch.jsonl"))
    a = p.parse_args()
    ns = argparse.Namespace(dataset=a.dataset, reaction=a.reaction, method="rk4", step_size=0.5, tol_scale=1.0,
                            max_nfe=20000, epoch=None, time=None)
    opt = R.build_opt(ns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt["device"] = device
    dataset = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"])

    for K in a.switch:
        for s in a.splits:
            R.set_seed(s)
            dataset.data = get_fixed_splits(dataset.data, opt["dataset"], s)
            data = dataset.data.to(device)
            o = copy.deepcopy(opt)
            model = GNN(o, dataset, device).to(device)
            optim = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=o["lr"], weight_decay=o["decay"])
            set_integrator(model, *(COARSE if K > 0 else FINE))
            best = dict(val=-1.0, test=None, epoch=None)
            best_state, status, fail_epoch = None, "ok", None
            for epoch in range(1, o["epoch"]):
                if epoch == K + 1 and K > 0:
                    set_integrator(model, *FINE)
                try:
                    loss, _, _ = R.train_step(model, optim, data)
                    if not np.isfinite(loss):
                        status, fail_epoch = "nan", epoch; break
                    (tr, va, te), _, finite = R.evaluate(model, data)
                except (MaxNFEException, AssertionError, RuntimeError):
                    status, fail_epoch = "solver_error", epoch; break
                if not finite:
                    status, fail_epoch = "nan", epoch; break
                if epoch > K and va > best["val"]:
                    best = dict(val=va, test=te, epoch=epoch)
                    best_state = copy.deepcopy(model.state_dict())
            rec = dict(reaction=a.reaction, switch=K, split=s, status=status, fail_epoch=fail_epoch,
                       best_val=best["val"] if best_state is not None else None,
                       test_acc=best["test"], best_epoch=best["epoch"])
            if best_state is not None:
                model.load_state_dict(best_state)
                set_integrator(model, *REF)
                _, lg, _, err = infer(model, data)
                rec["ref_finite"] = err is None
                if err is None:
                    tm = data.test_mask
                    rec["ref_test_acc"] = float((lg.argmax(1)[tm] == data.y[tm]).float().mean())
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            acc = f"{rec['test_acc']*100:.2f}" if rec["test_acc"] is not None else "-"
            print(f"[K={K:3d} split={s}] {status:12s} fail_epoch={fail_epoch} test(rk4 h=0.5)={acc} "
                  f"ref_finite={rec.get('ref_finite')}", flush=True)


if __name__ == "__main__":
    main()
