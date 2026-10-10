"""
cross_eval.py — 학습된 가중치를 고정하고 추론 시 적분 설정만 바꿔 보는 실험. GREAD/src 에서 실행.

학습 설정(train config) 하나로 split별 모델을 학습(best-val 가중치 사용)한 뒤,
같은 가중치로 여러 추론 설정(eval config)에서 다음을 측정한다.
  - test 정확도
  - 예측 변경률 P_flip : 기준해(ref, RK4 h=0.05) 대비 test 노드 예측이 바뀐 비율
  - 표현 오차 E = ||H_h(T) - H_ref(T)||_F / ||H_ref(T)||_F
  - 학습 이산화 대비 예측 변경률 flip_vs_train (기준해가 발산하는 AC에서 사용)
  - finite 여부 (NaN/inf/솔버 오류)
학습 설정과 같은 eval 설정은 "self" 로 표시 (학습 때와 동일한 이산화).

사용: python cross_eval.py --reaction bspm --method rk4 --step_size 0.5
      python cross_eval.py --reaction allen-cahn --method euler --step_size 1.0
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
import rd_sweep as R  # noqa: E402  (torch.load 호환 패치 포함)
from GNN import GNN  # noqa: E402
from data import get_dataset  # noqa: E402
from heterophilic import get_fixed_splits  # noqa: E402
from utils import ROOT_DIR, MaxNFEException  # noqa: E402

REF = ("rk4", 0.05, None)
EVALS = [("euler", 0.1, None), ("euler", 0.25, None), ("euler", 0.5, None), ("euler", 1.0, None), ("euler", 2.0, None),
         ("rk4", 0.25, None), ("rk4", 0.5, None), ("rk4", 1.0, None), ("rk4", 2.0, None),
         ("dopri5", None, 1e5), ("dopri5", None, 1e3)]


def set_integrator(model, method, h, tol):
    for o in {id(model.opt): model.opt, id(model.odeblock.opt): model.odeblock.opt,
              id(model.odeblock.odefunc.opt): model.odeblock.odefunc.opt}.values():
        o["method"] = method
        o["step_size"] = h if h is not None else 0.1
    t = 1.0 if tol is None else tol
    model.odeblock.atol, model.odeblock.rtol = 1e-7 * t, 1e-9 * t


@torch.no_grad()
def infer(model, data):
    model.eval()
    model.resetNFE()
    try:
        z = model.forward_XN(data.x)
        nfe = model.getNFE()
        logits = model.m2(model.GNN_postXN(z))
    except (MaxNFEException, AssertionError, RuntimeError) as e:
        model.resetNFE()
        return None, None, None, f"{type(e).__name__}: {str(e)[:80]}"
    model.resetNFE()
    if not (torch.isfinite(z).all() and torch.isfinite(logits).all()):
        return z, logits, nfe, "nonfinite"
    return z, logits, nfe, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora")
    p.add_argument("--reaction", required=True, choices=["bspm", "allen-cahn", "diffusion"])
    p.add_argument("--method", required=True, choices=["euler", "rk4", "dopri5"])
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--tol_scale", type=float, default=1.0)
    p.add_argument("--splits", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "crosseval.jsonl"))
    a = p.parse_args()
    a.max_nfe, a.epoch, a.time = 20000, None, None

    opt = R.build_opt(a)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt["device"] = device
    dataset = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"])
    train_tag = f"{a.reaction}|{a.method}|{a.step_size if a.method != 'dopri5' else a.tol_scale}"
    print("[train]", train_tag, flush=True)

    for s in a.splits:
        R.set_seed(s)
        dataset.data = get_fixed_splits(dataset.data, opt["dataset"], s)
        data = dataset.data.to(device)
        o = copy.deepcopy(opt)
        model = GNN(o, dataset, device).to(device)
        optim = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=o["lr"], weight_decay=o["decay"])
        best_val, best_state, status = -1.0, None, "ok"
        for epoch in range(1, o["epoch"]):
            try:
                loss, _, _ = R.train_step(model, optim, data)
                if not np.isfinite(loss):
                    status = "nan"; break
                (tr, va, te), _, finite = R.evaluate(model, data)
            except (MaxNFEException, AssertionError, RuntimeError):
                status = "solver_error"; break
            if not finite:
                status = "nan"; break
            if va > best_val:
                best_val, best_state = va, copy.deepcopy(model.state_dict())
        if best_state is None:
            print(f"[split {s}] 학습 실패 ({status}) — 건너뜀", flush=True)
            continue
        model.load_state_dict(best_state)
        model.opt["max_nfe"] = 20000
        tm = data.test_mask

        set_integrator(model, *REF)
        z_ref, lg_ref, nfe_ref, err_ref = infer(model, data)
        pred_ref = lg_ref.argmax(1) if err_ref is None else None

        tr_cfg = (a.method, a.step_size if a.method != "dopri5" else None, a.tol_scale if a.method == "dopri5" else None)
        set_integrator(model, *tr_cfg)
        _, lg_self, _, err_self = infer(model, data)
        pred_self = lg_self.argmax(1) if err_self is None else None

        for cfg in [REF] + EVALS:
            set_integrator(model, *cfg)
            z, lg, nfe, err = infer(model, data)
            rec = dict(train=train_tag, split=s, train_status=status, best_val=best_val,
                       eval_method=cfg[0], eval_h=cfg[1], eval_tol=cfg[2],
                       is_self=(cfg[0] == tr_cfg[0] and cfg[1] == tr_cfg[1] and cfg[2] == tr_cfg[2]),
                       is_ref=(cfg == REF), eval_error=err, eval_nfe=nfe, ref_error=err_ref)
            if err is None:
                pred = lg.argmax(1)
                rec["test_acc"] = float((pred[tm] == data.y[tm]).float().mean())
                if pred_ref is not None:
                    rec["flip_test"] = float((pred[tm] != pred_ref[tm]).float().mean())
                    rec["rel_err"] = float(torch.norm(z - z_ref) / (torch.norm(z_ref) + 1e-12))
                if pred_self is not None:
                    rec["flip_vs_train"] = float((pred[tm] != pred_self[tm]).float().mean())
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[split {s}] trained ok (val {best_val:.3f}); ref {'ok' if err_ref is None else err_ref}", flush=True)


if __name__ == "__main__":
    main()
