"""
rd_sweep.py — GREAD 반응항 x 수치 적분 설정 스윕 러너 (KSC 2026 학부생 논문용)

GREAD/src 안에 두고 실행한다. 원본 run_GNN.py는 건드리지 않는다.

원본 run_GNN.py와 다른 점 (공정성 통제 목적)
  1) best params를 먼저 적용한 뒤, 실험 변수(reaction / method / step_size / tol_scale)를
     명령줄 값으로 '마지막에' 덮어쓴다. (원본은 best params가 명령줄 값을 덮어씀)
  2) 실제 적용된 opt 값을 결과 JSON에 그대로 기록한다 (applied_*).
  3) 분할(split)마다 torch/numpy 시드를 고정한다.
  4) NaN / NFE 한도 초과 / 솔버 오류를 잡아서 '발산'으로 기록하고 다음 실험으로 넘어간다.
  5) 학습 forward NFE, 평가 forward NFE, epoch당 시간, Dirichlet energy를 따로 잰다.

사용 예
  python rd_sweep.py --reaction bspm --method rk4 --step_size 0.5 --splits 0 1 2
  python rd_sweep.py --reaction allen-cahn --method dopri5 --tol_scale 1e3 --splits 0 1 2
  python rd_sweep.py --reaction diffusion --method euler --step_size 1.0 --splits 0 1 2
"""
import argparse
import copy
import json
import os
import re
import sys
import textwrap
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from GNN import GNN  # noqa: E402
from data import get_dataset  # noqa: E402
from heterophilic import get_fixed_splits  # noqa: E402
from gread_params import best_params_dict, shared_gread_params, hetero_params  # noqa: E402
from utils import ROOT_DIR, MaxNFEException, dirichlet_energy  # noqa: E402


def default_opt():
    """run_GNN.py의 argparse 기본값을 그대로 가져온다 (원본 파일 수정 없이)."""
    src = open(os.path.join(HERE, "run_GNN.py"), encoding="utf-8").read()
    start = src.index("parser = argparse.ArgumentParser()")
    end = src.index("args = parser.parse_args()")
    block = textwrap.dedent("    " + src[start:end])
    ns = {"argparse": argparse}
    exec(block, ns)
    return vars(ns["parser"].parse_args([]))


def build_opt(a):
    opt = default_opt()
    opt.update(best_params_dict[a.dataset])
    opt["dataset"] = a.dataset
    opt = shared_gread_params(opt)
    opt = hetero_params(opt)

    # ---- 실험 변수: 반드시 best params 이후에 덮어쓴다 ----
    if a.reaction == "diffusion":
        # 순수 확산 대조군: 같은 GREAD 함수·같은 블록에서 반응항만 0으로
        opt["reaction_term"] = "none"
        opt["beta_diag"] = False
    else:
        opt["reaction_term"] = a.reaction
    opt["method"] = a.method
    opt["step_size"] = a.step_size
    opt["tol_scale"] = a.tol_scale
    opt["max_nfe"] = a.max_nfe
    if a.epoch is not None:
        opt["epoch"] = a.epoch
    opt["no_early"] = True
    opt["adjoint"] = False
    opt["wandb"] = False
    return opt


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    model.resetNFE()
    out = model(data.x)
    loss = F.cross_entropy(out[data.train_mask], data.y.squeeze()[data.train_mask])
    if model.odeblock.nreg > 0:
        reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
        loss = loss + sum(r * c for r, c in zip(reg_states, model.regularization_coeffs) if c != 0)
    fnfe = model.getNFE()
    model.resetNFE()
    if not torch.isfinite(loss):
        return float("nan"), fnfe, 0
    loss.backward()
    optimizer.step()
    bnfe = model.getNFE()
    model.resetNFE()
    return loss.item(), fnfe, bnfe


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    model.resetNFE()
    logits = model(data.x)
    enfe = model.getNFE()
    model.resetNFE()
    finite = bool(torch.isfinite(logits).all())
    accs = []
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        pred = logits[mask].max(1)[1]
        accs.append(pred.eq(data.y[mask]).sum().item() / mask.sum().item())
    return accs, enfe, finite


@torch.no_grad()
def energy(model, data):
    model.eval()
    n = data.num_nodes
    x0 = model.encoder(data.x)
    xN = model.forward_XN(data.x)
    model.resetNFE()
    out = {}
    for name, X in (("x0", x0), ("xN", xN)):
        Xr = X / torch.norm(X, p="fro")
        out[f"de_{name}_norm"] = float(dirichlet_energy(data.edge_index, None, n, Xr))
    return out


def run_one(opt, dataset, split, device):
    set_seed(split)
    dataset.data = get_fixed_splits(dataset.data, opt["dataset"], split)
    data = dataset.data.to(device)
    model = GNN(opt, dataset, device).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=opt["lr"], weight_decay=opt["decay"])

    rec = dict(split=split, status="ok", nan_epoch=None, error=None)
    best = dict(val=-1.0, test=0.0, train=0.0, epoch=0)
    best_state = None
    fnfes, bnfes, enfes, times, losses = [], [], [], [], []
    for epoch in range(1, opt["epoch"]):
        try:
            sync(); t0 = time.time()
            loss, fnfe, bnfe = train_step(model, optimizer, data)
            sync(); times.append(time.time() - t0)
            if not np.isfinite(loss):
                rec.update(status="nan", nan_epoch=epoch)
                break
            (tr, va, te), enfe, finite = evaluate(model, data)
        except MaxNFEException:
            rec.update(status="max_nfe", nan_epoch=epoch)
            break
        except (AssertionError, RuntimeError) as e:  # dopri5 underflow, CUDA nan 등
            rec.update(status="solver_error", nan_epoch=epoch, error=str(e)[:300])
            break
        if not finite:
            rec.update(status="nan", nan_epoch=epoch)
            break
        fnfes.append(fnfe); bnfes.append(bnfe); enfes.append(enfe); losses.append(loss)
        if va > best["val"]:
            best = dict(val=va, test=te, train=tr, epoch=epoch)
            best_state = copy.deepcopy(model.state_dict())

    rec.update(
        best_val=best["val"] if best["val"] >= 0 else None,
        test_acc=best["test"] if best["val"] >= 0 else None,
        train_acc=best["train"] if best["val"] >= 0 else None,
        best_epoch=best["epoch"],
        epochs_run=len(fnfes),
        final_loss=losses[-1] if losses else None,
        fwd_nfe_per_epoch=float(np.mean(fnfes)) if fnfes else None,
        bwd_nfe_per_epoch=float(np.mean(bnfes)) if bnfes else None,
        eval_nfe=float(np.mean(enfes)) if enfes else None,
        sec_per_epoch=float(np.mean(times)) if times else None,
        total_train_sec=float(np.sum(times)) if times else None,
    )
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
            rec.update(energy(model, data))
        except Exception as e:  # energy는 선택 지표이므로 실패해도 본 결과는 유지
            rec["energy_error"] = str(e)[:200]
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Cora")
    p.add_argument("--reaction", required=True, choices=["bspm", "allen-cahn", "fisher", "zeldovich", "diffusion"])
    p.add_argument("--method", required=True, choices=["euler", "rk4", "midpoint", "dopri5"])
    p.add_argument("--step_size", type=float, default=0.5)
    p.add_argument("--tol_scale", type=float, default=1.0)
    p.add_argument("--max_nfe", type=int, default=5000)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default=os.path.join(ROOT_DIR, "results", "sweep.jsonl"))
    p.add_argument("--tag", default="")
    a = p.parse_args()

    opt = build_opt(a)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt["device"] = device
    dataset = get_dataset(opt, os.path.join(ROOT_DIR, "data"), opt["not_lcc"])

    applied = {k: opt[k] for k in ["reaction_term", "beta_diag", "method", "step_size", "tol_scale",
                                    "time", "block", "hidden_dim", "lr", "decay", "epoch", "alpha_dim",
                                    "beta_dim", "data_norm", "max_nfe"]}
    applied["atol"] = 1e-7 * opt["tol_scale"]
    applied["rtol"] = 1e-9 * opt["tol_scale"]
    print("[applied]", json.dumps(applied, ensure_ascii=False), flush=True)
    assert opt["method"] == a.method and opt["step_size"] == a.step_size, "override 실패"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for s in a.splits:
        t0 = time.time()
        rec = run_one(copy.deepcopy(opt), dataset, s, device)
        rec.update(dataset=a.dataset, reaction=a.reaction, method=a.method,
                   step_size=a.step_size if a.method != "dopri5" else None,
                   tol_scale=a.tol_scale if a.method == "dopri5" else None,
                   tag=a.tag, wall_sec=time.time() - t0, device=str(device),
                   **{f"applied_{k}": v for k, v in applied.items()})
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        acc = f"{rec['test_acc']*100:.2f}" if rec["test_acc"] is not None else "-"
        print(f"[done] {a.reaction:10s} {a.method:6s} h={a.step_size} tol={a.tol_scale} split={s} "
              f"status={rec['status']} test={acc} fNFE/ep={rec['fwd_nfe_per_epoch']} "
              f"s/ep={rec['sec_per_epoch']}", flush=True)


if __name__ == "__main__":
    main()
