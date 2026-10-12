# Two Causes of Divergence in Continuous-Depth Networks and a Stable Reaction-Diffusion Integrator

**Distinguishing dynamical from numerical instability, exposing solver-induced false stability, and fixing both.**

This repository contains the full experimental code for a study of why training of continuous-depth models diverges (NaN). Using a reaction-diffusion GNN ([GREAD](https://github.com/jeongwhanchoi/GREAD), ICML 2023) and a graph-free Neural ODE, we separate divergence into **finite-time blow-up of the continuous equation (dynamical)** and **step sizes outside the solver's stability region (numerical)**, predict and verify each, and propose an integration scheme that removes both (**sign constraint + exact-flow splitting**). Every number in the accompanying paper (KSC 2026, undergraduate track, under review) can be reproduced from this repository.

**Key findings**

- **High accuracy obtained with a coarse solver does not guarantee that the trained model is a valid solution of the continuous ODE.** GREAD-AC trained with Euler h=1 (87.7%) and a cubic Neural ODE (73.0%) both blow up in finite time when the same weights are integrated accurately (*false stability*).
- Of 8 GREAD reaction settings, the 4 that are provably blow-up-free all trained with accurate solvers, and the 4 predicted to blow up by the scalar condition all diverged.
- The closed-form solution predicted blow-up for 25 of 159,040 (node, channel) elements; in the reaction-only model exactly those 25 blew up (median timing error 0.014).
- Shrinking only the step on the same checkpoint removes false stability, and the observed blow-up time converges to the reference (t = 2.66).
- Dynamically stable reactions (BS, diffusion) are solver-insensitive: Euler h=2 (NFE 2) matches RK4 h=0.25 (NFE 64).
- **Proposed method** (sign constraint + exact-flow splitting): trains every split on Cora and Pubmed settings where RK4 diverges 10/10, and on extremely stiff Film, with 1–4 function evaluations per forward pass. Accuracy differs from an accurate re-integration of the same weights by at most 0.23 pp.

<p align="center">
  <img src="figs/overview.png" width="85%">
</p>

## Contents

1. [Problem](#1-problem)
2. [Theory: telling the two causes apart](#2-theory-telling-the-two-causes-apart)
3. [Proposed method: sign constraint and exact-flow splitting](#3-proposed-method-sign-constraint-and-exact-flow-splitting)
4. [Experimental setup](#4-experimental-setup)
5. [Results](#5-results)
6. [Repository layout](#6-repository-layout)
7. [Reproduction](#7-reproduction)
8. [Limitations and future work](#8-limitations-and-future-work)
9. [References](#9-references)

## 1. Problem

Continuous-depth networks define the hidden state through an ODE `dh/dt = f_θ(h)` and compute the forward pass with a numerical solver. When training diverges, the usual fix is to shrink the step or swap the solver. But divergence has two causes with different remedies.

| | Dynamical instability | Numerical instability |
|---|---|---|
| What happens | The continuous solution goes to infinity inside the integration interval | The solution is finite, but the step leaves the solver's stability region and errors are amplified |
| Shrinking the step | Does not help | Fixes it |
| Remedy | Change the model (sign of reaction coefficients, integration time T) | Smaller step, adaptive or implicit solver |

A third phenomenon sits on top: **a coarse explicit solver can hide dynamical blow-up.** It approximates [0, T] by N = ⌈T/h⌉ polynomial maps, so with small N the discrete solution stays finite even when the continuous one blows up (always finite for N = 1). Training looks normal, but the model becomes a function tied to one particular discretization. We call this **false stability**.

## 2. Theory: telling the two causes apart

We consider vector fields made of a term g that grows at most linearly plus a channel-wise polynomial reaction. In GREAD, g is the diffusion α(A − I)H; in a plain Neural ODE, it is a tanh MLP.

```text
dh/dt = g(h) + b ⊙ r(h),    ‖g(h)‖ ≤ A + B‖h‖
```

**Dynamical instability.** If r = 0, Grönwall's inequality rules out finite-time blow-up (diffusion, BS, tanh MLP). With a polynomial reaction, blow-up is decided by the sign of the leading term and the region of the initial state.

| Reaction | r(x) | b < 0 | b > 0 |
|---|---|---|---|
| Allen-Cahn (AC) | x − x³ | blows up if \|x₀\| > 1 | no blow-up |
| Fisher (F) | x − x² | blows up if x₀ > 1 | blows up if x₀ < 0 |
| Zeldovich (Z) | x² − x³ | blows up if x₀ < 0 or x₀ > 1 | no blow-up |

For AC with b < 0 and |x₀| > 1, the scalar blow-up time is

```text
t* = ln( x₀² / (x₀² − 1) ) / (2|b|)
```

The table ignores diffusion, so for the full system it is used only as a **prediction**. In a coupled system diffusion can delay or even prevent blow-up (e.g. two nodes x₁ = u, x₂ = −u with b = −1, α = 1 give du/dt = u³ − 3u, which does not blow up for |u₀| < √3).

**Theorem 1 (lower bound on graph-level blow-up time).** GREAD computes the attention A once before integration and uses channel-wise coefficients, so each channel is an independent system dh/dt = α(A − I)h + b·r(h). Since α ≥ 0 (GREAD uses α = sigmoid(·)) and on Cora and Film A is a row-wise softmax, i.e. a nonnegative row-stochastic matrix, the upper right Dini derivative of M(t) = ‖h(t)‖∞ satisfies D⁺M ≤ b(M − M³) (M may be non-differentiable when the maximizing node changes). Hence for b > 0, ‖h(t)‖∞ ≤ max(M(0), 1); for b < 0 there is no blow-up if M(0) ≤ 1, and any blow-up happens no earlier than t*(M(0), b). The scalar t* is therefore only a **lower bound** on the blow-up time of the full system.

For b > 0, AC and Zeldovich also satisfy a one-sided bound x·b·r(x) ≤ b/4 (AC) and ≤ 27b/256 (Z). Because the diffusion term is linear in H during integration, ⟨H, f(H)⟩ ≤ a + c‖H‖² holds, which **guarantees** no finite-time blow-up for the full system including diffusion.

**Numerical instability.** Even with a finite solution, errors grow when hλ leaves the stability region for a linearization eigenvalue λ. On the real axis this is −2 ≤ hλ ≤ 0 for Euler and −2.785 ≤ hλ ≤ 0 for classical RK4. For AC, λ = b(1 − 3x²), so even with b > 0 (no blow-up) the system becomes stiff for large |x| and can diverge at large steps.

**Three-step diagnosis.**

1. If training diverges, or succeeds only with a coarse solver, integrate the same weights over [0, T] with a reference solver (small step or adaptive).
2. If the reference also blows up, the cause is dynamical: locate blow-up elements with the table and t*, and change the model rather than the step. If a coarse solver trained successfully, that result is false stability.
3. If the reference is finite and a smaller step recovers training, the cause is numerical.

## 3. Proposed method: sign constraint and exact-flow splitting

The diagnosis shows that AC diverges for two reasons: (1) finite-time blow-up in channels with b < 0, and (2) stiffness at large |x| even when b > 0. The method removes each one structurally.

1. **Sign constraint:** b = softplus(w) > 0, so the bounded case of Theorem 1 holds for any value reached during training.
2. **Exact-flow splitting:** each step is split into reaction dt/2 → diffusion dt → reaction dt/2 (a Strang-type splitting). The reaction is solved exactly with its closed form; diffusion uses explicit Euler.

<p align="center">
  <img src="figs/architecture.png" width="95%">
</p>

GREAD is an encoder → ODE block → decoder model; the method only replaces the solver inside the ODE block. Because the attention A is computed once before integration, the diffusion term is linear in H during integration. In control experiments the same block is solved with Euler, RK4 or dopri5 (torchdiffeq).

```text
x(t) = x·exp(bt) / sqrt(1 + x²(exp(2bt) − 1))
```

**Theorem 2 (discrete maximum principle).** If b > 0 and 0 ≤ dt·α ≤ 1, every step of the proposed integrator satisfies ‖H(n)‖∞ ≤ max(‖H(0)‖∞, 1). The exact reaction flow keeps |x| ≤ max(|x|, 1), and the diffusion step (1 − dt·α)H + dt·α·AH is a convex combination with a nonnegative row-stochastic matrix. The guarantee is **boundedness of the forward state** only; it does not cover backpropagation or optimization stability.

- The property is **independent of the reaction's stiffness**. The only remaining condition is 0 ≤ dt·α ≤ 1 for the diffusion step; since α = sigmoid(·) ∈ (0, 1) in GREAD, dt ≤ 1 suffices. It is not independent of the step size altogether.
- Because diffusion uses Euler, the scheme is **first-order** overall (measured: shrinking h by 100× reduces the error from 6.2×10⁻² to 6.1×10⁻⁴).
- For b < 0 the exact flow detects finite-time blow-up of the reaction subproblem exactly, so unlike coarse Euler it does not pass through it with finite values. This does not imply that the full system including diffusion must blow up.
- Operator splitting and maximum-principle preservation are classical numerical-analysis techniques. The contribution here is matching them to the two diagnosed causes, applying them to Neural ODE training, and validating the result against accurate re-integration of the same weights.

## 4. Experimental setup

| Item | Setting |
|---|---|
| Model | Official GREAD code (the original `run_GNN.py` is untouched; experiments use a separate runner, `rd_sweep.py`) |
| Data | Cora, Film, Texas, Wisconsin, plus Pubmed (19,717 nodes) for the proposed method; Geom-GCN 10 splits |
| Hyperparameters | Fixed to the published per-dataset BS optimum (Cora: T = 3.79, input dropout 0.50) to isolate the effect of reaction and solver |
| Reactions | Diffusion only, BS, AC, Fisher, Zeldovich, and sign-constrained variants b = softplus(w) > 0 |
| Reaction coefficients | Channel-wise b, initialized from U(−1, 1) |
| Solvers | Euler h ∈ {0.1, 0.25, 0.5, 1, 1.5, 2}, RK4 h ∈ {0.25, 0.5, 1, 2}, dopri5 (tol_scale 1e1, 1e3, 1e5) |
| Scale | About 1,370 training runs (sweep 1: 390, sweep 2: 984) |
| Reference solution | RK4 h = 0.05 (GREAD) or 0.01 (Neural ODE) on the same weights |
| Divergence criterion | NaN in loss or output, NFE limit exceeded, or solver error |
| Reproduction check | Cora BS 87.85 ± 0.81% (paper: 88.57 ± 0.66%) |

`rd_sweep.py` differs from the original `run_GNN.py` only to keep comparisons fair:

- Best params are applied first, and the experimental variables (reaction, solver, step, tolerance, T) from the command line are applied **last**. In the original, best params overwrite the command line.
- The settings actually applied are written to the result JSON.
- torch/numpy seeds are fixed per split.
- Divergence is caught and recorded, and the sweep moves on.
- Training and evaluation NFE, time per epoch and Dirichlet energy are measured separately.

**Graph-free Neural ODE.** Two-moons classification with `Linear(2, 32) → dh/dt = MLP_tanh(h) [+ b ⊙ (h − h³)] → Linear(32, 2)`, T = 4, 10 seeds. Three models: MLP (no cubic term), cubic (free sign of b) and cubic+ (b > 0).

## 5. Results

### 5.1 Predictions vs. observations (GREAD, Cora)

Test accuracy (%, mean over 10 splits). "div." = diverged on 10/10 splits. † no blow-up guaranteed for the full system; ‡ blow-up predicted by the scalar condition.

| Reaction | Theory | RK4 h=0.25 | dopri5 | RK4 h=1 | Euler h=1 |
|---|---|---:|---:|---:|---:|
| Diffusion | stable† | 87.79 | 87.93 | 87.93 | 87.77 |
| BS | stable† | 88.11 | 87.93 | 87.89 | 87.71 |
| AC | unstable‡ | div. | div. | div. | 87.71 |
| AC (b>0) | stable† | 87.89 | 87.85 | div. | 87.91 |
| F | unstable‡ | div. | div. | div. | 87.79 |
| F (b>0) | unstable‡ | div. | div. | div. | 87.34 |
| Z | unstable‡ | div. | div. | div. | 87.36 |
| Z (b>0) | stable† | 87.79 | 87.79 | div. | 87.34 |

- As predicted, Fisher blows up on the x < 0 side even with b > 0, and diverged with every accurate solver.
- AC(b>0) and Z(b>0) diverged 10/10 only at RK4 h=1 and recovered with smaller steps: the dynamics are stable and the failure is **numerical** (stiffness).
- All 4 dynamically unstable settings trained only with coarse Euler.

### 5.2 Element-level verification of blow-up

Weights of AC trained with RK4 h=0.5 for 8 epochs were frozen (in the sweep this setting hit NaN at epoch 9).

| Item | Result |
|---|---|
| Elements predicted to blow up within T (t* < T) | 25 of 159,040 |
| Elements that actually blew up in the reaction-only model | exactly the same 25 (all b < 0, \|x₀\| > 1) |
| Median error of predicted blow-up time | 0.014 (time grid 0.01) |
| First time a blow-up sign (\|H\| > 10 or non-finite) is observed in the full model under accurate integration (RK4 h=0.01) | t = 2.66, later than the Theorem 1 lower bound 1.47; the first element to blow up belongs to the predicted set |
| Same weights, RK4 h=0.5 / Euler h=1 | max\|H\| = ∞ / 2.58 (coarse Euler hides the blow-up) |

During training, max|x₀| in training mode (dropout on) exceeded 1 at epoch 3, and the reference solution blew up from epoch 8. Euler h=1 training passed through this phase with finite values up to 1.8×10⁸. Starting with coarse Euler and switching to RK4 diverged regardless of the switching epoch (0, 5, 20, 50).

<p align="center">
  <img src="results/figs/fig_mechanism.png" width="70%">
</p>

### 5.3 Shrinking the step on the same checkpoint

The checkpoint above was integrated from training-mode inputs (max|x₀| = 1.43), recording the first time |H| > 10.

| Solver | h | First \|H\| > 10 | max\|H(T)\| |
|---|---:|---:|---:|
| Euler | 1 | none | 2.58 |
| Euler | 0.5 | none | 6.14 |
| Euler | 0.25 | 3.50 | blow-up |
| Euler | 0.1 | 3.10 | blow-up |
| Euler | 0.05 | 2.85 | blow-up |
| Euler | 0.01 | 2.70 | blow-up |
| RK4 | 1, 0.5 | 3.0 | blow-up |
| RK4 | 0.25 | 2.75 | blow-up |
| RK4 | 0.1 | 2.70 | blow-up |
| RK4 (reference) | 0.01 | 2.66 | blow-up |

Shrinking the step removes false stability and the observed time converges to the reference value 2.66. On inference-mode inputs (max|x₀| = 1.09) every setting stayed finite (max|H| ≈ 1): for this checkpoint the blow-up only arises from the training-mode state enlarged by dropout.

### 5.4 Graph-free Neural ODE (two-moons)

Accuracy (%). In parentheses: the same weights integrated with the reference solver (RK4 h=0.01).

| Model | Theory | RK4 h ≥ 0.25 | RK4 h=0.1 | dopri5 | Euler h=1 |
|---|---|---:|---:|---:|---:|
| MLP | stable | 97.6 | — | — | 97.0 (72.1) |
| cubic | unstable | div. | div. | div. | 73.0 (blow-up 10/10) |
| cubic+ (b>0) | stable, stiff | div. | 96.9 | 97.3 | 69.9 (59.4) |

The same three phenomena appear in a plain Neural ODE: cubic shows false stability, cubic+ shows numerical instability. The MLP never diverges, but the Euler h=1 model drops from 97.0 to 72.1 under the reference solver; this is solver dependence [5], not blow-up.

### 5.5 Stable dynamics are solver-insensitive

- BS and diffusion did not diverge in any of the 13 solver settings; accuracy varied by at most 0.82 pp.
- With frozen weights and only the inference solver changed, the representation error scales as h for Euler and roughly h⁴ for RK4.
- Euler h=2 (NFE 2) matched RK4 h=0.25 (NFE 64). A BS model trained with Euler h=2 kept its accuracy under the reference solver (87.59 → 87.65%).

<p align="center">
  <img src="results/figs/fig_conv_div.png" width="70%">
</p>

### 5.6 Other datasets

- **Texas, Wisconsin, Film:** BS trained normally with every solver.
- **Film:** AC and AC(b>0) diverged with accurate solvers and trained only with single-step Euler h ≥ 0.5. Training-mode max|x₀| reaches 9.46.
- **Cause identified:** on the same weights, the RK4 h=0.05 reference diverged 10/10, while RK4 h=0.002 and splitting h=0.001 references were finite and agreed in accuracy. AC(b>0) diverging even with dopri5 is caused by **stiffness**, not blow-up.

### 5.7 Same weights, different solvers (direct test of numerical instability)

AC(b>0) weights were frozen and only the solver was changed (3 splits; error = relative error vs. RK4 h=0.01).

| Solver | Cora, training-mode input | Pubmed, training-mode input | Pubmed, inference input |
|---|---|---|---|
| RK4 h=1 | non-finite 3/3 | non-finite 3/3 | non-finite 3/3 |
| RK4 h=0.5 | finite, 2.0×10⁻⁴ | non-finite 3/3 | non-finite 3/3 |
| RK4 h=0.25 | finite, 9.5×10⁻⁶ | non-finite 3/3 | finite 1/3 |
| RK4 h=0.05 | finite, 2.0×10⁻⁷ | finite 2/3 | finite 3/3 |
| dopri5 | finite, 6.8×10⁻⁷ | finite 3/3 | finite 3/3 |
| Proposed h=1 | finite 3/3 | finite 3/3 | finite 3/3 |

The continuous solution is finite; only the step leaves the stability region. Inputs with larger states (|λ| ≈ b(3x² − 1)) needed smaller steps. Cora weights were trained with RK4 h=0.25, Pubmed weights with the proposed method.

### 5.8 Proposed method

Sign-constrained AC(b>0), accuracy (%, mean ± std over 10 splits). In parentheses: the same weights under the reference solver (fine-step RK4).

| Dataset | RK4 h=1 | RK4 h=0.25 | Euler h=1 | Proposed h=1 | Paired difference (proposed − Euler) |
|---|---|---:|---:|---:|---:|
| Cora | div. | 87.89±1.17 | 88.03±1.01 (87.93) | 87.42±1.38 (87.40) | −0.60±1.36 (proposed wins 3/10) |
| Pubmed | div. | div. | 89.97±0.37 (89.82) | 89.65±0.44 (89.88) | −0.32±0.44 (2/10) |
| Film | div. | div. | 31.62±2.73 (31.51) | 37.10±0.95 (37.22) | **+5.47±2.66 (10/10)** |

- NFE per forward pass for the proposed method and Euler h=1: Cora 4, Pubmed 2, Film 1. RK4 h=0.25: Cora 64, Pubmed 28. All h = 1 runs satisfy dt ≤ 1 (the condition of Theorem 2).
- Training time per epoch on Pubmed for the same AC(b>0) model (GPU-synchronized, averaged over epochs): proposed 0.008 s, Euler h=1 0.008 s, RK4 h=0.25 0.050 s (until divergence). For reference, BS with RK4 h=0.25 takes 0.076 s.
- Cora h=2 (NFE 2) also trained (87.53%), but with dt ≈ 1.9 it may fall outside the guarantee of Theorem 2.
- Without the sign constraint, the proposed integrator also diverged 10/10 on Cora and Film: it does not pass through the reaction subproblem's blow-up with finite values.
- On Cora and Pubmed Euler is 0.3–0.6 pp higher on average, within split-to-split variation. The benefit of the method is not accuracy but training where RK4 fails, the guarantee of Theorem 2, and the gain on stiff Film (+5.5 pp).

## 6. Repository layout

```text
.
├── rd_sweep.py        # main runner: reaction × solver × split; logs divergence, NFE, time, Dirichlet energy
├── run_sweep.sh       # sweep 1 (Cora, BS/AC/diffusion × 13 solver settings × 10 splits), resumable
├── run_sweep2.sh      # sweep 2 (Fisher/Zeldovich, sign constraints, other datasets)
├── analyze.py         # sweep.jsonl → tables (CSV/Markdown) + figures
├── cross_eval.py      # frozen weights, swap the inference solver: accuracy, prediction flips, representation error
├── cross_summary.py   # summary table for cross_eval
├── diag_ac.py         # AC divergence diagnosis: sweep summary + trajectories at initialization
├── diag_train.py      # per-epoch tracking of max|x₀|, reaction coefficients, reference blow-up
├── train_switch.py    # Euler → RK4 switching during training
├── diag_channel.py    # element-level (node, channel) blow-up check vs. closed-form t*
├── step_refine.py     # same checkpoint, shrinking step sizes
├── toy_node.py        # graph-free Neural ODE (two-moons)
├── make_figs.py       # paper figures (fig_conv_div, fig_mechanism)
├── split_run.py       # proposed method (sign constraint + exact-flow splitting) with reference checks; --plain euler for the control
├── run_split.sh       # main runs of the proposed method (Cora/Film/Pubmed + baseline solvers)
├── run_split2.sh      # proposed vs. Euler with two references (fine RK4, fine splitting)
├── fixed_ckpt.py      # same weights, swap only the solver: direct test of numerical instability
├── figs/              # README diagrams (overview.png, architecture.png) and Graphviz sources (.dot)
└── results/           # result jsonl files, tables, figures
```

| Result file | Produced by |
|---|---|
| `results/sweep.jsonl` | `run_sweep.sh` |
| `results/sweep2.jsonl` | `run_sweep2.sh` |
| `results/crosseval.jsonl` | `cross_eval.py` |
| `results/switch.jsonl` | `train_switch.py` |
| `results/diag_train.jsonl` | `diag_train.py` |
| `results/report/` | `analyze.py` |
| `results/figs/` | `make_figs.py` |
| `results/split.jsonl`, `results/split_base.jsonl` | `run_split.sh` |
| `results/split2.jsonl` | `run_split2.sh` |
| `results/fixed_ckpt.jsonl` | `fixed_ckpt.py` |

## 7. Reproduction

### 7.1 Environment

Tested with Python 3.9 and CUDA 12.8 (cu128 is needed for sm_120 on RTX 50-series GPUs).

```bash
conda create -n gread python=3.9 -y
conda activate gread
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install torch-geometric==2.3.1 "numpy<2" scikit-learn==1.2.2 torchdiffeq==0.2.3 \
            pykeops numba pandas matplotlib tabulate
```

- `scikit-learn` must be pinned to 1.2.2 for the `DistanceMetric` import in the GREAD code.
- `rd_sweep.py` handles the `torch.load(weights_only=True)` default of recent PyTorch.

### 7.2 Linking with GREAD

All scripts run from GREAD's `src/` directory. The original code is not modified; scripts are linked in.

```bash
git clone https://github.com/jeongwhanchoi/GREAD.git
git clone https://github.com/ghdwldbs04/gread-integration-sensitivity.git repo
for f in repo/*.py repo/*.sh; do ln -sf "$(realpath "$f")" GREAD/src/; done
ln -sfn "$(realpath repo/results)" GREAD/results
cd GREAD/src
```

Place the Geom-GCN split files (`*_split_0.6_0.2_{0..9}.npz`) in `GREAD/src/splits/`.

### 7.3 Running the experiments (from GREAD/src)

```bash
# Sweep 1 (Cora). Resumable: finished settings are skipped
setsid nohup bash run_sweep.sh > ../results/sweep.log 2>&1 &

# Sweep 2 (reaction degree, sign constraints, other datasets)
setsid nohup bash run_sweep2.sh > ../results/sweep2.log 2>&1 &

# Single run
python rd_sweep.py --reaction allen-cahn --beta_pos --method rk4 --step_size 0.25 --splits 0 1 2
```

Main arguments of `rd_sweep.py`:

| Argument | Description |
|---|---|
| `--dataset` | Cora, film, texas, wisconsin, ... |
| `--reaction` | `bspm`, `allen-cahn`, `fisher`, `zeldovich`, `diffusion` |
| `--beta_pos` | constrain reaction coefficients to b = softplus(w) > 0 |
| `--method` | `euler`, `rk4`, `midpoint`, `dopri5` |
| `--step_size` / `--tol_scale` | fixed step / dopri5 tolerance multiplier |
| `--time` | integration time T (default: best params) |
| `--max_nfe` | runs exceeding this NFE are recorded as diverged (default 5000) |
| `--splits` | split indices to run |

```bash
# Frozen weights, swap the inference solver
python cross_eval.py --reaction bspm --method rk4 --step_size 0.5
python cross_eval.py --reaction allen-cahn --method euler --step_size 1.0
python cross_summary.py ../results/crosseval.jsonl

# Blow-up mechanism
python diag_train.py --method rk4 --step_size 0.5 --epochs 15
python diag_train.py --method euler --step_size 1.0 --epochs 60
python train_switch.py --reaction allen-cahn --switch 0 5 20 50
python diag_channel.py --epochs 8
python step_refine.py

# Graph-free Neural ODE
python toy_node.py --out ../results/toy_node.jsonl \
  --solvers euler:1.0,rk4:0.25,rk4:0.1,dopri5:0 --kinds mlp,cubic,cubic+

# Proposed method
bash run_split.sh > ../results/split.log 2>&1
bash run_split2.sh > ../results/split2.log 2>&1
python split_run.py --dataset Cora --beta_pos --step_size 1.0            # single run
python fixed_ckpt.py --dataset Cora --train rk4 --train_h 0.25 --splits 0 1 2
python fixed_ckpt.py --dataset Pubmed --train split --train_h 1.0 --splits 0 1 2

# Tables and figures
python analyze.py ../results/sweep.jsonl ../results/report
python make_figs.py ../results ../results/figs
```

## 8. Limitations and future work

**Limitations**

- Theorem 1 only gives a lower bound on the blow-up time. Whether blow-up actually occurs is predicted by the scalar condition; a graph-level condition for blow-up (an upper bound) is still missing.
- Theorem 2 guarantees boundedness of the forward state only. Stability of backpropagation and optimization is left for future work.
- The proposed integrator is limited to the AC reaction, which has a closed-form flow; it is first-order and its guarantee requires dt·α ≤ 1. The sign constraint costs little expressiveness (0.22 pp below BS at RK4 h=0.25), but on homophilous graphs accuracy is slightly below Euler.
- All GREAD experiments share the per-dataset BS optimum so that only the reaction and solver change, so they are not directly comparable with the AC-specific settings of the original paper. The claim is therefore not "AC cannot be trained with accurate solvers" but "under identical settings, stability depends on the reaction–solver combination, and the cause can be predicted".
- The Neural ODE experiments are small (10 seeds).

**Future work**

1. **A condition for blow-up (upper bound).** Derive an upper bound on the graph-level blow-up time with Kaplan's eigenvector method, to pair with the lower bound of Theorem 1.
2. **Extending the method.** Exact-flow splitting for Fisher and Zeldovich, a second-order diffusion step, larger graphs, and comparison with ACMP and IMEX solvers.

## 9. References

1. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. Neural Ordinary Differential Equations. NeurIPS, 2018.
2. E. Haber, L. Ruthotto. Stable Architectures for Deep Neural Networks. Inverse Problems, 34(1), 2017.
3. B. P. Chamberlain et al. GRAND: Graph Neural Diffusion. ICML, 2021.
4. J. Choi, S. Hong, N. Park, S.-B. Cho. GREAD: Graph Neural Reaction-Diffusion Networks. ICML, 2023.
5. K. Ott, P. Katiyar, P. Hennig, M. Tiemann. ResNet After All? Neural ODEs and Their Numerical Solution. ICLR, 2021.
6. A. Zhu, P. Jin, B. Zhu, Y. Tang. On Numerical Integration in Neural Ordinary Differential Equations. ICML, 2022.
7. H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, B. Yang. Geom-GCN: Geometric Graph Convolutional Networks. ICLR, 2020.
8. Y. Wang, K. Yi, X. Liu, Y. G. Wang, S. Jin. ACMP: Allen-Cahn Message Passing with Attractive and Repulsive Forces for Graph Neural Networks. ICLR, 2023.
9. E. Hairer, G. Wanner. Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems. Springer, 1996.
10. J. Kelly, J. Bettencourt, M. J. Johnson, D. Duvenaud. Learning Differential Equations that are Easy to Solve. NeurIPS, 2020.
11. S. Kim, W. Ji, S. Deng, Y. Ma, C. Rackauckas. Stiff Neural Ordinary Differential Equations. Chaos, 31(9), 2021.
12. E. Haber, K. Lensink, E. Treister, L. Ruthotto. IMEXnet: A Forward Stable Deep Neural Network. ICML, 2019.
13. T. K. Rusch, B. Chamberlain, J. Rowbottom, S. Mishra, M. Bronstein. Graph-Coupled Oscillator Networks. ICML, 2022.
