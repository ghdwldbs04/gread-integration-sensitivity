# Results

## Setup

- **Model.** GREAD: dH/dt = α(A − I)H + r(H)·diag(b). A is a row-stochastic attention matrix computed once per forward pass.
  - Main model: Allen-Cahn r(x) = x − x³ with b = softplus(w) > 0. The reaction flow keeps [−1, 1] invariant, so the continuous solution is bounded.
- **Data.** Cornell, Wisconsin, Texas, Chameleon, Film, Cora; Geom-GCN splits, 10 per dataset.
- **Hyperparameters.** The published GREAD-BS values, shared by every integrator. The time horizon T is dataset-specific, so the largest step is dt = min(h, T).
- **Gap Δ.** Test accuracy with the training integrator minus test accuracy of the same weights under a reference integrator (eval mode).
- **AC reference.** First-order splitting (exact reaction + Euler diffusion), h = 0.001. Three checks:
  - Against h = 0.0005: accuracy difference 0.0 for every model.
  - Against an independent RK4 h = 0.01 on the same weights: 217 of 260 models stay finite. For those, accuracy differs by ≤ 0.22 %p and the final-state relative error is ≤ 1.5e-4. The other 43 make RK4 itself non-finite; they are mostly high-κ Euler-trained models.
- **Risk index.** κ = dt · max_{i,c} [b_c(3x_{0,ic}² − 1)]₊, the Euler instability risk of the reaction at the initial state.
  - κ > 2 means at least one reaction mode lies outside Euler's stability interval. This is a reaction-only, local condition.
  - A JVP power-iteration estimate of the full-Jacobian spectral scale agrees with the reaction Jacobian scale: rank correlation 0.997, median ratio 1.000.

## 1. Gap and risk index (Euler h = 1)

| Dataset | Homophily | κ Euler | κ Strang | Δ Euler (run 1 / run 2) | Δ Strang |
|---|---|---|---|---|---|
| Cornell | 0.123 | 8,619 | 7.5 | 42.2 / 42.7 | 0.0 / 0.0 |
| Wisconsin | 0.178 | 36,170 | 231 | 36.3 / 24.3 | 0.2 / 0.2 |
| Texas | 0.061 | 27,260 | 138 | 19.5 / 19.2 | 0.0 / 0.3 |
| Chameleon | 0.230 | 145 | 45 | 1.9 / 2.2 | 0.0 / −0.1 |
| Film | 0.217 | 4.6 | 51 | 0.0 / 0.2 | 0.0 / 0.0 |
| Cora | 0.804 | 2.3 | 2.5 | −0.1 / −0.2 | 0.0 / 0.0 |

**Spearman correlation of κ with the gap**

| Level | ρ | Notes |
|---|---|---|
| Dataset level | 0.83 | n = 6 |
| Model level, pooled | 0.80 | n = 60 |
| Within dataset (mean) | 0.27 | Positive in 5 of 6 datasets; stratified permutation p = 0.027 (10,000 permutations) |
| Within dataset, including h = 0.5 | 0.34 | p = 0.0004 |

Homophily also correlates with the gap at dataset level (ρ = −0.77, n = 6), so six datasets cannot separate the two. Two observations still point to κ: heterophilic Film and Chameleon have both a small gap and a small κ, and κ tracks the gap within a dataset, where homophily is fixed.

## 2. Training methods (AC, h = 1)

Reference test accuracy (%, mean ± sd over 10 splits), gap in parentheses. Every method uses the same rule: if the training loss becomes non-finite, training stops and the best-validation checkpoint is evaluated. [k] is the number of splits that finished 199 epochs without divergence, shown only when k < 10.

| Dataset | Euler | + consistency λ = 0.1 | + consistency λ = 1 | Strang |
|---|---|---|---|---|
| Cornell | 18.6±25.6 (42.7) | 24.3±27.2 (37.8) | 31.1±33.2 (32.2) | **72.7±6.8 (0.0)** |
| Wisconsin | 52.5±20.5 (24.3) | 48.0±7.0 (−4.1) [0] | 48.0±7.0 (−3.7) [0] | **79.4±4.2 (0.2)** |
| Texas | 56.2±7.1 (19.2) | 58.9±4.6 (−1.9) [0] | 58.9±4.6 (−2.7) [0] | **74.3±6.9 (0.3)** |
| Chameleon | 57.5±1.9 (2.2) | 52.9±2.6 (−0.2) [0] | 53.2±2.5 (0.7) [0] | **57.6±1.7 (−0.1)** |
| Film | 31.6±2.7 (0.2) | 31.4±3.0 (0.1) | 31.1±2.7 (0.1) | **37.1±1.3 (0.0)** |
| Cora | 87.9±1.0 (−0.2) [9] | 87.8±0.6 (−0.1) [8] | 86.9±1.3 (−0.5) [5] | 87.8±0.9 (0.0) |

**Strang vs Euler** (paired Wilcoxon, Holm-corrected, 12 tests)

- Against Euler's reference accuracy, Strang is higher on Cornell (+54.1), Wisconsin (+26.9), Texas (+18.1) and Film (+5.5); Holm p ≤ 0.046.
- Against the accuracy Euler reports with its own integrator, Strang is higher on Cornell (+11.4) and Film (+5.3), lower on Chameleon (−2.1), and not different elsewhere.

**Strang models are stiff, but Strang handles it**

- RK4 h = 0.25 on Strang-trained weights is non-finite on Wisconsin, Texas, Chameleon and Film (10/10 each).
- Strang itself agrees with the reference within ±0.9 %p for h = 2, 1, 0.5, 0.25.

**Ablation.** First-order splitting (exact reaction + Euler diffusion) gives reference accuracy and gaps similar to Strang. The key ingredient is the exact reaction flow; Strang adds a forward boundedness guarantee for every step size.

**Other generic fixes**

- Euler h = 0.5 keeps the gap (Texas 18.6, Cornell 22.7 %p) and diverges on Wisconsin.
- RK4 h = 0.25 diverges during training on Texas, Wisconsin and Film.

**Cost.** Strang needs 1.5–2.2× the total training time of Euler, 4.2× on Cora and 7.7× on Chameleon, because of the diffusion heat-kernel series.

**Correction (2026-10-16).** An earlier version of this file said consistency regularisation "closes the gap by lowering training accuracy". The status field shows that those runs diverged within 1–38 epochs, and the table above reports them as such.

## 3. Scope: other reactions

**Published GREAD-BS (linear reaction)**

- Gap −0.6 to 2.2 %p across the six datasets; κ_gen mostly < 1.
- The only weak gap is Chameleon: κ_gen 2.8–4.7, gap 1.4–2.2 %p.

**Fisher r = x(1 − x) and Zeldovich r = x²(1 − x)** (unconstrained b, Euler h = 1, splits 0–2)

| Reaction | Finite | Blow-up | RK4 failure | Unresolved |
|---|---|---|---|---|
| Fisher (18) | 8 | 10 | 0 | 0 |
| Zeldovich (18) | 2 | 13 | 0 | 3* |
| AC b > 0 (18, control) | 16 | 0 | 2 | 0 |

\* The blow-up time (≈ 1e−4) is shorter than the coarsest step.

**How a blow-up was classified.** The reference is non-finite at every N from 1,000 to 64,000, and the blow-up time changes by less than 5 % between the two largest N.

- Median relative difference to the closed-form reaction blow-up time: 2.6 %.
- 15 of 23 blow up before T/10.
- Euler still reports normal accuracy for these models, e.g. Wisconsin 76–82 %, Texas 69–83 %.
- This is a numerical observation, not a proof.
- For Fisher, b > 0 alone does not guarantee boundedness, because elements with x < 0 can still blow up.

## Limitations

- One model family (GREAD) and six small-to-medium graphs.
- κ is evaluated at the initial state.
- The boundedness theorem covers the forward pass only.
- No equal-compute-budget comparison.
- No fix proposed for Fisher / Zeldovich.
