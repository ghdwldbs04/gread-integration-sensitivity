# Results: solver-dependent dynamics in reaction-diffusion GNNs

Model: GREAD-AC with a sign-constrained reaction coefficient (b = softplus(w) > 0), so the continuous dynamics are provably bounded.
Hyperparameters: the published GREAD-BS settings, shared by every solver. 10 fixed geom-gcn splits per dataset.
**Solver-transfer gap** = test accuracy with the training solver − test accuracy of the *same weights* re-integrated with a converged reference (reaction-exact splitting, h = 0.001).

## RQ1. Does the stiffness of the learned dynamics predict the gap?

Split-level Spearman correlation between a risk index computed on the trained weights and the gap (`analyze_gap.py`, `transfer.jsonl`).

| Training solver | n models | ρ(κ, gap) | ρ(q_h, gap) |
|---|---|---|---|
| Euler h = 1 | 60 | **+0.84** | **+0.81** (n = 50) |
| Euler h = 1 and 0.5 | 110 | +0.71 | +0.64 |
| Structure-preserving (split, Strang) | 120 | −0.23 | −0.26 |

κ = h · max|λ(J_F − D L)| (dominated by the reaction term b(1 − 3x²)); q_h = (h²/4)‖J_F F‖, the leading term of the Euler step-doubling mismatch (one JVP). q_h is missing for Texas (its configuration uses a constant-attention ODE block; to be fixed).

The gap follows κ, not graph heterophily:

| Dataset | Edge homophily | κ (Euler h = 1) | Gap, Euler (pp) | Gap, Strang (pp) |
|---|---|---|---|---|
| Cornell | 0.123 | 74,528 | 42.2 | 0.0 |
| Wisconsin | 0.178 | 36,171 | 36.3 | 0.2 |
| Texas | 0.061 | 27,255 | 19.5 | 0.0 |
| Chameleon | 0.230 | 145 | 1.9 | 0.0 |
| Film | 0.217 | 15 | 0.0 | 0.0 |
| Cora | 0.804 | 2.3 | −0.1 | 0.0 |

**Reading:** stiffness predicts the collapse only under Euler training. Structure-preserving training reaches comparable stiffness (κ ≈ 45–230) without any gap, so the problem is how the integrator treats stiffness, not stiffness itself.

## RQ2–RQ3. Structure-preserving integration vs. generic fixes

Precise (reference) test accuracy, gap in parentheses, mean of 10 splits (`consist.py`, run `consist_1015_1731`). Step-doubling consistency adds λ‖z_h − z_{h/2}‖² on the logits (same weights, input and dropout mask).

| Dataset | Euler h = 1 | + consistency λ = 0.1 | + consistency λ = 1 | **Strang (proposed)** |
|---|---|---|---|---|
| Cornell | 18.6 (42.7) | 24.3 (37.8) | 31.1 (32.2) | **72.7 (0.0)** |
| Wisconsin | 52.5 (24.3) | 48.0 (−4.1) | 48.0 (−3.7) | **79.4 (0.2)** |
| Texas | 56.2 (19.2) | 58.9 (−1.9) | 58.9 (−2.7) | **74.3 (0.3)** |
| Chameleon | 57.5 (2.2) | 52.9 (−0.2) | 53.2 (0.7) | **57.6 (−0.1)** |
| Film | 31.6 (0.2) | 31.4 (0.1) | 31.1 (0.1) | **37.1 (0.0)** |
| Cora | 87.9 (−0.2) | 87.8 (−0.1) | 86.9 (−0.5) | **87.8 (0.0)** |

- Consistency regularization closes the gap mainly by lowering the training-solver accuracy (Wisconsin 76.9 → 44), not by raising the precise accuracy. h and h/2 both lie in Euler's unstable region, so they can agree with each other while both stay far from the exact flow.
- Other generic fixes (`transfer.jsonl`): Euler h = 0.5 keeps the gap (Texas 18.6, Cornell 22.7) and diverges on Wisconsin; RK4 h = 0.25 diverges during training on Texas, Wisconsin and Film and reaches 55.9% on Cornell.
- Strang closes the gap on all six datasets and gives the best precise accuracy (Cora tied).
- Cost (s/epoch): Euler 0.003–0.010, consistency 0.007–0.028, Strang 0.006–0.074 (largest on Chameleon).

## Reliability

- Reference convergence: |acc(h = 0.001) − acc(h = 0.0005)| = 0.0 for every model.
- Reproducibility: two independent training runs give the same Euler gaps (Cornell 42.2 / 42.7, Texas 19.5 / 19.2).

## Limitations

One model family (GREAD-AC); consistency regularization tested at two λ values; six small-to-medium graphs; guarantees cover forward boundedness only.
