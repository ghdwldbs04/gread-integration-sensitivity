# 연속 깊이 신경망 학습 발산의 두 원인

**동역학적·수치적 불안정성의 구분과 거친 이산화에 의한 거짓 안정**
*Two Causes of Divergence in Continuous-Depth Neural Networks: Distinguishing Dynamical from Numerical Instability and Solver-Induced False Stability*

반응-확산 GNN([GREAD](https://github.com/jeongwhanchoi/GREAD), ICML 2023)과 그래프가 없는 Neural ODE에서, 학습이 NaN으로 발산하는 원인을 **연속 방정식 자체의 폭주(동역학적)** 와 **적분기의 안정 영역 이탈(수치적)** 로 나누어 예측하고 검증한 실험 코드입니다. KSC 2026 학부생 논문(심사 중) 실험을 모두 재현할 수 있습니다.

**핵심 결과**

- **거친 적분기로 얻은 높은 정확도는 학습된 모델이 연속 ODE의 정상적인 해라는 것을 보장하지 않는다.** Euler h=1로 87.7%를 낸 GREAD-AC와 73.0%를 낸 cubic Neural ODE는, 같은 가중치를 정밀 적분기로 풀면 연속 해가 유한 시간에 폭주한다(거짓 안정).
- GREAD 반응항 8개 설정 중 폭주 없음이 이론적으로 보장되는 4개는 모두 정밀 적분으로 학습되었고, 스칼라 조건이 폭주를 예측한 4개는 모두 발산했다.
- 해석해가 폭주를 예측한 원소 25개(159,040개 중)는 반응만 남긴 모델에서 정확히 그 25개가 폭주했고, 시각 오차 중앙값은 0.014였다.
- 같은 체크포인트에서 step만 줄이면 거짓 안정이 사라지고, 폭주 시각이 기준해(t = 2.66)로 수렴한다.
- 동역학이 안정한 반응항(BS, 확산)은 Euler h=2(NFE 2)와 RK4 h=0.25(NFE 64)의 정확도가 같았다.

## 목차

1. [문제 설정](#1-문제-설정)
2. [이론: 두 원인을 가르는 조건](#2-이론-두-원인을-가르는-조건)
3. [실험 구성](#3-실험-구성)
4. [결과](#4-결과)
5. [저장소 구조](#5-저장소-구조)
6. [재현 방법](#6-재현-방법)
7. [한계와 후속 연구](#7-한계와-후속-연구)
8. [참고문헌](#8-참고문헌)

## 1. 문제 설정

연속 깊이 신경망은 은닉 상태를 ODE `dh/dt = f_θ(h)`로 정의하고 수치 적분기로 순전파를 계산한다. 학습이 NaN으로 발산하면 보통 step을 줄이거나 적분기를 바꾸지만, 발산에는 처방이 다른 두 원인이 있다.

| | 동역학적 불안정성 | 수치적 불안정성 |
|---|---|---|
| 무슨 일이 일어나나 | 연속 해가 적분 구간 안에서 무한대로 간다 | 해는 유한하지만 step이 적분기 안정 영역을 벗어나 오차가 증폭된다 |
| step을 줄이면 | 해결되지 않는다 | 해결된다 |
| 처방 | 모델을 바꾼다(반응 계수 부호, 적분 시간 T) | step을 줄이거나 적응형·암시적 적분기를 쓴다 |

여기에 세 번째 현상이 겹친다. **거친 명시적 적분기는 동역학적 폭주를 가릴 수 있다.** [0, T]를 N = ⌈T/h⌉번의 다항 사상으로 근사하므로, N이 작으면 연속 해가 폭주해도 이산 해는 유한하게 남는다(N = 1이면 항상 유한). 학습은 정상처럼 보이지만 모델은 특정 이산화에 묶인 함수가 된다. 이 저장소는 이를 **거짓 안정(false stability)** 이라고 부른다.

## 2. 이론: 두 원인을 가르는 조건

다루는 벡터장은 선형 이하로 자라는 항 g와 채널별 다항 반응항의 합이다. GREAD에서 g는 확산항 α(A − I)H, 일반 Neural ODE에서는 tanh MLP다.

```text
dh/dt = g(h) + b ⊙ r(h),    ‖g(h)‖ ≤ A + B‖h‖
```

**동역학적 불안정성.** r = 0이면 Grönwall 부등식에 의해 해는 유한 시간에 폭주하지 않는다(확산, BS, tanh MLP). 다항 반응항이 있으면 폭주 여부는 최고차항의 부호와 초기 상태의 영역이 정한다.

| 반응항 | r(x) | b < 0 | b > 0 |
|---|---|---|---|
| Allen-Cahn (AC) | x − x³ | \|x₀\| > 1이면 폭주 | 폭주 없음 |
| Fisher (F) | x − x² | x₀ > 1이면 폭주 | x₀ < 0이면 폭주 |
| Zeldovich (Z) | x² − x³ | x₀ < 0 또는 x₀ > 1이면 폭주 | 폭주 없음 |

AC에서 b < 0, |x₀| > 1이면 스칼라 폭주 시간은 다음과 같다.

```text
t* = ln( x₀² / (x₀² − 1) ) / (2|b|)
```

위 표는 확산을 무시한 스칼라 조건이라 전체 시스템에 대해서는 **예측**으로만 쓴다. 반면 b > 0인 AC·Zeldovich는 x·b·r(x)가 위로 유계(AC ≤ b/4, Z ≤ 27b/256)다. GREAD는 어텐션 가중치를 적분 전에 한 번 계산하므로 적분 중 확산항이 H에 선형이다. 따라서 ⟨H, f(H)⟩ ≤ a + c‖H‖²가 성립하고, 확산을 포함한 전체 시스템에서도 유한시간 폭주가 없음이 **보장**된다.

**수치적 불안정성.** 해가 유한해도 선형화 고유값 λ에 대해 hλ가 안정 영역을 벗어나면 오차가 증폭된다. 실수축에서 Euler는 −2 ≤ hλ ≤ 0, 고전 RK4는 −2.785 ≤ hλ ≤ 0이다. AC에서 λ = b(1 − 3x²)이므로 b > 0으로 폭주를 없애도 |x|가 크면 강성(stiff)이 되어 큰 step에서 발산할 수 있다.

**3단계 진단 절차.**

1. 발산하거나 거친 적분기로만 학습되면, 같은 가중치로 기준 적분(작은 step 또는 적응형)을 [0, T]에서 구한다.
2. 기준 해도 폭주하면 동역학적 불안정성이다. 표의 조건과 t*로 폭주 원소를 특정하고, step이 아니라 모델을 바꾼다. 거친 적분기로 학습에 성공했다면 그 결과는 거짓 안정이다.
3. 기준 해가 유한하고 step을 줄이면 회복되면 수치적 불안정성이다.

## 3. 실험 구성

| 항목 | 설정 |
|---|---|
| 모델 | GREAD 공개 코드(원본 `run_GNN.py`는 수정하지 않고 별도 러너 `rd_sweep.py` 사용) |
| 데이터 | Cora, Film, Texas, Wisconsin (Geom-GCN 10개 분할) |
| 하이퍼파라미터 | 데이터셋별 공개 BS 최적값으로 고정(Cora: T = 3.79, 입력 dropout 0.50) |
| 반응항 | 확산(반응 없음), BS, AC, Fisher, Zeldovich, 그리고 b = softplus(w) > 0 제약 버전 |
| 반응 계수 | 채널별 b, U(−1, 1) 초기화 |
| 적분 설정 | Euler h ∈ {0.1, 0.25, 0.5, 1, 1.5, 2}, RK4 h ∈ {0.25, 0.5, 1, 2}, dopri5(tol_scale 1e1, 1e3, 1e5) |
| 규모 | 약 1,370회 학습 (1차 스윕 390회 + 2차 스윕 984회) |
| 기준해 | 같은 가중치의 RK4 h = 0.05 (GREAD), 0.01 (Neural ODE) |
| 발산 판정 | 손실·출력의 NaN, NFE 한도 초과, 적분기 오류 |
| 재현 확인 | Cora BS 87.85 ± 0.81% (원 논문 88.57 ± 0.66%) |

`rd_sweep.py`가 원본 `run_GNN.py`와 다른 점은 공정성 통제를 위한 것이다.

- best params를 먼저 적용한 뒤 실험 변수(반응항, 적분기, step, tol, T)를 명령줄 값으로 **마지막에** 덮어쓴다. 원본은 best params가 명령줄 값을 덮어쓴다.
- 실제 적용된 설정값을 결과 JSON에 그대로 기록한다.
- 분할마다 torch/numpy 시드를 고정한다.
- 발산을 잡아서 기록하고 다음 실험으로 넘어간다.
- 학습·평가 forward NFE, epoch당 시간, Dirichlet energy를 따로 잰다.

**그래프 없는 Neural ODE.** two-moons 분류에서 `Linear(2, 32) → dh/dt = MLP_tanh(h) [+ b ⊙ (h − h³)] → Linear(32, 2)`, T = 4로 학습한다. 비교 모델은 셋이다: 3차항이 없는 MLP, b 부호가 자유로운 cubic, b > 0인 cubic+.

## 4. 결과

### 4.1 이론 예측과 관측 (GREAD, Cora)

테스트 정확도(%, 10분할 평균). '발산'은 10/10 발산. † 전체 시스템에서 폭주 없음 보장, ‡ 스칼라 조건에 의한 폭주 예측.

| 반응항 | 이론 | RK4 h=0.25 | dopri5 | RK4 h=1 | Euler h=1 |
|---|---|---:|---:|---:|---:|
| 확산 | 안정† | 87.79 | 87.93 | 87.93 | 87.77 |
| BS | 안정† | 88.11 | 87.93 | 87.89 | 87.71 |
| AC | 불안정‡ | 발산 | 발산 | 발산 | 87.71 |
| AC (b>0) | 안정† | 87.89 | 87.85 | 발산 | 87.91 |
| F | 불안정‡ | 발산 | 발산 | 발산 | 87.79 |
| F (b>0) | 불안정‡ | 발산 | 발산 | 발산 | 87.34 |
| Z | 불안정‡ | 발산 | 발산 | 발산 | 87.36 |
| Z (b>0) | 안정† | 87.79 | 87.79 | 발산 | 87.34 |

- Fisher는 b > 0으로 제약해도 x < 0 쪽에서 폭주한다는 예측대로 정밀 적분에서 모두 발산했다.
- AC(b>0)와 Z(b>0)는 RK4 h=1에서만 10/10 발산했고 step을 줄이면 회복됐다. 동역학은 안정하고 강성 때문에 생긴 **수치적** 불안정성이다.
- 동역학적으로 불안정한 4개 설정은 모두 거친 Euler에서만 학습되었다.

### 4.2 폭주의 원소 단위 검증

AC를 RK4 h=0.5로 8 epoch 학습한 가중치를 고정했다. 스윕에서는 이 설정이 9 epoch에 NaN이 됐다.

| 항목 | 결과 |
|---|---|
| 해석해 t* < T 로 폭주가 예측된 원소 | 159,040개 중 25개 |
| 반응만 남긴 모델에서 실제 폭주한 원소 | 정확히 같은 25개 (모두 b < 0, \|x₀\| > 1) |
| 폭주 시각 예측 오차 중앙값 | 0.014 (시간 격자 0.01 수준) |
| 확산 포함 전체 모델의 첫 폭주 시각 | t = 1.47 → 2.66 으로 늦춰짐, 첫 폭주 원소는 예측 집합에 속함 |
| 같은 가중치, RK4 h=0.5 / Euler h=1 | max\|H\| = ∞ / 2.58 (거친 Euler가 폭주를 가림) |

학습 중 추적에서는 dropout이 켜진 학습 모드의 max|x₀|가 3 epoch에 1을 넘었고, 8 epoch부터 기준해가 폭주했다. Euler h=1 학습은 이 구간을 최대 1.8×10⁸의 유한값으로 통과했다. 거친 Euler로 초반을 지나고 RK4로 바꾸는 학습은 전환 시점(0, 5, 20, 50 epoch)과 무관하게 모두 발산했다.

<p align="center">
  <img src="results/figs/fig_mechanism.png" width="70%">
</p>

### 4.3 같은 체크포인트에서 step 줄이기

위 체크포인트를 학습 모드 입력(max|x₀| = 1.43)으로 적분하며 |H| > 10이 처음 관측되는 시각을 쟀다.

| 적분기 | h | 첫 \|H\| > 10 시각 | max\|H(T)\| |
|---|---:|---:|---:|
| Euler | 1 | 없음 | 2.58 |
| Euler | 0.5 | 없음 | 6.14 |
| Euler | 0.25 | 3.50 | 폭주 |
| Euler | 0.1 | 3.10 | 폭주 |
| Euler | 0.05 | 2.85 | 폭주 |
| Euler | 0.01 | 2.70 | 폭주 |
| RK4 | 1, 0.5 | 3.0 | 폭주 |
| RK4 | 0.25 | 2.75 | 폭주 |
| RK4 | 0.1 | 2.70 | 폭주 |
| RK4 (기준) | 0.01 | 2.66 | 폭주 |

step을 줄이면 거짓 안정이 사라지고 폭주 시각이 기준해 2.66으로 수렴한다. 같은 가중치의 추론 모드 입력(max|x₀| = 1.09)에서는 모든 설정이 유한(max|H| ≈ 1)했다. 이 체크포인트의 폭주는 dropout으로 커진 학습 모드 상태에서만 생긴다.

### 4.4 그래프 없는 Neural ODE (two-moons)

정확도(%). 괄호는 같은 가중치를 기준해(RK4 h=0.01)로 풀었을 때.

| 모델 | 이론 | RK4 h ≥ 0.25 | RK4 h=0.1 | dopri5 | Euler h=1 |
|---|---|---:|---:|---:|---:|
| MLP | 안정 | 97.6 | — | — | 97.0 (72.1) |
| cubic | 불안정 | 발산 | 발산 | 발산 | 73.0 (10/10 폭주) |
| cubic+ (b>0) | 안정·강성 | 발산 | 96.9 | 97.3 | 69.9 (59.4) |

GNN이 아닌 일반 Neural ODE에서도 같은 세 현상이 나타났다. cubic은 거짓 안정, cubic+는 수치적 불안정성이다. MLP는 발산은 없지만 Euler h=1로 학습한 모델이 기준해에서 97.0 → 72.1로 떨어져, 이산화 의존이 남는다.

### 4.5 안정한 동역학은 적분기에 둔감하다

- 동역학이 안정한 BS·확산은 13개 적분 설정 모두에서 발산이 없었고, 정확도 차이는 0.82%p 이하였다.
- 가중치를 고정하고 추론 적분기만 바꾸면 표현 오차가 Euler에서 h에, RK4에서 약 h⁴에 비례했다.
- Euler h=2(NFE 2)는 RK4 h=0.25(NFE 64)와 같은 정확도를 냈다. Euler h=2로 학습한 BS 모델을 기준해로 풀어도 87.59 → 87.65%로 같았다.

<p align="center">
  <img src="results/figs/fig_conv_div.png" width="70%">
</p>

### 4.6 다른 데이터셋

- **Texas, Wisconsin, Film:** BS는 모든 적분 설정에서 정상 학습됐다.
- **Film:** AC와 AC(b>0)는 정밀 적분기에서 발산했고, 한 번의 step으로 끝나는 Euler h ≥ 0.5에서만 학습됐다. 학습 모드 max|x₀|가 9.46에 이른다.
- **미해결:** AC(b>0)가 dopri5에서도 발산한 이유는 아직 분석하지 못했다([한계](#7-한계와-후속-연구) 참고).

## 5. 저장소 구조

```text
.
├── rd_sweep.py        # 메인 러너: 반응항 × 적분 설정 × 분할, 발산·NFE·시간·Dirichlet energy 기록
├── run_sweep.sh       # 1차 스윕(Cora, BS/AC/확산 × 13개 적분 설정 × 10분할), 끊겨도 이어서 실행
├── run_sweep2.sh      # 2차 스윕(Fisher/Zeldovich/부호 제약, 다른 데이터셋)
├── analyze.py         # sweep.jsonl → 표(CSV/Markdown) + 그림
├── cross_eval.py      # 가중치 고정, 추론 적분기만 교체: 정확도·예측 변경률·표현 오차
├── cross_summary.py   # cross_eval 결과 요약 표
├── diag_ac.py         # AC 발산 진단: 스윕 요약 + 초기화 직후 궤적
├── diag_train.py      # 학습 중 epoch별 max|x₀|, 반응 계수, 기준해 폭주 추적
├── train_switch.py    # Euler → RK4 전환 학습
├── diag_channel.py    # (노드, 채널) 원소 단위 폭주 검증, 해석해 t*와 비교
├── step_refine.py     # 같은 체크포인트에서 step을 줄이며 폭주 관측 변화 확인
├── toy_node.py        # 그래프 없는 Neural ODE(two-moons) 일반화 실험
├── make_figs.py       # 논문 그림(fig_conv_div, fig_mechanism) 생성
└── results/           # 실험 결과 jsonl, 표, 그림
```

| 결과 파일 | 만드는 스크립트 |
|---|---|
| `results/sweep.jsonl` | `run_sweep.sh` |
| `results/sweep2.jsonl` | `run_sweep2.sh` |
| `results/crosseval.jsonl` | `cross_eval.py` |
| `results/switch.jsonl` | `train_switch.py` |
| `results/diag_train.jsonl` | `diag_train.py` |
| `results/report/` | `analyze.py` |
| `results/figs/` | `make_figs.py` |

## 6. 재현 방법

### 6.1 환경

Python 3.9, CUDA 12.8(RTX 50 시리즈의 sm_120 지원 때문에 cu128 사용)에서 실행했다.

```bash
conda create -n gread python=3.9 -y
conda activate gread
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install torch-geometric==2.3.1 "numpy<2" scikit-learn==1.2.2 torchdiffeq==0.2.3 \
            pykeops numba pandas matplotlib tabulate
```

- `scikit-learn`은 1.2.2로 고정해야 GREAD 코드의 `DistanceMetric` import가 동작한다.
- 최근 PyTorch의 `torch.load(weights_only=True)` 기본값 문제는 `rd_sweep.py`가 내부에서 처리한다.

### 6.2 GREAD 코드 연결

모든 스크립트는 GREAD의 `src/` 안에서 실행되도록 작성했다. 원본 코드는 수정하지 않고 심볼릭 링크로 연결한다.

```bash
git clone https://github.com/jeongwhanchoi/GREAD.git
git clone https://github.com/ghdwldbs04/gread-integration-sensitivity.git repo
for f in repo/*.py repo/*.sh; do ln -sf "$(realpath "$f")" GREAD/src/; done
ln -sfn "$(realpath repo/results)" GREAD/results
cd GREAD/src
```

Geom-GCN 분할 파일(`*_split_0.6_0.2_{0..9}.npz`)은 `GREAD/src/splits/`에 둔다.

### 6.3 실험 실행 (GREAD/src 에서)

```bash
# 1차 스윕 (Cora). 끊겨도 다시 실행하면 끝난 설정은 건너뜀
setsid nohup bash run_sweep.sh > ../results/sweep.log 2>&1 &

# 2차 스윕 (반응항 차수·부호 제약, 다른 데이터셋)
setsid nohup bash run_sweep2.sh > ../results/sweep2.log 2>&1 &

# 단일 실행 예시
python rd_sweep.py --reaction allen-cahn --beta_pos --method rk4 --step_size 0.25 --splits 0 1 2
```

`rd_sweep.py` 주요 인자:

| 인자 | 설명 |
|---|---|
| `--dataset` | Cora, film, texas, wisconsin 등 |
| `--reaction` | `bspm`, `allen-cahn`, `fisher`, `zeldovich`, `diffusion` |
| `--beta_pos` | 반응 계수를 b = softplus(w) > 0 으로 제약 |
| `--method` | `euler`, `rk4`, `midpoint`, `dopri5` |
| `--step_size` / `--tol_scale` | 고정 step / dopri5 허용오차 배율 |
| `--time` | 적분 종료 시간 T (기본: best params) |
| `--max_nfe` | 이 값을 넘으면 발산으로 기록 (기본 5000) |
| `--splits` | 사용할 분할 번호 |

```bash
# 가중치 고정 후 추론 적분기 교체
python cross_eval.py --reaction bspm --method rk4 --step_size 0.5
python cross_eval.py --reaction allen-cahn --method euler --step_size 1.0
python cross_summary.py ../results/crosseval.jsonl

# 폭주 메커니즘
python diag_train.py --method rk4 --step_size 0.5 --epochs 15
python diag_train.py --method euler --step_size 1.0 --epochs 60
python train_switch.py --reaction allen-cahn --switch 0 5 20 50
python diag_channel.py --epochs 8
python step_refine.py

# 그래프 없는 Neural ODE
python toy_node.py --out ../results/toy_node.jsonl \
  --solvers euler:1.0,rk4:0.25,rk4:0.1,dopri5:0 --kinds mlp,cubic,cubic+

# 표와 그림
python analyze.py ../results/sweep.jsonl ../results/report
python make_figs.py ../results ../results/figs
```

## 7. 한계와 후속 연구

**한계**

- 폭주 조건과 폭주 시간은 확산을 무시한 스칼라 조건이다. 확산은 폭주를 늦추며(1.47 → 2.66), 그래프 전체에 대한 정확한 조건은 아직 없다.
- GREAD 실험은 데이터셋별 BS 최적값을 공통으로 썼다. 원 논문의 AC 전용 설정과 직접 비교되지 않는다. 따라서 결과의 뜻은 "AC는 정밀 적분으로 학습되지 않는다"가 아니라 "같은 설정에서 반응항과 적분기 조합에 따라 안정성이 달라지며, 그 원인을 예측할 수 있다"이다.
- Film에서 AC(b>0)가 dopri5에서도 발산한 원인은 극단적 강성 외에 추가 분석이 필요하다.
- Neural ODE 실험은 소규모다(시드 10개).

**후속 연구**

1. **그래프 전체 동역학의 폭주 조건.** 채널별로 독립인 n차원 계 `dh/dt = α(A − I)h + b·r(h)`에서 최대값 원리로 폭주 시간의 하한을, Kaplan 고유벡터 방법으로 상한을 유도한다.
2. **폭주를 막는 반응항 매개변수화와 적분법.** 유한시간 해의 존재가 구조적으로 보장되는 반응항과, 반응항은 정확해·확산은 명시적으로 푸는 분할 적분기를 설계하고 GREAD·ACMP와 정확도, 발산율, NFE를 비교한다.

## 8. 참고문헌

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
