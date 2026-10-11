#!/usr/bin/env bash
# 제안법(부호 제약 AC + 정확해 분할 적분) 본 실험. GREAD/src 에서 실행. 끝난 설정은 건너뜀.
set -u
cd "$(dirname "$0")"
DONE=../results/done_split
mkdir -p "$DONE"
S="0 1 2 3 4 5 6 7 8 9"

sp() {  # 제안법: $1 dataset  $2 h  $3 bpos(1/0)
  key="split_$1_$2_$3"; [ -f "$DONE/$key" ] && return
  flag=""; [ "$3" = 1 ] && flag="--beta_pos"
  python split_run.py --dataset "$1" $flag --step_size "$2" --splits $S --out ../results/split.jsonl && touch "$DONE/$key"
}
base() {  # 기존 적분기 대조군: $1 dataset  $2 reaction  $3 method  $4 h
  key="base_$1_$2_$3_$4"; [ -f "$DONE/$key" ] && return
  rx=${2%+bpos}; flag=""; [ "$2" != "$rx" ] && flag="--beta_pos"
  python rd_sweep.py --dataset "$1" --reaction "$rx" $flag --method "$3" --step_size "$4" --splits $S --out ../results/split_base.jsonl && touch "$DONE/$key"
}

# 1) Cora: 제안법 (b>0) h = 0.5, 1, 2  /  부호 제약 없는 AC + 분할 h = 1
for h in 1.0 2.0 0.5; do sp Cora $h 1; done
sp Cora 1.0 0
# 2) Film: dopri5 에서도 발산하던 AC(b>0)
for h in 1.0 0.5; do sp film $h 1; done
sp film 1.0 0
# 3) Pubmed: 규모 확장 (제안법 + 대조군)
for h in 1.0 0.5; do sp Pubmed $h 1; done
base Pubmed bspm rk4 0.25
base Pubmed allen-cahn+bpos rk4 0.25
base Pubmed allen-cahn+bpos rk4 1.0
base Pubmed allen-cahn+bpos euler 1.0
base Pubmed allen-cahn rk4 0.25
echo "ALL DONE"
