#!/usr/bin/env bash
# 2차 스윕: (A) 이론 예측 검증 — 반응항 차수·계수 부호  (B) 데이터셋 일반화
# GREAD/src 에서 실행. 재실행하면 끝난 설정은 건너뜀. 결과: ../results/sweep2.jsonl
set -u
cd "$(dirname "$0")"
OUT=../results/sweep2.jsonl
DONE=../results/done2
mkdir -p "$DONE"
SPLITS="0 1 2 3 4 5 6 7 8 9"
SETTINGS="euler:0.1 euler:0.5 euler:1.0 euler:2.0 rk4:0.25 rk4:1.0 dopri5:1e3"

run() {  # $1 dataset  $2 reaction(+bpos)  $3 method:value
  ds=$1; rx=${2%+bpos}; flag=""; [ "$2" != "$rx" ] && flag="--beta_pos"
  m=${3%%:*}; v=${3##*:}
  key="${ds}_${2}_${m}_${v}"
  [ -f "$DONE/$key" ] && { echo "[skip] $key"; return; }
  if [ "$m" = "dopri5" ]; then
    python rd_sweep.py --dataset "$ds" --reaction "$rx" $flag --method dopri5 --tol_scale "$v" --splits $SPLITS --out "$OUT"
  else
    python rd_sweep.py --dataset "$ds" --reaction "$rx" $flag --method "$m" --step_size "$v" --splits $SPLITS --out "$OUT"
  fi && touch "$DONE/$key"
}

# (A) Cora: 이론 예측 — 2차(Fisher)·3차(Zeldovich) 반응항, 부호 제약(b>0)
for r in fisher zeldovich allen-cahn+bpos zeldovich+bpos fisher+bpos; do
  for s in $SETTINGS; do run Cora "$r" "$s"; done
done
# (B) 다른 데이터셋 일반화: BS / AC / AC(b>0)
for ds in Citeseer texas wisconsin chameleon; do
  for r in bspm allen-cahn allen-cahn+bpos; do
    for s in $SETTINGS; do run "$ds" "$r" "$s"; done
  done
done
echo "ALL DONE"
