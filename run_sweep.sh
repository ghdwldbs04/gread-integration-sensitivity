#!/usr/bin/env bash
# 반응항 x 적분 설정 전체 스윕. GREAD/src 에서 실행.  끊겨도 다시 실행하면 끝난 설정은 건너뜀.
set -u
cd "$(dirname "$0")"
OUT=../results/sweep.jsonl
DONE=../results/done
mkdir -p "$DONE"
SPLITS="${SPLITS:-0 1 2}"
REACTIONS="${REACTIONS:-bspm allen-cahn diffusion}"

run() {  # $1 reaction, $2 method, $3 value(step 또는 tol)
  key="$1_$2_$3"
  [ -f "$DONE/$key" ] && { echo "[skip] $key"; return; }
  if [ "$2" = "dopri5" ]; then
    python rd_sweep.py --reaction "$1" --method dopri5 --tol_scale "$3" --splits $SPLITS --out "$OUT"
  else
    python rd_sweep.py --reaction "$1" --method "$2" --step_size "$3" --splits $SPLITS --out "$OUT"
  fi && touch "$DONE/$key"
}

for r in $REACTIONS; do
  for h in 0.1 0.25 0.5 1.0 1.5 2.0; do run "$r" euler "$h"; done
  for h in 0.25 0.5 1.0 2.0;         do run "$r" rk4   "$h"; done
  for t in 1e1 1e3 1e5;              do run "$r" dopri5 "$t"; done
done
echo "ALL DONE"
