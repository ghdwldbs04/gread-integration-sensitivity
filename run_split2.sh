#!/usr/bin/env bash
# 2차: 제안법 vs 같은 부호 제약에서 Euler 학습, 두 가지 기준해(RK4 미세 step, 분할 미세 step)로 검증. GREAD/src 에서 실행.
set -u
cd "$(dirname "$0")"
O=../results/split2.jsonl
S="0 1 2 3 4 5 6 7 8 9"
for ds in Cora Pubmed; do
  python split_run.py --dataset $ds --beta_pos --step_size 1.0 --splits $S --ref_h 0.05 --ref2_h 0.01 --out $O
  python split_run.py --dataset $ds --beta_pos --step_size 1.0 --plain euler --splits $S --ref_h 0.05 --ref2_h 0.01 --out $O
done
# Film: 초기 상태가 커서(max|x0|~9) 강성이 크다 -> 기준해 step을 아주 작게
python split_run.py --dataset film --beta_pos --step_size 1.0 --splits $S --ref_h 0.002 --ref2_h 0.001 --out $O
python split_run.py --dataset film --beta_pos --step_size 1.0 --plain euler --splits $S --ref_h 0.002 --ref2_h 0.001 --out $O
python split_run.py --dataset film --beta_pos --step_size 0.5 --plain euler --splits $S --ref_h 0.002 --ref2_h 0.001 --out $O
echo "ALL DONE"
