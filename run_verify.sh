#!/bin/bash
# Verification run (review checks). Run from GREAD/src. One new time-stamped file per run.
RUN=$(date +%m%d_%H%M)
O=../results/verify_${RUN}.jsonl
echo "run_id=${RUN} -> ${O}"
for D in cornell wisconsin texas chameleon film Cora; do
  python verify_rx.py --dataset $D --reaction fisher    --splits 0 1 2 --out $O --run_id $RUN
  python verify_rx.py --dataset $D --reaction zeldovich --splits 0 1 2 --out $O --run_id $RUN
  python verify_rx.py --dataset $D --reaction allen-cahn --beta_pos --splits 0 1 2 --out $O --run_id $RUN
done
echo "DONE ${RUN}"
