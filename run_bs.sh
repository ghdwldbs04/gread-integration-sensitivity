#!/bin/bash
# GREAD-BS (published hyperparameters) solver-transfer run. Run from GREAD/src. One new time-stamped file per run.
RUN=$(date +%m%d_%H%M)
O=../results/bs_${RUN}.jsonl
echo "run_id=${RUN} -> ${O}"
for D in cornell wisconsin texas chameleon film Cora; do
  python bs_transfer.py --dataset $D --train published --out $O --run_id $RUN
  python bs_transfer.py --dataset $D --train euler --h 1.0 --out $O --run_id $RUN
done
echo "DONE ${RUN}"
