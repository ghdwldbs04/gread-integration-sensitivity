#!/usr/bin/env bash
# RQ3 generic fix vs structure-preserving integrator, plus reference-convergence check. Run from GREAD/src.
# Every invocation writes to a NEW time-stamped file; every line carries the run id.
set -u
cd "$(dirname "$0")"
RUN=$(date +%m%d_%H%M)
O=../results/consist_${RUN}.jsonl
echo "run_id=${RUN} out=${O}"
for ds in cornell wisconsin texas chameleon Cora film; do
  python consist.py --dataset $ds --train euler  --lam 0   --out $O --run_id $RUN
  python consist.py --dataset $ds --train euler  --lam 0.1 --out $O --run_id $RUN
  python consist.py --dataset $ds --train euler  --lam 1   --out $O --run_id $RUN
  python consist.py --dataset $ds --train strang           --out $O --run_id $RUN
done
echo "ALL DONE ${O}"
