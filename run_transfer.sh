#!/usr/bin/env bash
# Solver Transfer Analysis: train with one solver, re-evaluate the same weights with every solver. Small heterophilic graphs first so results arrive early. Run from GREAD/src.
set -u
cd "$(dirname "$0")"
O=../results/transfer.jsonl
for ds in texas wisconsin cornell chameleon Cora film; do
  python transfer.py --dataset $ds --train split --train_h 1.0 --out $O
  python transfer.py --dataset $ds --train strang --train_h 1.0 --out $O
  python transfer.py --dataset $ds --train euler --train_h 1.0 --out $O
  python transfer.py --dataset $ds --train euler --train_h 0.5 --out $O
  python transfer.py --dataset $ds --train rk4   --train_h 0.25 --out $O
done
echo "ALL DONE"
