#!/bin/bash
# Build every dysts system's block (25 real windows + 25 IAAFT fakes). Usage:
#   ./run_build.sh               natural sampling, 20 points per period -> data/
#   ./run_build.sh 2.0           fixed lambda*dt = 2.0                  -> data_ldt2.0/
# Systems that fail to integrate or take over 120 s are skipped (listed as "skipped").
cd "$(dirname "$0")"
if [ -z "$1" ]; then
  mkdir -p data
  awk '{print $1, NR}' systems.txt | xargs -P 2 -n 2 sh -c 'timeout 120 python3 build_one.py "$0" "$1" || echo "$0 skipped"'
else
  OUT="data_ldt$1"; mkdir -p "$OUT"
  awk -v L="$1" -v O="$OUT" '{print $1, NR, L, O}' systems.txt \
   | xargs -P 2 -n 4 sh -c 'timeout 120 python3 build_one.py "$0" "$1" "$2" "$3" || echo "$0 skipped"'
fi
