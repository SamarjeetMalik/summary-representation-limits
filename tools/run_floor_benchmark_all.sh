#!/bin/bash
# Semi-synthetic known-floor benchmark in the KASCADE geometry: three interaction models x
# (station sample + three decades), REPS noise replicates each. Four runs at a time per model.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for g in "" epos sibyll; do
  for d in "" 14-15 15-16 16-17; do
    KCDC_GEN=$g KCDC_DECADE=$d $PY analysis/kcdc/run_kcdc_floor_benchmark.py > analysis/kcdc/boxlogs/floor_bench_${g:-qgs4}_${d:-st}.log 2>&1 &
  done
  wait
done
if grep -l "Traceback\|control failed" analysis/kcdc/boxlogs/floor_bench_*_*.log; then exit 1; fi
echo "all 12 benchmark runs done"
