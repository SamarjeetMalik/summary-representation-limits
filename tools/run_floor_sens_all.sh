#!/bin/bash
# Floor-estimate sensitivity for three interaction models x (station sample + three decades).
# Four runs at a time per model; logs in analysis/kcdc/boxlogs/floor_sens_<model>_<sample>.log
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for g in "" epos sibyll; do
  for d in "" 14-15 15-16 16-17; do
    KCDC_GEN=$g KCDC_DECADE=$d $PY analysis/kcdc/run_kcdc_floor_sens.py > analysis/kcdc/boxlogs/floor_sens_${g:-qgs4}_${d:-st}.log 2>&1 &
  done
  wait
done
grep -l "Traceback\|control failed" analysis/kcdc/boxlogs/floor_sens_*_*.log && exit 1
echo "all 12 sensitivity runs done"
