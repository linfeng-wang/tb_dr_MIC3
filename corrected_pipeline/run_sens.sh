#!/bin/bash
# Sensitivity runs for drugs whose main CNN results exist. Two at a time while the main training is still running,
# six at a time afterwards. Re-run to pick up newly finished drugs.
cd "$(dirname "$0")"
mkdir -p logs results models tables figs
export PY=${PY:-python}
P=6; pgrep -f "r02_trai[n].py" > /dev/null && P=2
for d in EMB RIF INH ETH DLM LEV MXF AMI KAN LZD BDQ CFZ; do for m in perm tbptrain; do
  k=cnn_std; [ $m = tbptrain ] && k=cnn_bin
  [ -f results/${k}_$d.json ] && [ ! -f results/sens_${m}_$d.npz ] && echo "$d $m"; done; done | \
  xargs -P $P -L 1 bash -c '"$PY" r08_sensitivity.py $0 $1 > logs/sens_$0_$1.log 2>&1'
echo SENS_DONE
