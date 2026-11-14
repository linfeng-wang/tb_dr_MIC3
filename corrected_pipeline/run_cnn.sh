#!/bin/bash
# Run all 24 CNN jobs (12 drugs x standard/binary-supported), 6 at a time, largest feature sets first.
cd "$(dirname "$0")"
mkdir -p logs results models tables figs
PY=${PY:-python}
for d in EMB RIF INH ETH DLM LEV MXF AMI KAN LZD BDQ CFZ; do for k in cnn_bin cnn_std; do echo "$d $k"; done; done | \
  xargs -P 6 -L 1 bash -c '[ -f results/$1_$0.json ] || '"$PY"' r02_train.py $0 $1 > logs/$0_$1.log 2>&1'
echo ALL_DONE
