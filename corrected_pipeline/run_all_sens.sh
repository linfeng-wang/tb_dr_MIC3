#!/bin/bash
# Keep launching the sensitivity runs (run_sens.sh) as the main CNN results appear, until the main run has finished
# and one full pass has been made after it.
cd "$(dirname "$0")"
mkdir -p logs results models tables figs
while pgrep -f "run_sens.s[h]" > /dev/null; do sleep 60; done
while true; do
  done_cnn=0; grep -q ALL_DONE logs/run_cnn.out && done_cnn=1
  ./run_sens.sh >> logs/run_sens.out 2>&1
  [ $done_cnn = 1 ] && break
  sleep 120
done
echo SENS_ALL_DONE >> logs/run_sens.out
