#!/bin/bash
# Smoke test with confidence features, then full runs with and without confidence (ablation).
cd /home/vishnu/geostreamfusion_attribution_20261108
W=runs/selector_20261111
rm -rf runs/selector_smoke/online_conf runs/selector_smoke/FAILED.txt runs/selector_smoke/failures.txt runs/selector_smoke/faults
LOGTAG=conf ./run_stage.sh bin_memory_selector runs/selector_smoke 12 --limit 6 --iterations 2 --epochs 5 --depth-epochs 5 --out online_conf
if [ -e runs/selector_smoke/FAILED.txt ] || ! grep -q DONE runs/selector_smoke/online_conf/progress.txt; then
  echo "$(date +%T) smoke failed, full runs not started" > runs/conf_chain_status.txt; exit 1
fi
rm -f $W/FAILED.txt $W/failures.txt; rm -rf $W/faults $W/online_conf $W/online_noconf
LOGTAG=conf ./run_stage.sh bin_memory_selector $W 56 --out online_conf &
sleep 30
LOGTAG=noconf SELECTOR_NO_CONF=1 ./run_stage.sh bin_memory_selector $W 56 --out online_noconf &
wait
echo "$(date +%T) chain done" > runs/conf_chain_status.txt
