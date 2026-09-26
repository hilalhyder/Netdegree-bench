#!/bin/sh
# End-to-end demo on generated networks (no downloads).
set -e
python3 make_demo_data.py
for f in scalefree gnr_citation trophic_T0.3 trophic_T1.5; do
  flip=""; [ "$f" = gnr_citation ] && flip="--flip"
  python3 run_benchmark.py --edgelist data/demo/$f.txt $flip --name $f --out results_demo
done
python3 run_benchmark.py --edgelist data/demo/big_trophic.txt --name big_trophic_300k --out results_demo --no-celf
python3 run_agreement_sweep.py --out results_demo --reps 3
python3 run_higgs.py --fixture --out results_demo
python3 make_figures.py --results results_demo --figs figures_demo
