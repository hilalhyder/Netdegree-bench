@echo off
cd /d %~dp0
echo ==== main start >> extra_baselines.log
py -3.13 run_extra_baselines.py --datasets main --out results >> extra_baselines.log 2>&1
echo ==== heldout start >> extra_baselines.log
py -3.13 run_extra_baselines.py --datasets heldout --out results_heldout >> extra_baselines.log 2>&1
echo ALL DONE >> extra_baselines.log
