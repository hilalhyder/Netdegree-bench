@echo off
cd /d C:\Users\HP\Projects\netdegree_bench
for %%N in (soc-sign-bitcoin-alpha cit-HepTh sx-mathoverflow soc-Slashdot0811 sx-askubuntu sx-superuser ego-Gplus) do (
  echo ==== %%N start %TIME% >> heldout\run.log
  py -3.13 run_benchmark.py --datasets %%N --out results_heldout >> heldout\run.log 2>&1
  echo ==== %%N end %TIME% >> heldout\run.log
)
echo ALL DONE >> heldout\run.log
