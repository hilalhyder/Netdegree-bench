# netdegree-bench

This is the benchmark code for the paper *From net degree to trophic level: local and global
flow-imbalance measures for identifying influential spreaders in directed networks*.

It computes the netdegree family of measures, trophic levels and nine baselines on directed
networks. It then scores them against Monte Carlo diffusion (IC, LT, SIR) and against the
observed retweet cascades in the SNAP Higgs Twitter data. It writes the CSVs and draft figures
that the proposal lists.

## Direction convention (read this first)

Every arc is oriented **i → j = information can flow from i to j**. Raw files that point the
other way ("i follows j", "i cites j", "i trusts j") are reversed on load. Each dataset carries
this as a `flip` flag in `ndbench/datasets.py`.

Entries marked `verify=True` must be checked against that dataset's README before you report
results. For the Higgs follower graph, the direction is decided from the data itself by
`run_higgs.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                              # 21 checks, about 5 s after the first numba compile
```

## Run

```bash
# 1. RQ2 synthetic sweep: agreement between netdegree and trophic level vs F0 (about 1 min)
python run_agreement_sweep.py --out results

# 2. Main benchmark (RQ2-RQ4). Downloads from SNAP into ./data on first use.
python run_benchmark.py --datasets email-Eu-core CollegeMsg wiki-Vote soc-sign-bitcoin-otc --out results
python run_benchmark.py --datasets cit-HepPh soc-Epinions1 soc-Slashdot0922 email-EuAll --out results
python run_benchmark.py --datasets wiki-Talk soc-Pokec --out results --no-celf     # scalability runs

# Any local edge list (add --flip if arcs point against information flow)
python run_benchmark.py --edgelist mynet.txt --flip --name mynet --out results

# Quick smoke test with no downloads
python run_benchmark.py --synthetic 0.3 1.5 --quick --out results_test

# 3. Higgs real-cascade validation (RQ5). Downloads about 0.5 GB on first use.
python run_higgs.py --out results
python run_higgs.py --social data/higgs-social_network.edgelist.gz \
                    --activity data/higgs-activity_time.txt.gz --out results

# 4. Figures and statistics
python make_figures.py --results results --figs figures
```

`congress-Twitter` ships as JSON on SNAP. Convert it to an edge list, then use `--edgelist`.

## What each output answers

| File | Contents | Manuscript |
|---|---|---|
| `datasets.csv` | n, m, reciprocity, p_c, trophic incoherence F0, flip, arc meaning | Section 6 |
| `runtime.csv` | Wall-clock seconds per measure and heuristic | Section 13, Fig. 10 |
| `measures_<net>.csv` | Every score and the trophic level for every node | RQ2, Fig. 2 |
| `agreement.csv` | Spearman of each local measure with the converged trophic ranking, plus F0 | RQ1, RQ2, Fig. 3 |
| `groundtruth_<net>.csv` | Simulated single-node spreading ability per setting | RQ3 |
| `tau.csv` | Kendall τ_b and imprecision ε(1%, 5%, 10%) per measure × setting | RQ3, Table 1, Figs. 4-6 |
| `stats_summary.csv` | Friedman test, Nemenyi CD, Wilcoxon netdegree vs out-degree (Holm) | RQ3, Fig. 6 |
| `spread.csv` | Seed-set spread with bootstrap 95% CI, and redundancy, for k = 10, 25, 50, 100 | RQ4, Table 2, Figs. 7, 8 |
| `higgs_*.csv` | Direction check, metrics, deciles, precision@k curve, temporal reach | RQ5, Fig. 9 |

Figures are numbered in order of first citation in the manuscript and drawn to Elsevier's
artwork specifications: final printed width (140 mm = 1.5 columns, 190 mm = full width),
embedded Arial at 7-8 pt, no in-figure titles (titles go in the captions), (a)/(b) part labels.
Submit the PDFs (vector); the PNGs (600 dpi) are for Word drafts and previews.

| Figure | File | Contents | First cited |
|---|---|---|---|
| 1 | `fig1_risk_classification` | Degree-tail ratio (max / p99 total degree) vs. reciprocity, with both thresholds; needs `data/` | Section 5 |
| 2 | `fig2_netdegree_vs_trophic` | Normalized net-degree vs. trophic level, four networks | Section 7 |
| 3 | `fig3_agreement_vs_F0` | Agreement vs. F0, and vs. Jacobi depth | Section 7 |
| 4 | `fig4_tau_heatmap` | Kendall τ, every measure × network, IC at p_c | Section 8 |
| 5 | `fig5_tau_vs_p` | τ robustness to transmission probability | Section 8 |
| 6 | `fig6_critical_difference` | Friedman-Nemenyi critical-difference diagram | Section 8 |
| 7 | `fig7_capstone_regularization_fix` | Spread ratio to the best method (k = 100, IC p = p_c), grouped by failure mode; needs `data/` | Section 9 |
| 8 | `fig8_spread_vs_k` | Seed-set spread vs. budget k | Section 9 |
| 9 | `fig9_higgs` | Higgs deciles and precision@k | Section 10 |
| 10 | `fig10_runtime` | Runtime vs. edge count (log-log) | Section 13 |

## Settings used (defaults)

- **IC:** uniform p at 0.5, 1.0 and 1.5 × p_c, with p_c = ⟨k⟩/⟨k_in·k_out⟩. Also weighted
  cascade, with p = 1/k_in(v).
- **LT:** weights 1/k_in(v) and thresholds uniform on (0, 1).
- **SIR:** β at the same multiples of p_c, and recovery μ = 0.5 per step.
- **Networks under 10,000 nodes** (`--exact-below`): ground truth for every node from
  1,000 runs, and CELF as the reference.
- **Larger networks:** 2,000 nodes sampled across out-degree strata, 200 runs each.
- **Seed-set spread:** 1,000 runs.

Supercritical settings (1.5 × p_c, SIR) dominate the run time, because every run infects a
large part of the network. On a laptop, expect minutes for networks under 10k nodes and hours
for the largest ones. The simulators use every CPU core. Use `--quick` for a smoke test, and
lower `--runs-*` or `--sample` if needed.

## Layout

```
ndbench/graph.py       DiGraph container, edge-list reader (gz, labels, flip, duplicate merge)
ndbench/measures.py    netdegree family, trophic levels + Jacobi iterates, baselines
ndbench/diffusion.py   numba IC / LT / SIR, parallel single-node spreading
ndbench/seeds.py       top-k, top-k with exclusion, VoteRank, DegreeDiscountIC, CELF
ndbench/evaluate.py    Kendall, imprecision, bootstrap CI, redundancy, AUC, Friedman-Nemenyi, Holm
ndbench/datasets.py    SNAP registry with arc meaning and flip flags
ndbench/synthetic.py   trophic-coherence-tunable generator; Higgs-like fixture for testing
run_benchmark.py       RQ2-RQ4
run_agreement_sweep.py RQ1-RQ2 synthetic sweep
run_higgs.py           RQ5
make_figures.py        figures 1-10 and stats_summary.csv
tests/test_core.py     correctness checks
```

## Demo without downloads

```bash
./run_demo.sh      # generates 5 networks (up to 300k nodes / 3M arcs), runs everything, draws figures_demo/
```

This takes about 15 minutes on 2 cores. The Higgs step uses a small synthetic stand-in. The
demo checks that the pipeline runs end to end; its numbers are not results.
