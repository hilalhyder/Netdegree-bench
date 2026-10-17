# Held-out validation: pre-registered predictions

Written 2026-10-17, before any diffusion was run on these networks. The commit that adds this
file is the pre-registration; results go in `results_heldout/` and are compared against the
predictions below without changing them.

## What is frozen

- **Measures and α rules:** exactly as in `ndbench/measures.py` at this commit
  (`spring_rank_maxdeg`: α = maximum total degree; `spring_rank_p99`; `spring_rank_tailadaptive`).
- **Risk thresholds:** degree-tail ratio (max / p99 total degree) > 100, reciprocity > 0.75, as
  in Figure 1 (`make_figures.py`).
- **Protocol:** `run_benchmark.py` full mode, default settings, identical to the 14-network suite.
- **Primary outcome:** seed-set spread ratio to the best method, k = 100, IC at p = p_c (Table 2).

## Networks

Seven SNAP directed networks not used anywhere in the development of the α rules
(`heldout/preregistration.csv`, produced by `heldout/preregister.py` from degree data only):

| Network | Nodes | Arcs | Reciprocity | Tail ratio | Zero in-degree | Predicted class |
|---|---|---|---|---|---|---|
| soc-sign-bitcoin-alpha | 3,683 | 22,650 | 0.855 | 6.3 | 11.2% | reciprocity ceiling |
| cit-HepTh | 27,770 | 352,768 | 0.003 | 14.8 | 9.8% | normal |
| sx-mathoverflow | 24,818 | 227,991 | 0.351 | 9.0 | 11.7% | normal |
| sx-askubuntu | 159,316 | 544,621 | 0.327 | 91.3 | 21.0% | normal |
| sx-superuser | 194,085 | 854,377 | 0.327 | 145.0 | 23.8% | degree-tail outlier |
| soc-Slashdot0811 | 77,360 | 828,161 | 0.867 | 16.0 | 8.4% | reciprocity ceiling |
| ego-Gplus | 107,614 | 13,673,453 | 0.210 | 7.0 | 32.8% | normal |

Two caveats, stated now: soc-Slashdot0811 is an earlier snapshot of the same platform as
soc-Slashdot0922, and soc-sign-bitcoin-alpha comes from a marketplace similar to bitcoin-otc,
so they are held out from tuning but not independent platforms. sx-askubuntu (ratio 91) sits
just below the tail threshold.

## Hypotheses (IC at p_c, k = 100, ratio to best method unless stated)

- **H1 (boundary collapse).** Unregularized trophic level has ratio < 0.5 on every held-out
  network (all have at least 8% zero-in-degree nodes).
- **H2 (fix on normal networks).** On the four networks predicted normal, SpringRank
  (α = max degree) has ratio ≥ 0.95 and is within 0.02 of net-degree.
- **H3 (degree-tail outlier).** On sx-superuser, SpringRank (α = max degree) exceeds SpringRank
  (α = p99) by at least 0.05.
- **H4 (reciprocity ceiling).** On the two networks predicted reciprocity-limited, both
  net-degree and SpringRank (α = max degree) have ratio < 0.95.
- **H5 (α = max degree reproduces net-degree).** On every held-out network, SpringRank
  (α = max degree) orders fewer than 1% of node pairs with distinct weighted net-degree
  differently from weighted net-degree.
- **H6 (single-node ranking).** Unregularized trophic level has the highest top-1% imprecision
  (ε₁) of all measures, averaged over the held-out networks (IC at p_c).

Each hypothesis is reported as held or failed, network by network, with no post-hoc changes to
thresholds, α rules or protocol.
