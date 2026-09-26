#!/usr/bin/env python3
"""Companion check for proofs/positive_only_filtering_effect.md: recompute
netdegree-trophic agreement on the four signed datasets with EVERY signed
edge kept (sign ignored, presence-only) instead of positive_only=True, and
append properly to results/agreement.csv (net name suffixed '_unfiltered')
so the comparison is part of the permanent, traceable record rather than a
one-off terminal query.

    python run_signed_unfiltered_check.py --out results
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ndbench import datasets, evaluate, measures
from ndbench.graph import read_edgelist
from ndbench.io import append_csv

SPECS = [
    dict(net="soc-sign-epinions_unfiltered", path="data/soc-sign-epinions.txt.gz",
        flip=True, delimiter=None),
    dict(net="soc-sign-bitcoin-otc_unfiltered", path="data/soc-sign-bitcoinotc.csv.gz",
        flip=True, delimiter=","),
    dict(net="soc-sign-Slashdot090221_unfiltered", path="data/soc-sign-Slashdot090221.txt.gz",
        flip=True, delimiter=None),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = []

    for spec in SPECS:
        g = read_edgelist(spec["path"], flip=spec["flip"], weight_col=None,
                          delimiter=spec["delimiter"], name=spec["net"])
        h, info = measures.trophic_levels(g, return_info=True)
        ref = -h
        act = (g.out_degree() + g.in_degree()) > 0
        cand = {"netdegree": measures.netdegree(g), "netdegree_norm": measures.normalized_netdegree(g),
               "netdegree_2hop": measures.hop_netdegree_walk(g, 2), "out_degree": measures.out_degree(g)}
        for name, s in cand.items():
            rows.append(dict(net=spec["net"], source="real (sign-unfiltered)", measure=name,
                             spearman_vs_trophic=evaluate.spearman(s[act], ref[act]), F0=info["F0"]))
        print(f"{spec['net']:38s} n={g.n:<8,} m={g.m:<9,} F0={info['F0']:.3f}  "
              f"agreement(netdegree_norm)={[r['spearman_vs_trophic'] for r in rows if r['net']==spec['net'] and r['measure']=='netdegree_norm'][-1]:.3f}")

    # wiki-RfA needs the bespoke parser; add a keep-everything variant there too.
    g = datasets.load_wiki_rfa_variant(args.data_dir, positive_only=False)
    h, info = measures.trophic_levels(g, return_info=True)
    ref = -h
    act = (g.out_degree() + g.in_degree()) > 0
    cand = {"netdegree": measures.netdegree(g), "netdegree_norm": measures.normalized_netdegree(g),
           "netdegree_2hop": measures.hop_netdegree_walk(g, 2), "out_degree": measures.out_degree(g)}
    for name, s in cand.items():
        rows.append(dict(net="wiki-RfA_unfiltered", source="real (sign-unfiltered)", measure=name,
                         spearman_vs_trophic=evaluate.spearman(s[act], ref[act]), F0=info["F0"]))
    print(f"{'wiki-RfA_unfiltered':38s} n={g.n:<8,} m={g.m:<9,} F0={info['F0']:.3f}  "
          f"agreement(netdegree_norm)={[r['spearman_vs_trophic'] for r in rows if r['net']=='wiki-RfA_unfiltered' and r['measure']=='netdegree_norm'][-1]:.3f}")

    append_csv(pd.DataFrame(rows), os.path.join(args.out, "agreement.csv"))
    print(f"\nAppended {len(rows)} rows to {os.path.join(args.out, 'agreement.csv')}")


if __name__ == "__main__":
    main()
