#!/usr/bin/env python3
"""wiki-RfA through the standard benchmark pipeline (run_benchmark.run_one).

Unlike Higgs, wiki-RfA's raw file just needs re-parsing into a DiGraph
(ndbench.datasets.load_wiki_rfa) -- the rest of the pipeline (measures,
RQ1/RQ2 agreement, RQ3 ground truth, RQ4 seed-set spread) is identical to
every other registry dataset, so this reuses run_benchmark.run_one directly
rather than duplicating it.

    python run_wiki_rfa.py --out results [--quick]
"""
from __future__ import annotations

import argparse
import os

from ndbench import datasets
from run_benchmark import run_one


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--sample-super", type=int, default=1000)
    ap.add_argument("--runs-small", type=int, default=1000)
    ap.add_argument("--runs-large", type=int, default=200)
    ap.add_argument("--spread-runs", type=int, default=1000)
    ap.add_argument("--celf-runs", type=int, default=200)
    ap.add_argument("--no-celf", action="store_true")
    ap.add_argument("--exact-below", type=int, default=10_000)
    ap.add_argument("--k", type=int, nargs="*", default=[10, 25, 50, 100])
    args = ap.parse_args(argv)
    if args.quick:
        args.runs_small, args.runs_large, args.spread_runs, args.celf_runs = 100, 50, 200, 50
        args.sample = min(args.sample, 500)
    os.makedirs(args.out, exist_ok=True)

    g = datasets.load_wiki_rfa(args.data_dir)
    run_one(g, args)
    print(f"\nDone. Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
