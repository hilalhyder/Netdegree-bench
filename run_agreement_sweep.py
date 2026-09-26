#!/usr/bin/env python3
"""RQ1/RQ2 synthetic sweep: how local-global agreement (netdegree vs trophic
level) depends on trophic incoherence F0.  No downloads, no simulation.

  python run_agreement_sweep.py --out results
Appends rows with source='synthetic sweep' to results/agreement.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

from ndbench import evaluate, measures, synthetic
from ndbench.io import append_csv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--T", type=float, nargs="*",
                    default=[0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for T in args.T:
        for rep in range(args.reps):
            g, _ = synthetic.trophic_tunable(n=args.n, T=T, rng_seed=1000 * rep + int(100 * T))
            h, info = measures.trophic_levels(g, return_info=True)
            ref = -h
            act = (g.out_degree() + g.in_degree()) > 0
            cand = {"netdegree": measures.netdegree(g), "netdegree_norm": measures.normalized_netdegree(g),
                    "netdegree_2hop": measures.hop_netdegree_walk(g, 2), "out_degree": measures.out_degree(g)}
            for t in (1, 2, 5, 10):
                cand[f"jacobi_{t}"] = measures.jacobi_trophic(g, t)
            for name, s in cand.items():
                rows.append(dict(net=f"synthetic_T{T:g}_r{rep}", source="synthetic sweep", measure=name,
                                 spearman_vs_trophic=evaluate.spearman(s[act], ref[act]), F0=info["F0"], T=T))
        print(f"T={T:<4g} F0={info['F0']:.3f}  rho(nd_norm, -h)="
              f"{[r['spearman_vs_trophic'] for r in rows if r['measure'] == 'netdegree_norm'][-1]:.3f}")
    df = pd.DataFrame(rows)
    path = os.path.join(args.out, "agreement.csv")
    append_csv(df, path)


if __name__ == "__main__":
    main()
