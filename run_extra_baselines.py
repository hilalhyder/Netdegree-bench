#!/usr/bin/env python3
"""Stronger seed-selection baselines: IMM, adaptive Collective Influence and gravity centrality.

Evaluated exactly like the seed sets in run_benchmark.py (same IC p = p_c and LT 1/k_in
settings, budgets, 1,000 runs and simulation seed), and written to separate files so the
main and pre-registered held-out results stay untouched:

  <out>/spread_extra.csv   same columns as spread.csv
  <out>/imm_info.csv       RR-set count, lower bound, cap flag per network and model
  <out>/runtime_extra.csv

IMM is run once per diffusion model and evaluated only under that model ("model-matched").
Networks already present in spread_extra.csv are skipped, so an interrupted run resumes.

  python run_extra_baselines.py --datasets main --out results
  python run_extra_baselines.py --datasets heldout --out results_heldout
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from ndbench import datasets, diffusion, evaluate, imm, seeds
from ndbench.io import append_csv
from run_benchmark import settings_for

MAIN = ["email-Eu-core", "wiki-RfA", "CollegeMsg", "wiki-Vote", "soc-sign-bitcoin-otc", "soc-sign-epinions",
        "soc-sign-Slashdot090221", "cit-HepPh", "soc-Epinions1", "ego-Twitter", "soc-Slashdot0922",
        "email-EuAll", "wiki-Talk", "soc-Pokec"]
HELDOUT = ["soc-sign-bitcoin-alpha", "cit-HepTh", "sx-mathoverflow", "sx-askubuntu", "sx-superuser",
           "soc-Slashdot0811", "ego-Gplus"]


def load(name, data_dir):
    return datasets.load_wiki_rfa(data_dir) if name == "wiki-RfA" else datasets.load(name, data_dir)


def run_one(g, args):
    net = g.name
    print(f"\n=== {net}: n={g.n:,} m={g.m:,}", flush=True)
    _, sets = settings_for(g, quick=False)
    ic = [s for s in sets if s["setting"] == "IC p=1pc"][0]
    lt = [s for s in sets if s["model"] == "LT"][0]
    ks = [k for k in args.k if k <= max(1, g.n // 10)]
    if not ks:
        return
    kmax = max(ks)
    rt, info_rows, sets_by_setting = [], [], {ic["setting"]: {}, lt["setting"]: {}}

    def timed(label, fn):
        t0 = time.perf_counter()
        out = fn()
        sec = time.perf_counter() - t0
        rt.append(dict(net=net, n=g.n, m=g.m, method=label, seconds=sec))
        print(f"    {label:22s} {sec:8.1f}s", flush=True)
        return out

    ci = timed("collective_influence", lambda: seeds.collective_influence(g, kmax))
    gr = timed("gravity", lambda: seeds.top_k(seeds.gravity(g), kmax))
    for st in (ic, lt):
        sets_by_setting[st["setting"]][("collective_influence", "set")] = ci
        sets_by_setting[st["setting"]][("gravity", "top-k")] = gr
    for st, p in ((ic, ic["p"]), (lt, None)):
        s, info = timed(f"imm_{st['model']}", lambda: imm.imm(g, kmax, st["model"], p, eps=args.eps, rng_seed=17))
        sets_by_setting[st["setting"]][("imm", "model-matched")] = s
        info_rows.append(dict(net=net, model=st["model"], setting=st["setting"], k=kmax, **info))
        print(f"      theta={info['theta']:,} mean RR={info['mean_rr_size']:.1f} "
              f"est spread={info['est_spread']:.1f} capped={info['capped']}", flush=True)

    rows = []
    for st in (ic, lt):
        for (method, variant), S in sets_by_setting[st["setting"]].items():
            for k in ks:
                samples = diffusion.spread(g, S[:k], st["model"], st["param"], args.spread_runs, 0.5, rng_seed=5)
                lo, hi = evaluate.bootstrap_ci(samples)
                rows.append(dict(net=net, model=st["model"], setting=st["setting"], k=k, method=method,
                                 variant=variant, spread=samples.mean(), ci_low=lo, ci_high=hi,
                                 redundancy=evaluate.redundancy(g, S[:k]), runs=args.spread_runs))
    print(pd.DataFrame(rows).query("k == @kmax")[["setting", "method", "spread"]].to_string(index=False))
    append_csv(pd.DataFrame(info_rows), os.path.join(args.out, "imm_info.csv"))
    append_csv(pd.DataFrame(rt), os.path.join(args.out, "runtime_extra.csv"))
    append_csv(pd.DataFrame(rows), os.path.join(args.out, "spread_extra.csv"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["main"], help="names, or 'main' / 'heldout'")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--spread-runs", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--k", type=int, nargs="*", default=[10, 25, 50, 100])
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    names = {"main": MAIN, "heldout": HELDOUT}.get(args.datasets[0], args.datasets) \
        if len(args.datasets) == 1 else args.datasets
    done_path = os.path.join(args.out, "spread_extra.csv")
    done = set(pd.read_csv(done_path).net) if os.path.exists(done_path) else set()
    for name in names:
        if name in done:
            print(f"skip {name} (already in spread_extra.csv)")
            continue
        run_one(load(name, args.data_dir), args)
    print(f"\nDone. Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
