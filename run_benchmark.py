#!/usr/bin/env python3
"""Main benchmark (RQ2-RQ4): measures, local-global agreement, single-node
spreading (Kendall tau, imprecision) and seed-set spread.

Examples
--------
  python run_benchmark.py --datasets email-Eu-core wiki-Vote
  python run_benchmark.py --datasets all --out results
  python run_benchmark.py --edgelist mynet.txt --flip --name mynet
  python run_benchmark.py --synthetic 0.3 1.0 --quick          # no downloads needed

Outputs (appended, one row per result) in --out:
  datasets.csv  runtime.csv  agreement.csv  tau.csv  spread.csv
  measures_<net>.csv  groundtruth_<net>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from ndbench import datasets, diffusion, evaluate, measures, seeds, synthetic
from ndbench.graph import read_edgelist
from ndbench.io import append_csv

SMALL = 10_000          # exact / heavier settings below this many nodes
SEED_METHOD_BASE = ["netdegree", "netdegree_norm", "netdegree_2hop", "trophic", "spring_rank_p99",
                    "spring_rank_tailadaptive", "spring_rank_maxdeg",
                    "out_degree", "in_degree", "pagerank_rev", "leaderrank", "kshell", "betweenness"]




def settings_for(g, quick):
    pc = diffusion.percolation_threshold(g)
    if pc > 0.5:
        print(f"    note: p_c = {pc:.2f} (tree-like network); uniform-p settings capped at 0.5")
    mults = [1.0] if quick else [0.5, 1.0, 1.5]
    out = []
    for mlt in mults:
        p = min(0.5, mlt * pc) if pc > 0.5 else min(1.0, mlt * pc)
        out.append(dict(model="IC", setting=f"IC p={mlt:g}pc", p_over_pc=mlt,
                        param=diffusion.arc_params(g, "uniform", p), p=p))
    out.append(dict(model="IC", setting="IC weighted-cascade", p_over_pc=np.nan,
                    param=diffusion.arc_params(g, "wc"), p=np.nan))
    out.append(dict(model="LT", setting="LT 1/k_in", p_over_pc=np.nan,
                    param=diffusion.arc_params(g, "lt"), p=np.nan))
    for mlt in ([1.5] if quick else mults):
        T = min(0.5, mlt * pc) if pc > 0.5 else min(1.0, mlt * pc)
        b = diffusion.sir_beta_for_transmissibility(T, 0.5)
        out.append(dict(model="SIR", setting=f"SIR T={mlt:g}pc mu=0.5", p_over_pc=mlt,
                        param=diffusion.arc_params(g, "uniform", b), p=b))
    return pc, out


def run_one(g, args):
    net = g.name
    out = args.out
    print(f"\n=== {net}: n={g.n:,} m={g.m:,} flip={g.meta.get('flip')}")

    # ---- measures + runtime -------------------------------------------------
    scores, rt, info = measures.compute_all(g, verbose=True,
                                            exclude=() if g.n <= 50_000 else ("netdegree_2hop_reach",))
    scores["jacobi_1"] = measures.jacobi_trophic(g, 1)
    scores["jacobi_10"] = measures.jacobi_trophic(g, 10)
    rt_rows = [dict(net=net, n=g.n, m=g.m, method=k, seconds=v) for k, v in rt.items()]

    pc, sets = settings_for(g, args.quick)
    stats_row = dict(net=net, n=g.n, m=g.m, mean_degree=g.m / g.n, reciprocity=g.reciprocity(),
                     p_c=pc, F0=info.get("F0"), trophic_residual=info.get("residual"),
                     flip=g.meta.get("flip"), arc_meaning=g.meta.get("meaning"),
                     verify=g.meta.get("verify"), role=g.meta.get("role"))
    append_csv(pd.DataFrame([stats_row]), os.path.join(out, "datasets.csv"))
    print(f"    F0={info.get('F0'):.3f}  reciprocity={stats_row['reciprocity']:.3f}  p_c={pc:.4g}")

    mdf = pd.DataFrame(scores)
    mdf.insert(0, "trophic_level", info["levels"])
    mdf.insert(0, "node", g.labels if g.labels is not None else np.arange(g.n))
    mdf.to_csv(os.path.join(out, f"measures_{net}.csv"), index=False)

    # ---- RQ1/RQ2 agreement ----------------------------------------------------
    ref = scores["trophic"]
    active = (g.out_degree() + g.in_degree()) > 0
    ag = []
    for name in ["netdegree", "netdegree_norm", "netdegree_weighted", "netdegree_2hop",
                 "jacobi_1", "jacobi_2", "jacobi_5", "jacobi_10", "spring_rank_p99",
                 "spring_rank_tailadaptive", "spring_rank_maxdeg", "out_degree", "pagerank_rev"]:
        if name in scores:
            ag.append(dict(net=net, source="real" if not net.startswith("trophic_T") else "synthetic",
                           measure=name, spearman_vs_trophic=evaluate.spearman(scores[name][active], ref[active]),
                           F0=info["F0"]))
    append_csv(pd.DataFrame(ag), os.path.join(out, "agreement.csv"))

    # ---- RQ3 single-node spreading ability ------------------------------------
    small = g.n <= args.exact_below
    all_nodes = np.arange(g.n) if small else diffusion.stratified_sample(g, args.sample, rng_seed=1)
    # supercritical settings: every run infects much of the network, so use a node sample
    super_nodes = diffusion.stratified_sample(g, min(args.sample_super, all_nodes.size), rng_seed=2)
    gt_parts = []
    tau_rows = []
    for st in sets:
        supercrit = np.isfinite(st["p_over_pc"]) and st["p_over_pc"] > 1
        nodes = super_nodes if supercrit else all_nodes
        runs = args.runs_large if (supercrit or not small) else args.runs_small
        t0 = time.perf_counter()
        sigma = diffusion.single_node_spreading(g, nodes, st["model"], st["param"], runs, 0.5, rng_seed=11)
        gt_parts.append(pd.DataFrame({"node_index": nodes, "setting": st["setting"], "sigma": sigma}))
        print(f"    ground truth {st['setting']:26s} runs={runs} nodes={nodes.size:,} "
              f"({time.perf_counter() - t0:.1f}s)  mean sigma={sigma.mean():.2f}")
        for name, sc in scores.items():
            s = sc[nodes]
            tau_rows.append(dict(net=net, model=st["model"], setting=st["setting"], p_over_pc=st["p_over_pc"],
                                 measure=name, tau=evaluate.kendall_tau(s, sigma),
                                 eps_1=evaluate.imprecision(s, sigma, 0.01),
                                 eps_5=evaluate.imprecision(s, sigma, 0.05),
                                 eps_10=evaluate.imprecision(s, sigma, 0.10),
                                 n_sample=nodes.size, runs=runs))
    gt = pd.concat(gt_parts, ignore_index=True)
    gt.to_csv(os.path.join(out, f"groundtruth_{net}.csv"), index=False)
    append_csv(pd.DataFrame(tau_rows), os.path.join(out, "tau.csv"))

    # ---- RQ4 seed-set spread -------------------------------------------------
    ks = [k for k in args.k if k <= max(1, g.n // 10)]
    if not ks:
        append_csv(pd.DataFrame(rt_rows), os.path.join(out, "runtime.csv"))
        return
    kmax = max(ks)
    ic_main = [s for s in sets if s["setting"] == "IC p=1pc"][0]
    seed_sets = {}
    for name in SEED_METHOD_BASE:
        if name in scores:
            seed_sets[(name, "top-k")] = seeds.top_k(scores[name], kmax)
            seed_sets[(name, "top-k+exclusion")] = seeds.top_k_exclusion(g, scores[name], kmax)
    t0 = time.perf_counter()
    seed_sets[("voterank", "set")] = seeds.voterank(g, kmax)
    rt_rows.append(dict(net=net, n=g.n, m=g.m, method="voterank", seconds=time.perf_counter() - t0))
    t0 = time.perf_counter()
    seed_sets[("degree_discount", "set")] = seeds.degree_discount_ic(g, kmax, ic_main["p"])
    rt_rows.append(dict(net=net, n=g.n, m=g.m, method="degree_discount", seconds=time.perf_counter() - t0))
    seed_sets[("random", "set")] = np.random.default_rng(3).choice(g.n, kmax, replace=False)
    if small and not args.no_celf:
        pool = np.unique(np.concatenate([seeds.top_k(scores[m], 100) for m in SEED_METHOD_BASE if m in scores]))
        t0 = time.perf_counter()
        seed_sets[("celf", "set")] = seeds.celf(g, kmax, "IC", ic_main["param"], runs=args.celf_runs,
                                                candidates=pool)
        rt_rows.append(dict(net=net, n=g.n, m=g.m, method="celf", seconds=time.perf_counter() - t0))
        print(f"    CELF over {pool.size} candidates: {time.perf_counter() - t0:.1f}s")
    append_csv(pd.DataFrame(rt_rows), os.path.join(out, "runtime.csv"))

    spread_rows = []
    eval_sets = [ic_main] + [s for s in sets if s["model"] == "LT"]
    t0 = time.perf_counter()
    for st in eval_sets:
        for (method, variant), S in seed_sets.items():
            for k in ks:
                samples = diffusion.spread(g, S[:k], st["model"], st["param"], args.spread_runs, 0.5, rng_seed=5)
                lo, hi = evaluate.bootstrap_ci(samples)
                spread_rows.append(dict(net=net, model=st["model"], setting=st["setting"], k=k, method=method,
                                        variant=variant, spread=samples.mean(), ci_low=lo, ci_high=hi,
                                        redundancy=evaluate.redundancy(g, S[:k]), runs=args.spread_runs))
    append_csv(pd.DataFrame(spread_rows), os.path.join(out, "spread.csv"))
    print(f"    seed-set spread: {len(seed_sets)} sets x {len(ks)} budgets x {len(eval_sets)} models "
          f"({time.perf_counter() - t0:.1f}s)")
    best = pd.DataFrame(spread_rows).query("k == @kmax and model == 'IC'").sort_values("spread", ascending=False)
    print(best[["method", "variant", "spread", "redundancy"]].head(8).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=[], help="registry names, or 'all'")
    ap.add_argument("--edgelist", help="path to a local edge list instead of a registry dataset")
    ap.add_argument("--flip", action="store_true", help="with --edgelist: reverse arcs on load")
    ap.add_argument("--weight-col", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--synthetic", nargs="*", type=float, default=None,
                    help="temperatures T for trophic-tunable synthetic graphs")
    ap.add_argument("--synthetic-n", type=int, default=2000)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="fewer settings and runs (smoke test)")
    ap.add_argument("--sample", type=int, default=2000, help="nodes sampled for ground truth on big graphs")
    ap.add_argument("--sample-super", type=int, default=1000,
                    help="nodes sampled for supercritical settings (p > p_c)")
    ap.add_argument("--runs-small", type=int, default=1000)
    ap.add_argument("--runs-large", type=int, default=200)
    ap.add_argument("--spread-runs", type=int, default=1000)
    ap.add_argument("--celf-runs", type=int, default=200)
    ap.add_argument("--no-celf", action="store_true")
    ap.add_argument("--exact-below", type=int, default=SMALL,
                    help="networks with fewer nodes get ground truth for every node and CELF")
    ap.add_argument("--k", type=int, nargs="*", default=[10, 25, 50, 100])
    args = ap.parse_args(argv)
    if args.quick:
        args.runs_small, args.runs_large, args.spread_runs, args.celf_runs = 100, 50, 200, 50
        args.sample = min(args.sample, 500)
    os.makedirs(args.out, exist_ok=True)

    graphs = []
    names = list(datasets.REGISTRY) if args.datasets == ["all"] else args.datasets
    names = [n for n in names if n != "higgs-social"]  # handled by run_higgs.py
    for name in names:
        graphs.append(lambda name=name: datasets.load(name, args.data_dir))
    if args.edgelist:
        def _load():
            g = read_edgelist(args.edgelist, flip=args.flip, weight_col=args.weight_col,
                              name=args.name or os.path.splitext(os.path.basename(args.edgelist))[0])
            g.meta.update(dict(flip=args.flip, meaning="user supplied", verify=True, role="user"))
            return g
        graphs.append(_load)
    for T in args.synthetic or []:
        def _syn(T=T):
            g, _ = synthetic.trophic_tunable(n=args.synthetic_n, T=T, rng_seed=int(T * 1000))
            g.meta.update(dict(flip=False, meaning="synthetic", verify=False, role="synthetic"))
            return g
        graphs.append(_syn)
    if not graphs:
        ap.error("nothing to run: give --datasets, --edgelist or --synthetic")
    for loader in graphs:
        g = loader()
        run_one(g, args)
    print(f"\nDone. Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    sys.exit(main())
