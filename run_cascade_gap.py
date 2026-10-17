#!/usr/bin/env python3
"""Why does net-degree add information beyond follower count on Higgs but not on Digg?

On both follower graphs an arc points from the followed (watched) user to the follower,
so out-degree counts followers and in-degree counts the accounts a user follows. Given
out-degree, net-degree (out - in) can only add information through in-degree, so the
sign of its partial correlation with the outcome is set by how following many accounts
relates to the outcome. This script measures, per dataset:

  * structure: reciprocity, Spearman(in-degree, out-degree), node-level reciprocity;
  * activity: the user's own actions (Higgs: retweets, replies and mentions the user
    made; Digg: stories the user voted on) and its rank correlation with in-degree;
  * partial Spearman of net-degree and in-degree with the outcome, controlling for
    out-degree alone and for out-degree plus activity;
  * the same partial correlations within strata of node reciprocity and of activity.

Outcomes are those of run_higgs.py (distinct retweeters) and run_digg.py (distinct fans).

  python run_cascade_gap.py --out results     ->  <out>/cascade_gap_summary.csv
                                                  <out>/cascade_gap_strata.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import run_digg
import run_higgs
from ndbench import datasets
from ndbench.graph import from_edges, read_edgelist


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y controlling for every column of `controls`."""
    r = [stats.rankdata(v) for v in (x, y, *controls)]
    A = np.column_stack(r[2:] + [np.ones_like(r[0])])

    def resid(a):
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return float(np.corrcoef(resid(r[0]), resid(r[1]))[0, 1])


def node_reciprocity(g):
    """Share of a node's distinct neighbours linked in both directions (0 for isolates)."""
    key = g.src * g.n + g.dst
    mutual = np.isin(key, g.dst * g.n + g.src)
    kout = np.bincount(g.src, minlength=g.n)
    kin = np.bincount(g.dst, minlength=g.n)
    mu = np.bincount(g.src[mutual], minlength=g.n)
    nb = kout + kin - mu
    return np.divide(mu, nb, out=np.zeros(g.n), where=nb > 0), kout.astype(float), kin.astype(float)


def higgs(data_dir):
    social = datasets.download(datasets.REGISTRY["higgs-social"]["url"], data_dir)
    activity = datasets.download(datasets.HIGGS_ACTIVITY_URL, data_dir)
    raw = read_edgelist(social, flip=False, name="higgs-social")
    act = run_higgs.load_activity(activity, raw.labels)
    flip = run_higgs.direction_check(raw, act)["flip"]
    g = raw.reversed() if flip else raw
    y = run_higgs.distinct_retweeters(act, g.n)
    own = np.bincount(act.ia.to_numpy(), minlength=g.n).astype(float)   # actions the user made
    return g, np.arange(g.n), y, own, "distinct retweeters", "own retweets, replies and mentions"


def digg(data_dir, reverse=False):
    f = run_digg.read_konect(run_digg.fetch("digg-friends", data_dir), ["a", "b", "w", "t"])
    v = run_digg.read_konect(run_digg.fetch("digg-votes", data_dir), ["u", "s", "w", "t"])
    f = f[f.a != f.b].drop_duplicates(["a", "b"])
    v = v.groupby(["u", "s"], as_index=False).t.min()
    n = int(max(f.a.max(), f.b.max(), v.u.max())) + 1
    flip = run_digg.direction_check(f, v)["flip"] ^ reverse
    g = from_edges(f.a.to_numpy(), f.b.to_numpy(), flip=flip, n=n, labels=np.arange(n), name="digg-friends")
    keep = np.bincount(np.r_[g.src, g.dst], minlength=n) > 0
    keep[v.u.unique()] = True
    keep[0] = False
    y = run_digg.distinct_fans(run_digg.credited(g, v), n)
    own = np.bincount(v.u.to_numpy(), minlength=n).astype(float)        # stories the user voted on
    return g, np.flatnonzero(keep), y, own, "distinct fans", "stories voted on"


def analyse(name, g, idx, y, own, outcome, activity):
    rec, kout, kin = (a[idx] for a in node_reciprocity(g))
    y, own = y[idx], own[idx]
    nd = kout - kin
    summary = dict(
        dataset=name, outcome=outcome, activity=activity, n=idx.size, m=g.m, reciprocity=g.reciprocity(),
        spearman_in_out=stats.spearmanr(kin, kout)[0],
        median_node_reciprocity=float(np.median(rec[kin + kout > 0])),
        share_nodes_follow_none=float(np.mean(kin == 0)),
        share_active=float(np.mean(own > 0)),
        spearman_in_activity=stats.spearmanr(kin, own)[0],
        spearman_out_activity=stats.spearmanr(kout, own)[0],
        spearman_activity_outcome=stats.spearmanr(own, y)[0],
        partial_activity_ctrl_out=partial_spearman(own, y, [kout]),
        partial_netdegree_ctrl_out=partial_spearman(nd, y, [kout]),
        partial_in_ctrl_out=partial_spearman(kin, y, [kout]),
        partial_netdegree_ctrl_out_activity=partial_spearman(nd, y, [kout, own]),
        partial_in_ctrl_out_activity=partial_spearman(kin, y, [kout, own]),
    )
    act = own > 0
    summary.update(
        active_partial_netdegree_ctrl_out=partial_spearman(nd[act], y[act], [kout[act]]),
        active_partial_netdegree_ctrl_out_activity=partial_spearman(nd[act], y[act], [kout[act], own[act]]))

    strata = []
    rbin = pd.cut(rec, [-0.001, 0.0, 0.5, 1.0], labels=["0", "(0, 0.5]", "(0.5, 1]"])
    abin = np.where(own == 0, "none", pd.qcut(pd.Series(own).where(own > 0).rank(method="first"), 3,
                                               labels=["low", "mid", "high"]).astype(str))
    for kind, bins, order in (("node reciprocity", np.asarray(rbin, str), ["0", "(0, 0.5]", "(0.5, 1]"]),
                              ("activity tertile", abin, ["none", "low", "mid", "high"])):
        for b in order:
            s = bins == b
            if s.sum() < 100 or np.ptp(y[s]) == 0:
                continue
            strata.append(dict(dataset=name, stratum=kind, bin=b, n=int(s.sum()), mean_outcome=float(y[s].mean()),
                               partial_netdegree_ctrl_out=partial_spearman(nd[s], y[s], [kout[s]]),
                               partial_in_ctrl_out=partial_spearman(kin[s], y[s], [kout[s]])))
    return summary, strata


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows, strata = [], []
    for name, load in (("Higgs", lambda: higgs(args.data_dir)),
                       ("Digg", lambda: digg(args.data_dir)),
                       ("Digg (reversed)", lambda: digg(args.data_dir, reverse=True))):
        print(f"=== {name}", flush=True)
        s, st = analyse(name, *load())
        rows.append(s)
        strata += st
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out, "cascade_gap_summary.csv"), index=False)
    pd.DataFrame(strata).to_csv(os.path.join(args.out, "cascade_gap_strata.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.set_index("dataset").T.to_string())
    print(pd.DataFrame(strata).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
