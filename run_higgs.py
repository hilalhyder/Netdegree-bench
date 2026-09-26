#!/usr/bin/env python3
"""RQ5: do netdegree-family scores on the Higgs follower graph predict who
drives the observed retweet cascades?

Data (SNAP, De Domenico et al. 2013):
  higgs-social_network.edgelist.gz   follower graph, 456,626 nodes, 14,855,842 arcs
  higgs-activity_time.txt.gz         'userA userB timestamp RT|MT|RE'
                                     (RT: userA retweeted userB -> information flowed B -> A)

Steps
  1. Direction check: which way does the follower file point?  If a retweet
     'A retweeted B' mostly coincides with the raw arc A -> B, the file means
     'A follows B' and must be flipped to follow information flow.
  2. Scores on the (re-oriented) follower graph.
  3. Outcomes: y1 = distinct retweeters per user; y2 = time-respecting cascade
     reach in the retweet network (on a node sample).
  4. Metrics: Spearman, partial Spearman controlling for out-degree (follower
     count), ROC-AUC and precision@k for the top-1% most-retweeted users,
     decile tables.  Repeated for a time split: scores from before 4 July 2012,
     outcomes from 4 July onward.

  python run_higgs.py                                   # downloads from SNAP
  python run_higgs.py --social path/to/higgs-social_network.edgelist.gz \
                      --activity path/to/higgs-activity_time.txt.gz
  python run_higgs.py --fixture                         # tiny synthetic stand-in (testing)
"""
from __future__ import annotations

import argparse
import heapq
import os
import time

import numba as nb
import numpy as np
import pandas as pd

from ndbench import datasets, evaluate, measures, synthetic
from ndbench.graph import from_edges, read_edgelist

CUTOFF_4JULY = 1341360000  # 2012-07-04 00:00:00 UTC (announcement day)
MEASURES = ["netdegree", "netdegree_norm", "netdegree_2hop", "trophic", "spring_rank_p99",
            "spring_rank_tailadaptive", "spring_rank_maxdeg", "jacobi_2", "jacobi_5",
            "out_degree", "in_degree", "pagerank_rev", "leaderrank", "kshell"]


def load_activity(path, labels):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["a", "b", "t", "kind"],
                     dtype={"a": np.int64, "b": np.int64, "t": np.int64, "kind": str}, engine="c")
    ia = np.searchsorted(labels, df["a"].to_numpy())
    ib = np.searchsorted(labels, df["b"].to_numpy())
    ia[ia >= labels.size] = 0
    ib[ib >= labels.size] = 0
    ok = (labels[ia] == df["a"].to_numpy()) & (labels[ib] == df["b"].to_numpy())
    if (~ok).any():
        print(f"  note: {int((~ok).sum()):,} activity rows reference users missing from the follower graph; dropped")
    df = df[ok].copy()
    df["ia"], df["ib"] = ia[ok], ib[ok]
    return df


def direction_check(raw, act):
    """Share of 'A retweeted B' pairs matched by raw arc A->B vs B->A."""
    rt = act[act.kind == "RT"][["ia", "ib"]].drop_duplicates()
    n = raw.n
    keys = np.sort(raw.src * n + raw.dst)

    def frac(x, y):
        q = x * n + y
        pos = np.searchsorted(keys, q)
        pos[pos >= keys.size] = 0
        return float(np.mean(keys[pos] == q))
    fwd = frac(rt.ia.to_numpy(), rt.ib.to_numpy())   # raw arc retweeter -> author
    bwd = frac(rt.ib.to_numpy(), rt.ia.to_numpy())   # raw arc author -> retweeter
    flip = fwd > bwd
    return dict(rt_pairs=len(rt), share_raw_retweeter_to_author=fwd, share_raw_author_to_retweeter=bwd,
                decision="flip (file means 'A follows B')" if flip else "keep (file already author -> follower)",
                flip=flip)


def distinct_retweeters(act, n, t_from=None, t_to=None):
    rt = act[act.kind == "RT"]
    if t_from is not None:
        rt = rt[rt.t >= t_from]
    if t_to is not None:
        rt = rt[rt.t < t_to]
    pairs = rt[["ia", "ib"]].drop_duplicates()
    return np.bincount(pairs.ib.to_numpy(), minlength=n).astype(float)


def interaction_graph(act, n, t_to):
    """Information-flow graph from activity before t_to: RT reversed (B->A), MT/RE kept (A->B)."""
    a = act[act.t < t_to]
    rt = a.kind == "RT"
    src = np.where(rt, a.ib, a.ia)
    dst = np.where(rt, a.ia, a.ib)
    return from_edges(src, dst, n=n, name="higgs-interactions-pre")


@nb.njit(cache=True)
def _temporal_reach(n, tindptr, tdst, ttime, sources):
    """Earliest-arrival reach along time-respecting paths (edge usable if t >= arrival)."""
    out = np.zeros(sources.size, dtype=np.int64)
    arrival = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    stamp = np.full(n, -1, dtype=np.int64)
    for si in range(sources.size):
        s = sources[si]
        heap = [(np.int64(-1), s)]
        stamp[s] = si
        arrival[s] = -1
        cnt = 0
        while len(heap) > 0:
            a, u = heapq.heappop(heap)
            if a > arrival[u]:
                continue
            lo, hi = tindptr[u], tindptr[u + 1]
            # edges of u are sorted by time: binary search first t >= a
            L, R = lo, hi
            while L < R:
                mid = (L + R) // 2
                if ttime[mid] < a:
                    L = mid + 1
                else:
                    R = mid
            for k in range(L, hi):
                v = tdst[k]
                t = ttime[k]
                if stamp[v] != si:
                    stamp[v] = si
                    arrival[v] = t
                    cnt += 1
                    heapq.heappush(heap, (t, v))
                elif t < arrival[v]:
                    arrival[v] = t
                    heapq.heappush(heap, (t, v))
        out[si] = cnt
    return out


def temporal_reach(act, n, nodes):
    rt = act[act.kind == "RT"]
    src, dst, t = rt.ib.to_numpy(), rt.ia.to_numpy(), rt.t.to_numpy()
    order = np.lexsort((t, src))
    src, dst, t = src[order], dst[order], t[order]
    tindptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=tindptr[1:])
    return _temporal_reach(n, tindptr, dst.astype(np.int64), t.astype(np.int64), np.asarray(nodes, np.int64))


def score_block(scores, y, control, label, outcome, top_frac=0.01):
    rows = []
    pos = y >= np.quantile(y, 1 - top_frac) if (y > 0).any() else np.zeros(y.size, bool)
    pos &= y > 0
    npos = int(pos.sum())
    for name, s in scores.items():
        rows.append(dict(split=label, outcome=outcome, measure=name, n=y.size, positives=npos,
                         spearman=evaluate.spearman(s, y),
                         partial_spearman_ctrl_outdeg=evaluate.partial_spearman(s, y, control)
                         if name != "out_degree" else np.nan,
                         auc_top1pct=evaluate.auc(s, pos),
                         prec_at_100=evaluate.precision_at_k(s, pos, 100),
                         prec_at_1000=evaluate.precision_at_k(s, pos, 1000),
                         prec_at_npos=evaluate.precision_at_k(s, pos, max(npos, 1))))
    return rows


def deciles(scores, y, label):
    rows = []
    for name, s in scores.items():
        r = pd.Series(s).rank(method="first")
        d = pd.qcut(r, 10, labels=False) + 1
        g = pd.DataFrame({"d": d, "y": y}).groupby("d")["y"]
        for dec, grp in g:
            rows.append(dict(split=label, measure=name, decile=int(dec), mean_y=grp.mean(),
                             median_y=grp.median(), share_retweeted=float((grp > 0).mean()), n=len(grp)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--social")
    ap.add_argument("--activity")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cutoff", type=int, default=CUTOFF_4JULY)
    ap.add_argument("--reach-sample", type=int, default=3000)
    ap.add_argument("--fixture", action="store_true", help="use a small synthetic stand-in (pipeline test)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.fixture:
        folder = synthetic.higgs_like_fixture(folder=os.path.join(args.data_dir, "higgs_fixture"))
        args.social = os.path.join(folder, "higgs-social_network.edgelist")
        args.activity = os.path.join(folder, "higgs-activity_time.txt")
        args.cutoff = 1341100800 + 3 * 86400
    social = args.social or datasets.download(datasets.REGISTRY["higgs-social"]["url"], args.data_dir)
    activity = args.activity or datasets.download(datasets.HIGGS_ACTIVITY_URL, args.data_dir)

    t0 = time.perf_counter()
    raw = read_edgelist(social, flip=False, name="higgs-social")
    print(f"follower graph: n={raw.n:,} m={raw.m:,} ({time.perf_counter() - t0:.1f}s)")
    act = load_activity(activity, raw.labels)
    print(f"activity rows: {len(act):,}  ({act.kind.value_counts().to_dict()})")

    # 1. direction ----------------------------------------------------------
    chk = direction_check(raw, act)
    print("direction check:", {k: v for k, v in chk.items() if k != "flip"})
    pd.DataFrame([chk]).to_csv(os.path.join(args.out, "higgs_direction_check.csv"), index=False)
    g = raw.reversed() if chk["flip"] else raw
    g.name = "higgs-social"
    g.meta.update(dict(flip=chk["flip"], meaning="follower graph, oriented author -> follower", verify=False,
                       role="RQ5"))

    # 2. scores ---------------------------------------------------------------
    scores, rt, info = measures.compute_all(g, include=MEASURES, verbose=True)
    print(f"  F0 = {info['F0']:.3f}")
    pd.DataFrame([dict(net=g.name, n=g.n, m=g.m, method=k, seconds=v) for k, v in rt.items()]).to_csv(
        os.path.join(args.out, "higgs_runtime.csv"), index=False)

    # 3/4. outcomes and metrics ---------------------------------------------
    control = scores["out_degree"]
    y_all = distinct_retweeters(act, g.n)
    rows = score_block(scores, y_all, control, "all activity", "distinct retweeters")
    dec = deciles(scores, y_all, "all activity")

    # time split
    y_pre = distinct_retweeters(act, g.n, t_to=args.cutoff)
    y_post = distinct_retweeters(act, g.n, t_from=args.cutoff)
    split_scores = dict(scores)
    split_scores["past_retweeters (pre)"] = y_pre
    gi = interaction_graph(act, g.n, args.cutoff)
    split_scores["netdegree (pre interactions)"] = measures.netdegree(gi)
    split_scores["out_degree (pre interactions)"] = measures.out_degree(gi)
    split_scores["trophic (pre interactions)"] = measures.trophic_source_score(gi)
    rows += score_block(split_scores, y_post, control, "scores pre-cutoff -> outcome post-cutoff",
                        "distinct retweeters")
    dec += deciles({k: split_scores[k] for k in ["netdegree", "out_degree", "trophic", "past_retweeters (pre)"]},
                   y_post, "pre -> post")

    # y2: temporal reach on a sample
    rng = np.random.default_rng(0)
    active = np.flatnonzero(y_all > 0)
    pick = [rng.choice(active, min(args.reach_sample, active.size), replace=False)]
    pick += [np.argsort(-scores[m])[:200] for m in scores]
    nodes = np.unique(np.concatenate(pick))
    t1 = time.perf_counter()
    reach = temporal_reach(act, g.n, nodes).astype(float)
    print(f"temporal reach for {nodes.size:,} users: {time.perf_counter() - t1:.1f}s")
    pd.DataFrame({"node": g.labels[nodes], "temporal_reach": reach, "distinct_retweeters": y_all[nodes]}).to_csv(
        os.path.join(args.out, "higgs_reach.csv"), index=False)
    rows += score_block({k: v[nodes] for k, v in scores.items()}, reach, control[nodes], "sample",
                        "temporal cascade reach")

    # precision@k curve (all activity, top-1% most-retweeted users as positives)
    pos = (y_all >= np.quantile(y_all, 0.99)) & (y_all > 0)
    ks = np.unique(np.logspace(1, np.log10(max(20, min(g.n, 20 * max(pos.sum(), 1)))), 25).astype(int))
    curve = [dict(measure=m, k=int(k), precision=evaluate.precision_at_k(scores[m], pos, int(k)))
             for m in scores for k in ks]
    pd.DataFrame(curve).to_csv(os.path.join(args.out, "higgs_precision_curve.csv"), index=False)

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(args.out, "higgs_metrics.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(args.out, "higgs_deciles.csv"), index=False)
    pd.set_option("display.width", 160)
    print(met[["split", "outcome", "measure", "spearman", "partial_spearman_ctrl_outdeg", "auc_top1pct",
               "prec_at_100"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
