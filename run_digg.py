#!/usr/bin/env python3
"""RQ5 replication on a second real-cascade dataset: Digg 2009 (Lerman & Ghosh 2010;
Hogg & Lerman 2012), via KONECT.

Data (http://konect.cc/networks/digg-friends, .../digg-votes):
  digg-friends/out.digg-friends   'a b 1 t'  directed friendship link a -> b
  digg-votes/out.digg-votes       'u s 1 t'  user u voted on story s at time t

Digg showed each user the stories their friends voted on, so a vote can spread from a
user to the users who watch them. There is no explicit attribution (unlike a retweet),
so the outcome follows the standard 'fan vote' definition: a vote by v on story s is
credited to every user u that v watches who voted on s earlier.

Steps
  1. Direction check: which end of a raw link is the watched user (see direction_check)?
     Information flows from the watched user to the watcher.
  2. Scores on the re-oriented friendship graph (same measures as run_higgs.py).
  3. Outcomes: y1 = distinct fans credited per user; y2 = within-story cascade reach
     (users reachable from u in the story's credited-vote graph, summed over u's
     stories; computed on a node sample).
  4. Metrics as for Higgs. Time split: scores and past credited fans from stories that
     started before the cutoff; outcome from stories that started after it.

  python run_digg.py                       # downloads both KONECT archives
  python run_digg.py --direction reverse   # sensitivity: the opposite reading of every link,
                                           # written to <out>/digg_reversed_*.csv
"""
from __future__ import annotations

import argparse
import os
import tarfile
import time
import urllib.request

import numba as nb
import numpy as np
import pandas as pd

from ndbench import measures
from ndbench.graph import from_edges
from run_higgs import MEASURES, deciles, score_block

KONECT = "http://konect.cc/files/download.tsv.{}.tar.bz2"
CUTOFF = 1245283200  # 2009-06-18 00:00 UTC, the middle of the vote period


def fetch(name, data_dir):
    folder = os.path.join(data_dir, "digg")
    path = os.path.join(folder, name, f"out.{name}")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        arc = os.path.join(folder, f"{name}.tar.bz2")
        if not os.path.exists(arc):
            print(f"downloading {name}")
            urllib.request.urlretrieve(KONECT.format(name), arc)
        with tarfile.open(arc) as t:
            t.extractall(folder, filter="data")
    return path


def read_konect(path, cols):
    return pd.read_csv(path, sep=r"\s+", comment="%", header=None, names=cols, engine="c")


def direction_check(f, v, early=100):
    """Which end of a link a -> b is the watched user? Three signals:
      * degree tails: a user can watch a limited number of others, but can be watched
        by any number, so the watched side has the far heavier tail;
      * story submitters (first voter of each story) are typically watched users;
      * co-vote timing on one-way links among each story's first ``early`` votes: the
        watched user tends to vote first.
    The decision follows the degree tail and the submitters; the timing signal is
    reported as well."""
    da, db = f.a.value_counts(), f.b.value_counts()
    first = v.sort_values("t").drop_duplicates("s").u
    key, rkey = f.a * (f.b.max() + 1) + f.b, f.b * (f.b.max() + 1) + f.a
    one = f[~np.isin(rkey, key)]
    ve = v[v.groupby("s").t.rank(method="first") <= early]
    pairs = (ve.rename(columns={"u": "a", "t": "ta"}).merge(one[["a", "b"]], on="a")
             .merge(ve.rename(columns={"u": "b", "t": "tb"}), on=["b", "s"]))
    a_after_b, b_after_a = int((pairs.ta > pairs.tb).sum()), int((pairs.ta < pairs.tb).sum())
    sub_a, sub_b = float(first.map(da).fillna(0).median()), float(first.map(db).fillna(0).median())
    flip = bool(db.max() > da.max() and sub_b > sub_a)
    return dict(max_links_from_a=int(da.max()), max_links_to_b=int(db.max()),
                submitter_median_links_from=sub_a, submitter_median_links_to=sub_b,
                one_way_links=len(one), early_votes=early, covotes_source_after_target=a_after_b,
                covotes_target_after_source=b_after_a,
                decision="flip (a -> b means 'a watches b')" if flip else "keep (a -> b means 'b watches a')",
                flip=flip)


@nb.njit(cache=True)
def _credit(n, rindptr, rindices, vs, vu, vt):
    """Votes sorted by (story, time). For each vote (v, s, t), every u with an arc u -> v
    in the information-flow graph who voted on s before t gets credited (u, v, s, t)."""
    stamp = np.full(n, -1, dtype=np.int64)
    when = np.zeros(n, dtype=np.int64)
    cap = 1 << 20
    cu = np.empty(cap, np.int64)
    cv = np.empty(cap, np.int64)
    cs = np.empty(cap, np.int64)
    ct = np.empty(cap, np.int64)
    m = 0
    for i in range(vs.size):
        s, v, t = vs[i], vu[i], vt[i]
        for kk in range(rindptr[v], rindptr[v + 1]):
            u = rindices[kk]
            if stamp[u] == s and when[u] < t:
                if m == cap:
                    cap *= 2
                    cu2, cv2, cs2, ct2 = np.empty(cap, np.int64), np.empty(cap, np.int64), \
                        np.empty(cap, np.int64), np.empty(cap, np.int64)
                    cu2[:m], cv2[:m], cs2[:m], ct2[:m] = cu[:m], cv[:m], cs[:m], ct[:m]
                    cu, cv, cs, ct = cu2, cv2, cs2, ct2
                cu[m], cv[m], cs[m], ct[m] = u, v, s, t
                m += 1
        stamp[v] = s
        when[v] = t
    return cu[:m], cv[:m], cs[:m], ct[:m]


@nb.njit(cache=True)
def _story_reach(keys, dst, vote_u, vote_s, sources, n):
    """keys = story * n + u sorted, dst aligned; (vote_u, vote_s) sorted by user. For each source u, total number of users
    reachable from u inside each story u voted in (credited arcs only)."""
    out = np.zeros(sources.size, dtype=np.int64)
    stamp = np.full(n, -1, dtype=np.int64)
    tag = 0
    for si in range(sources.size):
        u0 = sources[si]
        lo_s, hi_s = np.searchsorted(vote_u, u0), np.searchsorted(vote_u, u0 + 1)
        for r in range(lo_s, hi_s):
            s = vote_s[r]
            tag += 1
            stack = [u0]
            stamp[u0] = tag
            while len(stack) > 0:
                u = stack.pop()
                q = s * n + u
                lo, hi = np.searchsorted(keys, q), np.searchsorted(keys, q + 1)
                for kk in range(lo, hi):
                    w = dst[kk]
                    if stamp[w] != tag:
                        stamp[w] = tag
                        out[si] += 1
                        stack.append(w)
    return out


def credited(g, votes):
    vv = votes.sort_values(["s", "t"], kind="stable")
    rindptr, rindices, _ = g.in_csr()
    cu, cv, cs, ct = _credit(g.n, rindptr, rindices, vv.s.to_numpy(np.int64), vv.u.to_numpy(np.int64),
                             vv.t.to_numpy(np.int64))
    return pd.DataFrame({"u": cu, "v": cv, "s": cs, "t": ct})


def distinct_fans(cr, n):
    pairs = cr[["u", "v"]].drop_duplicates()
    return np.bincount(pairs.u.to_numpy(), minlength=n).astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cutoff", type=int, default=CUTOFF)
    ap.add_argument("--reach-sample", type=int, default=3000)
    ap.add_argument("--direction", choices=["auto", "reverse"], default="auto",
                    help="'reverse' uses the opposite of the direction check's decision")
    args = ap.parse_args(argv)
    prefix = "digg" if args.direction == "auto" else "digg_reversed"
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    f = read_konect(fetch("digg-friends", args.data_dir), ["a", "b", "w", "t"])
    v = read_konect(fetch("digg-votes", args.data_dir), ["u", "s", "w", "t"])
    f = f[f.a != f.b].drop_duplicates(["a", "b"])
    v = v.groupby(["u", "s"], as_index=False).t.min()        # repeated votes: keep the first
    n = int(max(f.a.max(), f.b.max(), v.u.max())) + 1
    print(f"friend links {len(f):,}, votes {len(v):,} by {v.u.nunique():,} users on {v.s.nunique():,} stories "
          f"({time.perf_counter() - t0:.1f}s)")

    # 1. direction ----------------------------------------------------------
    chk = direction_check(f, v)
    print("direction check:", {k: x for k, x in chk.items() if k != "flip"})
    flip = chk["flip"] if args.direction == "auto" else not chk["flip"]
    print(f"orientation: flip={flip} ({args.direction})")
    pd.DataFrame([dict(chk, direction=args.direction, applied_flip=flip)]).to_csv(
        os.path.join(args.out, f"{prefix}_direction_check.csv"), index=False)
    g = from_edges(f.a.to_numpy(), f.b.to_numpy(), flip=flip, n=n, labels=np.arange(n), name="digg-friends")
    keep = np.bincount(np.r_[g.src, g.dst], minlength=n) > 0
    keep[v.u.unique()] = True
    keep[0] = False                                           # KONECT ids start at 1
    print(f"graph: n={int(keep.sum()):,} active users, m={g.m:,}, reciprocity={g.reciprocity():.3f}")

    # 2. scores ---------------------------------------------------------------
    scores, rt, info = measures.compute_all(g, include=MEASURES, verbose=True)
    print(f"  F0 = {info['F0']:.3f}")
    pd.DataFrame([dict(net=g.name, n=g.n, m=g.m, method=k, seconds=x) for k, x in rt.items()]).to_csv(
        os.path.join(args.out, f"{prefix}_runtime.csv"), index=False)

    # 3/4. outcomes and metrics ---------------------------------------------
    cr = credited(g, v)
    print(f"credited fan votes: {len(cr):,} ({cr[['v', 's']].drop_duplicates().shape[0] / len(v):.1%} of votes "
          f"follow an earlier vote by a watched user)")
    idx = np.flatnonzero(keep)
    sc = {k: x[idx] for k, x in scores.items()}
    control = sc["out_degree"]
    y_all = distinct_fans(cr, n)[idx]
    rows = score_block(sc, y_all, control, "all activity", "distinct fans")
    dec = deciles(sc, y_all, "all activity")

    start = v.groupby("s").t.min()
    early, late = start.index[start < args.cutoff], start.index[start >= args.cutoff]
    y_pre = distinct_fans(cr[cr.s.isin(early)], n)[idx]
    y_post = distinct_fans(cr[cr.s.isin(late)], n)[idx]
    print(f"time split: {len(early):,} stories before the cutoff, {len(late):,} after")
    split = dict(sc)
    split["past_fans (pre)"] = y_pre
    rows += score_block(split, y_post, control, "scores pre-cutoff -> outcome post-cutoff", "distinct fans")
    dec += deciles({k: split[k] for k in ["netdegree", "out_degree", "trophic", "past_fans (pre)"]}, y_post,
                   "pre -> post")

    # y2: within-story cascade reach on a sample
    rng = np.random.default_rng(0)
    voters = np.unique(v.u.to_numpy())
    pick = [rng.choice(voters, min(args.reach_sample, voters.size), replace=False)]
    pick += [idx[np.argsort(-sc[m])[:200]] for m in sc]
    nodes = np.unique(np.concatenate(pick))
    order = np.lexsort((cr.u.to_numpy(), cr.s.to_numpy()))
    keys = (cr.s.to_numpy() * n + cr.u.to_numpy())[order]
    dst = cr.v.to_numpy()[order]
    us = v[["u", "s"]].sort_values(["u", "s"]).to_numpy(np.int64)
    t1 = time.perf_counter()
    reach = _story_reach(keys, dst, np.ascontiguousarray(us[:, 0]), np.ascontiguousarray(us[:, 1]),
                         nodes.astype(np.int64), n).astype(float)
    print(f"story cascade reach for {nodes.size:,} users: {time.perf_counter() - t1:.1f}s")
    fans = distinct_fans(cr, n)
    pd.DataFrame({"node": nodes, "story_reach": reach, "distinct_fans": fans[nodes]}).to_csv(
        os.path.join(args.out, f"{prefix}_reach.csv"), index=False)
    rows += score_block({k: x[nodes] for k, x in scores.items()}, reach, scores["out_degree"][nodes], "sample",
                        "story cascade reach")

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(args.out, f"{prefix}_metrics.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(args.out, f"{prefix}_deciles.csv"), index=False)
    pd.set_option("display.width", 160)
    print(met[["split", "outcome", "measure", "spearman", "partial_spearman_ctrl_outdeg", "auc_top1pct",
               "prec_at_100"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
