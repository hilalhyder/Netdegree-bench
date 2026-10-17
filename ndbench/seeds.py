"""Seed-set selection strategies for a budget k."""
from __future__ import annotations

import heapq

import numba as nb
import numpy as np

from . import diffusion


def top_k(scores, k, rng_seed=0):
    """Top-k by score, random tie-breaking."""
    rng = np.random.default_rng(rng_seed)
    jitter = rng.random(scores.size) * 1e-9 * (np.abs(scores).max() + 1)
    return np.argsort(-(scores + jitter), kind="stable")[:k]


def top_k_tiebreak(scores, tiebreak, k, rng_seed=0):
    """Top-k by ``scores``, breaking exact ties by ``tiebreak`` (e.g. out-degree)
    instead of at random. See proofs/rq4_trophic_seed_selection_failure.md: on
    networks where the top of a measure's ranking is a large exact tie (e.g.
    raw trophic level, whose zero-in-degree nodes all sit at the same boundary
    value), plain top_k's random tie-break makes the resulting seed set a
    matter of luck rather than a real choice among the tied candidates.

    True lexicographic sort (primary=scores, secondary=tiebreak, tertiary=
    random) via np.lexsort -- NOT a linear combination of the two scores,
    which would risk the secondary key distorting the primary ordering
    between close-but-unequal primary values."""
    rng = np.random.default_rng(rng_seed)
    jitter = rng.random(scores.size)
    # lexsort sorts by the LAST key first (primary), ties broken by earlier
    # keys; negate for descending order.
    order = np.lexsort((-jitter, -tiebreak, -scores))
    return order[:k]


def top_k_exclusion(g, scores, k, rng_seed=0):
    """Top-k skipping candidates that are out-neighbours of an already chosen seed."""
    rng = np.random.default_rng(rng_seed)
    jitter = rng.random(scores.size) * 1e-9 * (np.abs(scores).max() + 1)
    order = np.argsort(-(scores + jitter), kind="stable")
    indptr, indices, _ = g.out_csr()
    covered = np.zeros(g.n, dtype=bool)
    chosen, skipped = [], []
    for v in order:
        if len(chosen) == k:
            break
        if covered[v]:
            skipped.append(v)
            continue
        chosen.append(v)
        covered[indices[indptr[v]:indptr[v + 1]]] = True
    if len(chosen) < k:
        chosen += skipped[: k - len(chosen)]
    return np.array(chosen, dtype=np.int64)


def voterank(g, k):
    """Directed VoteRank (Zhang et al. 2016): a node collects votes from the
    nodes it can inform (its out-neighbours); electing a node zeroes its voting
    ability and lowers that of its out-neighbours by 1/<k_out>."""
    indptr, indices, _ = g.out_csr()
    rindptr, rindices, _ = g.in_csr()
    f = 1.0 / max(g.m / g.n, 1e-12)
    return _voterank(g.n, indptr, indices, rindptr, rindices, k, f)


@nb.njit(cache=True)
def _voterank(n, indptr, indices, rindptr, rindices, k, f):
    va = np.ones(n)
    score = np.zeros(n)
    for u in range(n):
        score[u] = indptr[u + 1] - indptr[u]
    elected = np.zeros(n, dtype=np.bool_)
    out = np.empty(k, dtype=np.int64)
    for step in range(k):
        best, bs = -1, -1.0
        for u in range(n):
            if not elected[u] and score[u] > bs:
                bs = score[u]
                best = u
        if best < 0:
            return out[:step]
        out[step] = best
        elected[best] = True
        # changes in voting ability
        changed = [best]
        dv = [-va[best]]
        va[best] = 0.0
        for kk in range(indptr[best], indptr[best + 1]):
            j = indices[kk]
            old = va[j]
            va[j] = max(0.0, va[j] - f)
            changed.append(j)
            dv.append(va[j] - old)
        # propagate to voters' candidates: score[i] = sum_{j in out(i)} va[j]
        for c in range(len(changed)):
            j = changed[c]
            d = dv[c]
            if d == 0.0:
                continue
            for kk in range(rindptr[j], rindptr[j + 1]):
                score[rindices[kk]] += d
    return out


def degree_discount_ic(g, k, p):
    """DegreeDiscountIC (Chen, Wang & Yang 2009), directed version: selecting u
    discounts the out-neighbours v of u (u can already activate them)."""
    indptr, indices, _ = g.out_csr()
    d = g.out_degree().astype(float)
    t = np.zeros(g.n)
    dd = d.copy()
    heap = [(-dd[v], v) for v in range(g.n)]
    heapq.heapify(heap)
    chosen = np.zeros(g.n, dtype=bool)
    out = []
    while len(out) < k and heap:
        negv, v = heapq.heappop(heap)
        if chosen[v] or -negv != dd[v]:
            continue
        chosen[v] = True
        out.append(v)
        for w in indices[indptr[v]:indptr[v + 1]]:
            if chosen[w]:
                continue
            t[w] += 1
            dd[w] = d[w] - 2 * t[w] - (d[w] - t[w]) * t[w] * p
            heapq.heappush(heap, (-dd[w], w))
    return np.array(out, dtype=np.int64)


@nb.njit(cache=True, parallel=True)
def _two_hop_sums(n, indptr, indices, kout, blocks):
    """For every node i, sums of kout over the nodes at out-distance exactly 1 (s1) and
    exactly 2 (s2), and the number of nodes at distance 2 (c2)."""
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    c2 = np.zeros(n)
    for b in nb.prange(blocks):
        stamp = np.full(n, -1, dtype=np.int64)
        for i in range(b, n, blocks):
            stamp[i] = i
            for kk in range(indptr[i], indptr[i + 1]):
                j = indices[kk]
                if stamp[j] != i:
                    stamp[j] = i
                    s1[i] += kout[j]
            for kk in range(indptr[i], indptr[i + 1]):
                j = indices[kk]
                for k2 in range(indptr[j], indptr[j + 1]):
                    l = indices[k2]
                    if stamp[l] != i:
                        stamp[l] = i
                        s2[i] += kout[l]
                        c2[i] += 1
    return s1, s2, c2


def gravity(g):
    """Gravity centrality (Ma et al. 2016) truncated at radius R = 2 on out-distances:
    G(i) = k_out(i) * sum_{d(i,j) <= 2} k_out(j) / d(i,j)^2."""
    indptr, indices, _ = g.out_csr()
    kout = g.out_degree().astype(float)
    s1, s2, _ = _two_hop_sums(g.n, indptr, indices, kout, max(1, nb.get_num_threads()))
    return kout * (s1 + s2 / 4.0)


@nb.njit(cache=True)
def _ci_node(i, indptr, indices, kout, removed, stamp, tag):
    """CI_2(i) = (k_out(i) - 1) * sum over nodes at out-distance exactly 2 of (k_out - 1),
    on the graph with ``removed`` nodes deleted."""
    if kout[i] <= 1:
        return 0.0
    stamp[i] = tag
    for kk in range(indptr[i], indptr[i + 1]):
        j = indices[kk]
        if not removed[j]:
            stamp[j] = tag
    tot = 0.0
    for kk in range(indptr[i], indptr[i + 1]):
        j = indices[kk]
        if removed[j]:
            continue
        for k2 in range(indptr[j], indptr[j + 1]):
            l = indices[k2]
            if not removed[l] and stamp[l] != tag:
                stamp[l] = tag
                tot += kout[l] - 1
    return (kout[i] - 1) * tot


def collective_influence(g, k):
    """Adaptive Collective Influence (Morone & Makse 2015), ball radius l = 2 on out-arcs,
    out-degrees in place of degrees. After each pick the node is removed; stale CI values
    are re-evaluated lazily when they reach the top of the heap (a candidate is accepted
    once its fresh value is at least the best stored value)."""
    indptr, indices, _ = g.out_csr()
    rindptr, rindices, _ = g.in_csr()
    kout = g.out_degree().astype(float)
    _, s2, c2 = _two_hop_sums(g.n, indptr, indices, kout, max(1, nb.get_num_threads()))
    ci = np.maximum(kout - 1, 0) * (s2 - c2)
    removed = np.zeros(g.n, dtype=np.bool_)
    stamp = np.zeros(g.n, dtype=np.int64)
    heap = [(-ci[v], -kout[v], v) for v in np.flatnonzero(ci > 0)]
    heapq.heapify(heap)
    out, tag = [], 0
    while len(out) < k and heap:
        _, nk, v = heapq.heappop(heap)
        if removed[v]:
            continue
        tag += 1
        fresh = _ci_node(v, indptr, indices, kout, removed, stamp, tag)
        if heap and fresh < -heap[0][0]:
            heapq.heappush(heap, (-fresh, nk, v))
            continue
        out.append(v)
        removed[v] = True
        kout[rindices[rindptr[v]:rindptr[v + 1]]] -= 1
    if len(out) < k:  # CI exhausted (tree-like remainder): fill by out-degree
        rest = [v for v in np.argsort(-g.out_degree(), kind="stable") if not removed[v]]
        out += rest[: k - len(out)]
    return np.array(out, dtype=np.int64)


def celf(g, k, model="IC", param=None, runs=200, candidates=None, rng_seed=0, mu=0.5):
    """Lazy-greedy (CELF) seed selection with Monte Carlo spread estimates.

    ``candidates`` restricts the pool (e.g. top 500 by out-degree) to keep it tractable.
    """
    pool = np.arange(g.n) if candidates is None else np.asarray(candidates)
    gains = diffusion.single_node_spreading(g, pool, model, param, runs, mu, rng_seed)
    heap = [(-gains[i], int(pool[i]), 0) for i in range(pool.size)]
    heapq.heapify(heap)
    seeds, cur = [], 0.0
    while len(seeds) < k and heap:
        neg, v, it = heapq.heappop(heap)
        if it == len(seeds):
            seeds.append(v)
            cur += -neg
            continue
        s = diffusion.spread(g, seeds + [v], model, param, runs, mu, rng_seed + len(seeds)).mean()
        heapq.heappush(heap, (-(s - cur), v, len(seeds)))
    return np.array(seeds, dtype=np.int64)
