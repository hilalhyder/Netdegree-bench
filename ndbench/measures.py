"""Node-scoring measures.  Higher score = predicted stronger spreader.

All functions take a :class:`ndbench.graph.DiGraph` whose arcs follow the
information-flow convention (i -> j: i can pass information to j).
"""
from __future__ import annotations

import time

import numba as nb
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

# --------------------------------------------------------------------------
# Netdegree family (local)
# --------------------------------------------------------------------------

def netdegree(g):
    return (g.out_degree() - g.in_degree()).astype(float)


def normalized_netdegree(g):
    ko, ki = g.out_degree(), g.in_degree()
    tot = ko + ki
    out = np.zeros(g.n)
    nz = tot > 0
    out[nz] = (ko[nz] - ki[nz]) / tot[nz]
    return out


def weighted_netdegree(g):
    return g.out_strength() - g.in_strength()


def hop_netdegree_walk(g, h=2):
    """sum_{t=1..h} (A^t 1 - (A^T)^t 1): walk-count version, O(h m)."""
    A = g.adjacency()
    AT = A.T.tocsr()
    fo = np.ones(g.n)
    fi = np.ones(g.n)
    total = np.zeros(g.n)
    for _ in range(h):
        fo = A @ fo
        fi = AT @ fi
        total += fo - fi
    return total


def hop_netdegree_reach(g, h=2, max_work=5e8):
    """Exact version: |nodes reachable within h hops| - |nodes that reach it within h hops|.

    Uses BFS truncated at depth h from every node.  Refuses graphs where the
    estimated work exceeds ``max_work`` (returns None).
    """
    indptr, indices, _ = g.out_csr()
    rindptr, rindices, _ = g.in_csr()
    ko = np.diff(indptr)
    est = float(np.sum(ko * 1.0 * np.diff(rindptr))) + float(np.sum(ko.astype(float) ** 2))
    if est > max_work:
        return None
    fwd = _hop_reach_all(g.n, indptr, indices, h)
    bwd = _hop_reach_all(g.n, rindptr, rindices, h)
    return (fwd - bwd).astype(float)


@nb.njit(cache=True)
def _hop_reach_all(n, indptr, indices, h):
    out = np.zeros(n, dtype=np.int64)
    stamp = np.full(n, -1, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    for s in range(n):
        stamp[s] = s
        depth[s] = 0
        head, tail = 0, 1
        queue[0] = s
        cnt = 0
        while head < tail:
            u = queue[head]
            head += 1
            if depth[u] == h:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if stamp[v] != s:
                    stamp[v] = s
                    depth[v] = depth[u] + 1
                    queue[tail] = v
                    tail += 1
                    cnt += 1
        out[s] = cnt
    return out


# --------------------------------------------------------------------------
# Trophic levels (global) -- MacKay, Johnson & Sansom (2020)
# --------------------------------------------------------------------------

def trophic_levels(g, weighted=True, rtol=1e-10, maxiter=None, return_info=False):
    """Solve Lambda h = v with Lambda = diag(u) - W - W^T, u = s_in + s_out,
    v = s_in - s_out.  Levels are shifted so the minimum of each weakly
    connected component is 0.  Isolated nodes get level 0.

    Returns h (low = upstream source).  With return_info also returns
    (F0 trophic incoherence, residual norm, cg iterations).
    """
    n = g.n
    w = g.w if weighted else np.ones(g.m)
    W = sp.csr_matrix((w, (g.src, g.dst)), shape=(n, n))
    s_out = np.asarray(W.sum(axis=1)).ravel()
    s_in = np.asarray(W.sum(axis=0)).ravel()
    u = s_in + s_out
    v = s_in - s_out
    act = np.flatnonzero(u > 0)
    h = np.zeros(n)
    if act.size:
        S = (W + W.T).tocsr()[act][:, act]
        L = (sp.diags(u[act]) - S).tocsr()
        Minv = sp.diags(1.0 / u[act])
        iters = [0]

        def cb(_):
            iters[0] += 1
        x, info = cg(L, v[act], rtol=rtol, maxiter=maxiter or 20 * act.size, M=Minv, callback=cb)
        h[act] = x
        # shift per weak component
        comp = g.weak_components()
        mins = np.full(comp.max() + 1, np.inf)
        np.minimum.at(mins, comp[act], h[act])
        h[act] -= mins[comp[act]]
        resid = float(np.linalg.norm(L @ x - v[act]) / max(np.linalg.norm(v[act]), 1e-300))
    else:
        resid, iters = 0.0, [0]
    if not return_info:
        return h
    F0 = trophic_incoherence(g, h, weighted)
    return h, {"F0": F0, "residual": resid, "cg_iterations": iters[0]}


def trophic_incoherence(g, h, weighted=True):
    w = g.w if weighted else np.ones(g.m)
    if g.m == 0:
        return float("nan")
    return float(np.sum(w * (h[g.dst] - h[g.src] - 1.0) ** 2) / np.sum(w))


def set_aside_isolated(g, score):
    """Rank isolated nodes below every other node.

    Isolated nodes have no trophic level (Lemma 1 sets them aside), but
    trophic_levels stores 0 for them, which after the per-component shift is
    the top source score -- tied with each component's most upstream node.
    """
    score = np.asarray(score, dtype=float).copy()
    iso = (g.out_degree() + g.in_degree()) == 0
    if iso.any() and (~iso).any():
        score[iso] = score[~iso].min() - 1.0
    return score


def trophic_source_score(g, weighted=True):
    return set_aside_isolated(g, -trophic_levels(g, weighted=weighted))


def spring_rank(g, alpha=1.0, weighted=True, rtol=1e-10, maxiter=None):
    """SpringRank (De Bacco, Larremore & Moore 2018) with L2 regularization
    alpha: solve (diag(u) + alpha*I - S) rank = v. alpha=0 (in the limit)
    reduces to the same system as trophic_levels (proofs/rq1_jacobi_identity.md
    Sec. 6); alpha>0 shrinks every node's rank toward 0, proportionally MORE
    for small-u_i (low-degree) nodes than large-u_i (hub) nodes -- tested in
    proofs/normalization_is_the_actual_mechanism.md as a candidate fix for
    raw trophic level's boundary-collapse failure at zero-in-degree nodes."""
    n = g.n
    w = g.w if weighted else np.ones(g.m)
    W = sp.csr_matrix((w, (g.src, g.dst)), shape=(n, n))
    s_out = np.asarray(W.sum(axis=1)).ravel()
    s_in = np.asarray(W.sum(axis=0)).ravel()
    u = s_in + s_out
    v = s_in - s_out
    S = (W + W.T).tocsr()
    L = (sp.diags(u + alpha) - S).tocsr()
    Minv = sp.diags(1.0 / (u + alpha))
    x, info = cg(L, v, rtol=rtol, maxiter=maxiter or 20 * n, M=Minv)
    return -x  # source score convention: low rank/level = upstream = high score


def spring_rank_p99(g, weighted=True):
    """spring_rank with alpha = 99th percentile of total degree (out+in) --
    the data-driven rule from proofs/SUMMARY_the_regularization_fix.md, which
    fixed raw trophic level's boundary-collapse failure on every network
    tested there (2 signed social networks + the real Higgs graph) without
    per-dataset hand-tuning. An earlier version of this used the 90th
    percentile; a finer sweep (90/95/99/99.5/99.9/99.99) across 6 real
    networks found p99 has both the highest mean AND the lowest cross-network
    variance (std=0.013 vs p90's 0.093) -- p90 was a real but measurably
    suboptimal choice, corrected here rather than left as a caveat. Beyond
    p99, mean creeps up slightly on some networks (bitcoin-otc specifically)
    but variance roughly triples as other networks start mildly reversing --
    p99 is the more robust single non-tuned choice. Registered as a
    first-class measure (rather than left as an ad-hoc check) so it's
    evaluated everywhere trophic level is, including on networks that were
    not broken to begin with."""
    u = g.out_degree().astype(float) + g.in_degree().astype(float)
    active = u > 0
    alpha = float(np.quantile(u[active], 0.99)) if active.any() else 1.0
    return spring_rank(g, alpha=max(alpha, 1e-6), weighted=weighted)


def spring_rank_tailadaptive(g, weighted=True):
    """spring_rank with alpha = geometric mean of the 99th-percentile and the
    maximum of total degree (out+in) -- the degree-tail-adaptive rule flagged
    as "the natural next refinement" in proofs/scaling_to_12_networks_two_failure_modes.md
    and proofs/scaling_to_massive_networks_wiki_talk_pokec.md.

    spring_rank_p99's alpha is a fixed bulk percentile, blind to how far the
    true maximum sits above that bulk. On "normally" heavy-tailed networks
    (max/p99 ratio roughly 2-16) that's fine; on email-EuAll (ratio 509) and
    especially wiki-Talk (ratio 4001) it under-regularizes catastrophically
    -- p99 alpha=25 against a max degree of 100,032 on wiki-Talk. The
    geometric mean sqrt(p99 * max) scales up automatically with the tail's
    actual extent (it moves from p99 toward max as the ratio grows) while
    staying close to p99 itself when the ratio is already small, so it does
    not need a per-dataset tuning knob -- same "one non-tuned rule" framing
    as spring_rank_p99, just responsive to the tail shape p99 alone ignores.
    See proofs/degree_tail_adaptive_alpha.md for the empirical comparison
    against spring_rank_p99 across the full failure-mode spectrum (normal,
    degree-tail-outlier, reciprocity-ceiling, massive-scale)."""
    u = g.out_degree().astype(float) + g.in_degree().astype(float)
    active = u > 0
    if not active.any():
        return spring_rank(g, alpha=1.0, weighted=weighted)
    p99 = float(np.quantile(u[active], 0.99))
    umax = float(u[active].max())
    alpha = float(np.sqrt(max(p99, 1e-6) * max(umax, 1e-6)))
    return spring_rank(g, alpha=max(alpha, 1e-6), weighted=weighted)


def spring_rank_maxdeg(g, weighted=True):
    """spring_rank with alpha = the single maximum total degree (out+in) in
    the network -- no percentile involved at all.

    Follow-up to spring_rank_tailadaptive (see proofs/degree_tail_adaptive_alpha.md):
    a p99^a * max^(1-a) exponent sweep (a=1 recovers spring_rank_p99, a=0.5
    is spring_rank_tailadaptive) run on 6 stress-test networks spanning both
    known failure modes (email-EuAll/wiki-Talk, degree-tail outliers;
    soc-sign-bitcoin-otc/soc-Slashdot0922 unsigned, reciprocity ceiling; plus
    wiki-Vote/soc-Pokec as normal-tail controls) found the ratio-to-netdegree
    curve is monotonic in a, with a=0 (pure max, i.e. this function) at or
    above every other exponent tried on every single network -- including
    strict wins on the two hardest, most alpha-hungry cases in the whole
    study (bitcoin-otc: 1.117 vs spring_rank_tailadaptive's 1.102;
    soc-Slashdot0922 unsigned: 0.964 vs 0.878). Simpler than the geometric
    mean too: no percentile computation, no exponent choice, just the single
    largest total degree already computed for every other measure in this
    file. See proofs/degree_tail_adaptive_alpha.md for the full sweep and
    the 12-network+2-massive-network validation."""
    u = g.out_degree().astype(float) + g.in_degree().astype(float)
    active = u > 0
    alpha = float(u[active].max()) if active.any() else 1.0
    return spring_rank(g, alpha=max(alpha, 1e-6), weighted=weighted)


def jacobi_trophic(g, t, weighted=True):
    """t Jacobi iterations of the trophic system from h=0; returns the source score -h.

    t=1 gives exactly the (weighted) normalized netdegree.
    """
    w = g.w if weighted else np.ones(g.m)
    W = sp.csr_matrix((w, (g.src, g.dst)), shape=(g.n, g.n))
    S = (W + W.T).tocsr()
    s_out = np.asarray(W.sum(axis=1)).ravel()
    s_in = np.asarray(W.sum(axis=0)).ravel()
    u = s_in + s_out
    inv = np.zeros(g.n)
    inv[u > 0] = 1.0 / u[u > 0]
    b = (s_in - s_out) * inv
    h = np.zeros(g.n)
    for _ in range(t):
        h = inv * (S @ h) + b
    return -h


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def out_degree(g):
    return g.out_degree().astype(float)


def in_degree(g):
    return g.in_degree().astype(float)


def pagerank_reversed(g, d=0.85, tol=1e-10, maxiter=1000):
    """PageRank on the attention graph (arcs reversed w.r.t. information flow).

    A node scores high when many (high-scoring) nodes receive information from it.
    """
    n = g.n
    kin = g.in_degree().astype(float)            # attention out-degree
    # transition along attention arcs j -> i for every info arc i -> j
    P = sp.csr_matrix((np.ones(g.m), (g.src, g.dst)), shape=(n, n))  # rows i, cols j
    x = np.full(n, 1.0 / n)
    dangling = kin == 0
    invk = np.zeros(n)
    invk[~dangling] = 1.0 / kin[~dangling]
    for _ in range(maxiter):
        xn = d * (P @ (x * invk)) + (d * x[dangling].sum() + 1 - d) / n
        if np.abs(xn - x).sum() < tol:
            x = xn
            break
        x = xn
    return x


def leaderrank(g, tol=1e-10, maxiter=1000):
    """LeaderRank (Lu et al. 2011) on the attention graph with a ground node."""
    n = g.n
    katt = g.in_degree().astype(float) + 1.0   # attention out-degree incl. ground link
    P = sp.csr_matrix((np.ones(g.m), (g.src, g.dst)), shape=(n, n))
    s = np.ones(n)
    sg = 0.0
    for _ in range(maxiter):
        flow = s / katt
        sn = P @ flow + sg / n            # from attention in-neighbours + ground
        sgn = flow.sum()                  # every node sends 1/katt to ground
        if np.abs(sn - s).sum() + abs(sgn - sg) < tol:
            s, sg = sn, sgn
            break
        s, sg = sn, sgn
    return s + sg / n


def kshell(g):
    """Core number of the undirected simple projection (Batagelj-Zaversnik)."""
    a = np.minimum(g.src, g.dst)
    b = np.maximum(g.src, g.dst)
    key = np.unique(a * g.n + b)
    a, b = key // g.n, key % g.n
    s = np.concatenate([a, b])
    t = np.concatenate([b, a])
    order = np.argsort(s, kind="stable")
    indices = t[order]
    indptr = np.zeros(g.n + 1, dtype=np.int64)
    np.cumsum(np.bincount(s, minlength=g.n), out=indptr[1:])
    return _bz_core(g.n, indptr, indices).astype(float)


@nb.njit(cache=True)
def _bz_core(n, indptr, indices):
    deg = np.empty(n, dtype=np.int64)
    md = 0
    for v in range(n):
        deg[v] = indptr[v + 1] - indptr[v]
        if deg[v] > md:
            md = deg[v]
    bins = np.zeros(md + 1, dtype=np.int64)
    for v in range(n):
        bins[deg[v]] += 1
    start = 0
    for d in range(md + 1):
        num = bins[d]
        bins[d] = start
        start += num
    pos = np.empty(n, dtype=np.int64)
    vert = np.empty(n, dtype=np.int64)
    for v in range(n):
        pos[v] = bins[deg[v]]
        vert[pos[v]] = v
        bins[deg[v]] += 1
    for d in range(md, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 0
    for i in range(n):
        v = vert[i]
        for k in range(indptr[v], indptr[v + 1]):
            u = indices[k]
            if deg[u] > deg[v]:
                du = deg[u]
                pu = pos[u]
                pw = bins[du]
                w = vert[pw]
                if u != w:
                    pos[u] = pw
                    vert[pu] = w
                    pos[w] = pu
                    vert[pw] = u
                bins[du] += 1
                deg[u] -= 1
    return deg


def betweenness(g, samples=None, seed=0):
    """Directed, unweighted betweenness (Brandes).  ``samples`` sources if given."""
    indptr, indices, _ = g.out_csr()
    rindptr, rindices, _ = g.in_csr()
    rng = np.random.default_rng(seed)
    if samples is None or samples >= g.n:
        sources = np.arange(g.n, dtype=np.int64)
        scale = 1.0
    else:
        sources = rng.choice(g.n, size=samples, replace=False).astype(np.int64)
        scale = g.n / samples
    return _brandes(g.n, indptr, indices, rindptr, rindices, sources) * scale


@nb.njit(cache=True)
def _brandes(n, indptr, indices, rindptr, rindices, sources):
    bc = np.zeros(n)
    sigma = np.zeros(n)
    dist = np.full(n, -1, dtype=np.int64)
    delta = np.zeros(n)
    stack = np.empty(n, dtype=np.int64)
    for s in sources:
        # BFS
        head, tail = 0, 1
        stack[0] = s
        dist[s] = 0
        sigma[s] = 1.0
        while head < tail:
            v = stack[head]
            head += 1
            for k in range(indptr[v], indptr[v + 1]):
                w = indices[k]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    stack[tail] = w
                    tail += 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        # accumulation (predecessors found via in-arcs)
        for i in range(tail - 1, -1, -1):
            w = stack[i]
            for k in range(rindptr[w], rindptr[w + 1]):
                v = rindices[k]
                if dist[v] >= 0 and dist[v] == dist[w] - 1:
                    delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
        for i in range(tail):
            w = stack[i]
            dist[w] = -1
            sigma[w] = 0.0
            delta[w] = 0.0
    return bc


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def compute_all(g, *, include=None, exclude=(), betweenness_samples=500, big=50_000,
                verbose=False):
    """Compute every registered measure. Returns (scores: dict, runtimes: dict, info: dict)."""
    reg = {
        "netdegree": lambda: netdegree(g),
        "netdegree_norm": lambda: normalized_netdegree(g),
        "netdegree_weighted": lambda: weighted_netdegree(g),
        "netdegree_2hop": lambda: hop_netdegree_walk(g, 2),
        "netdegree_2hop_reach": lambda: hop_netdegree_reach(g, 2) if g.n <= big else None,
        "trophic": None,  # handled below (also yields F0)
        "spring_rank_p99": lambda: spring_rank_p99(g),
        "spring_rank_tailadaptive": lambda: spring_rank_tailadaptive(g),
        "spring_rank_maxdeg": lambda: spring_rank_maxdeg(g),
        "jacobi_2": lambda: jacobi_trophic(g, 2),
        "jacobi_5": lambda: jacobi_trophic(g, 5),
        "out_degree": lambda: out_degree(g),
        "in_degree": lambda: in_degree(g),
        "pagerank_rev": lambda: pagerank_reversed(g),
        "leaderrank": lambda: leaderrank(g),
        "kshell": lambda: kshell(g),
        "betweenness": lambda: betweenness(g, None if g.n <= 5_000 else betweenness_samples),
    }
    names = [k for k in reg if (include is None or k in include) and k not in exclude]
    scores, runtimes, info = {}, {}, {}
    for name in names:
        t0 = time.perf_counter()
        if name == "trophic":
            h, tinfo = trophic_levels(g, return_info=True)
            val = set_aside_isolated(g, -h)
            info.update(tinfo)
            info["levels"] = h
        else:
            val = reg[name]()
        runtimes[name] = time.perf_counter() - t0
        if val is None:
            continue
        scores[name] = np.asarray(val, dtype=float)
        if verbose:
            print(f"    {name:22s} {runtimes[name]:8.3f}s")
    return scores, runtimes, info
