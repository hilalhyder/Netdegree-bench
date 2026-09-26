"""Monte Carlo diffusion: Independent Cascade, Linear Threshold, discrete SIR.

Every simulator works on the out-CSR of a DiGraph plus a per-arc parameter
array aligned with the CSR ``indices`` (transmission probability for IC/SIR,
influence weight for LT).  Numba-compiled; single-node spreading ability for
many nodes runs in parallel.
"""
from __future__ import annotations

import numba as nb
import numpy as np

IC, LT, SIR = 0, 1, 2
MODELS = {"IC": IC, "LT": LT, "SIR": SIR}


# --------------------------------------------------------------------------
# Parameter helpers
# --------------------------------------------------------------------------

def percolation_threshold(g):
    """p_c ~ <k> / <k_in k_out> for a directed uncorrelated network."""
    kin, kout = g.in_degree().astype(float), g.out_degree().astype(float)
    denom = np.mean(kin * kout)
    if denom <= 0:
        return 1.0
    return float(min(1.0, np.mean(kout) / denom))


def sir_beta_for_transmissibility(T, mu):
    """Per-step infection probability beta giving overall transmissibility T per arc
    when the infectious period is geometric with recovery probability mu per step
    (T = 1 - mu(1-beta) / (1 - (1-mu)(1-beta)))."""
    T = min(max(T, 0.0), 1.0)
    q = (1 - T) / (mu + (1 - T) * (1 - mu))
    return 1.0 - q


def arc_params(g, kind, p=None):
    """Per-arc parameter array aligned with g.out_csr() indices.

    kind = "uniform"  -> constant p
           "wc"       -> weighted cascade, 1 / k_in(target)
           "lt"       -> LT weights 1 / k_in(target)
    """
    indptr, indices, _ = g.out_csr()
    if kind == "uniform":
        return np.full(indices.size, float(p))
    kin = g.in_degree().astype(float)
    return 1.0 / np.maximum(kin[indices], 1.0)


# --------------------------------------------------------------------------
# Core kernels
# --------------------------------------------------------------------------

@nb.njit(cache=True)
def _ic_once(indptr, indices, prob, seeds, stamp, run, queue):
    tail = 0
    for s in seeds:
        if stamp[s] != run:
            stamp[s] = run
            queue[tail] = s
            tail += 1
    head = 0
    while head < tail:
        u = queue[head]
        head += 1
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if stamp[v] != run and np.random.random() < prob[k]:
                stamp[v] = run
                queue[tail] = v
                tail += 1
    return tail


@nb.njit(cache=True)
def _lt_once(indptr, indices, weight, seeds, stamp, run, queue, acc, acc_stamp, thr):
    tail = 0
    for s in seeds:
        if stamp[s] != run:
            stamp[s] = run
            queue[tail] = s
            tail += 1
    head = 0
    while head < tail:
        u = queue[head]
        head += 1
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if stamp[v] == run:
                continue
            if acc_stamp[v] != run:
                acc_stamp[v] = run
                acc[v] = 0.0
                thr[v] = np.random.random()
            acc[v] += weight[k]
            if acc[v] >= thr[v]:
                stamp[v] = run
                queue[tail] = v
                tail += 1
    return tail


@nb.njit(cache=True)
def _sir_once(indptr, indices, beta, seeds, stamp, run, cur, nxt, mu):
    """Discrete-time SIR. stamp[v]==run marks 'ever infected'. Returns final size."""
    ncur = 0
    total = 0
    for s in seeds:
        if stamp[s] != run:
            stamp[s] = run
            cur[ncur] = s
            ncur += 1
            total += 1
    while ncur > 0:
        nn = 0
        for i in range(ncur):
            u = cur[i]
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if stamp[v] != run and np.random.random() < beta[k]:
                    stamp[v] = run
                    nxt[nn] = v
                    nn += 1
                    total += 1
            # u stays infectious with prob 1 - mu
            if np.random.random() >= mu:
                nxt[nn] = u
                nn += 1
        for i in range(nn):
            cur[i] = nxt[i]
        ncur = nn
    return total


@nb.njit(cache=True)
def _simulate(model, n, indptr, indices, param, seeds, runs, mu, rng_seed):
    np.random.seed(rng_seed)
    stamp = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    out = np.empty(runs, dtype=np.float64)
    if model == 1:
        acc = np.zeros(n)
        acc_stamp = np.full(n, -1, dtype=np.int64)
        thr = np.zeros(n)
        for r in range(runs):
            out[r] = _lt_once(indptr, indices, param, seeds, stamp, r, queue, acc, acc_stamp, thr)
    elif model == 2:
        nxt = np.empty(2 * n, dtype=np.int64)
        cur = np.empty(2 * n, dtype=np.int64)
        for r in range(runs):
            out[r] = _sir_once(indptr, indices, param, seeds, stamp, r, cur, nxt, mu)
    else:
        for r in range(runs):
            out[r] = _ic_once(indptr, indices, param, seeds, stamp, r, queue)
    return out


@nb.njit(parallel=True, cache=True)
def _simulate_parallel(model, n, indptr, indices, param, seeds, runs, mu, rng_seed, chunks):
    out = np.empty(runs, dtype=np.float64)
    bounds = np.linspace(0, runs, chunks + 1).astype(np.int64)
    for c in nb.prange(chunks):
        lo, hi = bounds[c], bounds[c + 1]
        if hi > lo:
            out[lo:hi] = _simulate(model, n, indptr, indices, param, seeds, hi - lo, mu, rng_seed + 104729 * c)
    return out


@nb.njit(parallel=True, cache=True)
def _single_node_all(model, n, indptr, indices, param, nodes, runs, mu, rng_seed):
    res = np.empty(nodes.size, dtype=np.float64)
    for i in nb.prange(nodes.size):
        seeds = np.empty(1, dtype=np.int64)
        seeds[0] = nodes[i]
        sizes = _simulate(model, n, indptr, indices, param, seeds, runs, mu, rng_seed + 7919 * i)
        res[i] = sizes.mean()
    return res


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def spread(g, seeds, model="IC", param=None, runs=1000, mu=0.5, rng_seed=0):
    """Final-size samples (array of length ``runs``) for a seed set."""
    indptr, indices, _ = g.out_csr()
    seeds = np.asarray(seeds, dtype=np.int64)
    chunks = min(int(runs), max(1, nb.get_num_threads()))
    if chunks == 1:
        return _simulate(MODELS[model], g.n, indptr, indices, np.asarray(param, float), seeds,
                         int(runs), float(mu), int(rng_seed))
    return _simulate_parallel(MODELS[model], g.n, indptr, indices, np.asarray(param, float), seeds,
                              int(runs), float(mu), int(rng_seed), chunks)


def single_node_spreading(g, nodes, model="IC", param=None, runs=200, mu=0.5, rng_seed=0):
    """Mean final outbreak size seeded at each node in ``nodes`` (parallel)."""
    indptr, indices, _ = g.out_csr()
    nodes = np.asarray(nodes, dtype=np.int64)
    return _single_node_all(MODELS[model], g.n, indptr, indices, np.asarray(param, float), nodes,
                            int(runs), float(mu), int(rng_seed))


def stratified_sample(g, size, rng_seed=0, strata=20):
    """Sample nodes stratified by out-degree quantile (always includes top hubs)."""
    rng = np.random.default_rng(rng_seed)
    if size >= g.n:
        return np.arange(g.n)
    ko = g.out_degree()
    order = np.argsort(ko, kind="stable")
    groups = np.array_split(order, strata)
    per = size // strata
    picked = [rng.choice(gr, size=min(per, gr.size), replace=False) for gr in groups]
    s = np.unique(np.concatenate(picked + [order[-max(1, size // 50):]]))
    return s
