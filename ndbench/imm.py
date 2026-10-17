"""IMM influence maximization (Tang, Shi & Xiao 2015) with the sampling fix of Chen (2018).

Reverse-reachable (RR) sets are drawn for the same two diffusion settings the
benchmark evaluates seed sets in:

  IC  uniform arc probability p (the benchmark's p = p_c setting)
  LT  weights 1 / k_in(target), so a reverse walk picks one in-neighbour uniformly

The final node selection uses a fresh batch of RR sets, independent of the batch
that estimated the lower bound (Chen 2018, "An issue in the martingale analysis of
the influence maximization algorithm IMM"). RR sets are generated in parallel
threads by a nogil Numba kernel.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numba as nb
import numpy as np

IC, LT = 0, 1


@nb.njit(cache=True, nogil=True)
def _rr_batch(n, rindptr, rindices, model, p, count, seed):
    np.random.seed(seed)
    offs = np.zeros(count + 1, dtype=np.int64)
    buf = np.empty(max(16, 4 * count), dtype=np.int32)
    stamp = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int32)
    logq = math.log(1.0 - p) if (model == IC and p < 1.0) else 0.0
    used = 0
    for r in range(count):
        root = np.random.randint(n)
        stamp[root] = r
        queue[0] = root
        tail = 1
        if model == IC:
            head = 0
            while head < tail:
                u = queue[head]
                head += 1
                lo, hi = rindptr[u], rindptr[u + 1]
                d = hi - lo
                if p >= 1.0:
                    for kk in range(lo, hi):
                        v = rindices[kk]
                        if stamp[v] != r:
                            stamp[v] = r
                            queue[tail] = v
                            tail += 1
                else:
                    # geometric skipping: visit each in-arc independently with prob p
                    pos = -1
                    while True:
                        pos += 1 + int(math.log(1.0 - np.random.random()) / logq)
                        if pos >= d:
                            break
                        v = rindices[lo + pos]
                        if stamp[v] != r:
                            stamp[v] = r
                            queue[tail] = v
                            tail += 1
        else:
            u = root
            while True:
                lo, hi = rindptr[u], rindptr[u + 1]
                if hi == lo:
                    break
                v = rindices[lo + np.random.randint(hi - lo)]
                if stamp[v] == r:
                    break
                stamp[v] = r
                queue[tail] = v
                tail += 1
                u = v
        if used + tail > buf.size:
            nb_ = np.empty(max(2 * buf.size, used + tail), dtype=np.int32)
            nb_[:used] = buf[:used]
            buf = nb_
        buf[used:used + tail] = queue[:tail]
        used += tail
        offs[r + 1] = used
    return buf[:used].copy(), offs


@nb.njit(cache=True)
def _greedy_cover(n, buf, offs, k, tiebreak):
    theta = offs.size - 1
    deg = np.zeros(n, dtype=np.int64)
    for x in buf:
        deg[x] += 1
    nptr = np.zeros(n + 1, dtype=np.int64)
    for v in range(n):
        nptr[v + 1] = nptr[v] + deg[v]
    fill = nptr[:-1].copy()
    nsets = np.empty(buf.size, dtype=np.int32)
    for s in range(theta):
        for idx in range(offs[s], offs[s + 1]):
            v = buf[idx]
            nsets[fill[v]] = s
            fill[v] += 1
    covered = np.zeros(theta, dtype=np.bool_)
    chosen = np.zeros(n, dtype=np.bool_)
    seeds = np.empty(k, dtype=np.int64)
    cov = 0
    for t in range(k):
        best, bd, bt = -1, -1, -1.0
        for v in range(n):
            if not chosen[v] and (deg[v] > bd or (deg[v] == bd and tiebreak[v] > bt)):
                best, bd, bt = v, deg[v], tiebreak[v]
        seeds[t] = best
        chosen[best] = True
        for j in range(nptr[best], nptr[best + 1]):
            s = nsets[j]
            if not covered[s]:
                covered[s] = True
                cov += 1
                for idx in range(offs[s], offs[s + 1]):
                    deg[buf[idx]] -= 1
    return seeds, cov


class _RRPool:
    def __init__(self, g, model, p, rng_seed, threads):
        self.n = g.n
        self.rindptr, self.rindices, _ = g.in_csr()
        self.model, self.p = model, float(p if p is not None else 1.0)
        self.next_seed = int(rng_seed) * 1_000_003 + 1
        self.threads = threads
        self.bufs, self.offs, self.count, self.entries = [], [], 0, 0

    def grow(self, target):
        need = int(math.ceil(target)) - self.count
        if need <= 0:
            return
        parts = [need // self.threads + (1 if i < need % self.threads else 0) for i in range(self.threads)]
        parts = [c for c in parts if c > 0]
        seeds = list(range(self.next_seed, self.next_seed + len(parts)))
        self.next_seed += len(parts)
        with ThreadPoolExecutor(len(parts)) as ex:
            res = list(ex.map(lambda a: _rr_batch(self.n, self.rindptr, self.rindices, self.model, self.p, *a),
                              zip(parts, seeds)))
        for b, o in res:
            self.bufs.append(b)
            self.offs.append(o)
            self.count += o.size - 1
            self.entries += b.size

    def flat(self):
        offs, base = [np.zeros(1, dtype=np.int64)], 0
        for b, o in zip(self.bufs, self.offs):
            offs.append(o[1:] + base)
            base += b.size
        return np.concatenate(self.bufs), np.concatenate(offs)


def imm(g, k, model="IC", p=None, eps=0.5, ell=1.0, rng_seed=0, threads=None, max_entries=4e8,
        tiebreak=None):
    """Seed set of size k from IMM; prefixes are the greedy sets for smaller budgets.

    Returns (seeds, info) where info records theta, the lower bound, the RR-set
    volume and whether the max_entries memory cap shortened theta."""
    n = g.n
    mcode = IC if model == "IC" else LT
    threads = threads or max(1, nb.get_num_threads())
    tiebreak = g.out_degree().astype(float) if tiebreak is None else np.asarray(tiebreak, float)
    ell = ell * (1 + math.log(2) / math.log(n))
    logcnk = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    eps_p = math.sqrt(2) * eps
    lam_p = (2 + 2 / 3 * eps_p) * (logcnk + ell * math.log(n) + math.log(math.log2(n))) * n / eps_p ** 2
    a = math.sqrt(ell * math.log(n) + math.log(2))
    b = math.sqrt((1 - 1 / math.e) * (logcnk + ell * math.log(n) + math.log(2)))
    lam_s = 2 * n * ((1 - 1 / math.e) * a + b) ** 2 / eps ** 2

    pool = _RRPool(g, mcode, p, rng_seed, threads)
    lb = 1.0
    for i in range(1, max(2, int(math.log2(n)))):
        x = n / 2 ** i
        pool.grow(lam_p / x)
        buf, offs = pool.flat()
        _, cov = _greedy_cover(n, buf, offs, k, tiebreak)
        est = n * cov / pool.count
        if est >= (1 + eps_p) * x:
            lb = est / (1 + eps_p)
            break
        if pool.entries > max_entries:
            lb = max(lb, est / (1 + eps_p))
            break
    mean_size = pool.entries / max(pool.count, 1)
    theta = int(math.ceil(lam_s / lb))
    capped = theta * mean_size > max_entries
    if capped:
        theta = int(max_entries / mean_size)
    del pool, buf, offs
    final = _RRPool(g, mcode, p, rng_seed + 7919, threads)
    final.grow(theta)
    buf, offs = final.flat()
    seeds, cov = _greedy_cover(n, buf, offs, k, tiebreak)
    info = dict(theta=final.count, lower_bound=lb, mean_rr_size=final.entries / max(final.count, 1),
                est_spread=n * cov / final.count, capped=bool(capped), eps=eps)
    return seeds, info
