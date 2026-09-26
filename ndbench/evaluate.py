"""Scoring helpers: rank agreement, imprecision, spread CIs, redundancy, statistics."""
from __future__ import annotations

import numpy as np
from scipy import stats


def kendall_tau(a, b):
    t = stats.kendalltau(a, b, variant="b")
    return float(t.statistic)


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def imprecision(scores, sigma, f):
    """Kitsak et al. (2010): 1 - <sigma of measure's top-f> / <sigma of true top-f>."""
    k = max(1, int(round(f * sigma.size)))
    top_m = np.argsort(-scores, kind="stable")[:k]
    top_t = np.argsort(-sigma, kind="stable")[:k]
    denom = sigma[top_t].mean()
    return float(1.0 - sigma[top_m].mean() / denom) if denom > 0 else float("nan")


def bootstrap_ci(samples, reps=1000, alpha=0.05, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    samples = np.asarray(samples, float)
    idx = rng.integers(0, samples.size, (reps, samples.size))
    means = samples[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def redundancy(g, seeds):
    """1 - |union of seeds' out-neighbourhoods| / sum of their sizes (0 = no overlap)."""
    indptr, indices, _ = g.out_csr()
    parts = [indices[indptr[s]:indptr[s + 1]] for s in seeds]
    total = sum(p.size for p in parts)
    if total == 0:
        return 0.0
    return float(1.0 - np.unique(np.concatenate(parts)).size / total)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank-residual method)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        A = np.vstack([rz, np.ones_like(rz)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def auc(scores, positives):
    """ROC-AUC via Mann-Whitney U (ties handled by average ranks)."""
    positives = np.asarray(positives, bool)
    npos, nneg = positives.sum(), (~positives).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = stats.rankdata(scores)
    return float((r[positives].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def precision_at_k(scores, positives, k):
    top = np.argsort(-scores, kind="stable")[:k]
    return float(np.mean(np.asarray(positives, bool)[top]))


def friedman_nemenyi(table, alpha=0.05):
    """table: 2-D array datasets x methods, higher = better.

    Returns (avg_ranks, friedman_stat, friedman_p, critical_difference)."""
    table = np.asarray(table, float)
    N, k = table.shape
    ranks = np.vstack([stats.rankdata(-row) for row in table])
    avg = ranks.mean(axis=0)
    if N >= 2 and k >= 3:
        fs = stats.friedmanchisquare(*table.T)
        stat, p = float(fs.statistic), float(fs.pvalue)
    else:
        stat, p = float("nan"), float("nan")
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * N)))
    return avg, stat, p, cd


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = p.size
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
