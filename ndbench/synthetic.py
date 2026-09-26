"""Synthetic directed graphs with tunable trophic coherence (for RQ2 sweeps and tests)."""
from __future__ import annotations

import numpy as np

from .graph import from_edges


def trophic_tunable(n=2000, mean_degree=8.0, levels=6.0, T=0.5, rng_seed=0):
    """Latent levels l_i ~ U(0, levels); arc i -> j accepted with probability
    exp(-(l_j - l_i - 1)^2 / (2 T^2)).  Small T -> coherent hierarchy (low F0),
    large T -> incoherent (F0 -> 1).  Returns (DiGraph, latent levels)."""
    rng = np.random.default_rng(rng_seed)
    lat = rng.uniform(0, levels, n)
    target = int(mean_degree * n)
    src, dst = [], []
    got = 0
    while got < target:
        b = max(4 * (target - got), 10_000)
        i = rng.integers(0, n, b)
        j = rng.integers(0, n, b)
        acc = rng.random(b) < np.exp(-((lat[j] - lat[i] - 1.0) ** 2) / (2 * T * T))
        acc &= i != j
        src.append(i[acc])
        dst.append(j[acc])
        got += int(acc.sum())
    src = np.concatenate(src)[:target]
    dst = np.concatenate(dst)[:target]
    g = from_edges(src, dst, n=n, name=f"trophic_T{T:g}")
    g.w[:] = 1.0  # ignore accidental duplicates' multiplicity
    return g, lat


def higgs_like_fixture(n=3000, rng_seed=1, folder="data/higgs_fixture"):
    """Write a tiny synthetic stand-in for the Higgs files (for testing the pipeline).

    social file: 'A B' means A FOLLOWS B (so information flows B -> A);
    activity file: 'A B t RT' means A retweeted B at time t.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    lat = rng.uniform(0, 5, n)
    # followers: A follows B mostly when B is "upstream" (lower level)
    m = 12 * n
    A = rng.integers(0, n, 6 * m)
    B = rng.integers(0, n, 6 * m)
    keep = (rng.random(A.size) < np.exp(-((lat[A] - lat[B] - 1) ** 2) / 0.5)) & (A != B)
    A, B = A[keep][:m], B[keep][:m]
    pop = np.bincount(B, minlength=n) + 1.0
    with open(os.path.join(folder, "higgs-social_network.edgelist"), "w") as fh:
        for a, b in zip(A, B):
            fh.write(f"{a} {b}\n")
    # activity: retweets travel along follow arcs, favouring popular upstream users
    t0 = 1341100800  # 1 July 2012 00:00 UTC
    rows = []
    fol = {}
    for a, b in zip(A, B):
        fol.setdefault(b, []).append(a)
    for b, followers in fol.items():
        rate = 0.02 * pop[b] ** 0.5
        for a in followers:
            if rng.random() < min(0.9, rate):
                t = t0 + int(rng.uniform(0, 7 * 86400))
                rows.append((a, b, t, "RT"))
            if rng.random() < 0.01:
                rows.append((a, b, t0 + int(rng.uniform(0, 7 * 86400)), "MT"))
    rows.sort(key=lambda r: r[2])
    with open(os.path.join(folder, "higgs-activity_time.txt"), "w") as fh:
        for a, b, t, x in rows:
            fh.write(f"{a} {b} {t} {x}\n")
    return folder
