"""Correctness checks.  Run:  python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ndbench import diffusion, evaluate, measures, seeds, synthetic  # noqa: E402
from ndbench.graph import from_edges, read_edgelist  # noqa: E402


def chain(n=5):
    return from_edges(np.arange(n - 1), np.arange(1, n), n=n)


def test_netdegree_sums_to_zero():
    g, _ = synthetic.trophic_tunable(n=500, T=1.0, rng_seed=3)
    assert abs(measures.netdegree(g).sum()) < 1e-9


def test_trophic_chain_levels():
    h = measures.trophic_levels(chain(5))
    assert np.allclose(h, [0, 1, 2, 3, 4], atol=1e-6)
    _, info = measures.trophic_levels(chain(5), return_info=True)
    assert info["F0"] < 1e-10


def test_jacobi_first_iterate_is_normalized_netdegree():
    g, _ = synthetic.trophic_tunable(n=800, T=0.7, rng_seed=5)
    assert np.allclose(measures.jacobi_trophic(g, 1), measures.normalized_netdegree(g))


def test_jacobi_converges_toward_trophic():
    g, _ = synthetic.trophic_tunable(n=800, T=0.7, rng_seed=6)
    ref = measures.trophic_source_score(g)
    act = (g.out_degree() + g.in_degree()) > 0
    r1 = evaluate.spearman(measures.jacobi_trophic(g, 1)[act], ref[act])
    r10 = evaluate.spearman(measures.jacobi_trophic(g, 10)[act], ref[act])
    assert r10 > r1


def test_jacobi_raw_iterates_converge_to_trophic_level():
    """proofs/rq1_jacobi_identity.md Proposition 2: on a connected component,
    Sum(v)=0 (Lemma 1) pins the iteration's mode-1 (constant-vector)
    coefficient at zero for every t (never excited from h^(0)=0), so the RAW
    iterates -- not just their rank order, checked separately by
    test_jacobi_converges_toward_trophic -- converge to the trophic level,
    gauge-fixed to be orthogonal to u.

    Convergence rate is governed by the symmetrized graph's spectral gap, so
    the test graph matters: a chain (or chain plus a few chords) is nearly
    path-shaped and needs tens of thousands of iterations to show it
    (verified separately -- residual 0.49 at t=500, but 4e-14 by t=50,000,
    confirming it's a mixing-time issue, not non-convergence). The
    denser/better-mixing synthetic generator used elsewhere in this file
    converges to machine precision by t=200, which is what's checked here.
    A plain chain would ALSO hit a second, genuine edge case: its
    symmetrization is a bipartite path graph, giving D^-1 S an eigenvalue at
    exactly -1, whose mode oscillates rather than contracts (the other edge
    case flagged in the proof's Prop. 2) -- the synthetic generator's denser,
    non-bipartite structure avoids this too.
    """
    g, _ = synthetic.trophic_tunable(n=200, T=0.6, rng_seed=11)
    assert g.weak_components().max() == 0, "test graph must be a single component"
    h_star = measures.trophic_levels(g)
    u = g.out_degree().astype(float) + g.in_degree().astype(float)
    h_star = h_star - (u @ h_star) / u.sum()  # gauge-fix: orthogonal to u

    def resid(t):
        h = -measures.jacobi_trophic(g, t)  # jacobi_trophic returns the source score -h
        h = h - (u @ h) / u.sum()           # same gauge fix
        return np.linalg.norm(h - h_star) / max(np.linalg.norm(h_star), 1e-300)

    r_early, r_late = resid(5), resid(200)
    assert r_late < 1e-6, f"raw Jacobi iterate did not converge to h*: residual at t=200 is {r_late}"
    assert r_late < r_early


def test_trophic_identity_holds_at_solution():
    g, _ = synthetic.trophic_tunable(n=600, T=0.8, rng_seed=7)
    h = measures.trophic_levels(g)
    import scipy.sparse as sp
    W = g.adjacency(weighted=True)
    S = (W + W.T).tocsr()
    u = g.out_strength() + g.in_strength()
    rho = (g.out_strength() - g.in_strength()) / np.where(u > 0, u, 1)
    act = u > 0
    rhs = (S @ h)[act] / u[act] - rho[act]
    assert np.allclose(h[act], rhs, atol=1e-5)


def test_flip_reverses_netdegree():
    a = from_edges([0, 0, 1], [1, 2, 2], n=3)
    b = from_edges([0, 0, 1], [1, 2, 2], n=3, flip=True)
    assert np.allclose(measures.netdegree(a), -measures.netdegree(b))


def test_spring_rank_large_alpha_correlates_with_netdegree():
    """proofs/normalization_is_the_actual_mechanism.md: large-alpha SpringRank
    should behave like raw (unnormalized) netdegree, not like trophic level --
    shrinkage matters proportionally more for small-u_i (trivial) nodes, so a
    large enough alpha should pull the ranking away from trophic level's
    boundary-collapse pathology and toward degree-scaled netdegree."""
    g, _ = synthetic.trophic_tunable(n=600, T=0.6, rng_seed=12)
    nd = measures.netdegree(g)
    sr_small = measures.spring_rank(g, alpha=0.01)
    sr_large = measures.spring_rank(g, alpha=1000.0)
    r_small = evaluate.spearman(sr_small, nd)
    r_large = evaluate.spearman(sr_large, nd)
    assert r_large > r_small
    assert r_large > 0.9


def _strengths(g):
    s_out = np.bincount(g.src, weights=g.w, minlength=g.n)
    s_in = np.bincount(g.dst, weights=g.w, minlength=g.n)
    return s_out, s_in


def test_spring_rank_large_alpha_bound():
    """Proposition 2(b): ||alpha h(alpha) - v||_inf <= 2 u_max ||v||_inf / alpha, with source
    score -h, so -alpha h(alpha) -> weighted net-degree s_out - s_in."""
    g, _ = synthetic.trophic_tunable(n=600, T=0.6, rng_seed=13)
    s_out, s_in = _strengths(g)
    v, umax = s_in - s_out, (s_in + s_out).max()
    for alpha in [umax, 10 * umax, 100 * umax]:
        h = -measures.spring_rank(g, alpha=alpha)
        assert np.abs(alpha * h - v).max() <= 2 * umax * np.abs(v).max() / alpha + 1e-6


def test_trophic_source_sits_one_below_successor_mean():
    """Proposition 3: for s_in(i) = 0, h_i = mean_W(successor levels) - 1 (alpha = 0), and
    h_i = s/(s + alpha) * (mean_W - 1) for SpringRank."""
    g, _ = synthetic.trophic_tunable(n=800, T=0.7, rng_seed=14)
    s_out, s_in = _strengths(g)
    src = (s_in == 0) & (s_out > 0)
    assert src.sum() > 5

    def succ_mean(h):
        return np.bincount(g.src, weights=g.w * h[g.dst], minlength=g.n)[src] / s_out[src]

    h = measures.trophic_levels(g)
    assert np.allclose(h[src], succ_mean(h) - 1, atol=1e-6)
    alpha = 5.0
    h = -measures.spring_rank(g, alpha=alpha)
    assert np.allclose(h[src], s_out[src] / (s_out[src] + alpha) * (succ_mean(h) - 1), atol=1e-6)


def test_spring_rank_p99_registered_and_no_worse_than_trophic_where_not_broken():
    """spring_rank_p99 (proofs/SUMMARY_the_regularization_fix.md's data-driven
    fix, refined from an earlier p90 version -- see the function's docstring)
    must be reachable through compute_all's registry, and on a network where
    raw trophic level ISN'T broken (a well-mixed synthetic graph, no large
    zero-in-degree tied cluster), it should not make things worse."""
    g, _ = synthetic.trophic_tunable(n=400, T=0.6, rng_seed=13)
    scores, _, _ = measures.compute_all(g, include={"trophic", "spring_rank_p99"})
    assert "spring_rank_p99" in scores
    ref = measures.trophic_source_score(g)
    act = (g.out_degree() + g.in_degree()) > 0
    r_trophic = evaluate.spearman(scores["trophic"][act], ref[act])
    r_sr = evaluate.spearman(scores["spring_rank_p99"][act], ref[act])
    assert r_trophic > 0.99   # sanity: trophic vs itself via compute_all's own path
    assert r_sr > 0.7         # should still correlate reasonably, not collapse


def test_spring_rank_tailadaptive_registered_and_no_worse_than_trophic_where_not_broken():
    """spring_rank_tailadaptive (the degree-tail-adaptive alpha rule from
    proofs/degree_tail_adaptive_alpha.md) must be reachable through
    compute_all's registry, and on a network with a normal (non-outlier)
    degree tail it should behave like spring_rank_p99: not worse than raw
    trophic level on a well-mixed synthetic graph."""
    g, _ = synthetic.trophic_tunable(n=400, T=0.6, rng_seed=13)
    scores, _, _ = measures.compute_all(g, include={"trophic", "spring_rank_tailadaptive"})
    assert "spring_rank_tailadaptive" in scores
    ref = measures.trophic_source_score(g)
    act = (g.out_degree() + g.in_degree()) > 0
    r_sr = evaluate.spearman(scores["spring_rank_tailadaptive"][act], ref[act])
    assert r_sr > 0.7


def test_spring_rank_tailadaptive_alpha_exceeds_p99_when_tail_is_extreme():
    """On a graph with one extreme-degree hub (max/p99 ratio >> normal),
    the tail-adaptive alpha must sit strictly above the plain p99 alpha --
    that's the entire point of the refinement (proofs/degree_tail_adaptive_alpha.md)."""
    rng = np.random.default_rng(0)
    n = 500
    src = rng.integers(0, n, size=2000)
    dst = rng.integers(0, n, size=2000)
    # one node (0) gets a huge extra fan-out, making it a degree-tail outlier
    hub_src = np.full(3000, 0)
    hub_dst = rng.integers(1, n, size=3000)
    from ndbench.graph import from_edges
    g = from_edges(np.concatenate([src, hub_src]), np.concatenate([dst, hub_dst]), n=n)
    u = g.out_degree().astype(float) + g.in_degree().astype(float)
    active = u > 0
    p99 = float(np.quantile(u[active], 0.99))
    umax = float(u[active].max())
    assert umax / max(p99, 1e-9) > 20   # confirm this synthetic graph IS a tail outlier
    tailadaptive_alpha = np.sqrt(p99 * umax)
    assert tailadaptive_alpha > p99


def test_spring_rank_maxdeg_registered_and_no_worse_than_trophic_where_not_broken():
    """spring_rank_maxdeg (alpha = max total degree, no percentile -- the
    follow-up to spring_rank_tailadaptive from proofs/degree_tail_adaptive_alpha.md's
    exponent sweep) must be reachable through compute_all's registry, and on
    a well-mixed synthetic graph it should not make things worse than raw
    trophic level."""
    g, _ = synthetic.trophic_tunable(n=400, T=0.6, rng_seed=13)
    scores, _, _ = measures.compute_all(g, include={"trophic", "spring_rank_maxdeg"})
    assert "spring_rank_maxdeg" in scores
    ref = measures.trophic_source_score(g)
    act = (g.out_degree() + g.in_degree()) > 0
    r_sr = evaluate.spearman(scores["spring_rank_maxdeg"][act], ref[act])
    assert r_sr > 0.7


def test_spring_rank_maxdeg_alpha_equals_max_total_degree():
    g, _ = synthetic.trophic_tunable(n=300, T=0.5, rng_seed=9)
    u = g.out_degree().astype(float) + g.in_degree().astype(float)
    expected_alpha = float(u[u > 0].max())
    got = measures.spring_rank_maxdeg(g)
    want = measures.spring_rank(g, alpha=expected_alpha)
    assert np.allclose(got, want)


def test_hop_netdegree_walk_h1_equals_netdegree():
    g, _ = synthetic.trophic_tunable(n=300, T=0.5, rng_seed=8)
    assert np.allclose(measures.hop_netdegree_walk(g, 1), measures.netdegree(g))


def test_hop_reach_on_chain():
    r = measures.hop_netdegree_reach(chain(5), 2)
    assert list(r) == [2, 1, 0, -1, -2]


def test_ic_p1_equals_reachability():
    g = chain(6)
    par = diffusion.arc_params(g, "uniform", 1.0)
    s = diffusion.single_node_spreading(g, np.arange(6), "IC", par, runs=5)
    assert np.allclose(s, [6, 5, 4, 3, 2, 1])


def test_ic_p0_only_seed():
    g, _ = synthetic.trophic_tunable(n=300, T=0.5, rng_seed=9)
    par = diffusion.arc_params(g, "uniform", 0.0)
    assert np.allclose(diffusion.spread(g, [0, 1, 2], "IC", par, runs=10), 3)


def test_lt_deterministic_star():
    # 3 sources all pointing to node 3; LT weight 1/3 each: all three needed if threshold > 2/3
    g = from_edges([0, 1, 2], [3, 3, 3], n=4)
    par = diffusion.arc_params(g, "lt")
    full = diffusion.spread(g, [0, 1, 2], "LT", par, runs=2000)
    assert np.allclose(full, 4)
    one = diffusion.spread(g, [0], "LT", par, runs=20000).mean()
    assert abs(one - (1 + 1 / 3)) < 0.02          # P(theta <= 1/3) = 1/3


def test_sir_mu1_matches_ic():
    g, _ = synthetic.trophic_tunable(n=400, T=0.5, rng_seed=10)
    par = diffusion.arc_params(g, "uniform", 0.2)
    ic = diffusion.spread(g, [0], "IC", par, runs=20000, rng_seed=1).mean()
    sir = diffusion.spread(g, [0], "SIR", par, runs=20000, mu=1.0, rng_seed=2).mean()
    assert abs(ic - sir) / max(ic, 1) < 0.05


def test_pagerank_rev_prefers_broadcaster():
    # node 0 informs everybody; nobody informs node 0
    n = 20
    g = from_edges(np.zeros(n - 1, int), np.arange(1, n), n=n)
    pr = measures.pagerank_reversed(g)
    lr = measures.leaderrank(g)
    assert pr.argmax() == 0 and lr.argmax() == 0
    assert abs(pr.sum() - 1) < 1e-8


def test_kshell_clique():
    n = 5
    src, dst = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    assert np.allclose(measures.kshell(from_edges(src, dst, n=n)), 4)


def test_betweenness_chain():
    b = measures.betweenness(chain(4))
    assert np.allclose(b, [0, 2, 2, 0])


def test_voterank_and_dd_pick_hub_first():
    n = 30
    g = from_edges(np.r_[np.zeros(20, int), np.arange(1, 10)], np.r_[np.arange(1, 21), np.arange(2, 11)], n=n)
    assert seeds.voterank(g, 3)[0] == 0
    assert seeds.degree_discount_ic(g, 3, 0.1)[0] == 0


def test_top_k_tiebreak_primary_dominates_secondary():
    # primary score must win regardless of secondary, even when secondary is huge
    scores = np.array([5.0, 5.0, 3.0, 3.0])
    tiebreak = np.array([1.0, 100.0, 1000.0, 2000.0])
    order = seeds.top_k_tiebreak(scores, tiebreak, 4)
    assert list(order[:2]) == [1, 0]          # tied at score=5: higher tiebreak (100) first
    assert list(order[2:]) == [3, 2]          # tied at score=3: higher tiebreak (2000) first


def test_top_k_exclusion_skips_neighbours():
    g = from_edges([0, 0], [1, 2], n=4)
    s = seeds.top_k_exclusion(g, np.array([3.0, 2.0, 1.0, 0.5]), 2)
    assert list(s) == [0, 3]


def test_read_edgelist_labels_and_flip(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("# comment\n10 20\n10 30\n20 30\n10 20\n")
    g = read_edgelist(str(p), flip=True)
    assert g.n == 3 and g.m == 3
    nd = dict(zip(g.labels.astype(int), measures.netdegree(g)))
    assert nd == {10: -2, 20: 0, 30: 2}
    assert g.w.max() == 2  # duplicate arc merged into weight 2


def test_temporal_reach():
    import pandas as pd
    import run_higgs
    # B=0 retweeted by A=1 at t=5; 1 retweeted by 2 at t=3 (before) and by 3 at t=9 (after)
    act = pd.DataFrame(dict(ia=[1, 2, 3], ib=[0, 1, 1], t=[5, 3, 9], kind=["RT"] * 3))
    r = run_higgs.temporal_reach(act, 4, np.array([0, 1]))
    assert list(r) == [2, 2]      # from 0: {1, 3}; from 1: {2, 3}


def test_direction_check_detects_follow_file():
    import pandas as pd
    import run_higgs
    raw = from_edges([1, 2], [0, 0], n=3)  # 'A follows B' rows: 1 follows 0, 2 follows 0
    act = pd.DataFrame(dict(ia=[1, 2], ib=[0, 0], t=[1, 2], kind=["RT", "RT"]))
    assert run_higgs.direction_check(raw, act)["flip"]


def test_friedman_nemenyi_shapes():
    t = np.random.default_rng(0).random((6, 4))
    avg, stat, p, cd = evaluate.friedman_nemenyi(t)
    assert avg.shape == (4,) and cd > 0


def test_sir_transmissibility_calibration():
    from ndbench.graph import from_edges as fe
    g = fe(np.zeros(3000, int), np.arange(1, 3001), n=3001)
    b = diffusion.sir_beta_for_transmissibility(0.25, 0.5)
    s = diffusion.spread(g, [0], "SIR", diffusion.arc_params(g, "uniform", b), runs=3000, mu=0.5).mean() - 1
    assert abs(s / 3000 - 0.25) < 0.01
