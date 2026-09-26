# RQ4: why naive top-k trophic-level seed selection fails badly

## The result

At k=100, IC diffusion, ratio of spread to the best method on each network:

| Network | netdegree | trophic |
|---|---|---|
| CollegeMsg | 1.000 | 0.801 |
| email-Eu-core | 0.988 | 0.843 |
| wiki-RfA | 1.000 | 0.508 |
| soc-sign-bitcoin-otc | 0.812 | 0.426 |
| wiki-Vote | 0.980 | 0.181 |
| soc-sign-epinions | 1.000 | **0.047** |
| soc-sign-Slashdot090221 | 1.000 | **0.040** |

Netdegree matches or nearly matches the best heuristic (DegreeDiscount/VoteRank/
CELF) on 6 of 7 networks and is within ~19% on the seventh — strongly
supports RQ4's hypothesis for netdegree. Trophic level fails everywhere,
catastrophically on two networks (4-5% of the best method's spread).

## The mechanism (not what RQ4's own hypothesis predicted)

RQ4's hypothesis attributed netdegree/trophic's expected benefit to reduced
**seed overlap** (redundancy). That is not what is driving trophic level's
failure. Checked directly on the two worst cases:

- `soc-sign-epinions`: trophic level's top-100 candidates have **1 distinct
  score value** among them (perfectly tied). Their mean out-degree is 1.0,
  mean in-degree 0.1 — these are near-isolated leaf nodes, not influential
  ones. Redundancy among them is *low* (0.049 — the opposite of the
  redundancy story), because near-isolated nodes trivially don't share
  neighborhoods; they're just individually useless as spreaders.
- `soc-sign-Slashdot090221`: same pattern (1 distinct score, out-degree
  ~1.1, in-degree ~0.2, redundancy 0.083).
- `seeds.top_k()`'s tie-break is uniform random jitter (`ndbench/seeds.py`)
  with no secondary criterion. This is **not the bug**: among a tied group
  of thousands of in-degree-0 nodes, there is no "better" node to break
  ties toward — every member of the tied set is structurally weak by
  construction, since trophic level pins any node with (near-)zero
  in-degree to the minimum boundary of its component regardless of that
  node's out-degree or actual reach.

## Why this happens structurally

Trophic level solves $\Lambda h = v$ globally; a node with in-degree 0
contributes nothing to any other node's level equation as a *target*, and
the boundary shift (`trophic_levels` subtracts each component's minimum) put
these zero-in-degree nodes at exactly $h=0$ regardless of their out-degree.
On networks where in-degree-0 nodes are common and vary widely in
out-degree (a mix of true broadcasters and true isolates all pinned to the
same score), the top of the trophic-level ranking is a degenerate tie that
top-k selection cannot usefully break — the score has no information left
to rank *within* that tied set.

Netdegree does not have this failure mode: a node's netdegree score is
`out_degree - in_degree`, which still varies continuously across
in-degree-0 nodes (a broadcaster with out-degree 50 and in-degree 0 scores
50; an isolate with out-degree 1 and in-degree 0 scores 1) — no boundary
collapse.

## Follow-up, now tested: does a degree-aware tie-break rescue it?

Added `seeds.top_k_tiebreak(scores, tiebreak, k)` (true lexicographic sort
via `np.lexsort`, not a linear combination of the two scores, which would
risk the secondary key distorting the primary ordering) and tested breaking
trophic level's ties by out-degree instead of randomly, then measured
actual IC spread (1,000 runs) on the two worst networks:

| Network | tied-at-top count | random-tiebreak spread | out-degree-tiebreak spread | netdegree reference |
|---|---|---|---|---|
| soc-sign-epinions | 6,767 (5.9% of the network) | 101.2 | 105.5 | 2125.2 |
| soc-sign-Slashdot090221 | 391 | 102.8 | 103.1 | 2554.8 |

**The tie-break does not rescue it.** On epinions, even the *highest*
out-degree nodes among the 6,767 tied-at-the-top nodes only average
out-degree 4.8 (median 4.0) — nowhere near netdegree's seed set (mean
out-degree 737). This confirms the deeper claim directly rather than just
by argument: the network's real hubs are **not members of the
minimum-trophic-level tied set at all** — they sit at other trophic levels
entirely. Raw trophic level's boundary value fails to localize influential
nodes on these networks, full stop; this is not a tie-breaking artifact and
there is no simple fix at the seed-selection stage.

## What this means for the paper

- Report this as a genuine, mechanistically explained and now directly
  tested finding, not folded into the redundancy story RQ4 originally
  hypothesized.
- This is a real property of using **raw trophic level for top-k seed
  selection specifically on these networks**, not a claim that trophic
  level itself is useless everywhere (Sections on RQ1-RQ3 already show it
  correlates reasonably with netdegree and with single-node spreading rank
  in a broad sense; and it did fine on CollegeMsg/email-Eu-core here).
- Any real fix would have to happen upstream of seed selection - e.g.
  weighting the trophic-level linear system by degree, or using a
  different global measure (SpringRank's regularized variant, which
  shrinks low-degree nodes toward the mean rather than pinning them to a
  hard boundary, might not have this failure mode) - not evaluated here,
  flagged as a natural next check if trophic level's poor RQ4 showing is
  discussed at length in the paper.

**Update 1:** this same failure mode was independently confirmed a third
time on real (non-simulated) data — see `proofs/rq5_higgs_findings.md`,
where raw trophic level's top-100 by score has `prec_at_100 = 0.00` against
real top-1%-retweeted Higgs users, despite a respectable overall AUC
(0.880). Three independent tasks (RQ3's Kendall-tau/imprecision, this
RQ4 seed-selection test, and RQ5's real-cascade precision) now show the
same pattern: decent bulk-ranking ability, catastrophic failure
specifically at the extreme top tier. State this as a general property of
raw trophic level in the paper, not a per-dataset quirk.

**Update 2 — mechanism refined, and a fix found:** the true mechanism is
narrower than "global measures fail" — see
`proofs/normalization_is_the_actual_mechanism.md`. It's specifically
*normalizing* netdegree by total degree (which trophic level's fixed-point
equation does internally, via the identity proved in RQ1) that creates a
hard ceiling reached exactly by any zero-in-degree node regardless of its
out-degree, demoting real hubs (which almost never have *exactly* zero
in-degree) below trivial ones. Mapping Jacobi depth $t=1$ through $50$
shows the exact tie resolves by $t=2$, but mean out-degree of the
top-selected nodes never recovers at any depth up to 50 — "more
iterations" does not fix this. **SpringRank's regularized variant does**:
swept $\alpha$ directly and confirmed spread recovers to match/exceed raw
netdegree's reference by $\alpha \approx 20$-$100$ on both networks (full
numbers in the mechanism doc). This turns "trophic level fundamentally
fails at seed selection" into "raw/unregularized trophic level fails, and
a specific, well-motivated, confirmed fix exists" — a much more
constructive result for the paper to report.

## Numbers this is based on

`results/spread.csv` (`net`, `method`, `k=100`, `model='IC'`), cross-checked
directly against `measures.trophic_source_score` / `measures.netdegree`
output and `evaluate.redundancy` on the top-100 sets for
`soc-sign-epinions` and `soc-sign-Slashdot090221` specifically. All from the
`--quick` benchmark run (reduced runs/samples) - the *ratio* pattern is
unlikely to be a small-sample artifact given how extreme it is (a 1-in-20
spread ratio), but a full non-quick run should confirm the exact numbers
before they go in the paper.
