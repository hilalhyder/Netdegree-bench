# The real mechanism: normalization, not "local vs global" — and it follows directly from RQ1's own identity

## What this corrects

`proofs/rq4_trophic_seed_selection_failure.md` framed the seed-selection
failure as something that emerges specifically for the "global" trophic
level (as opposed to the "local" netdegree). That framing is **imprecise**.
Mapping the Jacobi-iteration depth $t=1,2,3,5,10,20,50$ directly (same
epinions/Slashdot networks, actual IC spread):

| $t$ | epinions: n tied at max | mean out-degree of top-100 | spread |
|---|---|---|---|
| 1 | 26,313 | 1.3 | 150.9 |
| 2 | 1 | 1.0 | 310.0 |
| 5 | 1 | 1.0 | 224.7 |
| 20 | 1 | 1.1 | 124.9 |
| 50 | 1 | 1.2 | 108.3 |

The exact tie resolves by $t=2$ — but the mean out-degree of the selected
seeds **never recovers**, staying pinned near 1.0 all the way to $t=50$.
Resolving the tie doesn't fix anything, because the whole neighborhood
near the top of the ranking is populated by near-degree-1 nodes at every
depth, not just at the exact boundary. "More Jacobi iterations" does not
approach a good answer here; it just reshuffles which specific weak nodes
sit at the top.

## The actual mechanism: normalizing by degree, not "globalness"

Directly checked what happens to the network's single best real hub
(epinions: out-degree 3,338, in-degree 44 — genuinely the top broadcaster
by raw count):

- **Raw netdegree**: this node is the unique maximum (3,294), rank 1 of
  114,467. No competitors.
- **Normalized netdegree** ($\rho = \text{nd}/\text{degree}$): this node
  scores $\rho=0.974$ — close to the maximum of 1.0, but **not** it,
  because it has a nonzero (if small) in-degree. It is outranked by
  **26,313 other nodes** that happen to have *exactly zero* in-degree,
  regardless of how small their out-degree is (mean out-degree among
  those 26,313 tied nodes: 1.81). The node ranks **26,330th**.

This is a hard ceiling effect: $\rho_i \le 1$ with equality **iff**
in-degree is exactly zero, regardless of out-degree's magnitude. A genuine
hub essentially never has *exactly* zero in-degree (a few people will
always link back), so it can approach but not reach the ceiling; a trivial
node with one outgoing edge and no incoming ones reaches it exactly, for
free. Normalization doesn't just add noise near the top — it
systematically **demotes real hubs below trivial nodes**.

## This is not a new, separate finding — it IS RQ1's identity, applied

RQ1's Proposition 1 (`proofs/rq1_jacobi_identity.md`) proves
$h^{(1)}_i = v_i/u_i = \rho_i$ exactly. For any node with in-degree 0,
$v_i = -u_i$ (all of its degree is outgoing), so $\rho_i = v_i/u_i = -1$
**identically**, independent of the actual value of $u_i$. This is not a
coincidence discovered empirically after the fact — it is a direct,
inevitable algebraic consequence of the identity RQ1 set out to prove. The
same $\rho_i = v_i/u_i$ term is what full trophic level's fixed-point
equation ($h_i = (Sh)_i/u_i - \rho_i$, proof §4) carries forward at every
iteration, which is exactly why the pathology survives all the way to the
converged trophic level, not just $t=1$: the forcing term for every
zero-in-degree node is identically $-1$ at every step, by construction,
regardless of $t$.

## Why this matters for how the paper frames RQ1 vs. RQ3/RQ4/RQ5

**Raw (unnormalized) netdegree — the measure that actually wins in RQ3,
RQ4, and RQ5 — is not a member of the Jacobi-iterate family RQ1's
identity is about.** RQ1's unifying claim concerns *normalized* netdegree
($\rho$) as the first Jacobi iterate of trophic level; the empirically
best-performing measure across every downstream task is the *unnormalized*
one, which sits outside that family entirely and does not inherit its
boundary pathology. The paper's elegant unifying mathematical story (RQ1)
and its strongest empirical result (raw netdegree's dominance) are about
**two different quantities** that happen to share a name. This needs to be
stated explicitly and prominently, not blurred — a reviewer who notices
this before the paper addresses it will read it as an inconsistency in the
paper's own argument, when actually it's a real, interesting, explicable
fact about where the elegant math and the practical performance part ways.

## Tested: regularization (SpringRank-style) fixes it completely

Implemented `measures.spring_rank(g, alpha)` and swept $\alpha$ on both
worst-case networks, measuring actual IC seed-set spread (k=100):

| $\alpha$ | epinions spread | epinions mean out-deg of seeds | Slashdot spread |
|---|---|---|---|
| 0.01 | 106.6 | 1.3 | 117.1 |
| 1.0 | 1606.2 | 122.2 | 1117.4 |
| 5.0 | 1942.1 | 347.9 | 1892.2 |
| 20.0 | 2086.5 | 483.2 | 2423.4 |
| 100.0 | **2160.4** | 616.9 | **2582.8** |
| raw netdegree (reference) | 2128.7 | 736.8 | 2555.0 |

At $\alpha=100$, SpringRank's regularized seed selection **matches or
slightly exceeds raw netdegree's spread on both networks** — a complete
rescue, exactly as the shrinkage mechanism predicted (regularization
matters proportionally more for small-$u_i$ nodes, pulling trivial nodes'
scores down relative to real hubs). The transition is smooth and monotonic
in $\alpha$: mean out-degree of the selected top-100 climbs steadily from
~1 (indistinguishable from the unregularized failure) to ~500-700
(comparable to netdegree's own seed set) as $\alpha$ grows.

**Practical implication for the paper:** raw/unregularized trophic level
should not be recommended for seed selection, but this is fixable, not
fundamental — SpringRank with $\alpha$ on the scale of the network's hub
degrees (here, $\alpha \approx 20$-$100$ for hub out-degrees in the
hundreds-to-thousands) recovers competitive performance. $\alpha$ needs to
be chosen relative to the network's degree scale, not a fixed constant
across networks — not yet tested how to pick it in a principled,
data-driven way (e.g. proportional to median or 90th-percentile out-degree)
rather than by sweeping; worth doing before this goes in the paper as a
recommendation rather than just a diagnostic.
