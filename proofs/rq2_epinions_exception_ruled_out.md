# RQ2's epinions exception: the boundary-collapse mechanism is ruled out, and a bigger methodological point surfaces instead

## The question this closes

`proofs/positive_only_filtering_effect.md` flagged `soc-sign-epinions` as a
partial, unexplained exception to RQ2's F0-agreement relationship even
after correcting the sign-filtering artifact. Given
`proofs/normalization_is_the_actual_mechanism.md` and
`proofs/SUMMARY_the_regularization_fix.md` later found and fixed a related
boundary-collapse pathology (zero-in-degree nodes pinned to the same
normalized-netdegree/trophic-level boundary value) driving RQ3/RQ4/RQ5's
failures, the natural hypothesis was: is the *same* mechanism behind
epinions' RQ2 dip?

## Directly tested: no

Recomputed RQ2's agreement metric (Spearman of normalized netdegree vs.
trophic level) separately on boundary nodes (zero in- or out-degree) and
non-boundary nodes, across all 7 real networks (sorted by F0):

| Network | F0 | agreement, all nodes | agreement, boundary only | agreement, non-boundary only |
|---|---|---|---|---|
| wiki-Vote | 0.283 | 0.794 | 0.635 | 0.914 |
| wiki-RfA (unfiltered) | 0.334 | 0.731 | 0.539 | 0.810 |
| Slashdot (unfiltered) | 0.360 | 0.868 | 0.734 | 0.823 |
| **epinions (unfiltered)** | 0.572 | 0.714 | 0.666 | **0.699** |
| CollegeMsg | 0.823 | 0.887 | 0.421 | 0.767 |
| bitcoin-otc (unfiltered) | 0.897 | 0.832 | 0.233 | 0.700 |
| email-Eu-core | 0.907 | 0.962 | 0.552 | 0.939 |

**The dip persists, and is if anything sharper, when boundary nodes are
excluded entirely** (0.699 vs. its F0-neighbors' 0.81-0.82, rather than
being explained away). This rules out the RQ3/RQ4/RQ5 mechanism as the
cause here: whatever is different about epinions is a property of its
well-connected core specifically, not the degenerate tied-boundary
population that caused the other failures.

## Second hypothesis tested and also ruled out: network scale

Epinions is by far the largest network tested (131,828 nodes, 5-25x the
others), so size itself was the next candidate. Tested directly by
subsampling epinions down toward the other networks' scale (uniform random
node subsample, induced subgraph, five sizes):

| fraction kept | n | F0 | agreement (all nodes) | agreement (non-boundary) |
|---|---|---|---|---|
| 1.00 (full) | 131,828 | 0.572 | 0.714 | 0.699 |
| 0.50 | 65,914 | 0.556 | 0.657 | 0.652 |
| 0.25 | 32,957 | 0.507 | 0.619 | 0.617 |
| 0.10 | 13,182 | 0.477 | 0.552 | 0.563 |
| 0.05 | 6,591 | 0.386 | 0.537 | 0.477 |

**This rules out scale too, and in the opposite direction than "scale
explains it" would need.** At 10% (n=13,182 — almost exactly wiki-RfA's
10,010), epinions' F0 is 0.477 with agreement only 0.552-0.563. wiki-RfA,
at a *lower* F0 (0.338), has *higher* agreement (0.731). If F0 alone drove
agreement, the smaller epinions sample should beat wiki-RfA; it doesn't,
by a wide margin. Shrinking epinions doesn't make it look like its
same-sized peers — it stays anomalously low at every scale tested,
including scales matching its peers almost exactly. (Caveat: uniform
random node subsampling also tears apart real local structure, so this
isn't a perfectly clean "hold domain fixed, vary only scale" experiment —
but it is a real test of the specific "epinions is unusual because it's
big" hypothesis, and that hypothesis does not survive it.)

With both the boundary-collapse mechanism and network scale now
eliminated as explanations, the remaining live hypothesis is a genuine
**domain effect**: something about product-review trust specifically
(as opposed to financial-trading trust in bitcoin-otc, social tagging in
Slashdot, or Wikipedia adminship voting in wiki-RfA/wiki-Vote) that isn't
captured by F0, reciprocity, or scale. Not further diagnosed here — left
as a genuinely open question, but a much narrower one than at the start of
this investigation.

## The more important thing this check surfaced

`frac_boundary` (nodes with zero in-degree OR zero out-degree) is **61-81%
on 4 of the 7 real networks** (wiki-Vote 80.7%, wiki-RfA 77.8%, Slashdot
60.8%, epinions 63.8%) and 18.5-30.9% on the other three. RQ2's headline
agreement number, computed uniformly across all nodes, is therefore
**dominated by trivial-node agreement** on most of the real networks in
this study, not by how well the measures track each other on the
well-connected core the metric is implicitly meant to characterize. This
is a real methodological point independent of the epinions question:

- The paper should report RQ2's agreement metric **both ways** — over all
  nodes (as currently) and restricted to non-boundary nodes — the same way
  `proofs/positive_only_filtering_effect.md` argued for reporting both
  filtered and unfiltered signed-edge numbers. Both readings are legitimate
  and answer slightly different questions ("do the measures agree overall"
  vs. "do they agree on nodes where the comparison is actually
  informative").
- Note the boundary/non-boundary split itself is informative:
  non-boundary agreement is *higher* than the all-node number on 5 of 7
  networks (wiki-Vote, wiki-RfA, CollegeMsg, bitcoin-otc, email-Eu-core) —
  meaning trivial nodes are, on those networks, actively *dragging down*
  the headline agreement number, not inflating it. Slashdot and epinions
  are the two exceptions, where boundary-node agreement is close to or
  drags the number in the other direction.

## What this means for the paper

- State plainly that the epinions exception in RQ2 was investigated and
  the RQ3/RQ4/RQ5 mechanism was ruled out as its cause — an honest "we
  checked and it isn't this" is more credible than silence.
- Add the boundary/non-boundary breakdown as a table or footnote in RQ2's
  results — it's a genuinely new, general point (most nodes in these real
  networks are degree-degenerate for this purpose) that the F0-only framing
  doesn't surface on its own.
