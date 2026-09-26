# Scaling to two massive networks: one extreme confirmation, one clean resolution

Extends `scaling_to_12_networks_two_failure_modes.md`'s two failure modes
(degree-tail outlier; reciprocity ceiling) to two much larger real networks:
`wiki-Talk` (2,394,385 nodes, 5,021,410 edges) and `soc-Pokec` (1,632,803
nodes, 30,622,564 edges) — both run in quick mode, `--no-celf` (CELF greedy
is prohibitively expensive at this scale; seed selection uses the
non-CELF methods only).

Along the way, found and fixed a real registry bug: `soc-Pokec` had
`flip=False`, inconsistent with every other "i declares a relationship
toward j" dataset in this registry (i follows j, i trusts j, i tags j as
friend — all `flip=True`). SNAP's own description ("a directed edge from
user A to user B represents that user A considers user B as a friend") is
the identical declarer -> declared-friend structure, so corrected to
`flip=True`.

## Headline numbers (k=100, IC p=1pc, spread ratio to best method)

| Network | trophic | spring_rank_p99 | netdegree | max/p99 degree ratio | reciprocity |
|---|---|---|---|---|---|
| wiki-Talk | 0.022 | **0.293** | **1.000** | **4001** | 0.144 |
| soc-Pokec | 0.015 | **0.979** | 0.971 | 83 | 0.543 |
| mean, original 7 | 0.408 | 0.973 | 0.969 | 2-16 | (low-moderate) |
| mean, +5 more (12 total) | 0.101–0.280 | 0.789–0.897 | 0.916–0.947 | 4-509 | (mixed) |

`trophic` collapses on both, as always (the zero-in-degree boundary
mechanism doesn't care about scale). The other two split cleanly along the
axis `scaling_to_12_networks_two_failure_modes.md` already identified.

## wiki-Talk: the degree-tail-outlier failure mode, far more extreme

wiki-Talk's max/p99 total-degree ratio is **4001** — max degree 100,032
against a 99th-percentile of just 25. This dwarfs `email-EuAll`'s 509.1,
itself flagged there as "30-250x every other network." `spring_rank_p99`
picks alpha=25 (the bulk's 99th percentile) while the actual hub needs
~4000x more shrinkage headroom than that to be told apart from the bulk —
the fixed p99 rule is even further from adequate here than on email-EuAll.
Consequence: `spring_rank_p99`'s spread ratio (0.293) is *worse* than
`email-EuAll`'s own (0.377), scaling in the same direction as the ratio.

wiki-Talk also has an unusually skewed direction split: 93.8% of nodes have
zero out-degree (most editors only ever *receive* talk-page messages, never
send them) versus only 1.05% zero in-degree. `netdegree` is untouched by
this — it isn't subject to the boundary-collapse mechanism at all (see
`normalization_is_the_actual_mechanism.md`: raw netdegree sits outside the
Jacobi-iterate family the collapse comes from) — and it ties the best
method outright (ratio 1.000) on exactly the network where the "fix"
fails worst. This is the single cleanest illustration in the whole study of
why netdegree's own robustness, not just the trophic-level theory, belongs
in the paper's story.

## soc-Pokec: confirms the fix generalizes to massive scale when the tail isn't pathological

soc-Pokec's max/p99 ratio is 83 — higher than the original 7 networks'
2-16 range but nowhere near wiki-Talk's or email-EuAll's territory, and its
reciprocity (0.543) sits well below the ~0.8 ceiling identified in
`scaling_to_12_networks_two_failure_modes.md`. Both risk factors are absent,
and the fix performs exactly as the original theory predicts:
`spring_rank_p99` reaches 0.979 of the best method (`leaderrank`), actually
*edging out* netdegree's own 0.971 — the largest network in the whole study
(30.6M edges) is also one of the cleanest confirmations that regularized
SpringRank matches or beats raw netdegree once the two known risk
conditions are both absent.

## Updated scope statement

The two risk factors identified at 12-network scale — extreme degree-tail
weight (max/p99 ratio far above ~500) and high reciprocity (above ~0.8) —
correctly predict both massive-network outcomes *before* looking at any
spread numbers: wiki-Talk trips the first (ratio 4001), soc-Pokec trips
neither (ratio 83, reciprocity 0.543) and resolves cleanly. This is the
first genuinely out-of-sample test of the scope statement (both risk
thresholds were set using the original 12 networks, then applied here
without adjustment) and it held on both counts. State in the paper as: two
independent, checkable pre-conditions, confirmed predictive on 2 additional
networks after being fit on the first 12, not merely descriptive of the
networks that produced them.

## Resolved: the degree-tail-adaptive alpha (was "still open" below)

`degree_tail_adaptive_alpha.md` tested exactly the refinement flagged
below on these same two networks plus the two known reference cases
(wiki-Vote, email-EuAll). `spring_rank_tailadaptive`
(alpha = sqrt(p99 * max) instead of a fixed p99) brings wiki-Talk's spread
ratio from 0.293 to **0.976** (nearly matching netdegree's own 1.000) and
email-EuAll's from 0.373 to 0.689, while costing only -0.003/-0.005 on the
two networks that were already resolved. The degree-tail-outlier failure
mode identified in this doc is now a fixable case, not a hard exclusion —
see that doc for the full mechanism and numbers.

## Further update: `spring_rank_maxdeg` supersedes tailadaptive as the recommended default

`degree_tail_adaptive_alpha.md`'s follow-up exponent sweep found alpha =
max total degree (no percentile at all) beats tailadaptive's geometric
mean on wiki-Talk specifically (0.998 vs. 0.976 ratio-to-netdegree at
k=100/IC), essentially reaching netdegree parity on the network that
motivated this whole investigation. soc-Pokec's re-run (after an earlier
attempt was killed by a system memory-pressure safeguard mid-run) confirms
it lands exactly where a lighter-weight prediction said it would:
`spring_rank_maxdeg` 0.977, `spring_rank_tailadaptive` 0.974, `spring_rank_p99`
0.979 — all three statistically indistinguishable and all beating
netdegree's own 0.971. soc-Pokec neither strengthens nor weakens the
overall verdict; it's confirmed as another "already-resolved,
negligible-cost" network, and the full 14-network picture
(`degree_tail_adaptive_alpha.md`) is now complete.

## Still open (unchanged from `SUMMARY_the_regularization_fix.md`)

- No closed-form alpha rule yet — even `spring_rank_maxdeg` is a better
  empirical default, not a theoretically derived constant.
