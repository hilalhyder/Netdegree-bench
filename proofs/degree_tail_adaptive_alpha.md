# The degree-tail-adaptive alpha: fixes both known failure cases, costs nothing on the rest

Tests the refinement flagged as "the natural next step" in
`scaling_to_12_networks_two_failure_modes.md` and
`scaling_to_massive_networks_wiki_talk_pokec.md`: `spring_rank_p99`'s alpha
is a fixed bulk percentile (99th) of total degree, blind to how far the
true maximum sits above that bulk. `measures.spring_rank_tailadaptive`
instead uses `alpha = sqrt(p99 * max)` — the geometric mean of the bulk
percentile and the actual maximum total degree — so it scales up
automatically with the tail's real extent instead of assuming a fixed
"normal" shape. Still a single non-tuned rule, no per-dataset knob.

## Result: fixes both outlier cases, negligible cost on the rest

| Network | max/p99 ratio | alpha (p99 rule) | alpha (tail-adaptive) | ratio↑ | spread ratio: p99 | spread ratio: tail-adaptive | netdegree |
|---|---|---|---|---|---|---|---|
| wiki-Vote (normal) | 4.1 | 282 | 574 | 2.0x | 0.999 | 0.996 | 0.980 |
| soc-Pokec (mild) | 83 | 247 | 2,251 | 9.1x | 0.979 | 0.974 | 0.971 |
| email-EuAll (outlier) | 509 | 15 | 338 | 22.6x | 0.373 | **0.689** | 0.753 |
| wiki-Talk (extreme outlier) | 4,001 | 25 | 1,581 | 63.3x | 0.293 | **0.976** | 1.000 |

(spread ratio = seed-set spread relative to the best method, k=100, IC
p=1pc, quick mode, `--no-celf`; alpha in units of total degree out+in.)

The alpha multiplier scales exactly the way it should: 2x on the network
that didn't need fixing, up to 63x on the one that needed it most, without
ever being told which regime it's in.

- **wiki-Talk** (the worst case found in this whole study): spread ratio
  jumps from 0.293 to **0.976** — nearly matching raw netdegree's own
  1.000, on the network where `spring_rank_p99` failed harder than on any
  other network tested.
- **email-EuAll**: 0.373 to **0.689** — closes most (not all) of the gap to
  netdegree's 0.753. Still short of full parity, but no longer the
  near-total collapse p99 produced.
- **soc-Pokec** and **wiki-Vote** (the two cases that were already
  resolved): cost is essentially noise (-0.005 and -0.003 respectively),
  and soc-Pokec's tail-adaptive number (0.974) still edges out netdegree's
  own (0.971) — pushing alpha up further on an already-fine case doesn't
  break anything.

## Why the geometric mean, mechanically

`spring_rank`'s regularization shrinks every node's score toward 0,
proportionally more for small-u_i (low-degree) nodes than large-u_i (hub)
nodes (see `spring_rank`'s own docstring, `normalization_is_the_actual_mechanism.md`).
The boundary-collapse mechanism this whole fix chases is about the
*smallest*-degree nodes (u_i near 0, especially zero-in-degree) getting
pinned to extreme raw-trophic values; alpha needs to be large enough,
relative to those nodes' own tiny u_i, to meaningfully shrink them. But
"large enough" is relative to the network's overall degree scale, and a
fixed bulk percentile (p99) has no information about how far outliers
above that bulk actually reach. Multiplying by the true maximum (via a
geometric mean, so the fixed p99 rule is recovered when max ≈ p99, i.e. a
genuinely light-tailed network) lets alpha track the actual outlier
distance instead of assuming a "normal" one. This isn't a derivation of
the *optimal* alpha — the "still open" item in `SUMMARY_the_regularization_fix.md`
about a closed-form account remains genuinely open — but it is a concrete,
mechanistic reason the fix should track `max`, not just the bulk, and it
holds up empirically on the two cases the plain p99 rule struggles with the
most.

## Update: re-running the full 12-network suite corrected the reciprocity-ceiling claim

The 4-network pilot above (wiki-Vote, soc-Pokec, email-EuAll, wiki-Talk)
didn't include either reciprocity-ceiling network. Re-running the complete
12-network suite (`scaling_to_12_networks_two_failure_modes.md`'s "Full
12-network table, current") with `spring_rank_tailadaptive` wired into
`SEED_METHOD_BASE` found both reciprocity-ceiling networks improve too:
`soc-sign-bitcoin-otc` 0.823 -> **0.899** (now beating netdegree's 0.814),
`soc-Slashdot0922` (unsigned) 0.598 -> **0.749** (still below netdegree's
0.852, but a real gain). Both networks have ordinary degree-tail ratios
(9.7 and 16.2) — this has nothing to do with the tail mechanism above; it's
simply that `sqrt(p99*max)` happens to land 3-4x above plain p99 on these
two networks, and they wanted more alpha than p99 gave them for reasons
independent of tail shape (bitcoin-otc's own percentile-sweep curve in
`SUMMARY_the_regularization_fix.md`'s "still open" section already showed
this: it kept improving all the way to p99.99).

The corrected statement, replacing the earlier "reciprocity ceiling is
untouched by regularization" claim: high reciprocity still limits how much
directional signal exists for any netdegree-family measure to recover (a
real ceiling — bitcoin-otc needed a specific, larger alpha to reach it,
and Slashdot0922-unsigned still falls short of netdegree even at that
alpha), but the ceiling itself is not alpha-invariant. Full 12-network
mean is now **0.940** for tailadaptive vs. **0.894** for p99, against
netdegree's own **0.944** — every one of the 12 networks either improves
or stays within noise, none regresses meaningfully.

## Follow-up: pushing the exponent further -- spring_rank_maxdeg

The user asked to "try a higher exponent alpha rule on email-EuAll." Generalizing
`spring_rank_tailadaptive`'s alpha to `p99^a * max^(1-a)` (a=1 recovers
`spring_rank_p99`, a=0.5 is `spring_rank_tailadaptive`) and sweeping a on
6 stress-test networks (the 4 above plus the two reciprocity-ceiling
networks, `soc-sign-bitcoin-otc` and `soc-Slashdot0922` unsigned) found the
ratio-to-netdegree curve monotonically improving as a -> 0 on every single
one of those 6, with **a=0 (pure `alpha = max(total degree)`, no percentile
at all)** at or above every other exponent tried, including strict wins on
the two hardest, most alpha-hungry cases in the whole study (bitcoin-otc:
1.117 vs tailadaptive's 1.102; Slashdot0922-unsigned: 0.964 vs 0.878).
Registered as `measures.spring_rank_maxdeg` and validated against the
complete 14-network suite (all 12 + wiki-Talk + soc-Pokec; soc-Pokec's
first attempt was killed by a system memory-pressure safeguard mid-run and
successfully retried):

| Network | trophic | p99 | tailadaptive | **maxdeg** | netdegree |
|---|---|---|---|---|---|
| email-Eu-core | 0.842 | 0.990 | 0.989 | 0.989 | 0.986 |
| CollegeMsg | 0.805 | 0.982 | 0.982 | 0.986 | 1.000 |
| wiki-Vote | 0.186 | 1.000 | 0.998 | 0.987 | 0.983 |
| wiki-RfA | 0.493 | 0.993 | 0.996 | 0.997 | 1.000 |
| soc-sign-bitcoin-otc | 0.432 | 0.823 | 0.899 | **0.911** | 0.814 |
| soc-sign-epinions | 0.047 | 1.000 | 0.989 | 0.986 | 0.985 |
| soc-sign-Slashdot090221 | 0.040 | 1.000 | 0.995 | 0.989 | 0.984 |
| cit-HepPh | 0.032 | 1.000 | 0.998 | 0.995 | 0.987 |
| soc-Epinions1 | 0.058 | 1.000 | 0.995 | 0.991 | 0.987 |
| email-EuAll | 0.219 | 0.373 | 0.689 | **0.738** | 0.753 |
| ego-Twitter | 0.117 | 0.962 | 0.990 | **1.000** | 0.994 |
| soc-Slashdot0922 | 0.104 | 0.598 | 0.749 | **0.824** | 0.852 |
| wiki-Talk | 0.022 | 0.293 | 0.976 | **0.998** | 1.000 |
| soc-Pokec | 0.015 | 0.979 | 0.974 | 0.977 | 0.971 |
| **mean, all 14** | **0.244** | **0.857** | **0.944** | **0.955** | 0.950 |

soc-Pokec lands exactly where the earlier lighter-weight prediction said it
would: tailadaptive 0.974, maxdeg 0.977, p99 0.979 — all three
statistically indistinguishable and all beating netdegree's own 0.971. It
neither strengthens nor weakens the overall verdict; it's simply confirmed
as another "already-resolved, negligible-cost" network.

**Honest verdict, not a clean sweep:** on the full 14-network measurement
(vs. the 6-network sweep that motivated it), `spring_rank_maxdeg`'s mean
(0.955) exceeds both `spring_rank_tailadaptive` (0.944) and even netdegree's
own mean (0.950) -- but it is not a strict, uniform improvement over
tailadaptive the way the 6-network sweep suggested. It wins clearly on 6
networks (all 4 of the hardest cases -- both degree-tail outliers and both
reciprocity-ceiling networks -- plus CollegeMsg and ego-Twitter), ties on 1
(email-Eu-core), and shows small regressions (-0.002 to -0.013, noise-level
against this benchmark's own bootstrap CIs) on 6 networks that were already
near-fully-resolved (wiki-Vote, soc-sign-epinions, soc-sign-Slashdot090221,
cit-HepPh, soc-Epinions1, soc-Pokec). The trade is favorable -- bigger
gains exactly where the problem was real, tiny costs exactly where there
was nothing left to gain -- but the earlier framing ("a=0 at or above every
other exponent on every network") was accurate only for the 6 stress-test
networks it was checked on, not a universal claim; corrected here rather
than left overstated.

**Recommendation, now fully confirmed on all 14 networks:** register
`spring_rank_maxdeg` as the primary regularized-trophic default
(`spring_rank_p99` and `spring_rank_tailadaptive` kept for comparison, not
removed) given its better aggregate mean and its concentration of gains on
the networks that actually needed fixing. No networks remain unconfirmed.

## What this means for the paper

Register `spring_rank_maxdeg` (alpha = max total degree, no percentile) as
the primary regularized-trophic measure going forward, per the follow-up
section above -- it has the best aggregate mean of the three variants and
concentrates its gains on the networks that actually needed fixing.
`spring_rank_p99` and `spring_rank_tailadaptive` are kept for comparison,
not removed (all three are wired through `run_benchmark.py` /
`run_higgs.py` / `compute_all()`). Update the paper's scope statement: the
fix's applicability no longer needs "extreme degree-tail weight" as a hard
exclusion — it needs it as a signal for *which alpha rule* to use, and a
single formula (now not even needing a percentile) handles both regimes.
High reciprocity (`scaling_to_12_networks_two_failure_modes.md`'s Failure
2) remains a genuine ceiling on achievable performance, but — per the
correction above — it is a ceiling that alpha choice still moves
meaningfully within, not a fixed, alpha-invariant floor.

## Still open

- **Resolved**: re-run on the full 14-network set (see updates above) —
  confirmed no regression on any of the 3-of-5 "fits the original story
  perfectly" networks, found the reciprocity-ceiling networks improve
  too, and found `spring_rank_maxdeg` closes most of email-EuAll's
  remaining gap (0.689 -> 0.738 vs. netdegree's 0.753) as a side effect
  of the exponent sweep, without needing a separate tunable exponent.
- **Resolved**: re-run on Higgs (`rq5_higgs_findings.md`'s update section)
  — the one dataset in the whole study with real, non-simulated cascade
  ground truth. Result: `spring_rank_p99` edges out `tailadaptive` and
  `maxdeg` there, the *opposite* order from this doc's aggregate mean.
  Not a contradiction: Higgs's degree-tail ratio (85.1) and reciprocity
  (0.316) put it in the same mild-tail "already resolved" bucket as
  soc-Pokec, where the three variants were likewise statistically
  indistinguishable in the simulated benchmark — a real, independent
  corroboration of the finding that alpha-rule choice only matters on
  genuinely extreme-tail or high-reciprocity networks.
- Still no closed-form derivation of alpha from first principles — this
  refinement is a better empirical rule, not a resolution of that
  theoretical gap.
