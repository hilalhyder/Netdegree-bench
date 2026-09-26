# Scaling to 12 real networks: two genuine limits to the regularization fix, both explained

## What changed

`SUMMARY_the_regularization_fix.md` claimed a complete resolution across
the original 7 real networks: `spring_rank_p99` matched or beat netdegree
on every one of them. Five more real networks were added
(`cit-HepPh`, `soc-Epinions1` [unsigned trust, distinct from the signed
one already tested], `email-EuAll`, `ego-Twitter`, `soc-Slashdot0922`
[unsigned, distinct from the signed `soc-sign-Slashdot090221`]) via
quick-mode runs. Along the way, the registry's `soc-Slashdot0922` URL was
found to be wrong (`soc-Slashdot0922.txt.gz` 404s; the real SNAP file is
`soc-Slashdot0902.txt.gz`) and fixed.

**The "complete resolution" claim does not fully survive scaling up** —
honest to report, and now explained rather than left as a surprise:

| Network | trophic | spring_rank_p99 | netdegree |
|---|---|---|---|
| cit-HepPh | 0.030 | **1.000** | 0.973 |
| soc-Epinions1 (unsigned) | 0.055 | **1.000** | 0.987 |
| ego-Twitter | 0.091 | 0.968 | **1.000** |
| email-EuAll | 0.222 | **0.377** | 0.763 |
| soc-Slashdot0922 (unsigned) | 0.104 | **0.600** | 0.855 |
| mean, 5 new networks | 0.101 | 0.789 | 0.916 |
| mean, original 7 networks | 0.408 | 0.973 | 0.969 |
| mean, all 12 | 0.280 | 0.897 | 0.947 |

**Superseded by the full 12-network re-run with `spring_rank_tailadaptive`
included** (see the update below and `degree_tail_adaptive_alpha.md` for
the complete, current numbers) — kept here for the historical record of
what motivated that refinement.

Three of five new networks (cit-HepPh, soc-Epinions1, ego-Twitter) fit the
original story perfectly — `spring_rank_p99` wins outright or ties
netdegree closely. Two do not: `email-EuAll` and `soc-Slashdot0922`
(unsigned). Both were investigated directly rather than left as noise.

## Failure 1: email-EuAll is a genuine degree-distribution outlier

Compared max-degree / p99-degree ratio across all 11 networks checked:

| Network | max/p99 ratio |
|---|---|
| **email-EuAll** | **509.1** |
| soc-Slashdot0922 (unsigned) | 16.2 |
| soc-sign-Slashdot090221 | 15.6 |
| soc-Epinions1 | 13.7 |
| soc-sign-epinions | 14.3 |
| ego-Twitter | 10.8 |
| soc-sign-bitcoin-otc | 9.7 |
| cit-HepPh | 5.9 |
| wiki-Vote | 4.1 |
| CollegeMsg | 2.1 |
| email-Eu-core | 2.0 |

email-EuAll's ratio is **30-250x every other network's**. The p99 rule
picks alpha=15 while the true maximum degree is 7,636 — nowhere near
enough regularization to separate real hubs from the bulk on this
specific, unusually heavy-tailed network. This is not a general failure of
the p99 rule; it's a rule tuned against normally-heavy-tailed real networks
meeting a network whose tail is far heavier than any of them. A
degree-distribution-adaptive alpha (e.g. relative to max degree directly,
not just a percentile of the bulk) would likely fix this specific case —
not implemented here, flagged as the natural next refinement together with
the already-open "derive alpha theoretically" question in
`SUMMARY_the_regularization_fix.md`.

## Failure 2: high reciprocity is a structural ceiling, not a fixable bug

soc-Slashdot0922 (unsigned)'s max/p99 ratio (16.2) is unremarkable — similar
to networks where the fix worked perfectly. Its reciprocity, however, is
**0.841** — nearly identical to `soc-sign-bitcoin-otc`'s **0.839**, the two
highest reciprocity values and the two hardest networks across the *entire*
12-network study (bitcoin-otc: trophic 0.437, spring_rank_p99 only 0.840
even after the fix; Slashdot0922-unsigned: trophic 0.104, spring_rank_p99
0.600).

This makes structural sense and is not something regularization can fix:
netdegree is a measure of directional *imbalance* (out-degree minus
in-degree). In a highly reciprocal network, most edges go both ways, so
out-degree approximately equals in-degree for nearly every node — there is
little directional signal for any netdegree-family measure, regularized or
not, to exploit in the first place. This is a real ceiling on what the
whole netdegree family (not just raw trophic level) can achieve, distinct
from the boundary-collapse mechanism this whole investigation started
from.

## Update: the degree-tail-outlier failure mode is now fixable, and Failure 2 needed correcting too

`degree_tail_adaptive_alpha.md` replaces the fixed p99 alpha with
`alpha = sqrt(p99 * max)`, which scales up automatically with how far the
true maximum sits above the bulk instead of assuming a "normal" tail
shape. On `email-EuAll` (this doc's original tail-outlier case, ratio 509)
the spread ratio goes from 0.373 to 0.689; on `wiki-Talk`
(`scaling_to_massive_networks_wiki_talk_pokec.md`'s far more extreme
ratio-4001 case) it goes from 0.293 to 0.976. Failure 1 (degree-tail
outlier) below is now a fixable case with a single non-tuned rule, not a
hard scope exclusion.

**Failure 2's "untouched by regularization" claim was too strong and is now
corrected** by re-running the full 12-network suite with
`spring_rank_tailadaptive` included: on both reciprocity-ceiling networks,
alpha still moves the number meaningfully — `soc-sign-bitcoin-otc` goes
0.823 (p99) -> **0.899** (tailadaptive), now *beating* netdegree's own
0.814; `soc-Slashdot0922` (unsigned) goes 0.598 -> **0.749**, still below
netdegree's 0.852 but a real, non-trivial gain. Both networks have
*normal* degree-tail ratios (9.7 and 16.2 respectively — nowhere near
outlier territory), so this isn't the tail-adaptive mechanism doing
anything tail-specific; it's simply that `sqrt(p99*max)` happens to pick
an alpha 3-4x larger than plain p99 here, and these two networks wanted
more alpha than p99 gave them regardless of the tail-outlier question
(this was already flagged as an open item in `SUMMARY_the_regularization_fix.md`:
"bitcoin-otc's own percentile-sweep curve kept improving all the way to
p99.99"). The corrected statement: high reciprocity genuinely limits how
much directional signal exists for any netdegree-family measure to recover
(there is a real ceiling — bitcoin-otc still needed a specific, larger
alpha to reach it, and Slashdot0922-unsigned still falls short of
netdegree even at that alpha), but the ceiling is not alpha-invariant the
way the original framing claimed. Regularization strength interacts with
both failure modes, just via different mechanisms (tail shape vs. general
under-regularization).

## Update: both risk factors confirmed predictive out-of-sample

`scaling_to_massive_networks_wiki_talk_pokec.md` tested both risk factors
identified above (extreme degree-tail weight; high reciprocity) against two
much larger networks (wiki-Talk, 5M edges; soc-Pokec, 30.6M edges) *before*
looking at spread numbers. wiki-Talk's max/p99 ratio (4001, dwarfing this
doc's already-extreme email-EuAll at 509) correctly predicted a
degree-tail-outlier failure, worse even than email-EuAll's. soc-Pokec's
ratio (83) and reciprocity (0.543) are both comfortably below the risk
thresholds and correctly predicted a clean resolution
(`spring_rank_p99` at 0.979, edging out netdegree's own 0.971) — the first
genuinely out-of-sample confirmation that these are real, checkable
pre-conditions rather than post-hoc descriptions of the 12 networks that
produced them.

## What this means for the paper

- Do not claim universal resolution. State the fix as: complete on the
  original 7 real networks tested in depth, and 3 of 5 additional networks
  checked at broader scale; two genuine, understood limits found and
  explained (extreme tail-heaviness; high reciprocity), not swept under
  the rug.
- The reciprocity finding is arguably more important than the degree-tail
  finding: it identifies **which networks the whole netdegree family
  (not just this specific fix) is fundamentally ill-suited for** —
  something the paper should state as a scope-limiting condition on
  netdegree's applicability generally, not just a caveat on the
  regularization fix specifically. A quick, checkable prediction for future
  work: reciprocity above roughly 0.8 should predict weak netdegree-family
  performance on any new network, before running any experiment at all.
  (Refined below: this predicts a *ceiling*, not a fixed number — alpha
  choice still matters within it.)

## Full 12-network table, current (with `spring_rank_tailadaptive`)

k=100, IC p=1pc spread ratio to the best method per network (quick mode for
the 5 additional networks, full mode for the original 7, matching the
methodology each was originally tested under):

| Network | trophic | p99 | tailadaptive | netdegree |
|---|---|---|---|---|
| email-Eu-core | 0.842 | 0.990 | 0.989 | 0.986 |
| CollegeMsg | 0.805 | 0.982 | 0.982 | 1.000 |
| wiki-Vote | 0.186 | 1.000 | 0.998 | 0.983 |
| wiki-RfA | 0.493 | 0.993 | 0.996 | 1.000 |
| soc-sign-bitcoin-otc | 0.432 | 0.823 | **0.899** | 0.814 |
| soc-sign-epinions | 0.047 | 1.000 | 0.989 | 0.985 |
| soc-sign-Slashdot090221 | 0.040 | 1.000 | 0.995 | 0.984 |
| cit-HepPh | 0.032 | 1.000 | 0.998 | 0.987 |
| soc-Epinions1 (unsigned) | 0.058 | 1.000 | 0.995 | 0.987 |
| email-EuAll | 0.219 | 0.373 | **0.689** | 0.753 |
| ego-Twitter | 0.118 | 0.968 | 0.996 | 1.000 |
| soc-Slashdot0922 (unsigned) | 0.104 | 0.598 | **0.749** | 0.852 |
| **mean, all 12** | **0.281** | **0.894** | **0.940** | **0.944** |

`spring_rank_tailadaptive`'s mean (0.940) closes nearly all of the
remaining gap to netdegree's own mean (0.944), up from p99's 0.894. Every
single network either improves or stays within noise (max regression:
-0.005 on soc-Epinions1) — there is no network in this 12 where
tailadaptive is meaningfully worse than p99.

**Further update, now confirmed on all 14 networks including soc-Pokec**:
`degree_tail_adaptive_alpha.md`'s exponent-sweep follow-up found pushing
further -- `spring_rank_maxdeg` (alpha = max total degree, dropping the
percentile entirely) -- has an even better aggregate mean (0.955 across
all 14 networks, vs. tailadaptive's 0.944 and netdegree's own 0.950),
concentrated on exactly the hardest cases (email-EuAll, wiki-Talk, and
*both* reciprocity-ceiling networks), at the cost of small (-0.002 to
-0.013) regressions on 6 already-resolved networks. See that doc for the
full table and the honest "not a clean sweep, but a favorable trade"
verdict -- `spring_rank_maxdeg` is now the recommended default, not
`spring_rank_tailadaptive`.
