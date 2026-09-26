# A methodological finding: `positive_only=True` was a real confound

## What happened

`soc-sign-bitcoin-otc`'s registry entry sets `positive_only=True` (keep only
edges with a positive trust rating, drop negative ones) — a choice inherited
from the original proposal, applied by precedent to the three other signed
datasets added later (`soc-sign-epinions`, `soc-sign-Slashdot090221`,
`wiki-RfA`). Running all four this way produced what looked like a genuine
anomaly: bitcoin-otc's F0=0.860 with netdegree-trophic agreement of only
0.663 — *lower* than CollegeMsg's agreement (0.887) despite CollegeMsg
having a *lower* F0 (0.823). Across all 7 real networks tested, F0 vs.
agreement looked essentially non-monotonic (oscillating, not a clean trend),
apparently falsifying RQ2's hypothesis on real data even though the
synthetic sweep supported it almost perfectly.

## What it actually was

Re-loading the three positive-only-filtered datasets with **every** signed
edge kept (sign ignored, presence-only) instead:

| Dataset | F0 (filtered → unfiltered) | agreement (filtered → unfiltered) |
|---|---|---|
| soc-sign-bitcoin-otc | 0.860 → 0.897 | 0.663 → **0.832** |
| soc-sign-epinions | 0.552 → 0.572 | 0.671 → **0.714** |
| soc-sign-Slashdot090221 | 0.363 → 0.360 | 0.849 → **0.868** |

Agreement rises in every case when negative/oppose edges are kept, while F0
barely moves. **Dropping negative edges throws away real structural
information** — it isn't a neutral "keep only the meaningful signal" choice,
it measurably degrades the netdegree-trophic correspondence, most severely
on bitcoin-otc specifically (+0.169), which is exactly the dataset that
first looked anomalous.

With the three datasets' unfiltered numbers substituted back into the full
7-network comparison, Spearman(F0, agreement) goes from oscillating with no
clear sign to **+0.6** — a real, moderate positive relationship. Not as
clean as the synthetic sweep's near-1.0, and `soc-sign-epinions` remains a
partial exception (a dip relative to its neighbors even unfiltered), so
this is not a case of "the hypothesis was right all along and the anomaly
vanishes completely." It's a case of **the filtering choice was a real
confound that was partly, not wholly, responsible** for what first looked
like a clean falsification.

## Why this happened

A plausible mechanism, supported by the wiki-Vote/wiki-RfA comparison
(the same underlying Wikipedia adminship-voting process — wiki-Vote keeps
every vote as a plain edge; the wiki-RfA loader here filters to
support-only): an "oppose" or "distrust" edge is still a real, socially
meaningful interaction between two specific nodes. Removing it doesn't just
remove noise — it removes real edges from the graph that the netdegree/
trophic-level system was computed on, changing degree, reciprocity and
component structure in ways that are not accounted for by treating F0 alone
as the explanatory variable.

## What this means for the registry and the paper

- `positive_only=True` remains the registry default for these four datasets
  (unchanged, to match the original bitcoin-otc precedent and avoid
  silently rewriting already-reported numbers), but it is now a
  **documented, tested choice with a known, non-trivial effect**, not an
  unexamined default. See the comment pointing here in
  `ndbench/datasets.py`.
- The paper should report **both** filtered and unfiltered agreement for
  the signed datasets, not just one — the difference is itself a finding
  (arguably a more interesting one than the raw agreement numbers), and
  hiding it by picking one silently would be exactly the kind of choice
  RQ2's own falsification framing exists to prevent.
- ~~Open question, not yet resolved: why does `soc-sign-epinions` remain a
  partial exception even unfiltered?~~ **Investigated in
  `proofs/rq2_epinions_exception_ruled_out.md`**: the boundary-collapse
  mechanism found later (RQ3/RQ4/RQ5) was the natural suspect and was
  directly ruled out — the dip persists (is even sharper) when boundary
  nodes are excluded, so it's a property of epinions' well-connected core
  specifically, not the degenerate tied-node population. Root cause still
  genuinely open (scale effect vs. domain effect not distinguished); that
  same investigation also surfaced a more general, independent finding —
  60-81% of nodes in most of these real networks are degree-boundary cases,
  meaning RQ2's headline agreement number is usually dominated by
  trivial-node agreement rather than the well-connected core.
