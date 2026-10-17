"""Score the held-out runs against the pre-registered hypotheses (heldout/PREREGISTRATION.md).

    python heldout/evaluate.py --results results_heldout

Writes <results>/hypotheses.csv (one row per hypothesis x network, held or failed), copies it
to heldout/hypotheses.csv so the outcome is versioned next to the predictions, and prints a
summary. Thresholds and classes are read from the pre-registration, never from the results.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import make_figures  # noqa: E402

SETTING, K = "IC p=1pc", 100


def discordance(m, n_pairs=500_000, seed=0):
    """Fraction of node pairs with distinct weighted net-degree that SpringRank (alpha = max
    degree) orders differently from weighted net-degree."""
    nd, x = m.netdegree_weighted.to_numpy(), m.spring_rank_maxdeg.to_numpy()
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(m), n_pairs), rng.integers(0, len(m), n_pairs)
    keep = nd[i] != nd[j]
    return float(np.mean(np.sign(x[i][keep] - x[j][keep]) != np.sign(nd[i][keep] - nd[j][keep])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_heldout")
    args = ap.parse_args()
    pre = pd.read_csv(os.path.join(HERE, "preregistration.csv")).set_index("net")
    make_figures.seed_stats(args.results)
    ratio = pd.read_csv(os.path.join(args.results, "seed_ratio.csv"))
    r = ratio[ratio.setting == SETTING].set_index("net")
    nets = [n for n in pre.index if n in r.index]
    rows = []

    def add(h, net, value, held, note=""):
        rows.append(dict(hypothesis=h, net=net, value=value, held=bool(held), note=note))

    for net in nets:
        cls, x = pre.loc[net, "predicted"], r.loc[net]
        add("H1", net, x.trophic, x.trophic < 0.5, "trophic ratio < 0.5")
        if cls == "normal":
            gap = x.spring_rank_maxdeg - x.netdegree
            add("H2", net, x.spring_rank_maxdeg, x.spring_rank_maxdeg >= 0.95 and abs(gap) <= 0.02,
                f"SR(max) >= 0.95 and |SR(max) - net-degree| = {abs(gap):.3f} <= 0.02")
        if cls == "degree-tail outlier":
            d = x.spring_rank_maxdeg - x.spring_rank_p99
            add("H3", net, d, d >= 0.05, "SR(max) - SR(p99) >= 0.05")
        if cls == "reciprocity ceiling":
            add("H4", net, max(x.netdegree, x.spring_rank_maxdeg), x.netdegree < 0.95 and x.spring_rank_maxdeg < 0.95,
                "net-degree and SR(max) both < 0.95")
        m = pd.read_csv(os.path.join(args.results, f"measures_{net}.csv"))
        dis = discordance(m)
        add("H5", net, dis, dis < 0.01, "discordant pair fraction < 1%")
    tau = pd.read_csv(os.path.join(args.results, "tau.csv"))
    t = tau[(tau.setting == SETTING) & tau.net.isin(nets)]
    eps = t.groupby("measure").eps_1.mean().sort_values(ascending=False)
    add("H6", "all", eps.get("trophic", np.nan), eps.index[0] == "trophic",
        f"highest mean eps_1: {eps.index[0]} ({eps.iloc[0]:.3f}); trophic {eps.get('trophic', np.nan):.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results, "hypotheses.csv"), index=False)
    out.to_csv(os.path.join(HERE, "hypotheses.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\n{len(nets)} of {len(pre)} held-out networks evaluated")
    print(out.groupby("hypothesis").held.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
