#!/usr/bin/env python3
"""Draft the paper's figures and the statistics summary from the result CSVs.

  python make_figures.py --results results --figs figures

Writes fig2 ... fig8 (PNG, 300 dpi, plus PDF) and stats_summary.csv.
Figures whose input CSV is missing are skipped with a note.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ndbench import evaluate  # noqa: E402

# Fixed method -> colour/marker mapping (colour follows the method, never its rank).
# The spring_rank_* family shares a violet hue (distinct from trophic's orange and
# netdegree's blue) so the three regularized-trophic variants read as one family in
# every figure, while remaining distinguishable from each other by shade + marker.
STYLE = {
    "netdegree":               ("#2a78d6", "o", "Netdegree"),
    "trophic":                 ("#eb6834", "s", "Trophic source (−h), unregularized"),
    "netdegree_2hop":          ("#1baf7a", "^", "2-hop netdegree"),
    "out_degree":              ("#eda100", "D", "Out-degree"),
    "pagerank_rev":            ("#e87ba4", "v", "PageRank (reversed)"),
    "leaderrank":              ("#008300", "P", "LeaderRank"),
    "voterank":                ("#4a3aa7", "X", "VoteRank"),
    "degree_discount":         ("#e34948", "*", "DegreeDiscount"),
    "celf":                    ("#0b0b0b", "h", "CELF (reference)"),
    "spring_rank_p99":         ("#c9b3f2", "p", "SpringRank (α = p99 degree)"),
    "spring_rank_tailadaptive": ("#8f5fe0", "H", "SpringRank (α = √(p99·max))"),
    "spring_rank_maxdeg":      ("#4a1e9e", "8", "SpringRank (α = max degree)"),
}
GRAY = "#8a8985"
LABEL = {"netdegree_norm": "Normalized netdegree", "netdegree_weighted": "Weighted netdegree",
         "netdegree_2hop_reach": "2-hop netdegree (reach)", "in_degree": "In-degree", "kshell": "k-shell",
         "betweenness": "Betweenness", "random": "Random", "jacobi_1": "Jacobi t=1", "jacobi_2": "Jacobi t=2",
         "jacobi_5": "Jacobi t=5", "jacobi_10": "Jacobi t=10"}

# The 14 real networks this study's headline numbers are built from (excludes the
# RQ1-RQ2 synthetic sweep). Degree-tail ratio (max/p99 total degree) and reciprocity
# classify each into the two known failure modes from proofs/scaling_to_12_networks_two_failure_modes.md.
NETS_14 = ["email-Eu-core", "CollegeMsg", "wiki-Vote", "wiki-RfA", "soc-sign-bitcoin-otc",
           "soc-sign-epinions", "soc-sign-Slashdot090221", "cit-HepPh", "soc-Epinions1",
           "email-EuAll", "ego-Twitter", "soc-Slashdot0922", "wiki-Talk", "soc-Pokec"]
TAIL_RATIO_THRESHOLD = 100    # max/p99 degree ratio above this = "degree-tail outlier"
RECIPROCITY_THRESHOLD = 0.75  # reciprocity above this = "reciprocity ceiling"

# Hand-picked, maximally-distinct qualitative palette for per-network colouring
# (e.g. fig3's 13-network Jacobi-iterate panel) -- a sampled colormap like tab20
# is built from adjacent similar-hue pairs (two blues, two oranges, ...), which
# is exactly wrong for a line plot where every curve must read apart at a glance.
NET_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8f5fe0", "#eda100", "#e87ba4",
               "#4a3aa7", "#0fa3b1", "#a3572b", "#c9b3f2", "#5c8001", "#d6336c", "#495057"]


def lab(m):
    return STYLE[m][2] if m in STYLE else LABEL.get(m, m)


def col(m):
    return STYLE[m][0] if m in STYLE else GRAY


def mk(m):
    return STYLE[m][1] if m in STYLE else "."


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
        "legend.frameon": False, "font.family": "DejaVu Sans"})


def save(fig, figs, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figs, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def read(results, name):
    p = os.path.join(results, name)
    if not os.path.exists(p):
        print(f"  skip: {name} not found")
        return None
    return pd.read_csv(p)


def fig2(results, figs, ds):
    # Curated, not alphabetical: one "normal" network, one signed/reciprocity-ceiling
    # network, one degree-tail outlier, one massive-scale network -- spans the whole
    # story rather than whatever four filenames happen to sort first.
    curated = ["wiki-Vote", "soc-sign-bitcoin-otc", "email-EuAll", "wiki-Talk"]
    files = [os.path.join(results, f"measures_{n}.csv") for n in curated]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        files = sorted(glob.glob(os.path.join(results, "measures_*.csv")))[:4]
    if not files:
        return print("  skip fig2: no measures_*.csv")
    fig, axes = plt.subplots(1, len(files), figsize=(3.0 * len(files) + 0.5, 2.8), squeeze=False)
    hb = None
    for ax, f in zip(axes[0], files):
        net = os.path.basename(f)[9:-4]
        d = pd.read_csv(f)
        hb = ax.hexbin(d["netdegree_norm"], d["trophic_level"], gridsize=40, bins="log", cmap="Blues", mincnt=1,
                        linewidths=0)
        F0 = ds.loc[ds.net == net, "F0"].iloc[-1] if ds is not None and (ds.net == net).any() else np.nan
        r = evaluate.spearman(d["netdegree_norm"], -d["trophic_level"])
        ax.set_title(f"{net}\nF0 = {F0:.2f}, Spearman = {r:.2f}")
        ax.set_xlabel("Normalized netdegree ρ")
        ax.grid(False)
    axes[0][0].set_ylabel("Trophic level h (0 = source)")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    if hb is not None:
        cax = fig.add_axes((0.955, 0.16, 0.012, 0.68))
        fig.colorbar(hb, cax=cax, label="Nodes per bin (log)")
    save(fig, figs, "fig2_netdegree_vs_trophic")


def _place_labels(ax, xs, ys, texts, fontsize=7, color="#52514e"):
    """Greedy, render-aware label placement: try a fixed ring of offsets per
    point (near first, then farther / other side) and keep the first one whose
    actual rendered bounding box doesn't collide with an already-placed label.
    Handles both point clusters and long labels overlapping distant points --
    plain fixed-offset annotate can't, since it only reasons about point
    positions, not label extents."""
    xs, ys, texts = list(xs), list(ys), list(texts)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    candidates = [(6, 3), (6, -11), (-6, 3), (-6, -11), (6, 15), (6, -22), (-6, 15), (-6, -22),
                  (11, 26), (11, -34), (-11, 26), (-11, -34)]
    for idx in np.argsort(xs):
        x, y, text = xs[idx], ys[idx], texts[idx]
        chosen = False
        for dx, dy in candidates:
            ha = "left" if dx >= 0 else "right"
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                             fontsize=fontsize, color=color, ha=ha, va="center",
                             arrowprops=dict(arrowstyle="-", color="#d8d7d2", lw=0.6, shrinkA=0, shrinkB=2))
            fig.canvas.draw()
            bbox = t.get_window_extent(renderer=renderer).expanded(1.08, 1.3)
            if not any(bbox.overlaps(b) for b in placed):
                placed.append(bbox)
                chosen = True
                break
            t.remove()
        if not chosen:
            ax.annotate(text, (x, y), xytext=candidates[-1], textcoords="offset points",
                        fontsize=fontsize, color=color, ha="left", va="center")


def fig3(results, figs):
    ag = read(results, "agreement.csv")
    if ag is None:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    syn = ag[(ag.source == "synthetic sweep") & (ag.measure == "netdegree_norm")]
    if len(syn):
        gsyn = syn.groupby("T").agg(F0=("F0", "mean"), r=("spearman_vs_trophic", "mean"),
                                    sd=("spearman_vs_trophic", "std")).sort_values("F0")
        a1.plot(gsyn.F0, gsyn.r, color=GRAY, lw=1.6, label="Synthetic sweep (mean ± sd)")
        a1.fill_between(gsyn.F0, gsyn.r - gsyn.sd.fillna(0), gsyn.r + gsyn.sd.fillna(0), color=GRAY, alpha=0.2, lw=0)
    real = ag[(ag.source == "real") & (ag.measure == "netdegree_norm")].drop_duplicates("net", keep="last")
    a1.scatter(real.F0, real.spearman_vs_trophic, color=STYLE["netdegree"][0], s=36, zorder=3,
               edgecolor="white", linewidth=1.5, label="Real networks")
    _place_labels(a1, real.F0.to_numpy(), real.spearman_vs_trophic.to_numpy(), real.net.to_numpy())
    a1.set_xlabel("Trophic incoherence F0")
    a1.set_ylabel("Spearman(ρ, −h)")
    a1.set_title("Local–global agreement vs coherence")
    a1.margins(x=0.12)
    a1.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9,
              edgecolor="#e6e5e0", facecolor="white")
    src = ag[ag.source != "synthetic sweep"]
    jac = src[src.measure.str.startswith("jacobi_")].copy()
    if len(jac):
        jac["t"] = jac.measure.str.split("_").str[1].astype(int)
        nets = list(dict.fromkeys(jac.net))
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8", "<", ">", "p", "d"]
        for i, net in enumerate(nets):
            grp = jac[jac.net == net].drop_duplicates("t", keep="last").sort_values("t")
            a2.plot(grp.t, grp.spearman_vs_trophic, marker=markers[i % len(markers)], ms=4.5,
                    color=NET_PALETTE[i % len(NET_PALETTE)], label=net)
        a2.legend(fontsize=6.3, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
        a2.set_xscale("log")
        a2.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    a2.set_xlabel("Jacobi iterations t (t = 1 is normalized netdegree)")
    a2.set_ylabel("Spearman with converged −h")
    a2.set_title("From local to global")
    fig.tight_layout()
    save(fig, figs, "fig3_agreement_vs_F0")


def fig4(results, figs, tau, setting="IC p=1pc"):
    d = tau[tau.setting == setting]
    if not len(d):
        return print(f"  skip fig4: no rows for {setting}")
    piv = d.pivot_table(index="measure", columns="net", values="tau")
    # A measure only computed on a minority of networks (e.g. netdegree_2hop_reach,
    # restricted to smaller graphs) leaves a heatmap row that's mostly blank cells --
    # visual noise, not signal. Drop rows present on fewer than half the networks.
    sparse = piv.index[piv.notna().sum(axis=1) < 0.5 * piv.shape[1]]
    if len(sparse):
        print(f"  fig4: dropping sparse rows {list(sparse)}")
    piv = piv.drop(index=sparse)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * piv.shape[1], 0.32 * piv.shape[0] + 1.2))
    im = ax.imshow(piv.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=40, ha="right")
    ax.set_yticks(range(piv.shape[0]), [lab(m) for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Kendall τ with simulated spreading")
    ax.set_title(f"Single-node spreading, {setting}")
    save(fig, figs, "fig4_tau_heatmap")


def fig5(results, figs, tau):
    d = tau[(tau.model == "IC") & tau.p_over_pc.notna()]
    if not len(d):
        return print("  skip fig5")
    mean = d.groupby(["measure", "p_over_pc"]).tau.mean().unstack()
    top = mean.mean(axis=1).sort_values(ascending=False).index[:6]
    top = list(top) + [m for m in ("netdegree", "out_degree") if m not in top]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m in top:
        ax.plot(mean.columns, mean.loc[m], marker=mk(m), color=col(m), label=lab(m), ms=5)
    ax.set_xlabel("Transmission probability p / p_c")
    ax.set_ylabel("Mean Kendall τ across networks")
    ax.set_title("Robustness to transmission probability (IC)")
    ax.legend(fontsize=7, loc="best")
    save(fig, figs, "fig5_tau_vs_p")


def fig6(results, figs):
    sp = read(results, "spread.csv")
    if sp is None:
        return
    sp = sp[sp.model == "IC"]
    # Curated: the two clearest failure-mode cases (wiki-Talk, bitcoin-otc) plus one
    # normal-tail network and the largest-scale network, rather than append order.
    curated = ["wiki-Vote", "soc-sign-bitcoin-otc", "wiki-Talk", "soc-Pokec"]
    nets = [n for n in curated if n in set(sp.net)] or list(dict.fromkeys(sp.net))[:4]
    keep = ["netdegree", "trophic", "spring_rank_p99", "spring_rank_tailadaptive", "spring_rank_maxdeg",
            "out_degree", "pagerank_rev", "leaderrank", "voterank", "degree_discount", "celf"]
    fig, axes = plt.subplots(1, len(nets), figsize=(3.1 * len(nets), 2.9), squeeze=False)
    for ax, net in zip(axes[0], nets):
        d = sp[sp.net == net]
        for m in keep:
            dm = d[d.method == m]
            if not len(dm):
                continue
            # best variant for the method at the largest k
            kmax = dm.k.max()
            v = dm[dm.k == kmax].sort_values("spread").variant.iloc[-1]
            dm = dm[dm.variant == v].sort_values("k")
            ax.plot(dm.k, dm.spread, marker=mk(m), ms=4, color=col(m), label=lab(m),
                    ls="--" if m == "celf" else "-")
            ax.fill_between(dm.k, dm.ci_low, dm.ci_high, color=col(m), alpha=0.15, lw=0)
        ax.set_title(net)
        ax.set_xlabel("Seed budget k")
    axes[0][0].set_ylabel("Expected spread (IC, p = p_c)")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, figs, "fig6_spread_vs_k")


def fig7_and_stats(results, figs, tau):
    rows = []
    for setting, d in tau.groupby("setting"):
        piv = d.pivot_table(index="net", columns="measure", values="tau").dropna(axis=1)
        if piv.shape[0] < 2 or piv.shape[1] < 3:
            continue
        avg, fstat, fp, cd = evaluate.friedman_nemenyi(piv.values)
        for m, r in zip(piv.columns, avg):
            rows.append(dict(setting=setting, test="avg_rank", measure=m, value=r, n_nets=piv.shape[0]))
        rows.append(dict(setting=setting, test="friedman_chi2", measure="", value=fstat, p=fp, n_nets=piv.shape[0]))
        rows.append(dict(setting=setting, test="nemenyi_CD", measure="", value=cd, n_nets=piv.shape[0]))
        if {"netdegree", "out_degree"} <= set(piv.columns) and piv.shape[0] >= 5:
            w = stats.wilcoxon(piv["netdegree"], piv["out_degree"])
            rows.append(dict(setting=setting, test="wilcoxon_netdegree_vs_outdegree", measure="",
                             value=float(w.statistic), p=float(w.pvalue), n_nets=piv.shape[0]))
        if setting == "IC p=1pc":
            order = np.argsort(avg)
            fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(avg) + 1.0))
            names = [lab(piv.columns[i]) for i in order]
            ax.hlines(range(len(avg)), 1, avg[order], color="#e6e5e0", lw=1)
            ax.scatter(avg[order], range(len(avg)), c=[col(piv.columns[i]) for i in order], s=30, zorder=3)
            ax.set_yticks(range(len(avg)), names)
            ax.invert_yaxis()
            best = avg[order][0]
            ax.axvspan(best, best + cd, color="#2a78d6", alpha=0.10, lw=0)
            ax.set_xlabel(f"Average rank across {piv.shape[0]} networks (lower = better)\n"
                          f"Shaded band: within the Nemenyi critical difference ({cd:.2f}) of the best")
            ax.set_title(f"Critical difference, Kendall τ, {setting}  (Friedman p = {fp:.2g})")
            save(fig, figs, "fig7_critical_difference")
    if rows:
        out = pd.DataFrame(rows)
        w = out.test == "wilcoxon_netdegree_vs_outdegree"
        if w.any():
            out.loc[w, "p_holm"] = evaluate.holm(out.loc[w, "p"].values)
        out.to_csv(os.path.join(results, "stats_summary.csv"), index=False)
        print("  wrote stats_summary.csv")


def fig8(results, figs):
    dec = read(results, "higgs_deciles.csv")
    cur = read(results, "higgs_precision_curve.csv")
    if dec is None:
        return
    fig, axes = plt.subplots(1, 2 if cur is not None else 1, figsize=(7.4, 3.0), squeeze=False)
    a1 = axes[0][0]
    d = dec[dec.split == "all activity"]
    for m in ["netdegree", "trophic", "spring_rank_p99", "spring_rank_maxdeg", "out_degree", "pagerank_rev"]:
        dm = d[d.measure == m].sort_values("decile")
        if len(dm):
            a1.plot(dm.decile, dm.mean_y, marker=mk(m), color=col(m), label=lab(m), ms=4)
    a1.set_yscale("symlog", linthresh=0.1)
    a1.set_xlabel("Score decile (10 = highest)")
    a1.set_ylabel("Mean distinct retweeters")
    a1.set_title("Higgs: retweeters by score decile")
    a1.legend(fontsize=6.5)
    if cur is not None:
        a2 = axes[0][1]
        for m in ["netdegree", "trophic", "spring_rank_p99", "spring_rank_maxdeg", "out_degree", "netdegree_2hop"]:
            c = cur[cur.measure == m]
            if len(c):
                a2.plot(c.k, c.precision, color=col(m), label=lab(m))
        a2.set_xscale("log")
        a2.legend(fontsize=7)
        a2.set_xlabel("k (top-k users by score)")
        a2.set_ylabel("Precision for top-1% retweeted")
        a2.set_title("Recovering the most-retweeted users")
    fig.tight_layout()
    save(fig, figs, "fig8_higgs")


def _degree_tail_stats(data_dir):
    """max/p99 total-degree ratio and reciprocity for each of NETS_14, loaded fresh
    from the cached SNAP files in data_dir (fast: only degree arrays, no diffusion) --
    the same two numbers proofs/scaling_to_12_networks_two_failure_modes.md uses to
    classify a network's failure mode, computed here rather than hardcoded so this
    figure can't silently drift from the registry."""
    from ndbench import datasets as dsmod
    stats = {}
    for name in NETS_14:
        try:
            g = dsmod.load_wiki_rfa(data_dir) if name == "wiki-RfA" else dsmod.load(name, data_dir=data_dir)
        except Exception as e:
            print(f"  fig9: could not load {name} ({e}), skipping its tail stats")
            continue
        u = g.out_degree().astype(float) + g.in_degree().astype(float)
        active = u > 0
        p99 = float(np.quantile(u[active], 0.99)) if active.any() else 1.0
        umax = float(u[active].max()) if active.any() else 1.0
        stats[name] = dict(ratio=umax / max(p99, 1e-9), reciprocity=float(g.reciprocity()))
    return stats


def fig9_capstone(results, figs, data_dir="data"):
    """The regularization-fix story in one figure: seed-set spread ratio-to-best-method
    (k=100, IC p=1pc) for trophic/p99/tailadaptive/maxdeg/netdegree across every real
    network this study covers, grouped by the two known failure modes from
    proofs/scaling_to_12_networks_two_failure_modes.md and proofs/degree_tail_adaptive_alpha.md."""
    sp = read(results, "spread.csv")
    if sp is None:
        return
    d = sp[(sp.setting == "IC p=1pc") & (sp.k == 100)]
    nets = [n for n in NETS_14 if n in set(d.net)]
    if not nets:
        return print("  skip fig9: no k=100/IC p=1pc rows for the 14-network suite")
    measures_of_interest = ["trophic", "spring_rank_p99", "spring_rank_tailadaptive",
                             "spring_rank_maxdeg", "netdegree"]
    rows = []
    for net in nets:
        dn = d[d.net == net]
        best_per_method = dn.groupby("method").spread.max()
        best = best_per_method.max()
        if not np.isfinite(best) or best <= 0:
            continue
        for m in measures_of_interest:
            if m in best_per_method.index:
                rows.append(dict(net=net, measure=m, ratio=best_per_method[m] / best))
    if not rows:
        return print("  skip fig9: no measures matched")
    ratios = pd.DataFrame(rows).pivot(index="net", columns="measure", values="ratio")

    tail = _degree_tail_stats(data_dir)

    def category(net):
        s = tail.get(net)
        if s is None:
            return "normal"
        if s["reciprocity"] > RECIPROCITY_THRESHOLD:
            return "reciprocity ceiling"
        if s["ratio"] > TAIL_RATIO_THRESHOLD:
            return "degree-tail outlier"
        return "normal"

    cats = {n: category(n) for n in ratios.index}
    groups = {
        "Degree-tail outliers": sorted([n for n in ratios.index if cats[n] == "degree-tail outlier"],
                                        key=lambda n: -tail.get(n, {}).get("ratio", 0)),
        "Reciprocity ceiling": sorted([n for n in ratios.index if cats[n] == "reciprocity ceiling"],
                                       key=lambda n: -tail.get(n, {}).get("reciprocity", 0)),
        "Normal (already resolved)": sorted([n for n in ratios.index if cats[n] == "normal"]),
    }
    groups = {k: v for k, v in groups.items() if v}
    heights = [max(len(v), 1) for v in groups.values()]

    measures_of_interest = [m for m in measures_of_interest if m in ratios.columns]
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.6, 0.40 * sum(heights) + 1.4 * len(groups)),
                              gridspec_kw={"height_ratios": heights}, squeeze=False)
    axes = axes[:, 0]
    bar_h = 0.8 / len(measures_of_interest)
    handles, labels = [], []
    for ax, (title, grp_nets) in zip(axes, groups.items()):
        y = np.arange(len(grp_nets))
        for i, m in enumerate(measures_of_interest):
            vals = [ratios.loc[n, m] if n in ratios.index and pd.notna(ratios.loc[n, m]) else 0
                    for n in grp_nets]
            offset = (i - (len(measures_of_interest) - 1) / 2) * bar_h
            bars = ax.barh(y + offset, vals, height=bar_h * 0.92, color=col(m),
                            edgecolor="white", linewidth=0.4)
            if ax is axes[0]:
                handles.append(bars[0])
                labels.append(lab(m))
        ax.axvline(1.0, color="#8a8985", lw=0.8, ls=":", zorder=0)
        ax.set_yticks(y, grp_nets, fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color="#0b0b0b", pad=6)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Seed-set spread, ratio to the best method (k = 100, IC, p = p_c)")
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.05, top=0.86, hspace=0.55)
    fig.suptitle("The regularization fix, all 14 real networks", fontsize=12, y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.58, 0.935), frameon=False)
    save(fig, figs, "fig9_capstone_regularization_fix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figs", default="figures")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)
    setup()
    ds = read(args.results, "datasets.csv")
    tau = read(args.results, "tau.csv")
    fig2(args.results, args.figs, ds)
    fig3(args.results, args.figs)
    if tau is not None:
        fig4(args.results, args.figs, tau)
        fig5(args.results, args.figs, tau)
        fig7_and_stats(args.results, args.figs, tau)
    fig6(args.results, args.figs)
    fig9_capstone(args.results, args.figs, args.data_dir)
    fig8(args.results, args.figs)


if __name__ == "__main__":
    main()
