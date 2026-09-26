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
STYLE = {
    "netdegree":       ("#2a78d6", "o", "Netdegree"),
    "trophic":         ("#eb6834", "s", "Trophic source (−h)"),
    "netdegree_2hop":  ("#1baf7a", "^", "2-hop netdegree"),
    "out_degree":      ("#eda100", "D", "Out-degree"),
    "pagerank_rev":    ("#e87ba4", "v", "PageRank (reversed)"),
    "leaderrank":      ("#008300", "P", "LeaderRank"),
    "voterank":        ("#4a3aa7", "X", "VoteRank"),
    "degree_discount": ("#e34948", "*", "DegreeDiscount"),
    "celf":            ("#0b0b0b", "h", "CELF (reference)"),
}
GRAY = "#8a8985"
LABEL = {"netdegree_norm": "Normalized netdegree", "netdegree_weighted": "Weighted netdegree",
         "netdegree_2hop_reach": "2-hop netdegree (reach)", "in_degree": "In-degree", "kshell": "k-shell",
         "betweenness": "Betweenness", "random": "Random", "jacobi_1": "Jacobi t=1", "jacobi_2": "Jacobi t=2",
         "jacobi_5": "Jacobi t=5", "jacobi_10": "Jacobi t=10"}


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
    files = sorted(glob.glob(os.path.join(results, "measures_*.csv")))
    if not files:
        return print("  skip fig2: no measures_*.csv")
    files = files[:4]
    fig, axes = plt.subplots(1, len(files), figsize=(3.0 * len(files), 2.8), squeeze=False)
    for ax, f in zip(axes[0], files):
        net = os.path.basename(f)[9:-4]
        d = pd.read_csv(f)
        ax.hexbin(d["netdegree_norm"], d["trophic_level"], gridsize=40, bins="log", cmap="Blues", mincnt=1,
                  linewidths=0)
        F0 = ds.loc[ds.net == net, "F0"].iloc[-1] if ds is not None and (ds.net == net).any() else np.nan
        r = evaluate.spearman(d["netdegree_norm"], -d["trophic_level"])
        ax.set_title(f"{net}\nF0 = {F0:.2f}, Spearman = {r:.2f}")
        ax.set_xlabel("Normalized netdegree ρ")
        ax.grid(False)
    axes[0][0].set_ylabel("Trophic level h (0 = source)")
    save(fig, figs, "fig2_netdegree_vs_trophic")


def fig3(results, figs):
    ag = read(results, "agreement.csv")
    if ag is None:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    syn = ag[(ag.source == "synthetic sweep") & (ag.measure == "netdegree_norm")]
    if len(syn):
        gsyn = syn.groupby("T").agg(F0=("F0", "mean"), r=("spearman_vs_trophic", "mean"),
                                    sd=("spearman_vs_trophic", "std")).sort_values("F0")
        a1.plot(gsyn.F0, gsyn.r, color=GRAY, lw=1.6, label="Synthetic sweep (mean)")
        a1.fill_between(gsyn.F0, gsyn.r - gsyn.sd.fillna(0), gsyn.r + gsyn.sd.fillna(0), color=GRAY, alpha=0.2, lw=0)
    real = ag[(ag.source == "real") & (ag.measure == "netdegree_norm")].drop_duplicates("net", keep="last")
    a1.scatter(real.F0, real.spearman_vs_trophic, color=STYLE["netdegree"][0], s=36, zorder=3,
               edgecolor="white", linewidth=1.5, label="Real networks")
    for _, r in real.iterrows():
        a1.annotate(r.net, (r.F0, r.spearman_vs_trophic), fontsize=7, xytext=(4, 2), textcoords="offset points",
                    color="#52514e")
    a1.set_xlabel("Trophic incoherence F0")
    a1.set_ylabel("Spearman(ρ, −h)")
    a1.set_title("Local–global agreement vs coherence")
    a1.legend(loc="lower right", fontsize=7)
    src = ag[ag.source != "synthetic sweep"]
    jac = src[src.measure.str.startswith("jacobi_")].copy()
    if len(jac):
        jac["t"] = jac.measure.str.split("_").str[1].astype(int)
        net_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for i, (net, grp) in enumerate(jac.groupby("net", sort=False)):
            grp = grp.drop_duplicates("t", keep="last").sort_values("t")
            c = net_colors[i] if i < len(net_colors) else GRAY
            a2.plot(grp.t, grp.spearman_vs_trophic, marker=markers[i % 8], ms=4, color=c, label=net)
        a2.legend(fontsize=7, loc="lower right")
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
    nets = list(dict.fromkeys(sp.net))[:4]
    keep = ["netdegree", "trophic", "netdegree_2hop", "out_degree", "pagerank_rev", "leaderrank",
            "voterank", "degree_discount", "celf"]
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
    for m in ["netdegree", "trophic", "out_degree", "pagerank_rev"]:
        dm = d[d.measure == m].sort_values("decile")
        if len(dm):
            a1.plot(dm.decile, dm.mean_y, marker=mk(m), color=col(m), label=lab(m), ms=4)
    a1.set_yscale("symlog", linthresh=0.1)
    a1.set_xlabel("Score decile (10 = highest)")
    a1.set_ylabel("Mean distinct retweeters")
    a1.set_title("Higgs: retweeters by score decile")
    a1.legend(fontsize=7)
    if cur is not None:
        a2 = axes[0][1]
        for m in ["netdegree", "trophic", "out_degree", "pagerank_rev", "netdegree_2hop"]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figs", default="figures")
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
    fig8(args.results, args.figs)


if __name__ == "__main__":
    main()
