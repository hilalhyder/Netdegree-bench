"""Pre-register predictions for the held-out networks before any diffusion is run.

Computes each network's two risk factors (degree-tail ratio = max / p99 total degree,
reciprocity) with the same code and thresholds as Figure 1, and writes the predicted
failure mode to heldout/preregistration.csv. Commit that file before running
run_benchmark.py on these networks; the git timestamp is the pre-registration.

    python heldout/preregister.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ndbench import datasets  # noqa: E402

HELDOUT = ["soc-sign-bitcoin-alpha", "cit-HepTh", "sx-mathoverflow", "sx-askubuntu",
           "sx-superuser", "soc-Slashdot0811", "ego-Gplus"]
TAIL_RATIO_THRESHOLD = 100     # frozen: same values as make_figures.py / Figure 1
RECIPROCITY_THRESHOLD = 0.75


def category(ratio, rec):
    if rec > RECIPROCITY_THRESHOLD:
        return "reciprocity ceiling"
    if ratio > TAIL_RATIO_THRESHOLD:
        return "degree-tail outlier"
    return "normal"


def main():
    rows = []
    for name in HELDOUT:
        g = datasets.load(name)
        u = g.out_degree().astype(float) + g.in_degree().astype(float)
        a = u > 0
        p99, umax = float(np.quantile(u[a], 0.99)), float(u[a].max())
        rec = float(g.reciprocity())
        src = float(np.mean((g.in_degree() == 0)[a]))
        cat = category(umax / p99, rec)
        rows.append(dict(net=name, n=g.n, m=g.m, reciprocity=round(rec, 4), p99_degree=p99, max_degree=umax,
                         tail_ratio=round(umax / p99, 2), frac_zero_in=round(src, 4), predicted=cat))
        print(rows[-1], flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preregistration.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
