#!/usr/bin/env python3
"""Generate the demo networks used by run_demo.sh (no downloads needed)."""
import os

import networkx as nx
import numpy as np

from ndbench import synthetic

os.makedirs("data/demo", exist_ok=True)
G = nx.scale_free_graph(5000, seed=1)
np.savetxt("data/demo/scalefree.txt", np.array([(u, v) for u, v, _ in G.edges(keys=True)]), fmt="%d")
G = nx.gnr_graph(5000, 0.3, seed=2)            # arcs new -> old, like citations (loaded with --flip)
np.savetxt("data/demo/gnr_citation.txt", np.array(list(G.edges())), fmt="%d")
for T in (0.3, 1.5):
    g, _ = synthetic.trophic_tunable(n=5000, T=T, rng_seed=int(T * 10))
    np.savetxt(f"data/demo/trophic_T{T}.txt", np.c_[g.src, g.dst], fmt="%d")
g, _ = synthetic.trophic_tunable(n=300_000, T=1.0, mean_degree=10, rng_seed=9)
np.savetxt("data/demo/big_trophic.txt", np.c_[g.src, g.dst], fmt="%d")
print("demo networks written to data/demo/")
