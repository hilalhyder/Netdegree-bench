"""Compact directed-graph container and loaders.

Convention used everywhere in this package: an arc i -> j means that
information can flow FROM i TO j.  Datasets whose raw arcs point the other way
(e.g. "i follows j", "i cites j", "i trusts j") are loaded with ``flip=True``.
"""
from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class DiGraph:
    n: int
    src: np.ndarray            # int64, arc sources (deduplicated, no self-loops)
    dst: np.ndarray            # int64, arc targets
    w: np.ndarray              # float64, arc weights (multiplicity if unweighted file)
    labels: np.ndarray | None = None   # original node ids, labels[i] = raw id of node i
    name: str = "graph"
    meta: dict = field(default_factory=dict)

    # ---- derived structures (built lazily) ----
    _out: tuple | None = None
    _in: tuple | None = None

    @property
    def m(self) -> int:
        return int(self.src.size)

    def out_csr(self):
        """(indptr, indices, weights) for out-neighbours, sorted by source."""
        if self._out is None:
            self._out = _csr(self.n, self.src, self.dst, self.w)
        return self._out

    def in_csr(self):
        """(indptr, indices, weights) for in-neighbours, sorted by target."""
        if self._in is None:
            self._in = _csr(self.n, self.dst, self.src, self.w)
        return self._in

    def adjacency(self, weighted: bool = False) -> sp.csr_matrix:
        data = self.w if weighted else np.ones(self.m)
        return sp.csr_matrix((data, (self.src, self.dst)), shape=(self.n, self.n))

    def out_degree(self) -> np.ndarray:
        return np.bincount(self.src, minlength=self.n).astype(np.int64)

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.dst, minlength=self.n).astype(np.int64)

    def out_strength(self) -> np.ndarray:
        return np.bincount(self.src, weights=self.w, minlength=self.n)

    def in_strength(self) -> np.ndarray:
        return np.bincount(self.dst, weights=self.w, minlength=self.n)

    def reversed(self) -> "DiGraph":
        return DiGraph(self.n, self.dst.copy(), self.src.copy(), self.w.copy(),
                       self.labels, self.name + "_rev", dict(self.meta))

    def reciprocity(self) -> float:
        if self.m == 0:
            return 0.0
        key = self.src * self.n + self.dst
        rkey = self.dst * self.n + self.src
        key_sorted = np.sort(key)
        pos = np.searchsorted(key_sorted, rkey)
        pos[pos >= key_sorted.size] = 0
        return float(np.mean(key_sorted[pos] == rkey))

    def weak_components(self) -> np.ndarray:
        from scipy.sparse.csgraph import connected_components
        _, lab = connected_components(self.adjacency(), directed=True, connection="weak")
        return lab


def _csr(n, a, b, w):
    order = np.argsort(a, kind="stable")
    indices = b[order].astype(np.int64)
    weights = w[order].astype(np.float64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(a, minlength=n), out=indptr[1:])
    return indptr, indices, weights


def from_edges(src, dst, w=None, *, flip=False, n=None, labels=None, name="graph",
               keep_self_loops=False, sum_duplicates=True) -> DiGraph:
    """Build a DiGraph from integer arrays (already 0..n-1 indexed)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.ones(src.size) if w is None else np.asarray(w, dtype=np.float64)
    if flip:
        src, dst = dst, src
    if not keep_self_loops:
        mask = src != dst
        src, dst, w = src[mask], dst[mask], w[mask]
    if n is None:
        n = int(max(src.max(initial=-1), dst.max(initial=-1)) + 1)
    if sum_duplicates and src.size:
        key = src * n + dst
        uniq, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=w)
        src, dst = uniq // n, uniq % n
    return DiGraph(n, src, dst, w, labels, name)


def _open_text(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_edgelist(path, *, flip=False, weight_col=None, positive_only=False,
                  comment_chars="#%", delimiter=None, name=None, usecols=(0, 1)) -> DiGraph:
    """Read a whitespace/comma separated edge list (optionally .gz).

    Node ids may be arbitrary strings or integers; they are relabelled 0..n-1
    and the originals kept in ``labels``.  Duplicate arcs are merged and their
    weights summed (multiplicity when the file is unweighted).
    """
    cols = list(usecols) + ([weight_col] if weight_col is not None else [])
    df = _fast_read(path, cols, comment_chars, delimiter)
    a = df[0]
    b = df[1]
    w = df[2].astype(float) if weight_col is not None else None
    if positive_only and w is not None:
        keep = w > 0
        a, b, w = a[keep], b[keep], w[keep]
    labels, codes = np.unique(np.concatenate([a, b]), return_inverse=True)
    src = codes[: a.size]
    dst = codes[a.size:]
    g = from_edges(src, dst, None if w is None else np.asarray(w), flip=flip,
                   n=labels.size, labels=labels, name=name or os.path.basename(path))
    return g


def _fast_read(path, cols, comment_chars, delimiter):
    """Read selected columns; integer fast path, string fallback, pure-Python last resort."""
    import pandas as pd
    sep = delimiter if delimiter else r"\s+"
    for dtype in (np.int64, str):
        try:
            df = pd.read_csv(path, sep=sep, comment=comment_chars[0], header=None,
                             usecols=cols, dtype=dtype if dtype is str else None,
                             engine="c", compression="infer", skip_blank_lines=True)
            if dtype is np.int64:
                if not all(np.issubdtype(df[c].dtype, np.integer) for c in cols[:2]):
                    continue
            else:
                first = df[cols[0]].astype(str)
                mask = np.ones(len(df), dtype=bool)
                for c in comment_chars[1:]:
                    mask &= ~first.str.startswith(c).to_numpy()
                df = df[mask]
            return {i: df[c].to_numpy() for i, c in enumerate(cols)}
        except (ValueError, pd.errors.ParserError):
            continue
    rows = []
    with _open_text(path) as fh:
        for line in fh:
            if not line.strip() or line[0] in comment_chars:
                continue
            parts = line.replace(",", " ").split()
            rows.append([parts[c] for c in cols])
    arr = np.array(rows, dtype=object)
    return {i: arr[:, i].astype(str) for i in range(len(cols))}
