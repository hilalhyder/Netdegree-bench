"""Dataset registry: where to get each network and which way information flows.

``flip=True`` means the raw arc i -> j carries information from j to i (e.g.
"i follows j", "i cites j", "i trusts j"), so arcs are reversed on load.

``verify=True`` marks datasets whose arc meaning must be confirmed against the
dataset's own README before results are reported.  Do not skip this: the whole
study depends on arc direction.
"""
from __future__ import annotations

import gzip
import io
import os
import urllib.request

import numpy as np

from .graph import from_edges, read_edgelist

SNAP = "https://snap.stanford.edu/data/"

REGISTRY = {
    "congress-Twitter": dict(url=None, flip=False, verify=True, weight_col=None,
                              meaning="weighted interaction; ships as JSON - convert to an edge list first",
                              role="small, weighted"),
    "email-Eu-core": dict(url=SNAP + "email-Eu-core.txt.gz", flip=False, verify=False,
                          meaning="i emailed j", role="small; department labels"),
    "CollegeMsg": dict(url=SNAP + "CollegeMsg.txt.gz", flip=False, verify=False,
                       meaning="i messaged j (temporal; multi-arcs merged to weights)", role="small; temporal"),
    "soc-sign-bitcoin-otc": dict(url=SNAP + "soc-sign-bitcoinotc.csv.gz", flip=True, verify=False,
                                 weight_col=2, positive_only=True, delimiter=",",
                                 meaning="i rated j (trust score -10..10)", role="weighted; positive arcs only"),
    "wiki-Vote": dict(url=SNAP + "wiki-Vote.txt.gz", flip=False, verify=False,
                      meaning="i voted on j", role="control: endorsement, not diffusion"),
    "sx-mathoverflow": dict(url=SNAP + "sx-mathoverflow.txt.gz", flip=False, verify=True,
                            meaning="i answered/commented on j", role="temporal Q&A"),
    "cit-HepPh": dict(url=SNAP + "cit-HepPh.txt.gz", flip=True, verify=False,
                      meaning="paper i cites j", role="knowledge flow"),
    "soc-Epinions1": dict(url=SNAP + "soc-Epinions1.txt.gz", flip=True, verify=False,
                          meaning="i trusts j", role="trust"),
    "ego-Twitter": dict(url=SNAP + "twitter_combined.txt.gz", flip=True, verify=False,
                        meaning="i follows j -- confirmed 2026-09-26 (SNAP egonets-Twitter page: "
                                "'edges are directed (a follows b)'); flip=True matches this project's "
                                "own empirically-confirmed Higgs convention for the same 'follows' semantics",
                        role="follower network"),
    "soc-Slashdot0922": dict(url=SNAP + "soc-Slashdot0902.txt.gz", flip=True, verify=False,
                             # filename corrected 2026-09-26: registry originally had "0922", the real
                             # SNAP file is "0902" (confirmed via the dataset's own page) -- the "0922" in
                             # this key name is retained since it matches the original proposal's naming
                             meaning="i tagged j as friend/foe -- unsigned snapshot of the same Slashdot Zoo "
                                     "collection as soc-sign-Slashdot090221, whose raw direction (tagger -> "
                                     "tagged) was confirmed 2026-09-26 against Leskovec, Huttenlocher & "
                                     "Kleinberg (CHI 2010); flip=True is the same judgment call documented "
                                     "there (flow interpretation, not raw direction, is the unverifiable part)",
                             role="social tagging"),
    "email-EuAll": dict(url=SNAP + "email-EuAll.txt.gz", flip=False, verify=False,
                        meaning="i emailed j", role="mid-size communication"),
    "wiki-Talk": dict(url=SNAP + "wiki-Talk.txt.gz", flip=False, verify=False,
                      meaning="i wrote on j's talk page", role="scalability"),
    "soc-Pokec": dict(url=SNAP + "soc-pokec-relationships.txt.gz", flip=True, verify=False,
                      # BUG FOUND AND FIXED 2026-09-26: this was flip=False, inconsistent with every
                      # other "i declares a relationship toward j" dataset in this registry (i follows j,
                      # i trusts j, i tags j as friend -- all flip=True). Confirmed via SNAP's own source
                      # ("a directed edge from user A to user B represents that user A considers user B
                      # as a friend") that Pokec has the identical declarer -> declared-friend structure;
                      # flip=True matches this project's established convention for that structure.
                      meaning="i declares j a friend (i.e. i befriended j)", role="scalability"),
    "higgs-social": dict(url=SNAP + "higgs-social_network.edgelist.gz", flip=None, verify=True,
                         meaning="follow relation - direction decided by run_higgs.py's data check",
                         role="RQ5 structure"),
    "soc-sign-epinions": dict(url=SNAP + "soc-sign-epinions.txt.gz", flip=True, verify=False,
                              weight_col=2, positive_only=True,
                              meaning="i trusts j (signed: +1/-1, verified via raw-file peek 2026-09-26; "
                                      "same convention as soc-sign-bitcoin-otc)",
                              role="signed trust; other domain/scale, follow-up on the bitcoin-otc anomaly"),
    "soc-sign-Slashdot090221": dict(url=SNAP + "soc-sign-Slashdot090221.txt.gz", flip=True, verify=True,
                                    weight_col=2, positive_only=True,
                                    meaning="FromNodeId tags ToNodeId friend(+1)/foe(-1) -- raw direction confirmed "
                                            "2026-09-26 against the dataset's own source (Leskovec, Huttenlocher "
                                            "& Kleinberg, 'Signed Networks in Social Media', CHI 2010: 'edge "
                                            "direction ... annotates that the origin user tagged the target'). "
                                            "flip=True (treating the tagged/friended node as the information "
                                            "source) is NOT independently verified the way Higgs's direction was "
                                            "(no timestamp-style ground truth exists for friend/foe tags) -- it "
                                            "is a judgment call by analogy to this project's own soc-sign-epinions/"
                                            "bitcoin-otc 'i trusts j' convention. verify=True kept for this reason: "
                                            "the RAW direction is settled, the FLOW INTERPRETATION is not.",
                                    role="signed but UNweighted (ratings are +-1 only) - isolates whether the "
                                         "bitcoin-otc anomaly is about signedness alone or weighted ratings"),
}

# positive_only=True above is NOT a neutral default -- it measurably lowers
# netdegree-trophic agreement on all three of these datasets (and on
# bitcoin-otc) versus keeping every signed edge. Kept as the default to
# match the original bitcoin-otc precedent, but report BOTH filtered and
# unfiltered numbers in the paper. See proofs/positive_only_filtering_effect.md.


def load_wiki_rfa(data_dir="data"):
    """Default load: positive-only (see load_wiki_rfa_variant for the toggle,
    and proofs/positive_only_filtering_effect.md for why this default is not
    a neutral choice)."""
    return load_wiki_rfa_variant(data_dir, positive_only=True)


def load_wiki_rfa_variant(data_dir="data", positive_only=True):
    """wiki-RfA does not use REGISTRY/read_edgelist: its raw file is a multi-line
    SRC:/TGT:/VOT:/... record block per vote (verified via raw-file peek
    2026-09-26), not a plain edge list. VOT is signed (+1 support, -1 oppose,
    0 neutral). i supports/opposes j -> flip=True, same convention as the
    other signed/endorsement datasets. ``positive_only=False`` keeps oppose
    votes too (sign otherwise ignored, presence-only) -- see
    proofs/positive_only_filtering_effect.md for why this matters."""
    path = download(SNAP + "wiki-RfA.txt.gz", data_dir)
    with io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    src, dst = [], []
    for block in text.split("\n\n"):
        s = d = v = None
        for line in block.splitlines():
            if line.startswith("SRC:"):
                s = line[4:]
            elif line.startswith("TGT:"):
                d = line[4:]
            elif line.startswith("VOT:"):
                v = line[4:].strip()
        if not s or not d or v is None:
            # blank SRC (anonymous/logged-out voter, 1,661 such records as of the
            # 2026-09-26 file -- verified via raw-file peek) is not a real,
            # identifiable node; merging all of them into one shared "" node
            # would fabricate a single node with a spurious in-degree (real
            # candidates who received an anonymous vote would get a spurious
            # extra out-edge to it). Drop rather than merge.
            continue
        if positive_only and v != "1":
            continue
        if not positive_only and v == "0":  # still drop neutral; keep support+oppose
            continue
        src.append(s)
        dst.append(d)
    labels, codes = np.unique(np.concatenate([np.array(src, dtype=object), np.array(dst, dtype=object)]),
                              return_inverse=True)
    n_edges = len(src)
    s_idx = codes[:n_edges]
    d_idx = codes[n_edges:]
    name = "wiki-RfA" if positive_only else "wiki-RfA_unfiltered"
    g = from_edges(s_idx, d_idx, flip=True, n=labels.size, labels=labels, name=name)
    g.meta.update(dict(flip=True, verify=False,
                       meaning="i supports j (RfA vote)" if positive_only else "i supports/opposes j (RfA vote, sign ignored)",
                       role="signed voting; direct signed counterpart to wiki-Vote, already run unsigned"))
    return g

HIGGS_ACTIVITY_URL = SNAP + "higgs-activity_time.txt.gz"


def download(url, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"  downloading {url}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def load(name, data_dir="data", path=None, flip=None):
    """Load a registered dataset (downloads on first use unless ``path`` is given)."""
    spec = REGISTRY[name]
    if path is None:
        if spec["url"] is None:
            raise ValueError(f"{name}: no download URL; pass path= to a local edge list")
        path = download(spec["url"], data_dir)
    f = spec["flip"] if flip is None else flip
    if f is None:
        raise ValueError(f"{name}: direction must be decided explicitly (flip=True/False)")
    g = read_edgelist(path, flip=f, weight_col=spec.get("weight_col"),
                      positive_only=spec.get("positive_only", False),
                      delimiter=spec.get("delimiter"), name=name)
    g.meta.update(dict(flip=f, meaning=spec["meaning"], verify=spec["verify"], role=spec["role"]))
    return g
