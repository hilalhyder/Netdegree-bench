# RQ1: Netdegree as the first Jacobi iterate of trophic level

This is the written proof behind `ndbench.measures.jacobi_trophic` and the
claim in the proposal's Summary and RQ1 sections. Notation matches
`ndbench/measures.py` and MacKay, Johnson & Sansom (2020), "How directed is
a directed network?", *R. Soc. Open Sci.* 7:201138 (their eq. [5]-[6],
verified directly against the paper).

## 1. Setup

Let $G=(N,E,w)$ be a directed, weighted network with no self-loops, arcs
oriented so that $i\to j$ means information can flow from $i$ to $j$. Write
$W$ for the weighted adjacency matrix ($W_{ij}=w_{ij}$ if $i\to j$, else 0)
and $S = W+W^\top$ (so $S_{ij}=w_{ij}+w_{ji}$, symmetric, zero diagonal).
Define, per node $i$,

$$
s_{\text{out}}(i)=\sum_j w_{ij},\qquad s_{\text{in}}(i)=\sum_j w_{ji},\qquad
u_i = s_{\text{out}}(i)+s_{\text{in}}(i),\qquad v_i = s_{\text{in}}(i)-s_{\text{out}}(i).
$$

Note $u_i$ is exactly the $i$-th row sum of $S$, and $v_i = -\,\text{nd}^w_i$
where $\text{nd}^w_i = s_{\text{out}}(i)-s_{\text{in}}(i)$ is *weighted
netdegree*. Weighted normalized netdegree is $\rho_i = \text{nd}^w_i/u_i$
(with $\rho_i:=0$ where $u_i=0$); in the unweighted case this reduces to the
usual $\rho_i=(k^{\text{out}}_i-k^{\text{in}}_i)/(k^{\text{out}}_i+k^{\text{in}}_i)$.

**Trophic level** (MacKay et al. 2020) is any solution $h$ of

$$
\Lambda h = v, \qquad \Lambda = \operatorname{diag}(u) - S. \tag{1}
$$

$\Lambda$ is singular — it is exactly the (weighted, generalized) graph
Laplacian of the undirected graph with edge weights $S_{ij}$, so on each
weakly connected component $\Lambda\mathbf 1=0$ and (1) determines $h$ only
up to an additive constant per component, as MacKay et al. note explicitly.
The source score reported by the benchmark is $-h$ (low level = upstream).

## 2. The system is always consistent (needed for §4)

**Lemma 1.** $\sum_i v_i = 0$ on every weakly connected component.

*Proof.* Every edge $(m,n)\in E$ contributes $w_{mn}$ to $s_{\text{out}}(m)$
and the same $w_{mn}$ to $s_{\text{in}}(n)$. Summing $v_i=s_{\text{in}}(i)-s_{\text{out}}(i)$
over all $i$ in a component, every edge weight is added once (as part of
some node's $s_{\text{in}}$) and subtracted once (as part of some node's
$s_{\text{out}}$), so the total is zero. $\blacksquare$

This is just "total in-flow = total out-flow" and is why (1) has a solution
at all despite $\Lambda$ being singular: $v$ is automatically orthogonal to
$\Lambda$'s left null space (the constant vector) on each component.

## 3. Proposition (RQ1 core claim)

**Proposition 1.** Solve (1) by Jacobi iteration — split $\Lambda=D-R$ with
$D=\operatorname{diag}(u)$, $R=S$, and iterate
$h^{(t+1)} = D^{-1}\big(v + R\,h^{(t)}\big) = D^{-1}v + D^{-1}S\,h^{(t)}$ —
starting from $h^{(0)}=0$. Then

$$
h^{(1)} = D^{-1}v, \qquad -h^{(1)}_i = -v_i/u_i = \rho_i \quad\text{(weighted normalized netdegree).}
$$

*Proof.* Immediate substitution: $h^{(1)} = D^{-1}v + D^{-1}S\cdot 0 = D^{-1}v$,
so $h^{(1)}_i = v_i/u_i$. Since $v_i=-\text{nd}^w_i$ and $u_i$ is the total
weighted degree, $-h^{(1)}_i = \text{nd}^w_i/u_i = \rho_i$. $\blacksquare$

This matches `jacobi_trophic(g, 1)` exactly (`b = (s_in - s_out) * inv`,
returned negated), and is checked numerically in
`tests/test_core.py::test_jacobi_first_iterate_is_normalized_netdegree`.

As stated, this proof is two lines — it is an immediate instance of "the
first Jacobi iterate from zero is $D^{-1}b$" applied to a linear system that
was already published. **The genuinely new step is §4: showing the
iteration is even guaranteed to converge**, which is not automatic for a
singular system.

## 4. Convergence is not free — and follows from Lemma 1

Jacobi iteration for $\Lambda h = v$ converges from an arbitrary start iff
the spectral radius of $D^{-1}S$ is less than 1. Here it is **not**: $D^{-1}S$
is the transition matrix of the (weighted) random walk on the symmetrized
graph, so it is row-stochastic and has $\mathbf 1$ as an eigenvector with
eigenvalue exactly $1$. A naive claim that "Jacobi converges to the trophic
level" would therefore be false in general — this needs an argument, not an
assertion.

**Proposition 2.** Starting from $h^{(0)}=0$, the Jacobi iterates $h^{(t)}$
stay bounded and, if $1$ is a simple eigenvalue of $D^{-1}S$ on the
component (true for any connected, non-bipartite weighting, the generic
case), converge to the unique solution of (1) satisfying $\mathbf u^\top h=0$.

*Proof sketch.* $D^{-1}S$ is similar to the symmetric matrix
$P=D^{-1/2}SD^{-1/2}$ (via $D^{1/2}$), so it is diagonalizable with real
eigenvalues $1=\lambda_1>\lambda_2\ge\dots\ge\lambda_n\ge -1$ and an
orthonormal eigenbasis $\{\phi_k\}$ for $P$, giving $\Lambda$-eigenbasis
$\{D^{-1/2}\phi_k\}$ with $\phi_1\propto D^{1/2}\mathbf 1$. Decompose
$h^{(t)}=\sum_k c_k(t)\,D^{-1/2}\phi_k$. The Jacobi recursion becomes
$c_k(t+1)=\lambda_k c_k(t) + \tilde v_k$ for each mode, where $\tilde v_k$
is $v$'s coefficient in this basis. By Lemma 1, $\mathbf 1^\top v = 0$ on
the component, which is exactly $\tilde v_1=0$ (the mode-1 forcing term
vanishes). Since $c_1(0)=0$ (from $h^{(0)}=0$) and $\tilde v_1=0$, the
recursion $c_1(t+1)=1\cdot c_1(t)+0$ gives $c_1(t)=0$ for every $t$ — the
one non-contracting mode is never excited. Every other mode has
$|\lambda_k|<1$ (strict, for $k\ge2$, generically), so $c_k(t)\to
\tilde v_k/(1-\lambda_k)$, the corresponding coefficient of the true
solution. Hence $h^{(t)}\to h^\star$, the solution of (1) orthogonal to
$\mathbf 1$ in the $D$-inner product. $\blacksquare$

(The $\lambda_k=-1$ edge case, exact bipartiteness of the symmetrized graph
restricted to a component, causes that one mode to oscillate rather than
converge rather than diverge — still bounded, and negligible for the real
and synthetic networks used here; `test_jacobi_converges_toward_trophic`
checks rank-agreement rather than raw convergence for exactly this reason.)

**Rate, checked empirically.** Convergence speed is governed by the
symmetrized graph's spectral gap ($1-|\lambda_2|$), not just its sign. A
chain (whose symmetrization is a path graph — nearly the worst case for
mixing, and exactly bipartite besides) needs on the order of $10^4$-$10^5$
Jacobi steps to reach machine precision on 60 nodes; a denser, better-mixing
graph of the kind used in the RQ2 synthetic sweep reaches it by $t\approx
200$. This matters directly for Fig. 3: real networks with long, chain-like
propagation paths (citation chains, reply trees) will show slower
local-to-global convergence than densely-connected ones at the same node
count, independent of trophic incoherence $F_0$ — worth checking as a
second explanatory variable if Fig. 3's curve is noisier than $F_0$ alone
predicts.

## 5. Corollary: local-to-global is a measured continuum

Combining §3 and §4: $t=1$ recovers netdegree exactly, and $t\to\infty$
recovers the trophic level exactly (up to the harmless gauge freedom fixed
by $\mathbf 1^\top h=0$, which does not affect rank). Every intermediate
$t$ is a well-defined, computable point on the same path — each step adds
exactly one more hop of neighbor-averaging to the previous iterate. This is
what `run_agreement_sweep.py` reports at $t=1,2,5,10$ (RQ1/RQ2, Fig. 3):
not a binary local-vs-global choice, but a curve.

## 6. Remark: SpringRank and HodgeRank

SpringRank (De Bacco, Larremore & Moore 2018) minimizes
$E(\text{rank})=\tfrac12\sum_{ij} w_{ij}(\text{rank}_j-\text{rank}_i-1)^2 + \tfrac{\alpha}{2}\sum_i \text{rank}_i^2$.
Setting $\nabla E=0$ gives $(\operatorname{diag}(u)+\alpha I - S)\,\text{rank} = v$
(sign of $v$ depends on the edge-direction convention used for "beats");
at $\alpha=0$ this is exactly (1). The regularization $\alpha I$ exists only
for numerical stability on disconnected/tree-like graphs and vanishes in
the limit — so SpringRank and trophic level are the same object with a
different solver default, not two different theories. HodgeRank (Jiang,
Lim, Yao & Ye 2011) is the gradient (divergence-free-orthogonal) component
of the edge flow in the Hodge decomposition; for a flow built from
node potentials this component *is* the potential itself, which satisfies
the same normal equations. All three, and netdegree, are different
vantage points on one linear system.

## 7. What this proof does and doesn't claim

- **Does claim:** the $t=1$ identity (§3) and that Jacobi converges here at
  all (§4) are both correct and, as far as a direct check of MacKay et al.
  (2020) and De Bacco et al. (2018) could confirm, not previously stated in
  print (see the project's earlier literature check).
- **Does not claim:** that this required new numerical machinery. §3 is a
  restatement of Jacobi's method. §4 is a standard spectral argument for
  graph Laplacians (essentially the same argument used to show simple
  random walks on connected graphs converge to stationarity), applied here
  to a not-previously-connected setting.
- **Numerically verified:** `tests/test_core.py` — `test_jacobi_first_iterate_is_normalized_netdegree`
  (§3), `test_trophic_identity_holds_at_solution` (the fixed-point identity
  underlying §4), `test_jacobi_converges_toward_trophic` (rank-agreement
  improves with $t$), and `test_jacobi_raw_iterates_converge_to_trophic_level`
  (§4's Proposition 2 directly: raw iterates, not just their rank order,
  converge to $h^\star$ to $<10^{-6}$ relative error). 23/23 passing as of
  this writeup. The last test went through one real debugging cycle worth
  recording: it initially failed on a chain graph, which turned out to be
  the $\lambda=-1$ edge case above; retrying on a chain-plus-chords graph
  still failed, which turned out to be the separate slow-mixing issue
  described just above (confirmed by running to $t=50{,}000$, where the
  residual does reach $4\times10^{-14}$). Both are genuine properties of
  the underlying math, not test bugs, and are now documented rather than
  hidden by a more forgiving graph choice.
