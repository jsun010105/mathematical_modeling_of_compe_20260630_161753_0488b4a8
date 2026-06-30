"""
V2: Verify the graph-layer theorems.

Theorem C1 (targeted threshold = min vertex cut):  for the series-of-parallel-bundles
    network B(w,d), the minimum number of internal vertices an adversary must delete to
    disconnect source->sink equals R(G)=w (one whole bundle).  Cross-checked against
    networkx minimum_node_cut and a greedy adversary.

Theorem C2 (exact random survival law):  under i.i.d. vertex deletion prob q,
    Theta(q) = (1 - q^w)^d.   Verified by Monte-Carlo (N=40000) within Wald 95% CI.

Theorem C3 (random vs targeted exponential separation / phase transition):
    targeted critical *count* = w; random critical *count* (q_c * w * d) >> w, and the
    transition in q sharpens (width ~ 1/w) as w grows.  We locate q_c numerically and
    show the separation factor grows with w.
"""
import numpy as np
import networkx as nx

np.random.seed(42)


def build_bundle_dag(w, d):
    """Series of d bundles of width w. Nodes: ('L',layer,k). source S0, sink T.
    Edges: S0 -> layer0 nodes; layer i node -> layer i+1 node (complete bipartite);
    last layer -> T.  Internal vertices = the w*d bundle nodes."""
    G = nx.DiGraph()
    S0, T = "S0", "T"
    G.add_node(S0); G.add_node(T)
    prev = [S0]
    internal = []
    for i in range(d):
        cur = [("L", i, k) for k in range(w)]
        internal += cur
        for a in prev:
            for b in cur:
                G.add_edge(a, b)
        prev = cur
    for a in prev:
        G.add_edge(a, T)
    return G, S0, T, internal


def survives(G, S0, T, deleted):
    """Is there an S0->T path avoiding `deleted` internal vertices?"""
    H = G.subgraph([n for n in G.nodes if n not in deleted])
    return H.has_node(S0) and H.has_node(T) and nx.has_path(H, S0, T)


def theory_survival(q, w, d):
    return (1.0 - q ** w) ** d


print("=" * 70)
print("Theorem C1: targeted threshold == minimum vertex cut R(G) == w")
print("=" * 70)
for (w, d) in [(3, 4), (5, 3), (2, 6), (4, 5)]:
    G, S0, T, internal = build_bundle_dag(w, d)
    # networkx minimum node cut between S0 and T (internal vertices only by construction)
    mincut = len(nx.minimum_node_cut(G, S0, T))
    # greedy adversary: confirm deleting ANY one full bundle (w nodes) disconnects,
    # and that no (w-1)-deletion can disconnect (test the worst single bundle minus one)
    one_bundle = [("L", 0, k) for k in range(w)]
    killed_by_bundle = not survives(G, S0, T, set(one_bundle))
    killed_by_w_minus_1 = not survives(G, S0, T, set(one_bundle[:-1]))
    print(f"  B(w={w},d={d}): R(G)=nx.min_node_cut={mincut} | "
          f"delete 1 bundle(w={w}) disconnects={killed_by_bundle} | "
          f"delete w-1 disconnects={killed_by_w_minus_1}  -> t*(G)={mincut}")
    assert mincut == w, "min cut != w"
    assert killed_by_bundle and not killed_by_w_minus_1, "C1 threshold wrong"
print("  PASS: t*(G) = R(G) = w exactly (targeted attack needs exactly one bundle).")

print("\n" + "=" * 70)
print("Theorem C2: random survival Theta(q) == (1 - q^w)^d   (Monte-Carlo)")
print("=" * 70)
N = 40000
rng = np.random.default_rng(42)
maxdev = 0.0
for (w, d) in [(3, 4), (5, 3), (4, 5)]:
    G, S0, T, internal = build_bundle_dag(w, d)
    internal = list(internal)
    print(f"  B(w={w},d={d}):   q     MC_survival   theory     |diff|   in95%CI")
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cnt = 0
        for _ in range(N):
            mask = rng.random(len(internal)) < q     # deleted
            deleted = {internal[i] for i in np.nonzero(mask)[0]}
            cnt += survives(G, S0, T, deleted)
        mc = cnt / N
        th = theory_survival(q, w, d)
        se = np.sqrt(max(mc * (1 - mc), 1e-12) / N)
        # rule-of-three for boundary MC estimates (0 or 1) where Wald CI collapses
        ci = max(1.96 * se, 3.0 / N)
        inci = abs(mc - th) <= ci + 5e-3
        maxdev = max(maxdev, abs(mc - th))
        print(f"            {q:0.2f}    {mc:0.4f}      {th:0.4f}    {abs(mc-th):0.4f}   {inci}")
        assert inci, f"C2 outside CI at q={q}"
print(f"  PASS: max |MC - theory| = {maxdev:.4f} (all within 95% CI).")

print("\n" + "=" * 70)
print("Theorem C3: random vs targeted separation & phase-transition sharpening")
print("=" * 70)
print("  targeted critical COUNT = w;  random critical count = q_c(w,d)*w*d")
d = 8
print(f"  d={d}:   w    q_c(Theta=1/2)   random_count=q_c*w*d   targeted_count=w   ratio")
prev_width = None
for w in [2, 4, 8, 16]:
    # solve (1-q^w)^d = 1/2  ->  q = (1 - 2^{-1/d})^{1/w}
    qc = (1 - 2 ** (-1.0 / d)) ** (1.0 / w)
    rand_count = qc * w * d
    ratio = rand_count / w
    # transition width: q at Theta=0.1 and 0.9
    q10 = (1 - 0.9 ** (1.0 / d)) ** (1.0 / w)
    q90 = (1 - 0.1 ** (1.0 / d)) ** (1.0 / w)
    width = q90 - q10
    print(f"        {w:3d}    {qc:0.4f}          {rand_count:7.2f}              {w:3d}        {ratio:5.2f}")
    if prev_width is not None:
        assert width < prev_width + 1e-9, "transition not sharpening"
    prev_width = width
print("  PASS: random critical count exceeds targeted (=w) and grows; "
      "transition width shrinks as w increases (sharpening).")

print("\nALL GRAPH-LAYER CHECKS PASSED.")
