# Theorems and Proofs — Compensatory Robustness of LLM Alignment

Notation and objects are those of `definitions.md`. Three layers: **Information**
(Thm A1–A2, B1, Prop D1), **Graph/percolation** (Thm C1–C3, Cor E1). Every nontrivial
claim is cross-checked computationally (`src/verify_information_layer.py`,
`src/verify_graph_layer.py`); see `results/v1_v3_information_layer.txt`,
`results/v2_graph_layer.txt`.

---

## Layer I — Information-theoretic ablation bounds

### Theorem A1 (Ablation cost = non-redundant information).
For any joint law of `(S, R_1,…,R_n)` and any ablated subset `T`,
$$\Delta I_T \;=\; I(S;R_1,\dots,R_n) - I(S;R_{T^c}) \;=\; I(S;\,R_T \mid R_{T^c}).$$
In particular `ΔI_i = I(S; R_i | R_{-i}) ≥ 0`, with **`ΔI_i = 0` iff `S ⊥ R_i | R_{-i}`**.

*Proof.* By the chain rule for mutual information, splitting the full pathway tuple
`R = (R_{T^c}, R_T)`,
$$I(S;R) = I(S;R_{T^c}) + I(S;R_T\mid R_{T^c}).$$
Subtract `I(S;R_{T^c})`. Non-negativity of conditional mutual information gives
`ΔI_T ≥ 0`; `I(S;R_T|R_{T^c})=0` is by definition equivalent to `S ⊥ R_T | R_{T^c}`. ∎

**Remark (vs McGrath `CE`).** Unlike the logit-level compensatory-effect `CE`
(Lit. Def. 7), which Rushing–Nanda show is ~30% LayerNorm-scaling artifact (R3), `ΔI`
is an exact information identity with no scaling nuisance: it isolates *genuine*
non-redundant content. The redundant part `R_i` shares with `R_{-i}` cancels exactly.

*Verification.* `verify_information_layer.py`: identity error `0.00e+00` on all four
canonical laws (redundant, unique, XOR-synergistic, partial). PID cross-check (`dit`):
redundant case `Rdn=1.000, ΔI_i=0`; unique case `Unq_R1=1.000, ΔI_1=1.000`; XOR
`Syn=1.000, ΔI_i=1.000`; partial `Rdn=0.531, Unq=0.469, ΔI_1=0.469` — i.e. the ablation
loss equals `Unq_i + Syn` and the redundant atom is free, consistent with A1.

### Theorem A2 (k-redundancy ⇒ resilience to any k−1 ablations).
If `S` is k-redundant w.r.t. `R` (Def. 3), then `ΔI_A = 0` for **every** `A` with
`|A| ≤ k−1`; equivalently the optimal behavior is recoverable after any `k−1` ablations.

*Proof.* Let `|A| ≤ k−1` and `T = A^c`. Then `|T| = n − |A| ≥ n − (k−1) = n−k+1`, so by
k-redundancy `I(S;R_T) = I(S;R)`, hence `ΔI_A = I(S;R) − I(S;R_{A^c}) = 0`. By Thm A1
this is `I(S;R_A | R_{A^c}) = 0`. ∎

**Remark (combinatorial design view).** k-redundancy is a *covering* condition: every
`(n−k+1)`-subset of pathways must already determine `S`. This is exactly an
`(n,k)`-style redundancy code on pathways, and gives the discrete analogue of refusal
**concept cones** (Wollschläger R7) and **backup heads** (Wang R1): multiple subsets
each suffice.

### Theorem B1 (Behavioral-fidelity floor — Fano).
Let `Ŝ = g(R_{T^c})` be **any** decoder of `S` from the surviving pathways, with error
`P_e = Pr[Ŝ ≠ S]` and `|S|` the support size. Then
$$H_2(P_e) + P_e\,\log_2(|S|-1)\;\ge\;H(S\mid R_{T^c}) \;=\; H(S) - I(S;R) + \Delta I_T.$$
Consequently, if alignment was fully recoverable before ablation (`I(S;R)=H(S)`),
$$\boxed{\,H_2(P_e) + P_e\log_2(|S|-1)\;\ge\;\Delta I_T\,}.$$
Thus ablating pathways that carry conditional information `ΔI_T>0` forces a strictly
positive behavioral error that grows with `ΔI_T`; and `ΔI_T = 0` (redundant ablation)
admits a zero-error decoder, so behavior is fully preserved.

*Proof.* `S → R_{T^c} → Ŝ` is a Markov chain, so Fano's inequality gives
`H_2(P_e) + P_e \log_2(|S|−1) ≥ H(S | R_{T^c})`. Then
`H(S|R_{T^c}) = H(S) − I(S;R_{T^c}) = H(S) − I(S;R) + ΔI_T` by Thm A1. If `I(S;R)=H(S)`
the right side is `ΔI_T`. For the last clause, `ΔI_T=0` ⇒ `H(S|R_{T^c})=0` ⇒ `S` is an
a.s.-deterministic function of `R_{T^c}`, so the MAP decoder achieves `P_e=0`. ∎

*Verification.* In all four cases the floor holds with equality at the boundary cases:
unique-ablation forces `P_e*=0.5` with floor `1.000`; partial gives `P_e*=0.10`,
`H(S|R₂)=0.469` ≤ floor `0.469`; redundant gives floor `0` and `P_e*=0` (behavior
preserved). No violations.

### Proposition D1 (Observed compensation ⇒ *pre-existing* latent redundancy).
Suppose `S` is determined by the full network (`H(S|R)=0`) and, after ablating `T`, the
behavior is fully preserved (the optimal post-ablation decoder has `P_e=0`). Then
`I(S;R_{T^c}) = H(S)`: the surviving pathways **already contained all information about
`S` before the ablation**. Hence "dynamic rerouting" *reveals* latent redundancy; it
does not create robustness.

*Proof.* `P_e=0` for the optimal decoder is equivalent to `H(S|R_{T^c})=0`, i.e. `S` is
an a.s.-function of `R_{T^c}`; therefore `I(S;R_{T^c}) = H(S) − H(S|R_{T^c}) = H(S)`. The
quantity `I(S;R_{T^c})` is a property of the *original* joint law (the surviving
pathways' readouts are unchanged by deleting the others), so the redundancy is
pre-existing. ∎

**Remark.** This formalizes the empirical reading of self-repair/backup behavior
(Wang R1; McGrath R2; Joad R8) and the Iterative-Inference Hypothesis (Rushing–Nanda):
the task signal persists in the residual stream and a downstream pathway completes it
*because the information was already redundantly present* — consistent with Arditi's
finding that the refusal direction is present in base models ("repurposed, not learned").

---

## Layer II — Graph / percolation: targeted vs random thresholds

### Theorem C1 (Targeted critical threshold = pathway redundancy `R(G)`).
For a computation DAG `G` with source `σ`, sink `τ`, the minimum number of internal
vertices whose deletion makes `S` non-computable equals `R(G)=κ_{στ}(G)` (Def. 8), and
`S` remains computable after deleting **any** set of fewer than `R(G)` internal
vertices. For `B(w,d)`, `R(B(w,d)) = w`.

*Proof.* By Menger's theorem (vertex form), the minimum `σ`–`τ` vertex cut size equals
the maximum number of internally vertex-disjoint `σ`–`τ` paths, both `= R(G)`. If
`|A| < R(G)`, then `A` is not a `σ`–`τ` separator (every separator has size `≥ R(G)`),
so some `σ`–`τ` path avoids `A` and `S` is computable. Deleting a minimum cut (size
`R(G)`) is by definition a separator, so `S` becomes non-computable; hence the minimal
disconnecting set has size exactly `R(G)`. For `B(w,d)`: each of the `d` bundles is a
`σ`–`τ` separator of size `w`, so `R ≤ w`; and the `w` "column" paths
`σ → (\text{layer }0,k) → (\text{layer }1,k) → \dots → τ` are internally vertex-disjoint,
so `R ≥ w`. Thus `R = w`. ∎

*Verification.* `verify_graph_layer.py`: for `B(3,4),B(5,3),B(2,6),B(4,5)`,
`nx.minimum_node_cut` returns `w` exactly; deleting one full bundle disconnects, deleting
`w−1` never does ⇒ `t*(G)=R(G)=w`.

### Theorem C2 (Exact random-ablation survival law).
For `B(w,d)` under i.i.d. internal-vertex deletion with probability `q` (survival
`p=1−q`),
$$\Theta(q)\;=\;\Pr[\sigma\!-\!\tau \text{ connected}]\;=\;(1-q^{\,w})^{d}.$$

*Proof.* Between consecutive bundles the network is complete bipartite, so a `σ`–`τ`
path exists **iff** every bundle retains at least one live vertex: pick any survivor in
each layer and the complete-bipartite edges connect them in series; conversely a fully
deleted bundle is a separator. Bundle `i` is fully deleted with probability `q^w`, hence
survives with probability `1−q^w`; the `d` bundles are deleted independently, so the
survival events multiply, giving `(1−q^w)^d`. ∎

*Verification.* Monte-Carlo (`N=40000`) over `B(3,4),B(5,3),B(4,5)` at
`q∈{0.1,0.3,0.5,0.7,0.9}`: max `|MC − theory| = 0.0033`, every point inside the 95% CI.

### Theorem C3 (Phase transition; exponential random-vs-targeted separation).
Let `d` be fixed.
1. **Location.** `Θ(q)=½` at `q_c = (1 − 2^{-1/d})^{1/w}`, and `q_c → 1` as `w → ∞`.
2. **Sharpening.** The transition window `W(w) := q_{0.1} − q_{0.9}`, where
   `Θ(q_{0.1})=0.1` and `Θ(q_{0.9})=0.9`, satisfies `W(w) → 0` as `w → ∞`; quantitatively
   `W(w) = Θ(1/w)`. Hence `Θ` approaches a step function (a percolation phase
   transition) as the redundancy width grows.
3. **Separation.** The *targeted* critical count is `R(G)=w` (Thm C1); the *random*
   critical count is `q_c\,w\,d`. The deletion **fraction** required at random,
   `q_c → 1`, while the targeted attacker removes only the fraction `1/d`. The
   adversary advantage `(\text{random count})/(\text{targeted count}) = q_c\,d` is
   increasing in `w`.

*Proof.* (1) Solve `(1−q^w)^d = ½`: `q^w = 1 − 2^{-1/d}`, so
`q_c=(1−2^{-1/d})^{1/w}`. The base `c_d := 1−2^{-1/d} ∈ (0,1)` is fixed, and
`c_d^{1/w} → c_d^0 = 1` as `w→∞`.
(2) Put `x = q^w ∈ [0,1]`; then `Θ = (1−x)^d`, a fixed decreasing bijection in `x`
independent of `w`. The levels `Θ=0.9,0.1` occur at fixed
`x_{0.9}=1−0.9^{1/d}` and `x_{0.1}=1−0.1^{1/d}` with `0<x_{0.9}<x_{0.1}<1`. Mapping back,
`q_{0.1}=x_{0.1}^{1/w}` and `q_{0.9}=x_{0.9}^{1/w}`, so
`W(w)=x_{0.1}^{1/w}-x_{0.9}^{1/w}`. Writing `a^{1/w}=e^{(\ln a)/w}=1+(\ln a)/w+O(1/w^2)`
for fixed `a∈(0,1)`,
`W(w) = \frac{\ln x_{0.1} - \ln x_{0.9}}{w} + O(1/w^2) = \Theta(1/w) → 0.`
Since `x_{0.1}>x_{0.9}` gives `\ln x_{0.1}-\ln x_{0.9}>0`, `W(w)>0` and strictly
decreasing for large `w`.
(3) Targeted count is `w` by Thm C1. At `q_c` the expected number of deleted vertices is
`q_c·(wd)`; the ratio to the targeted count is `q_c·d`, and since `q_c` is increasing in
`w` (as `c_d^{1/w}` with `c_d<1`), the ratio increases in `w`. The random deletion
*fraction* needed is `q_c → 1`. ∎

*Verification.* `d=8`: `q_c` rises `0.288 → 0.537 → 0.733 → 0.856` for `w=2,4,8,16`;
random critical count `4.6 → 17.2 → 46.9 → 109.6` vs targeted `2,4,8,16`; ratio
`2.30 → 4.29 → 5.86 → 6.85` (increasing); transition window strictly shrinks
(assertion passed). See `figures/fig1_phase_transition.png`,
`figures/fig2_random_vs_targeted.png`.

### Corollary E1 (Minimal adversarial-training budget to misalign).
Any process (continued/adversarial training, pruning, quantization) that renders `S`
non-computable must degrade at least `R(G)` internal pathway-units, i.e. a full
`σ`–`τ` separator. Hence the minimal budget to misalign is `R(G)`: shallow alignment
(small `R(G)`) is removable cheaply; redundant alignment (large `R(G)`) costs
proportionally more.

*Proof.* If `S` is non-computable afterwards, the set `A` of degraded units must contain
a `σ`–`τ` separator (else a path survives and `S` is computable, Thm C1), so
`|A| ≥ κ_{στ}(G) = R(G)`. ∎

**Reading.** Qi 2023's "~5–10 gradient steps remove safety" and Qi 2024's shallow,
first-few-tokens localization correspond to a **small effective `R(G)`** for the safety
behavior — the minimum cut is tiny, so the targeted budget is tiny. Conversely the
backup-head / concept-cone / "redundant paths survived" phenomena correspond to a
**large random threshold** `q_c → 1` for incidental damage.
