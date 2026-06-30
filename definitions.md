# Definitions and Notation

All random variables are discrete with finite support unless stated otherwise; logs are
base 2 so information is in bits. `H(·)` is Shannon entropy, `I(·;·)` mutual information,
`I(·;·|·)` conditional mutual information. `X ⊥ Y | Z` means conditional independence.

## The model of computation

**Definition 1 (Pathways and aligned behavior).**
Let `S` be the random variable encoding an *aligned behavior* of an LLM on a prompt
distribution — e.g. the refusal/comply decision, or a quantized refusal logit-difference
(Lit. Def. 2, McGrath `CE` target). Let `R = (R_1,…,R_n)` be a finite tuple of
**pathway readouts**: each `R_i` is a (possibly vector-valued) deterministic function of
the model's internal activations on the same prompt — an attention head's output, the
projection onto a refusal direction (Lit. Def. 3), an SAE-latent group, or a layer.
The joint law `p(s, r_1,…,r_n)` is induced by the prompt distribution and the (fixed)
weights.

**Definition 2 (Ablation).**
Ablating a subset `T ⊆ {1,…,n}` of pathways replaces `{R_i : i∈T}` by a fixed
intervention value (mean/resample ablation, Lit. Def. 3–5). The post-ablation behavior is
the best behavior recoverable from the *surviving* pathways `R_{T^c} = (R_i)_{i∉T}`.
We measure the *informational cost* of ablating `T` by
`ΔI_T := I(S; R_1,…,R_n) − I(S; R_{T^c})`  (information about `S` lost), and write
`ΔI_i := ΔI_{{i}}`.

**Definition 3 (Conditional redundancy).**
Pathway `R_i` is **(conditionally) redundant** given the rest if `S ⊥ R_i | R_{-i}`,
equivalently `I(S; R_i | R_{-i}) = 0`. A behavior `S` is **k-redundant** w.r.t. `R` if
`I(S; R_T) = I(S; R)` for every `T` with `|T| ≥ n−k+1` (any `k−1` pathways may be
ablated without informational loss).

## Information-theoretic compensation measures

**Definition 4 (Compensation index).**
Following the proposed `C(M)=MI(A_pre,A_post)/H(A_pre)`, for activation patterns
`A_pre, A_post` (pre/post intervention or training) define
`C := I(A_pre; A_post) / H(A_pre) ∈ [0,1]`. `C=1` ⇔ activations are preserved up to a
deterministic relabeling (perfect compensation); `C=0` ⇔ post-intervention activations
are independent of the originals. (Well-defined when `H(A_pre)>0`.)

**Definition 5 (Output robustness ρ).**
For behavior vectors `b_pre, b_post ∈ ℝ^m` (e.g. per-prompt refusal scores),
`ρ := 1 − ‖b_post − b_pre‖₂ / ‖b_pre‖₂`. `ρ=1` ⇔ behavior unchanged. For a
classification behavior we also use the error rate `P_e := Pr[Ŝ ≠ S]` of the optimal
decoder from surviving pathways.

**Definition 6 (Rerouting / JS score).**
For attention (or activation) distributions `P_pre, P_post`, the rerouting score is the
Jensen–Shannon divergence `JS(P_pre‖P_post) = ½D(P_pre‖M)+½D(P_post‖M)`,
`M=½(P_pre+P_post)`, `JS∈[0,1]` bits. Large `JS` with small behavioral change ⇒
*internal rerouting that preserves output* (the signature of compensation).

## Graph model

**Definition 7 (Computation DAG).**
Model the network as a DAG `G=(V,E)` with a source `σ` (prompt) and sink `τ` (the unit
emitting `S`); internal vertices are computational units (heads/neurons). A directed
`σ→τ` path is a **computational pathway**. `S` is *computable* iff at least one `σ→τ`
path is intact.

**Definition 8 (Pathway redundancy `R(G)`).**
`R(G) := κ_{στ}(G)`, the minimum number of *internal* vertices whose removal destroys
every `σ→τ` path (the minimum `σ`–`τ` vertex cut). By Menger's theorem this equals the
maximum number of internally vertex-disjoint `σ→τ` paths.

**Definition 9 (Critical ablation thresholds).**
- *Targeted:* `t*(G) :=` the minimum number of vertices an adversary must remove to make
  `S` non-computable `= R(G)`.
- *Random:* under i.i.d. vertex deletion with deletion probability `q` (survival
  `p=1−q`), the **survival probability** is `Θ(q) := Pr[some σ→τ path intact]`. The
  random critical fraction `q_c` is the deletion level at which `Θ` crosses ½ (finite
  size) or its sharp-threshold location (asymptotic).

**Definition 10 (Series-of-parallel-bundles network `B(w,d)`).**
`d` layers in series, each a **bundle** of `w` parallel vertices; `S` is computable iff
*every* layer retains ≥1 live vertex. (Each layer is the same OR-redundant
sub-computation; layers are composed in series.) This is the canonical exactly-solvable
redundant alignment circuit used for the percolation results.
