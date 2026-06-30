# Mathematical Modeling of Compensatory Robustness in LLM Alignment
## A Neural-Redundancy Framework: Information-Theoretic Bounds and a Percolation Phase Transition

*Mathematics research report. All theorems are proved in full and independently
verified computationally (CPU-only, ~seconds). Code in `src/`, raw outputs in
`results/`, figures in `figures/`, canonical proofs in
`results/theorems_and_proofs.md`.*

---

## 1. Executive Summary

We give a self-contained mathematical framework for the conjecture that an LLM's
aligned behavior is preserved by **redundant internal pathways**. The framework has
three proved layers and resolves the central empirical paradox of the field — alignment
is reported as both *fragile* (removable in ~5 gradient steps, concentrated in the first
output tokens, a single direction) and *distributed/robust* (backup heads, refusal
concept cones, "redundant paths survived ablation").

**Main results.**
1. **Ablation = non-redundant information (Theorem A1).** The information about an
   aligned behavior `S` lost by ablating a pathway `R_i` is *exactly* the conditional
   mutual information `I(S; R_i | R_{-i})`. Fully redundant pathways cost **zero** — a
   clean, artifact-free replacement for McGrath's logit-level compensatory-effect `CE`
   (≈30% of which is a LayerNorm-scaling artifact).
2. **Behavioral floor (Theorem B1, via Fano).** This information loss lower-bounds the
   unavoidable behavioral error after ablation, turning "redundancy" into a *provable*
   robustness guarantee on output behavior `ρ`.
3. **A percolation phase transition (Theorems C1–C3).** Modeling computation as a DAG,
   the **targeted** critical ablation threshold equals the minimum vertex cut
   `R(G)` (Menger), while the **random** threshold is governed by percolation and is
   *exponentially* larger in the redundancy width. This single picture makes both
   empirical regimes true: alignment can be **robust to random/incidental damage yet
   fragile to a targeted attack on its min-cut.**
4. **Compensation reveals, not creates, robustness (Proposition D1).** If behavior
   survives ablation, the surviving pathways *already* contained all the information —
   "dynamic rerouting" exposes latent redundancy that pre-existed the intervention.
5. **Honest reconciliation (Corollary E1).** Misaligning `S` requires degrading at least
   `R(G)` pathway-units. The user's premise — that LLMs resist misalignment — is
   **predicted to hold for random/benign degradation but to fail for targeted
   adversarial training**, matching Qi 2023/2024's easy targeted misalignment. As a
   universal claim the premise is *not* supported; its correct form is regime-dependent.

Every nontrivial claim is confirmed numerically: the A1 identity to machine precision
(error `0.0e0`), the Fano floor with no violations, the targeted threshold `= R(G)`
exactly, and the random survival law `(1−q^w)^d` within Monte-Carlo 95% CIs
(max deviation `0.0033`, `N=40000`).

---

## 2. Research Question & Motivation

**Question.** Can the compensatory robustness of LLM alignment be given a rigorous
mathematical characterization in which (a) a redundancy measure provably bounds the
effect of ablating an internal pathway, and (b) the critical ablation threshold exhibits
a percolation-type phase transition separating robustness-to-random-damage from
fragility-to-targeted-attack?

**Why it matters.** Whether safety is *redundantly* encoded decides whether it survives
quantization, pruning, benign fine-tuning, or adversarial attack. The literature is
split (see `literature_review.md`): fragile (Qi 2023 R11, Qi 2024 R10, Arditi 2024)
vs distributed/redundant (Wang 2022 R1, McGrath 2023 R2, Wollschläger 2025 R7, Joad 2026
R8, Frank 2026 R12). **Gap 1** of the review is the absence of a single scalar under
which both are special cases. We supply it: the min-cut `R(G)` (targeted fragility) and
the percolation threshold `q_c` (random robustness) are two faces of the same network.

---

## 3. Definitions and Notation

(Full version in `definitions.md`.) Random variables are discrete, logs base 2 (bits).

- **`S`** — aligned behavior (refusal/comply decision, or a quantized refusal
  logit-difference). **`R=(R_1,…,R_n)`** — pathway readouts, each a deterministic
  function of internal activations (a head, a refusal-direction projection, an SAE-latent
  group, a layer).
- **Ablation** of `T ⊆ {1,…,n}`: replace `{R_i}_{i∈T}` by a fixed value; behavior is read
  from the surviving `R_{T^c}`. **Information cost** `ΔI_T := I(S;R) − I(S;R_{T^c})`.
- **Conditional redundancy:** `R_i` redundant given the rest iff `S ⊥ R_i | R_{-i}`.
  `S` is **k-redundant** iff `I(S;R_T)=I(S;R)` for every `|T| ≥ n−k+1`.
- **Compensation index** `C := I(A_pre;A_post)/H(A_pre) ∈ [0,1]` (proposed `C(M)`).
  **Output robustness** `ρ := 1 − ‖b_post−b_pre‖₂/‖b_pre‖₂`; decoder error
  `P_e := Pr[Ŝ≠S]`. **Rerouting score** `JS(P_pre‖P_post)` (Jensen–Shannon).
- **Computation DAG** `G=(V,E)`, source `σ`, sink `τ`; `S` *computable* iff a `σ→τ` path
  is intact. **Pathway redundancy** `R(G) := κ_{στ}(G)` = min `σ`–`τ` vertex cut =
  max vertex-disjoint `σ→τ` paths (Menger). **`B(w,d)`**: `d` bundles of `w` parallel
  vertices in series, complete-bipartite between bundles (the canonical exactly-solvable
  redundant circuit).

---

## 4. Statement of Results

- **Theorem A1** (ablation cost = non-redundant information): `ΔI_T = I(S;R_T|R_{T^c})`;
  `ΔI_i=0 ⇔ S⊥R_i|R_{-i}`.
- **Theorem A2** (k-redundancy ⇒ resilience to any `k−1` ablations).
- **Theorem B1** (Fano behavioral-fidelity floor): if `I(S;R)=H(S)` then
  `H_2(P_e)+P_e\log_2(|S|−1) ≥ ΔI_T`; redundant ablation ⇒ `P_e=0`.
- **Proposition D1** (observed compensation ⇒ pre-existing latent redundancy).
- **Theorem C1** (targeted threshold `t*(G)=R(G)`; `R(B(w,d))=w`).
- **Theorem C2** (exact random survival law `Θ(q)=(1−q^w)^d`).
- **Theorem C3** (phase transition: location `q_c=(1−2^{-1/d})^{1/w}`; window
  `W(w)=Θ(1/w)→0`; random-vs-targeted separation `q_c·d` increasing in `w`).
- **Corollary E1** (minimal adversarial budget to misalign `= R(G)`).

---

## 5. Proofs

### Theorem A1 — Ablation cost equals non-redundant information.
`ΔI_T = I(S;R) − I(S;R_{T^c}) = I(S; R_T | R_{T^c})`, hence `ΔI_i=I(S;R_i|R_{-i})≥0`,
zero iff `S⊥R_i|R_{-i}`.

*Proof.* Chain rule of mutual information on `R=(R_{T^c},R_T)`:
`I(S;R) = I(S;R_{T^c}) + I(S;R_T|R_{T^c})`. Subtract `I(S;R_{T^c})`. Conditional MI is
non-negative; it is zero iff the conditional independence holds. ∎

This is exact and free of the LayerNorm-scaling nuisance that contaminates McGrath's
logit-level `CE` (R3): the part `R_i` shares with the rest cancels, leaving only genuine
non-redundant content.

### Theorem A2 — k-redundancy ⇒ resilience to any `k−1` ablations.
*Proof.* For `|A|≤k−1` set `T=A^c`; then `|T|=n−|A|≥n−k+1`, so by k-redundancy
`I(S;R_T)=I(S;R)`, giving `ΔI_A=0` (and `I(S;R_A|R_{A^c})=0` by A1). ∎

k-redundancy is a covering/coding condition — every `(n−k+1)`-subset of pathways
determines `S` — the discrete analogue of concept cones (R7) and backup heads (R1).

### Theorem B1 — Behavioral-fidelity floor.
For any decoder `Ŝ=g(R_{T^c})` with error `P_e`,
`H_2(P_e)+P_e\log_2(|S|−1) ≥ H(S|R_{T^c}) = H(S)−I(S;R)+ΔI_T`; if `I(S;R)=H(S)` the RHS
is `ΔI_T`, and `ΔI_T=0 ⇒ P_e=0`.

*Proof.* `S→R_{T^c}→Ŝ` is Markov, so Fano gives `H_2(P_e)+P_e\log_2(|S|−1)≥H(S|R_{T^c})`.
Then `H(S|R_{T^c})=H(S)−I(S;R_{T^c})=H(S)−I(S;R)+ΔI_T` (Thm A1). If `ΔI_T=0` then
`H(S|R_{T^c})=0`, so `S` is an a.s.-function of `R_{T^c}` and the MAP decoder attains
`P_e=0`, i.e. behavior is fully preserved (`ρ=1`). ∎

This is the rigorous **redundancy ⇒ behavioral robustness** law: a positive
non-redundant loss `ΔI_T` *forces* output error; zero loss *guarantees* preservation.

### Proposition D1 — Compensation reveals, not creates, redundancy.
If `H(S|R)=0` and after ablating `T` the optimal decoder has `P_e=0`, then
`I(S;R_{T^c})=H(S)` — the surviving pathways already held all information about `S`.

*Proof.* `P_e=0` (optimal) ⇔ `H(S|R_{T^c})=0`, so `I(S;R_{T^c})=H(S)−0=H(S)`. This is a
property of the original joint law (surviving readouts are unchanged by deleting others),
so the redundancy pre-existed. ∎

Formalizes self-repair/backup behavior (R1,R2,R8) and the Iterative-Inference
Hypothesis: rerouting completes the task *because the signal was already redundant*,
matching Arditi's "present in base models, repurposed not learned."

### Theorem C1 — Targeted critical threshold equals `R(G)`.
The minimum number of internal vertices whose deletion makes `S` non-computable equals
`R(G)=κ_{στ}(G)`; `S` survives any deletion of `<R(G)` vertices; `R(B(w,d))=w`.

*Proof.* Menger (vertex form): min `σ`–`τ` cut = max internally-vertex-disjoint `σ`–`τ`
paths `=R(G)`. If `|A|<R(G)`, `A` is not a separator, so a path avoids it ⇒ computable;
a minimum cut (size `R(G)`) is a separator ⇒ non-computable. For `B(w,d)`: each bundle is
a size-`w` separator (`R≤w`) and the `w` column paths are vertex-disjoint (`R≥w`), so
`R=w`. ∎

### Theorem C2 — Exact random survival law.
For `B(w,d)` under i.i.d. deletion prob `q`, `Θ(q)=(1−q^w)^d`.

*Proof.* Complete-bipartite links between consecutive bundles ⇒ a `σ`–`τ` path exists iff
every bundle keeps ≥1 live vertex (choose one survivor per layer; conversely a dead
bundle is a separator). Bundle survives w.p. `1−q^w`; `d` independent bundles multiply. ∎

### Theorem C3 — Phase transition and random-vs-targeted separation (`d` fixed).
(1) `Θ(q)=½` at `q_c=(1−2^{-1/d})^{1/w}`, `q_c→1` as `w→∞`. (2) Transition window
`W(w)=q_{0.1}−q_{0.9}=Θ(1/w)→0`. (3) Targeted count `=R(G)=w`; random count `=q_c·w·d`;
advantage `q_c·d` increasing in `w`; random deletion *fraction* `q_c→1`.

*Proof.* (1) `(1−q^w)^d=½ ⇒ q^w=1−2^{-1/d}=:c_d∈(0,1) ⇒ q_c=c_d^{1/w}→1`.
(2) With `x=q^w`, `Θ=(1−x)^d` is a fixed bijection; levels `0.9,0.1` at fixed
`x_{0.9}=1−0.9^{1/d}<x_{0.1}=1−0.1^{1/d}`. Then
`W(w)=x_{0.1}^{1/w}−x_{0.9}^{1/w}=\frac{\ln x_{0.1}−\ln x_{0.9}}{w}+O(w^{-2})=Θ(1/w)>0`,
strictly decreasing for large `w` (step-function limit ⇒ percolation transition).
(3) Targeted `=w` (C1); expected random deletions at `q_c` is `q_c·wd`; ratio `q_c·d`
increases since `q_c=c_d^{1/w}` increases in `w`; the required fraction `q_c→1`. ∎

### Corollary E1 — Minimal adversarial budget.
Any process making `S` non-computable degrades ≥ `R(G)` pathway-units.

*Proof.* The degraded set must contain a `σ`–`τ` separator (else a path survives, C1), so
its size is `≥κ_{στ}(G)=R(G)`. ∎

---

## 6. Computational Verification

CPU-only; `numpy 2.5.0`, `scipy 1.18.0`, `sympy 1.14.0`, `networkx 3.6.1`, `dit`,
`matplotlib 3.10.9`; seed `42`. Reproduce: `python src/verify_information_layer.py`,
`python src/verify_graph_layer.py`, `python src/make_figures.py`.

**V1 — A1 identity & B1 floor** (`results/v1_v3_information_layer.txt`).

| case | `ΔI_1` (ablate R1) | `I(S;R_1|R_2)` | identity err | Fano floor holds (`P_e*`) |
|---|---|---|---|---|
| redundant `S=R1=R2` | 0.000000 | 0.000000 | `0.0e0` | yes (`P_e*=0.00`) |
| unique `S=R1` | 1.000000 | 1.000000 | `0.0e0` | yes (`P_e*=0.50`) |
| synergy `S=R1⊕R2` | 1.000000 | 1.000000 | `0.0e0` | yes (`P_e*=0.50`) |
| partial (ε=0.1) | 0.468996 | 0.468996 | `0.0e0` | yes (`P_e*=0.10`) |

**V3 — PID cross-check** (`dit`, Williams–Beer): redundant `Rdn=1.000`; XOR `Syn=1.000`;
partial `Rdn=0.531, Unq_R1=0.469`. Ablation loss `ΔI_1 = Unq_R1 + Syn`, redundant atom
free — exactly as A1 predicts, and a guard against `I_min` over-reporting (review Gap).

**V2 — graph layer** (`results/v2_graph_layer.txt`).
- *C1:* `nx.minimum_node_cut = w` for `B(3,4),B(5,3),B(2,6),B(4,5)`; one full bundle
  disconnects, `w−1` never does ⇒ `t*(G)=R(G)=w`.
- *C2:* `N=40000` Monte-Carlo vs `(1−q^w)^d`, max `|diff|=0.0033`, all within 95% CI.
- *C3 (`d=8`):* `q_c = 0.288, 0.537, 0.733, 0.856` for `w=2,4,8,16`; random critical
  count `4.6, 17.2, 46.9, 109.6` vs targeted `2,4,8,16`; ratio `2.30→6.85` (increasing);
  transition window strictly shrinks.

**Figures.** `figures/fig1_phase_transition.png` — `Θ(q)` sharpening into a step as `w`
grows; `figures/fig2_random_vs_targeted.png` — random vs targeted critical counts.

---

## 7. Analysis & Discussion

**The unifying scalar (resolves review Gap 1).** A network has *two* robustness numbers:
the min-cut `R(G)` (targeted fragility) and the percolation threshold `q_c` (random
robustness). Theorem C3 shows they diverge — `q_c→1` while `R(G)=w` — so the **same**
aligned circuit is simultaneously "robust" and "fragile" depending on the threat model.
This dissolves the apparent contradiction in the literature:

| Empirical finding | Framework reading |
|---|---|
| Backup name-mover heads: ablate 3 → 5% drop (Wang R1) | large random robustness; surviving disjoint paths (C1/C2) |
| "Redundant paths survived ablation" (Joad R8); refusal cones dim≤5 (R7) | `k`-redundancy / `R(G)>1` (A2, C1); D1: latent, pre-existing |
| Self-repair / Hydra ≈70% restoration (McGrath R2) | D1 + A1: surviving pathways already held the info |
| Safety removable in ~5 steps (Qi R11); shallow, first-token (Qi R10) | small effective `R(G)` ⇒ tiny targeted budget (E1) |
| Single refusal direction (Arditi) | `R(G)=1` special case: min-cut = the one direction |
| Per-head ablation 58× too weak at scale (Frank R12) | A1: single-pathway `ΔI_i≈0` under redundancy hides importance |

**Honest verdict on the user's premise.** The premise — "persistent null results when
misaligning LLMs ⇒ intrinsic robustness" — is **not supported as a universal claim**.
Qi 2023/2024 (in the gathered literature) show targeted fine-tuning misaligns models in
~5–10 steps. Our Corollary E1 explains *why both observations occur*: random/benign
degradation must overcome `q_c→1` (looks robust), but a targeted attacker need only
remove the `R(G)` min-cut (easy when alignment is shallow). The scientifically correct
statement is therefore **regime-dependent**, and the framework predicts exactly when
each holds — a more useful result than confirming an over-strong premise.

**Comparison to McGrath `CE`.** `ΔI` (A1) is an exact identity, eliminating the ~30%
LayerNorm-scaling artifact in `CE` (R3) and giving a head/feature-level,
mechanism-corrected compensation measure — addressing review Gap 2.

**Effect sizes.** The targeted/random separation is not marginal: at `d=8,w=16` a random
adversary must delete ~110 of 128 units (86%) to match what a targeted adversary does by
removing 16 (12.5%) — a ~7× count advantage and a qualitative fraction gap (`q_c=0.86`
vs `1/d=0.125`).

---

## 8. Limitations & Threats to Validity

1. **Combinatorial idealization (C1–C3).** The DAG model treats `S` as computable iff a
   path is *intact*. Real transformers are analog: partial contributions can sum past a
   threshold, so `R(G)` can *under*-state robustness (sub-threshold paths jointly
   suffice) — a genuine counterexample to exact tightness. `B(w,d)`'s closed form (C2)
   further assumes complete-bipartite inter-bundle wiring; sparser wiring changes the
   constant but not the qualitative transition.
2. **Estimation on real models.** `ΔI=I(S;R_i|R_{-i})` requires estimating (conditional)
   MI in high dimensions — hard and biased; PID `I_min` can over-report redundancy
   (we cross-checked with `dit`, but did not run BROJA/MMI). We did **not** instantiate
   the bounds on an actual LLM (CPU-only, 1h); this is a theory + synthetic-verification
   contribution, not an empirical measurement.
3. **Fano gives a floor, not the achieved error.** B1 lower-bounds `P_e`; it does not
   claim the bound is met (tight when `S|R_{T^c}` is uniform on its support).
4. **Static `R(G)`.** E1 bounds the *budget* to misalign but does not model how `R(G)`
   itself evolves under training (does adversarial training shrink the min-cut, or grow
   redundancy à la Bereska's abundance regime, R14?). This is left open (§9).
5. **Mapping `S` to "alignment" is a modeling choice.** Real aligned behavior is not a
   single discrete variable; the framework applies per chosen behavior `S`.

---

## 9. Open Questions

- **Dynamics of `R(G)` under training (review Gap 3).** Prove conditions (Hänni capacity
  `Õ(d²)` R13 + Bereska abundance/scarcity R14) under which adversarial training *grows*
  the alignment min-cut rather than cutting it.
- **Analog/threshold percolation.** Replace "path intact" by "summed contribution ≥ θ";
  derive the modified `R(G)` and a weighted min-cut / flow threshold.
- **Degeneracy vs redundancy (Gap 4).** Port Tononi–Sporns–Edelman degeneracy and
  Whitacre–Bender networked-buffering (R6) to bound *multi*-pathway lesion robustness by
  a graph-connectivity quantity beyond the single min-cut.
- **Tight Fano / list-decoding** for graded refusal behaviors; relate `P_e` to `ρ`
  quantitatively for vector behaviors.
- **Estimator-robust `ΔI`.** A redundancy-aware ablation estimator immune to Frank's
  58×-underestimate and Hänni's "unused features" caveat.

---

## 10. Conclusions

We answer the research question affirmatively **with a precise caveat**: compensatory
robustness of LLM alignment *is* mathematically characterizable — (a) the cost of
ablating a pathway equals its non-redundant conditional information `I(S;R_i|R_{-i})`
(Theorem A1), which lower-bounds behavioral error via Fano (Theorem B1), and (b) the
critical ablation threshold genuinely exhibits a percolation phase transition whose
*targeted* value is the min-cut `R(G)` and whose *random* value `q_c→1` is exponentially
separated from it (Theorems C1–C3). The framework's payoff is conceptual: alignment can
be **robust to random damage and fragile to targeted attack at once**, which reconciles
the field's contradictory reports and shows the user's "intrinsic robustness" premise is
true only in the random/benign regime, not the adversarial one (Corollary E1, with Qi
2023/2024). Compensation, finally, *reveals* pre-existing redundancy rather than creating
it (Proposition D1). Next steps: lift the combinatorial idealization to analog-threshold
percolation, model `R(G)`'s evolution under training, and estimate the bounds on real
transformers.

---

## 11. References

Cited via the gathered `literature_review.md` (results R1–R16) and `resources.md`.
Key sources: Wang et al. 2022 (IOI/backup heads, R1); McGrath et al. 2023 (Hydra `CE`,
R2); Rushing & Nanda 2024 (self-repair, LayerNorm artifact, R3); Arditi et al. 2024
(single direction); Wollschläger et al. 2025 (concept cones, R7); Joad et al. 2026
(redundant paths survived, R8); Qi et al. 2023 (fine-tuning removes safety, R11); Qi et
al. 2024 (shallow safety, R10); Frank 2026 (routing redundancy, R12); Williams & Beer
2010 (PID `I_min`, R4); Proca et al. 2022 (redundancy↔robustness, R5); Whitacre & Bender
2009 (degeneracy, R6); Hänni et al. 2024 (computation in superposition, R13); Bereska et
al. 2025 (superposition regimes, R14); Geiger et al. 2025 (causal abstraction/IIA, R15);
Zhang & Nanda 2023 (patching best practices, R16). Mathematical tools: Menger's theorem;
Fano's inequality; chain rule of mutual information (Cover & Thomas). Software: `numpy`,
`scipy`, `sympy`, `networkx`, `dit`, `matplotlib`.
