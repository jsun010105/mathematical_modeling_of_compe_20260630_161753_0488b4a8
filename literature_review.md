# Literature Review
## Mathematical Modeling of Compensatory Robustness in LLM Alignment: A Neural Redundancy Framework

### Research Area Overview

The hypothesis under study is that LLMs possess **mathematically characterizable
compensatory mechanisms** — redundant computational pathways that preserve aligned
behavior under adversarial training through dynamic rerouting of internal
representations — formalizable with information-theoretic measures and detectable
through ablation studies, analogous to biological neural compensation.

The relevant literature sits at the intersection of **five** mature subfields, each
supplying one ingredient of the framework:

1. **Self-repair / compensatory rerouting** (mechanistic interpretability): the
   *empirical phenomenon* the hypothesis names — ablate one component, a downstream
   component increases its effect to recover the output (Wang 2022 → McGrath 2023 →
   Rushing & Nanda 2024).
2. **Geometry of refusal directions**: the *aligned behavior* (refusal/safety) is
   shown to be mediated by **multiple, redundant directions/subspaces**, not one
   (Arditi 2024 vs Wollschläger 2025, Joad 2026, Piras 2025).
3. **Alignment robustness under (adversarial) training**: *where* alignment lives and
   whether further training reroutes or destroys it (Qi 2023, Qi 2024, Frank 2026).
4. **Information-theoretic & biological formalism for redundancy**: the *mathematics
   of measuring* redundancy — Partial Information Decomposition (Williams & Beer 2010;
   Proca 2022) and biological **degeneracy** (Whitacre & Bender 2009).
5. **Superposition & causal-ablation methodology**: the *source* of representational
   redundancy (Hänni 2024; Bereska 2025) and the *rigorous detection toolkit*
   (Zhang & Nanda 2024; Geiger 2025).

The central scientific tension the framework must resolve: alignment is sometimes
reported as **fragile and localized** (a few output tokens / a single direction /
~5 gradient steps to remove) yet sometimes as **distributed and redundant** (backup
heads, concept cones, surviving ablation). The redundancy framework's job is to give
a single quantitative account — a *compensation/redundancy measure* — under which both
observations are special cases.

---

### Key Definitions

**Definition 1 (Residual-stream model).** For a decoder-only transformer, token *i*,
layer *l*: `x_i^{(l+1)} = x_i^{(l)} + Attn^{(l)}(x_{1:i}^{(l)}) + MLP^{(l)}(x̃_i^{(l)})`.
Logits `π_t = RMSNorm(x_t^{(L)}) W_U`. *(Arditi 2024 Eq.1; McGrath 2023 Eq.3–4.)*

**Definition 2 (Difference-in-means / refusal direction).** For activation sets on
harmful vs harmless prompts, `r_i^{(l)} = μ_i^{(l)} − ν_i^{(l)}` with
`μ = mean(harmful acts)`, `ν = mean(harmless acts)`; unit `r̂ = r/‖r‖`.
*(Arditi 2024 Eq.2.)*

**Definition 3 (Directional ablation / projection).** Remove a direction's component
everywhere: `x' = x − r̂ r̂ᵀ x`. Weight-space equivalent (orthogonalization):
`W_out' = W_out − r̂ r̂ᵀ W_out`. *(Arditi 2024 Eq.4–5.)*

**Definition 4 (Activation patching = interchange intervention).** Three forward
passes — clean `X_clean`, corrupt `X_corrupt`, and **patched** (corrupt run with one
component's activation restored from the clean cache). Patching effect = restored
performance ⇒ that component's causal importance. Formally
`IntInv(M, sources, targets) = ∪_j Proj_{X_j}(Solve(M_{s_j}))`.
*(Zhang & Nanda 2024 §2; Geiger 2025 Def.44.)* **Path patching** = *recursive*
interchange intervention isolating direct effect (Geiger 2025 Def.45; Wang 2022 App.B).

**Definition 5 (Causal effects, neural-net-as-SCM).** Total effect = ablation effect
`TE = π̂_t(x | do(A=ã)) − π̂_t(x)`; direct effect `DE(z→z',Y,u) =
Y(u|do(Z=z',M=m(u))) − Y(u|do(Z=z))`; indirect effect = total − direct.
*(McGrath 2023 Def.3.2–3.4.)*

**Definition 6 (Self-repair).** `self-repair = Δlogit − ΔDE_head`, where
`Δlogit = logit_ablated − logit_clean`. A head "self-repairs" when
`−1 < Δlogit/DE_head < 0`. *(Rushing & Nanda 2024 Eq.1.)*

**Definition 7 (Compensatory effect — the Hydra metric).** After ablating component
`m`, sum the change in downstream direct effects:
`CE(ã^m, u) = Σ_{l>m} ΔDE(a^l,u,ã^m) + Σ_{l≥m} ΔDE(m^l,u,ã^m)`, with
`ΔDE = DE_ablated − DE`. *(McGrath 2023 Eq.33–34.)* **This is the canonical formal
measure of compensatory rerouting** and a prime template for the project's redundancy
measure.

**Definition 8 (Specific information & I_min redundancy — PID).** For target `S` and
sources `A_i`: specific information `I(S=s;A) = Σ_a p(a|s)[log 1/p(s) − log 1/p(s|a)]`;
redundancy `I_min(S;{A_1,…,A_k}) = Σ_s p(s) min_i I(S=s;A_i)`. The mutual information
`I(S;R_1,R_2)` decomposes into **Unique** `U(S;R_i)`, **Redundant** `Rdn`, **Synergistic**
`Syn` atoms via the redundancy lattice and Möbius inverse `Π_R`. *(Williams & Beer 2010
Eq.1–8.)* Property: `I_min(S;{A_i}) ≤ I(S;A_i)` for all i — redundancy lower-bounds each
pathway's information.

**Definition 9 (Redundancy vs Degeneracy).** *Redundancy* = coexistence of *identical*
components with identical function (robust only to "more of the same"). *Degeneracy*
= coexistence of *structurally distinct* components performing **similar roles under
some conditions, distinct roles under others** (trait-set intersection non-empty and
≠ union). *(Whitacre & Bender 2009.)* Degeneracy — not pure redundancy — is what yields
*distributed* robustness + evolvability via "networked buffering."

**Definition 10 (Superposition measure).** Effective features `F = e^{H(p)}`,
`p_i = (Σ_s|z_{i,s}|)/(Σ_j Σ_s|z_{j,s}|)` (SAE-budget allocation); superposition
`ψ = F/N` (N physical neurons). `ψ=1` ⇒ lossless; `ψ>1` ⇒ lossy compression / shared
dimensions. *(Bereska 2025 Eq.6–8.)*

**Definition 11 (Concept cone / representational independence).** A refusal-mediating
**concept cone** is `R_N = {Σ_{i=1}^N λ_i b_i : λ_i ≥ 0}\{0}` (orthonormal basis B)
such that *every* `r ∈ R_N` triggers refusal. Two directions are
**representationally independent** if ablating one does not change the per-layer
expression (cosine) of the other. *(Wollschläger 2025 §5–6.)*

**Definition 12 (Interchange Intervention Accuracy, IIA).** For a hypothesized
high-level causal model `H` aligned to network `L` by `(Π,π)`:
`IIA = E_i[ 1[ Proj_{out}(τ^π(L_i)) = Proj_{out}(H_{ω^π(i)}) ] ]` — the fraction of
interchange interventions on which the network's output matches the algorithm's. The
quantitative criterion that a (possibly distributed) subspace **realizes** an
aligned-behavior variable. *(Geiger 2025 Def.48.)*

---

### Key Papers (grouped; full notes per cluster)

#### Cluster 1 — Self-repair / compensatory rerouting

- **Wang et al. 2022 (IOI circuit, arXiv:2211.00593).** Reverse-engineers a 26-head
  circuit in GPT-2-small for indirect-object identification. **Load-bearing result:**
  ablating *all three* primary Name Mover Heads drops the logit difference by only
  **5%** because dormant **Backup Name Mover Heads** activate to replace them — the
  founding empirical demonstration of redundant backup pathways and ablation-triggered
  rerouting. Methods: path patching (direct vs indirect effect), mean ablation,
  faithfulness/completeness/minimality circuit criteria. Originally conjectured
  dropout as cause.
- **McGrath et al. 2023 (Hydra Effect, arXiv:2307.15771).** Formalizes compensation in
  a causal-SCM language and introduces the **compensatory-effect metric `CE`** (Def.7).
  In **Chinchilla-7B trained without dropout** (refuting the dropout hypothesis),
  ablating an attention layer causes downstream attention to increase its effect plus
  a counter-balancing late-MLP "erasure" response, jointly **restoring ≈70% of the
  logit reduction**; compensation explains **r²=0.92** of variance at layer 23
  (slope 0.69 < 1 ⇒ partial repair). Resample ablation; layer-granular.
- **Rushing & Nanda 2024 (Explorations of Self-Repair, arXiv:2402.15390).** Generalizes
  to the full pretraining distribution and to *individual heads* across 7 model
  families. Gives `self-repair = Δlogit − ΔDE` (Def.6) and the LayerNorm decomposition
  `Δlogit = (S/S'−1)·logit + (S/S')·ΔDE`. **Deflationary findings:** repair is
  imperfect & noisy; **≈30% of self-repair is a LayerNorm-scaling artifact** (ablation
  shrinks residual norm S>S', amplifying surviving logits), and MLP erasure is driven
  by sparse **Anti-Erasure neurons** that differ across prompts. Proposes the
  **Iterative Inference Hypothesis** as the rerouting mechanism: the task signal
  persists in the residual stream, so a downstream capable head completes the task.

#### Cluster 2 — Geometry of refusal: redundancy of the alignment representation

- **Arditi et al. 2024 (Single Direction, arXiv:2406.11717).** The **1-D baseline**:
  a single difference-in-means direction, when ablated (Def.3), removes refusal across
  13 models; when added, induces it. The refusal direction is already present in *base*
  models (repurposed, not learned). Adversarial suffixes work by *suppressing* the
  direction's expression.
- **Wollschläger et al. 2025 (Concept Cones, arXiv:2502.17420).** Refutes 1-D:
  **multiple representationally-independent directions and polyhedral concept cones up
  to dimension 5** mediate refusal; larger models support higher-dimensional cones.
  **Compositional ablation of k≥4 independent directions raises ASR monotonically,
  surpassing the single-direction baseline** ⇒ additive, distinct, redundant mechanisms.
  Introduces RDO/RCO optimization and the representational-independence loss.
- **Joad et al. 2026 (More than a Single Direction, arXiv:2602.02132).** Across **11
  refusal categories**, directions are geometrically distinct (pairwise cosine 0.4–0.6,
  several near-orthogonal) but linear steering collapses them to **one effective control
  knob**. SAE analysis reveals a **small shared latent core (591/517/421 latents at
  layers 9/20/31) + long polysemous tail**. **Llama survives safety-direction ablation**,
  which the authors attribute explicitly to *"a richer internal refusal structure with
  redundant paths that survived the ablation"* — direct evidence of compensatory
  redundancy invisible to linear control.
- **Piras et al. 2025 (SOM Directions, arXiv:2511.08379).** Refusal as a
  **low-dimensional manifold of multiple non-orthogonal directions** found via
  Self-Organizing Maps. **Proposition 1 (proven):** a 1-neuron SOM converges to the
  difference-in-means centroid, so Arditi's single direction is the degenerate limit.
  Ablating multiple directions beats single-direction and dedicated jailbreaks (e.g.
  Llama-2-7B 59.1% vs 0.0%), **rises monotonically with #directions**, and partially
  reverses circuit-breaker defenses. ASR correlates strongly (Pearson −0.83…−0.98) with
  the *collapse of harmful-cluster variance* under ablation.

#### Cluster 3 — Alignment robustness under (adversarial) training

- **Qi et al. 2023 (Fine-tuning Compromises Safety, arXiv:2310.03693).** Alignment is
  **not robust to further training**: ~5–10 gradient steps (≤100 examples, <$0.20) push
  harmfulness from <2% to ~80–90%; even *benign* fine-tuning regresses safety. The
  "destroy, not reroute" baseline — naive fine-tuning *removes* aligned behavior
  (catastrophic forgetting); per-category fragility is uneven.
- **Qi et al. 2024 (More than a Few Tokens Deep, arXiv:2406.05946).** Diagnoses **shallow
  safety alignment**: the per-token KL between aligned and base models is concentrated
  in the **first few output tokens**; prefilling a refusal prefix collapses base-model
  harmfulness to ~1–2%. This token-depth localization is *why* fine-tuning succeeds in
  a few steps. Mitigations *manufacture depth-redundancy*: a safety-recovery
  data-augmentation (refuse even after a harmful prefix) and a **token-wise constrained
  fine-tuning objective** that strongly anchors early tokens (β₁=0.5 vs β_{t>5}=0.1),
  cutting attack ASR from ~89% to ~5%.
- **Frank 2026 (How Alignment Routes, arXiv:2604.04385).** A **Detect→Route→Output**
  gate/amplifier circuit (detection at layers 15–17; a sparse **gate head** triggers,
  **amplifier heads** boost). Provides the affirmative redundancy evidence:
  **partial redundancy** with counter-routing "coalitions," gate **relocation** under
  continued training (Jaccard ≤0.05) with capability preserved, and steering that
  flips behavior — *"the safety-trained capability is gated by routing, not removed."*
  **Critical methodological warning:** per-head ablation underestimates importance up
  to **58× at 72B** and *misses redundant gates*; interchange/patching is required at
  scale.

#### Cluster 4 — Information-theoretic & biological formalism

- **Williams & Beer 2010 (PID, arXiv:1004.2515).** Founding Partial Information
  Decomposition: the `I_min` redundancy measure (Def.8), the redundancy lattice over
  antichains, and the nonnegative Unique/Redundant/Synergistic atom decomposition.
  Notes that interaction information *confounds* synergy and redundancy
  (`I(S;R1;R2) = Syn − Rdn`). Known weakness (later literature): `I_min` can over-report
  redundancy when sources carry *different* information about the same outcome.
- **Proca et al. 2022 (Synergistic Information in NNs, arXiv:2210.02996).** The
  **operational template**: computes redundancy/synergy/unique per layer (sources =
  neurons, target = next layer; `dit` plug-in for discrete, GCMI for continuous;
  2nd-order pairwise averaging to bypass the super-exponential atom count). Empirical
  laws: **redundancy ↔ robustness to lesion** (removing maximally-synergistic neurons
  hurts most); **dropout increases redundancy**; synergy rises with task integration.
  Functional reading: redundancy→robustness, unique→specialization, synergy→flexible
  integration (noise-vulnerable).
- **Whitacre & Bender 2009 (Degeneracy, arXiv:0907.0510).** Sharpens the framework's
  central concept (Def.9): *degeneracy* (structurally distinct, partially-overlapping
  pathways) — not pure redundancy — produces **distributed robustness** via "networked
  buffering," and uniquely also yields **evolvability** (~20× more accessible
  phenotypes than a redundant control of equal size). Supplies a complementary
  *graph-theoretic* measurement toolkit: neutral-network size, local robustness,
  differential robustness under multi-component lesion, 1-neighborhood phenotype
  accessibility. Cites Tononi–Sporns–Edelman (1999) as the information-theoretic
  measure of degeneracy — the bridge to PID.

#### Cluster 5 — Superposition & causal-ablation methodology

- **Hänni et al. 2024 (Computation in Superposition, arXiv:2408.05451).** Proves
  networks can not only *store* but *compute* with >n features in n dims: a single MLP
  layer ε-linearly represents a Universal-AND of all feature pairs with width
  `d = Õ(√m/ε²)`; information-theoretic capacity ≈ `Õ(d²)`. **Error-correction layers**
  (round-and-rectify) are a concrete *robustness/compensation* mechanism. The
  **"unused features" caveat (Thm 3):** random nets linearly represent features they
  do not *use* — so probing/SAE-detecting a pathway ≠ that pathway being load-bearing.
- **Bereska et al. 2025 (Superposition as Lossy Compression, arXiv:2512.13568).**
  Quantifies superposition via SAE entropy (Def.10) and tests the
  superposition⇄adversarial-vulnerability link. **Adversarial training does *not*
  universally reduce superposition**: an *abundance regime* (low task complexity / high
  capacity) *adds* redundant features, a *scarcity regime* reduces them — robustness can
  come *through added redundant pathways* when capacity permits. Dropout reduces effective
  feature count (redundant encoding consumes capacity).
- **Zhang & Nanda 2024 (Best Practices of Activation Patching, arXiv:2309.16042).** The
  practical detection toolkit (Def.4). Key methodological results for this project:
  prefer **Symmetric Token Replacement** over Gaussian Noising (which pushes activations
  **OOD and can break internal mechanisms**, misattributing effects); prefer **logit
  difference** over probability (probability is *blind to negative/suppressing
  components*); sliding-window patching inflates localization via joint effects. Explicit
  **self-repair / backup-behavior confound** for ablation studies.
- **Geiger et al. 2025 (Causal Abstraction, arXiv:2301.04709).** The unifying theory:
  activation/path patching = (recursive) interchange interventions; ablation/causal
  scrubbing = abstraction by a 3-variable collider; SAE/DAS/probing = (approximate)
  bijective translations for **modular features**. **IIA** (Def.12) is the quantitative
  criterion that a distributed subspace *realizes* an aligned-behavior variable —
  agnostic to the linear-representation hypothesis. Distributed interchange
  interventions formalize intervening on superposed pathways.

---

### Known Results (prerequisite theorems we can cite or build on)

| # | Result | Source | Statement (summary) | Use for our work |
|---|--------|--------|---------------------|------------------|
| R1 | Backup-head redundancy | Wang 2022 | Ablating all 3 Name Movers → only 5% logit-diff drop (backups activate) | Empirical anchor for "redundant pathways preserve behavior" |
| R2 | Compensatory-effect metric `CE` | McGrath 2023 Eq.33 | Downstream ΔDE sum quantifies rerouting; ≈70% restoration, r²=0.92 | Template for the project's *compensation measure* |
| R3 | Self-repair decomposition | Rushing–Nanda 2024 | `self-repair = Δlogit − ΔDE`; ~30% is LayerNorm artifact | Forces a *mechanism-corrected* compensation measure |
| R4 | I_min redundancy & PID atoms | Williams–Beer 2010 | `I_min`, nonnegative Unq/Rdn/Syn decomposition on the redundancy lattice | Core information-theoretic formalism for redundancy |
| R5 | Redundancy↔robustness; dropout↑redundancy | Proca 2022 | Per-layer PID; lesioning synergistic neurons hurts most | Operational law linking measure → robustness |
| R6 | Degeneracy ⇒ distributed robustness + evolvability | Whitacre–Bender 2009 | Degeneracy (≠ redundancy) gives networked buffering; ~20× evolvability | Sharpens target quantity; graph-theoretic measures |
| R7 | Multi-direction / concept-cone refusal | Wollschläger 2025 | Cones up to dim 5; compositional ablation k≥4 beats 1-D | Aligned behavior is *redundantly* represented |
| R8 | Redundant paths survive ablation | Joad 2026 | Llama survives safety-ablation; shared SAE core + tail | Direct evidence of compensation in alignment |
| R9 | Single direction = 1-neuron SOM limit | Piras 2025 Prop.1 | Difference-in-means is the degenerate case of a manifold | Unifies 1-D vs multi-D views |
| R10 | Shallow safety alignment | Qi 2024 | Alignment concentrated in first few output tokens (per-token KL) | The fragility/single-point-of-failure axis |
| R11 | Alignment removable in ~5 steps | Qi 2023 | <100 examples flip safety; benign FT regresses too | "Destroy not reroute" baseline to contrast |
| R12 | Gated-not-removed; ablation 58× weak at scale | Frank 2026 | Routing redundancy; interchange needed to detect it | Method caveat + distributed-redundancy evidence |
| R13 | Compute-in-superposition capacity Õ(d²); error correction | Hänni 2024 | >n features computed in n dims; round-rectify robustness | Mechanistic *source* of representational redundancy |
| R14 | Superposition measure ψ=e^{H(p)}/N; adversarial training non-monotone | Bereska 2025 | Abundance vs scarcity regimes | Capacity-dependent redundancy under adversarial training |
| R15 | Patching = interchange intervention; IIA criterion | Geiger 2025 | Causal-abstraction unification; IIA realization test | Rigorous detection/verification of pathways |
| R16 | STR > GN; logit-diff > probability; OOD/self-repair confounds | Zhang–Nanda 2024 | Patching best practices | Guards our ablation methodology |

---

### Proof Techniques in the Literature

- **Causal mediation / SCM analysis** (McGrath 2023; Geiger 2025): treat the network as
  a structural causal model; define total/direct/indirect effects via `do`-interventions.
  *Applicable* to defining compensation rigorously as an indirect-effect / rerouting term.
- **Activation & path patching / interchange interventions** (Wang 2022; Zhang–Nanda
  2024; Geiger 2025): the empirical detection method; recursive (path) patching isolates
  direct effect and is needed to find backups.
- **Difference-in-means + directional ablation/projection** (Arditi 2024; all of
  Cluster 2): the linear-algebraic primitive for extracting and removing an
  aligned-behavior direction; generalized to cones (projected-gradient + Gram–Schmidt,
  Wollschläger) and manifolds (SOM/SGD, Piras Prop.1).
- **Partial Information Decomposition** (Williams–Beer 2010; Proca 2022): lattice-Möbius
  construction; `I_min`/MMI/GCMI estimators; 2nd-order pairwise averaging to tame the
  Dedekind-number atom explosion.
- **Neutral-network / networked-buffering graph analysis** (Whitacre–Bender 2009):
  measure distributed robustness combinatorially over the graph of fitness-neutral
  configurations.
- **SAE entropy / Hill-number diversity** (Bereska 2025) and **ε-linear representation
  capacity bounds** (Hänni 2024): quantify and bound representational redundancy.

---

### Related Open Problems

- **No canonical compensation measure.** McGrath's `CE` is logit-level and
  layer-granular; Rushing–Nanda show ~30% is a LayerNorm artifact. A
  *mechanism-corrected, head/feature-level* compensation measure is open.
- **Reroute vs destroy under training.** Qi 2023/2024 (destroy) vs Frank 2026 (gated,
  reroute, relocate). *Under what capacity/objective conditions does adversarial
  training reroute alignment rather than erase it?* Bereska's abundance/scarcity
  bifurcation is a candidate predictor — untested for alignment.
- **Redundancy vs degeneracy in LLMs.** Whitacre–Bender argue degeneracy (distinct
  pathways) — not identical redundancy — gives distributed robustness. *Which do refusal
  cones / SOM manifolds instantiate, and how to measure it information-theoretically*
  (Tononi–Sporns–Edelman degeneracy measure) is open.
- **Detectability limits of ablation.** Frank's 58× underestimate and Hänni's
  "unused features" caveat mean ablation can both under- and over-state a pathway's role
  when backups/superposition are present. A *redundancy-aware ablation estimator* is open.
- **Dimensionality of the alignment subspace.** Cones up to dim 5 (Wollschläger),
  11-category distinct-but-collapsing directions (Joad), non-orthogonal manifolds
  (Piras). *The true dimension/structure of the refusal representation and its scaling
  with model size* is unresolved.

---

### Gaps and Opportunities

- **Gap 1 — A unifying scalar.** No measure currently maps both "shallow/fragile" and
  "distributed/redundant" findings onto one axis. **Opportunity:** define
  *compensatory capacity* of an aligned behavior `S` as the redundancy atom
  `Rdn(S; pathways)` (Def.8) — predicting robustness to single-pathway ablation — and
  *synergy* as its single-point-of-failure complement; reconcile with McGrath's `CE`.
- **Gap 2 — Mechanism-corrected compensation.** Subtract the LayerNorm-scaling and
  OOD/self-repair artifacts (Rushing–Nanda; Zhang–Nanda) to isolate *genuine* rerouting.
- **Gap 3 — Dynamics under training.** Model how `Rdn(S; pathways)` evolves under
  adversarial fine-tuning (does redundancy buffer alignment, à la Whitacre buffering, or
  collapse, à la Qi?). Bereska's capacity regimes give a testable predictor.
- **Gap 4 — From identical redundancy to degeneracy.** Port Whitacre–Bender's
  graph/buffering measures and Tononi–Sporns–Edelman degeneracy to transformer pathways.

---

### Recommendations for Proof Strategy

- **Recommended formal object.** Model an aligned behavior as a target variable `S`
  (e.g. refusal logit-difference, or Geiger's high-level "refuse" variable) and a set of
  *pathways* `R_1,…,R_n` (heads, directions, layers, or SAE-latent groups). Define:
  - **Compensatory capacity** `C(S) := Rdn(S; R_1,…,R_n)` (PID redundancy atom, Def.8) —
    information about aligned behavior simultaneously available in multiple pathways.
  - **Fragility** `Φ(S) := Syn(S; ·) + Unq` carried by any single indispensable pathway.
  - Link to the **causal compensation** `CE` (Def.7): conjecture/prove that high `Rdn`
    ⇒ small total-effect of single-pathway ablation (because `I_min ≤ I(S;R_i)` and
    backups retain the shared information), i.e. an inequality bounding ablation effect
    by the *non-redundant* information.
- **Key lemmas to establish.**
  1. *Ablation-effect bound:* the total effect of ablating pathway `R_i` is bounded by
     its **unique** information `Unq(S;R_i)` (the part no other pathway holds) — making
     redundant alignment provably robust to single ablation; relate to McGrath `CE`.
  2. *Rerouting realizability (Geiger IIA):* if two pathways are representationally
     independent (Def.11) and each realizes `S` with high IIA, the surviving one
     preserves `S` under the other's ablation (formalizes Joad's "redundant paths
     survived").
  3. *Capacity condition for redundancy growth under training* (from Hänni capacity
     `Õ(d²)` + Bereska abundance/scarcity): conditions under which adversarial training
     *adds* redundant alignment pathways rather than erasing them.
  4. *Degeneracy ⇒ distributed buffering:* port Whitacre–Bender networked-buffering to
     bound multi-pathway lesion robustness by a graph-connectivity quantity.
- **Potential obstacles.** (a) `I_min` over-reports redundancy (use BROJA/MMI as cross-
  checks; `dit` supports all). (b) PID atom count is super-exponential — restrict to
  2nd-order pairwise averaging (Proca). (c) Ablation OOD/self-repair confounds (Def.6,
  R16) — use STR + logit-diff + LayerNorm-freezing/path-patching. (d) Linear-
  representation assumption may fail — Geiger's framework and distributed interchange
  interventions are explicitly non-linear-friendly.
- **Computational support.** `dit` for PID atoms (scaffold in
  `code/pid_redundancy_demo.py`, verified: redundant→Rdn=1.0, XOR→Syn=1.0); `sympy` for
  symbolic verification of `CE` / LayerNorm algebra; `networkx` for buffering/neutral-
  network measures; (optional) `transformer_lens`/`sae_lens` to instantiate ablation
  experiments on GPT-2/Pythia for empirical validation of the proved bounds.
