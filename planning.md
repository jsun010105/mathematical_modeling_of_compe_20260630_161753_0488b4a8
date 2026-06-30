# Planning: Mathematical Modeling of Compensatory Robustness in LLM Alignment

## Motivation & Novelty Assessment

### Why This Research Matters
Whether an LLM's aligned behavior is *redundantly* encoded determines whether safety
survives degradation: random damage, quantization, benign fine-tuning, or adversarial
attack. The empirical literature is sharply split — alignment is reported as both
**fragile** (removable in ~5 gradient steps; concentrated in the first output tokens;
a single direction) and **distributed/redundant** (backup name-mover heads, refusal
concept cones, "redundant paths survived ablation"). A *mathematical* account that
makes both true as special cases would tell practitioners exactly which kind of
robustness a model has and which it lacks.

### Gap in Existing Work (from literature_review.md)
- **No unifying scalar** maps "shallow/fragile" and "distributed/redundant" onto one
  axis (Gap 1).
- McGrath's compensatory-effect `CE` is logit-level and ~30% LayerNorm artifact
  (Gap 2); no clean, assumption-light bound linking a redundancy measure to the
  *behavioral* effect of ablation exists.
- The proposed methodology (compensation index `C(M)`, redundancy `R(G)=`min vertex
  cut, percolation phase transition) is stated but **not formalized or proven**.

### Our Novel Contribution
A self-contained, theorem-grade framework with three linked layers, each *proved* and
*computationally verified*:
1. **Information layer.** An exact identity: the information lost by ablating a pathway
   equals a conditional mutual information (its non-redundant content). Corollary:
   fully redundant pathways are ablation-invariant. A Fano bound converts this into a
   guaranteed floor on behavioral fidelity ρ.
2. **Graph layer.** Modeling computation as a layered DAG, the **targeted** critical
   ablation threshold equals the minimum vertex cut `R(G)` (Menger), while the
   **random** threshold is governed by percolation and is exponentially larger in the
   bundle width. This *is* the unifying scalar: small min-cut ⇒ fragile to attack;
   large random-percolation threshold ⇒ robust to incidental damage.
3. **Reconciliation.** The framework predicts the user's premise ("can't misalign
   LLMs") holds for *random/benign* degradation but **fails** for *targeted* adversarial
   training — exactly matching Qi 2023/2024 (fragile to targeted) vs Wang 2022 / Joad
   2026 (robust to incidental). We report this honestly: the premise as stated is not
   supported; the *correct* statement is regime-dependent.

### Experiment (computational-verification) Justification
- **V1 — MI ablation identity & Fano floor:** verify `ΔI_i = I(S;R_i|R_{-i})` and the
  behavioral-fidelity floor on synthetic joint distributions (redundant, unique,
  synergistic/XOR). Confirms Theorems A1–A2, B1.
- **V2 — Menger vs percolation:** build series-of-parallel-bundles DAGs in `networkx`,
  compute min vertex cut, then Monte-Carlo random ablation vs greedy targeted ablation;
  verify targeted threshold `= R(G)` and the random phase-transition curve
  `[1-q^w]^d` with its predicted location/width. Confirms Theorems C1–C3.
- **V3 — PID cross-check:** use `dit` to confirm the redundant/unique/synergy
  decomposition matches the ablation losses (guards against `I_min` over-reporting).

## Research Question
Can the compensatory robustness of LLM alignment be given a rigorous mathematical
characterization in which (a) a redundancy measure provably bounds the effect of
ablating an internal pathway, and (b) the critical ablation threshold exhibits a
percolation-type phase transition separating robustness-to-random-damage from
fragility-to-targeted-attack?

## Hypothesis Decomposition
- **H1 (information):** the behavioral/informational cost of ablating a pathway is its
  *non-redundant* (conditional) information; redundant pathways cost nothing.
- **H2 (graph):** the worst-case (targeted) critical ablation threshold equals the
  network's minimum vertex cut `R(G)`.
- **H3 (percolation):** under random ablation the survival probability has a sharp
  threshold, exponentially separated from the targeted threshold in the redundancy
  width — reconciling fragile vs robust reports.

## Proposed Methodology
### Approach
Pure-math formalization (definitions → lemmas → theorems), with every nontrivial
claim cross-checked by an independent computational experiment. We use only standard,
non-contested tools where possible: Shannon mutual information and the chain rule
(rather than a specific contested PID operator) for the main bounds, Fano's inequality
for the information→behavior link, and Menger's theorem + exact percolation algebra for
the graph layer. PID (`dit`) is used only as a *cross-check / interpretation*.

### Experimental Steps
1. State definitions (behavior `S`, pathways, ablation, compensation index `C`,
   redundancy `R(G)`, robustness `ρ`). — rationale: precision first.
2. Prove information-layer theorems (A1 identity, A2 k-redundancy, B1 Fano floor).
3. Prove graph-layer theorems (C1 Menger threshold, C2 exact survival law, C3
   random-vs-targeted exponential separation / phase transition).
4. Verify V1–V3 computationally; record numbers.
5. Discuss reconciliation with empirical literature; honest assessment of premise.

### Baselines / comparisons
McGrath `CE` (logit-level compensation), Arditi single-direction (1-D special case),
Qi fragility results, Wang backup heads. We compare *predictions* of our bounds to
these reported phenomena.

### Evaluation Metrics
Correctness of each proof (independent step-check + computational confirmation);
agreement between predicted thresholds and Monte-Carlo estimates (target: predicted
phase-transition location within Monte-Carlo error; identity ΔI=I(S;R_i|R_{-i}) to
floating-point tolerance).

### Statistical Analysis Plan
Monte-Carlo survival probabilities with N≥20000 trials per point; report mean ± Wald
95% CI; compare to closed-form `[1-q^w]^d`. Identities checked to tol 1e-9.

## Expected Outcomes
ΔI identity holds exactly; Fano floor never violated; targeted threshold matches min
vertex cut exactly; random survival matches closed form within CI; transition sharpens
with width `w`. These would support H1–H3.

## Timeline (≈60 min)
- Setup + definitions: ~8 min
- Proofs (information + graph): ~20 min
- Verification scripts + runs: ~18 min
- Report + README + validation: ~14 min

## Potential Challenges
- PID `I_min` over-reporting → avoid in main theorems (use Shannon MI), use only as
  cross-check. - Percolation exact solvability → restrict to series-of-parallel-bundles
  (exactly solvable) rather than general directed percolation. - Honesty about premise
  → state clearly that targeted misalignment is *easy* (Qi), so the framework's value
  is delineating regimes, not validating an over-strong premise.

## Success Criteria
All stated theorems proved with complete logical steps and confirmed by computation;
REPORT.md delivers the unifying scalar and the reconciliation, with limitations stated.
