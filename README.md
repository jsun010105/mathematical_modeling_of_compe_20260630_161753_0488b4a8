# Compensatory Robustness in LLM Alignment — A Neural-Redundancy Framework

A theorem-grade mathematical framework for the conjecture that LLM alignment is
preserved by **redundant internal pathways**. It unifies the field's contradictory
reports (alignment as both *fragile* and *robust*) and is fully verified on CPU.

## Key results (all proved + computationally verified)
- **Theorem A1.** Information lost by ablating a pathway = its non-redundant conditional
  MI, `ΔI_i = I(S; R_i | R_{-i})`. Redundant pathways cost **zero** (artifact-free
  replacement for McGrath's compensatory-effect `CE`). *Identity verified to `0.0e0`.*
- **Theorem B1 (Fano).** This loss lower-bounds post-ablation behavioral error ⇒ a
  provable redundancy⇒robustness law. *No violations.*
- **Theorems C1–C3.** Computation-as-DAG: **targeted** critical ablation threshold =
  minimum vertex cut `R(G)` (Menger); **random** survival `Θ(q)=(1−q^w)^d` undergoes a
  percolation phase transition with `q_c→1`, exponentially separated from `R(G)`.
  *Min-cut matched exactly; survival law within 95% CI (max dev 0.0033, N=40000).*
- **Proposition D1.** Observed compensation ⇒ redundancy *pre-existed*; rerouting
  reveals, not creates, robustness.
- **Corollary E1.** Misaligning costs ≥ `R(G)` pathway-units ⇒ the "can't misalign LLMs"
  premise holds for random/benign damage but **fails** for targeted attacks (matches Qi
  2023/2024). Honest, regime-dependent verdict.

## Reproduce (CPU, ~seconds)
```bash
source .venv/bin/activate
python src/verify_information_layer.py   # Theorems A1, A2, B1; PID cross-check (dit)
python src/verify_graph_layer.py         # Theorems C1, C2, C3 (Menger + Monte-Carlo)
python src/make_figures.py               # figures/fig1_*.png, fig2_*.png
```

## Files
- `REPORT.md` — full report (executive summary, definitions, **complete proofs**,
  verification tables, discussion, limitations, open questions). **Primary deliverable.**
- `results/theorems_and_proofs.md` — canonical proofs with inline verification notes.
- `definitions.md` — precise definitions/notation. `planning.md` — plan & novelty.
- `src/` — verification scripts. `results/` — raw run logs. `figures/` — plots.
- `literature_review.md`, `resources.md` — pre-gathered background (17 papers).

See `REPORT.md` for full details.
