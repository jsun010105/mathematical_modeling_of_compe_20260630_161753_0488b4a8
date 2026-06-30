# Resources Catalog

## Summary
Resources gathered for **"Mathematical Modeling of Compensatory Robustness in LLM
Alignment: A Neural Redundancy Framework."** The research specification listed **no**
user-specified `code_references`, `papers`, or `resources`, so all items below were
discovered through systematic literature search. 17 papers were downloaded and read
in full; an information-theoretic computational scaffold was built and verified.

## Papers
Total papers downloaded: **17** (all read in full). See `papers/README.md` for grouped
descriptions and `literature_review.md` for precise extractions.

| Title | Authors | Year | File | Key results |
|-------|---------|------|------|-------------|
| The Hydra Effect: Emergent Self-repair | McGrath et al. | 2023 | papers/2307.15771_*.pdf | Compensatory-effect metric `CE`; ≈70% logit restoration, r²=0.92 (no dropout) |
| Explorations of Self-Repair | Rushing & Nanda | 2024 | papers/2402.15390_*.pdf | `self-repair = Δlogit−ΔDE`; ~30% is LayerNorm artifact |
| Interpretability in the Wild (IOI circuit) | Wang et al. | 2022 | papers/2211.00593_*.pdf | Backup Name Mover Heads: ablate 3 movers → only 5% drop |
| Refusal Is Mediated by a Single Direction | Arditi et al. | 2024 | papers/2406.11717_*.pdf | Difference-in-means direction; directional ablation; 1-D baseline |
| Geometry of Refusal: Concept Cones | Wollschläger et al. | 2025 | papers/2502.17420_*.pdf | Refusal cones up to dim 5; representational independence |
| More to Refusal than a Single Direction | Joad et al. | 2026 | papers/2602.02132_*.pdf | 11 categories; shared SAE core + tail; "redundant paths survived ablation" |
| SOM Directions are Better than One | Piras et al. | 2025 | papers/2511.08379_*.pdf | Refusal manifold; Prop.1 (1-direction = 1-neuron SOM limit) |
| Safety More Than a Few Tokens Deep | Qi et al. | 2024 | papers/2406.05946_*.pdf | Shallow safety alignment (first-token KL); depth-redundancy fix |
| Fine-tuning Compromises Safety | Qi et al. | 2023 | papers/2310.03693_*.pdf | ~5–10 gradient steps remove alignment ("destroy not reroute") |
| How Alignment Routes (policy circuits) | Frank | 2026 | papers/2604.04385_*.pdf | Gate/amplifier routing; partial redundancy; ablation 58× weak at scale |
| Nonnegative Decomposition of Multivariate Info (PID) | Williams & Beer | 2010 | papers/1004.2515_*.pdf | `I_min`; Unq/Rdn/Syn atoms; redundancy lattice |
| Synergistic Information in Neural Networks | Proca et al. | 2022 | papers/2210.02996_*.pdf | Per-layer PID; redundancy↔robustness; dropout↑redundancy |
| Degeneracy: Robustness and Evolvability | Whitacre & Bender | 2009 | papers/0907.0510_*.pdf | Degeneracy vs redundancy; networked buffering; ~20× evolvability |
| Computation in Superposition | Hänni et al. | 2024 | papers/2408.05451_*.pdf | Compute >n features in n dims; Õ(d²) capacity; error-correction robustness |
| Superposition as Lossy Compression | Bereska et al. | 2025 | papers/2512.13568_*.pdf | ψ=e^{H(p)}/N measure; adversarial training non-monotone (abundance/scarcity) |
| Best Practices of Activation Patching | Zhang & Nanda | 2023 | papers/2309.16042_*.pdf | STR>GN, logit-diff>probability; OOD/self-repair confounds |
| Causal Abstraction (foundation for MI) | Geiger et al. | 2023 | papers/2301.04709_*.pdf | Interchange interventions; IIA realization criterion |

## Prior Results Catalog
Key theorems/results available for our proofs (full statements & sources in
`literature_review.md`, "Known Results" table R1–R16):

| Result | Source | Statement summary | Used for |
|--------|--------|-------------------|----------|
| Backup-head redundancy (R1) | Wang 2022 | Ablate 3 Name Movers → 5% drop | Empirical anchor for redundant pathways |
| Compensatory-effect `CE` (R2) | McGrath 2023 | Downstream ΔDE quantifies rerouting | Template for compensation measure |
| `I_min` & PID atoms (R4) | Williams–Beer 2010 | Nonnegative Unq/Rdn/Syn decomposition | Core redundancy formalism |
| Redundancy↔robustness (R5) | Proca 2022 | Lesion synergistic neurons hurts most | Measure→robustness law |
| Degeneracy⇒distributed robustness (R6) | Whitacre 2009 | Networked buffering | Sharpen target quantity |
| Concept cones (R7) | Wollschläger 2025 | Cones dim≤5; compositional ablation | Aligned behavior redundantly represented |
| Single dir = SOM limit (R9) | Piras 2025 | Prop.1 | Unify 1-D vs multi-D |
| Patching = interchange; IIA (R15) | Geiger 2025 | Realization criterion | Rigorous pathway detection |

## Computational Tools

| Tool | Purpose | Location | Notes |
|------|---------|----------|-------|
| `dit` | Partial Information Decomposition (I_min/MMI/BROJA) | `.venv` (uv) | Computes redundancy/synergy/unique atoms |
| `numpy`, `scipy` | Numerical linear algebra / optimization | `.venv` | Activation algebra, projections, stats |
| `sympy` | Symbolic verification | `.venv` | `CE` / LayerNorm `Δlogit` algebra |
| `networkx` | Graph theory | `.venv` | Networked-buffering / neutral-network measures |
| `pid_redundancy_demo.py` | Framework scaffold (verified) | `code/` | Redundant→Rdn=1.0, XOR→Syn=1.0, partial→Rdn≈0.531 |
| `search_arxiv.py` | arXiv literature search | `code/` | Used (paper-finder service was down) |
| `download_papers.py` | Paper downloader | `code/` | Idempotent; fetched all 17 PDFs |

**Verified scaffold output** (`python code/pid_redundancy_demo.py`):
- Fully redundant pathways `R1=R2=S` → redundancy atom `π{0}{1}=1.0`, synergy `0`.
- Synergistic `S=R1 XOR R2` → synergy atom `π{0:1}=1.0`, redundancy `0`.
- Partially redundant (ε=0.1) → redundancy atom `≈0.531`.

No external repositories were cloned (none specified by the user; the work is
theory-driven). Optional heavier tooling (`transformer_lens`, `sae_lens`, `nnsight`)
is noted in `code/README.md` for an empirical-validation phase.

## Resource Gathering Notes

### Search Strategy
The paper-finder service at `localhost:8000` was **unavailable** (fallback returned
empty results, even after installing `httpx`). I therefore searched the **arXiv API
directly** via a custom helper (`code/search_arxiv.py`), running ~11 targeted queries
across the five subfields: self-repair/hydra/backup heads; refusal directions;
shallow safety / fine-tuning attacks; partial information decomposition; superposition;
degeneracy/biological redundancy; activation/path patching & causal abstraction; IOI
backup heads; and distributed/safety circuits. A generic combined query returned mostly
irrelevant computer-vision "compensation/redundancy" papers, confirming that
subfield-specific queries were necessary.

### Selection Criteria
Papers were chosen to cover each ingredient of the hypothesis with foundational +
recent work: the *phenomenon* (self-repair/backup pathways), the *aligned behavior*
(refusal geometry, single vs redundant directions), the *training-robustness* axis
(shallow alignment, fine-tuning attacks, routing), and the *mathematics*
(PID/redundancy/synergy, biological degeneracy, superposition capacity, causal-
abstraction methodology). Priority on precise, citable formal statements over breadth.

### Challenges Encountered
- Paper-finder service down → built a direct arXiv-API client.
- `pypdf`/`hatchling` build error in the initial `pyproject.toml` (src-layout
  requirement) → removed the build-system section; `uv add` then worked.
- All 17 PDFs downloaded cleanly (verified `%PDF` magic + size); no paywalls (all on
  arXiv). Deep reading of all 17 done in full by five parallel reading agents.

## Recommendations for Proof Construction

1. **Proof strategy.** Define an aligned behavior as a target `S` and pathways
   `R_1,…,R_n`; set *compensatory capacity* `C(S) = Rdn(S;·)` (PID redundancy atom)
   and *fragility* via synergy/unique. Prove an **ablation-effect bound**: the total
   effect of ablating `R_i` is controlled by its *unique* information `Unq(S;R_i)`,
   making redundant alignment provably robust to single-pathway ablation; connect to
   McGrath's `CE`. Use Geiger's **IIA** to formalize "the surviving pathway still
   realizes `S`" (rerouting).
2. **Key prerequisites to cite.** Williams–Beer PID (R4), Proca redundancy↔robustness
   (R5), McGrath `CE` (R2), Geiger interchange/IIA (R15), Whitacre degeneracy (R6),
   Hänni capacity + Bereska regimes (R13/R14) for the under-training dynamics.
3. **Computational tools.** `dit` (PID atoms, cross-check `I_min` vs BROJA/MMI), `sympy`
   (symbolic `CE`/LayerNorm checks), `networkx` (buffering); optionally
   `transformer_lens`/`sae_lens` for empirical ablation validation.
4. **Potential difficulties.** `I_min` over-reports redundancy (cross-check estimators);
   super-exponential PID atoms (use 2nd-order pairwise averaging); ablation OOD/self-
   repair confounds (STR + logit-diff + path-patching, freeze LayerNorm); linear-
   representation assumption may fail (use Geiger's non-linear distributed interventions).
