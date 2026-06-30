# Downloaded Papers

Curated literature for **"Mathematical Modeling of Compensatory Robustness in LLM
Alignment: A Neural Redundancy Framework."** Papers are grouped by the role they
play in the redundancy framework. All PDFs were retrieved from arXiv.

---

## Cluster 1 — Self-repair / compensatory rerouting (the empirical core of the hypothesis)

1. **[The Hydra Effect: Emergent Self-repair in Language Model Computations](2307.15771_McGrath2023_hydra_effect_self_repair.pdf)**
   - Authors: T. McGrath, M. Rahtz, J. Kramár, V. Mikulik, S. Legg (Google DeepMind)
   - Year: 2023 — arXiv:2307.15771
   - Why relevant: Canonical formalization of compensatory rerouting. Defines the
     **compensatory-effect metric CE** (downstream attention + MLP increase their
     effect when an upstream layer is ablated). Causal SCM / direct-vs-total-effect
     framing. Shows ~70% logit restoration, r²=0.92 compensation at layer 23 of
     Chinchilla-7B trained **without dropout** (refutes dropout-causation).

2. **[Explorations of Self-Repair in Language Models](2402.15390_RushingNanda2024_explorations_self_repair.pdf)**
   - Authors: C. Rushing, N. Nanda
   - Year: 2024 (ICML) — arXiv:2402.15390
   - Why relevant: Generalizes self-repair to full pretraining distribution and to
     individual heads across 7 model families. Provides the formula
     `self-repair = Δlogit − ΔDE`, the LayerNorm decomposition, and a **deflationary
     caveat** (~30% of self-repair is a LayerNorm-scaling artifact). Iterative
     Inference Hypothesis as the rerouting mechanism.

3. **[Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](2211.00593_Wang2022_IOI_circuit_backup_name_movers.pdf)**
   - Authors: K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt (Redwood)
   - Year: 2022 (ICLR 2023) — arXiv:2211.00593
   - Why relevant: Origin of the **Backup Name Mover Head**. Ablating all 3 primary
     Name Movers drops logit difference only 5% because backup heads activate.
     Path-patching methodology; faithfulness/completeness/minimality circuit criteria.

## Cluster 2 — Refusal direction(s): redundancy of the alignment representation

4. **[Refusal in Language Models Is Mediated by a Single Direction](2406.11717_Arditi2024_refusal_single_direction.pdf)**
   - Authors: A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, N. Nanda
   - Year: 2024 (NeurIPS) — arXiv:2406.11717
   - Why relevant: The **single-direction baseline**. Difference-in-means refusal
     direction; directional ablation `a' = a − (r̂·a)r̂`; weight orthogonalization.

5. **[The Geometry of Refusal in LLMs: Concept Cones and Representational Independence](2502.17420_Wollschlager2025_geometry_refusal_concept_cones.pdf)**
   - Authors: T. Wollschläger, J. Elstner, S. Geisler, V. Cohen-Addad, S. Günnemann, J. Gasteiger
   - Year: 2025 (ICML) — arXiv:2502.17420
   - Why relevant: Refutes 1-D view. Multiple **representationally-independent**
     directions and polyhedral **concept cones up to dimension 5**. Compositional
     ablation outperforms single-direction → additive redundant mechanisms.

6. **[There Is More to Refusal in LLMs than a Single Direction](2602.02132_Joad2026_more_to_refusal_than_single_direction.pdf)**
   - Authors: F. Joad, M. Hawasly, S. Boughorbel, N. Durrani, H. T. Sencar (QCRI)
   - Year: 2026 — arXiv:2602.02132
   - Why relevant: 11 refusal categories with geometrically distinct directions over
     a **small shared SAE latent core (591/517/421 latents) + long tail**. Explicitly
     attributes a model's survival of ablation to "redundant paths that survived."

7. **[SOM Directions are Better than One: Multi-Directional Refusal Suppression](2511.08379_Piras2025_SOM_directions_multidirectional_refusal.pdf)**
   - Authors: G. Piras, R. Mura, F. Brau, L. Oneto, F. Roli, B. Biggio (Cagliari/Genova)
   - Year: 2025 — arXiv:2511.08379
   - Why relevant: Refusal as a **low-dimensional manifold of multiple non-orthogonal
     directions** (SOM). Proposition 1: difference-in-means is the 1-neuron SOM limit.
     ASR rises monotonically with #directions ablated; partially reverses circuit-breaker defenses.

## Cluster 3 — Alignment robustness under (adversarial) training: where alignment lives

8. **[Safety Alignment Should Be Made More Than Just a Few Tokens Deep](2406.05946_Qi2024_safety_more_than_few_tokens_deep.pdf)**
   - Authors: X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, P. Henderson
   - Year: 2024 — arXiv:2406.05946
   - Why relevant: "**Shallow safety alignment**" — alignment concentrated in the
     first few output tokens (per-token KL). The token-depth single-point-of-failure;
     data-augmentation builds depth-redundancy; token-wise constrained FT objective.

9. **[Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](2310.03693_Qi2023_finetuning_compromises_safety.pdf)**
   - Authors: X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, P. Henderson
   - Year: 2023 — arXiv:2310.03693
   - Why relevant: Alignment is **not robust to further training** — ~5–10 gradient
     steps undo it (the "destroy, not reroute" baseline). Per-category fragility.

10. **[How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models](2604.04385_Frank2026_how_alignment_routes_policy_circuits.pdf)**
    - Authors: G. N. Frank
    - Year: 2026 (ICML MI Workshop) — arXiv:2604.04385
    - Why relevant: **Detect → Route → Output** gate/amplifier circuits; **partial
      redundancy** (counter-routing coalitions, gate relocation under continued
      training, capability gated-not-removed). Warns ablation underestimates importance
      58× at scale — interchange/patching required to detect redundant gates.

## Cluster 4 — Information-theoretic & biological formalism for redundancy

11. **[Nonnegative Decomposition of Multivariate Information](1004.2515_WilliamsBeer2010_nonnegative_decomposition_multivariate_info.pdf)**
    - Authors: P. L. Williams, R. D. Beer
    - Year: 2010 — arXiv:1004.2515
    - Why relevant: Founding **Partial Information Decomposition (PID)**. The `I_min`
      redundancy measure, the redundancy lattice, unique/redundant/synergistic atoms.

12. **[Synergistic Information Supports Modality Integration and Flexible Learning in Neural Networks](2210.02996_Proca2022_synergistic_information_neural_networks.pdf)**
    - Authors: A. M. Proca, F. E. Rosas, A. I. Luppi, D. Bor, M. Crosby, P. A. M. Mediano
    - Year: 2022 — arXiv:2210.02996
    - Why relevant: Operational template — computes redundancy/synergy per layer with
      `dit`/GCMI; shows **redundancy ↔ robustness to lesion** and that dropout raises
      redundancy. The concrete recipe (sources = neurons, target = next layer).

13. **[Degeneracy: A Design Principle for Achieving Robustness and Evolvability](0907.0510_WhitacreBender2009_degeneracy_robustness_evolvability.pdf)**
    - Authors: J. M. Whitacre, A. Bender
    - Year: 2009 (J. Theor. Biol.) — arXiv:0907.0510
    - Why relevant: The **biological-degeneracy** half of the analogy. Distinguishes
      *redundancy* (identical parts) from *degeneracy* (structurally distinct,
      partially-overlapping pathways) — the latter yields distributed robustness via
      "networked buffering." Graph-theoretic robustness/evolvability measures.

## Cluster 5 — Superposition (distributed codes) & causal-ablation methodology

14. **[Mathematical Models of Computation in Superposition](2408.05451_Hanni2024_math_models_computation_superposition.pdf)**
    - Authors: K. Hänni, J. Mendel, D. Vaintrob, L. Chan
    - Year: 2024 — arXiv:2408.05451
    - Why relevant: Math of how >n features/computations fit in n dimensions; capacity
      and error bounds — the source of representational redundancy.

15. **[Superposition as Lossy Compression: ... Connect to Adversarial Vulnerability](2512.13568_Bereska2025_superposition_lossy_compression_adversarial.pdf)**
    - Authors: L. Bereska, Z. Tzifa-Kratira, R. Samavi, E. Gavves
    - Year: 2025 — arXiv:2512.13568
    - Why relevant: Quantifies superposition via SAEs and links it to adversarial
      vulnerability/robustness — connects representational redundancy to robustness.

16. **[Towards Best Practices of Activation Patching in Language Models](2309.16042_ZhangNanda2023_best_practices_activation_patching.pdf)**
    - Authors: F. Zhang, N. Nanda
    - Year: 2023 — arXiv:2309.16042
    - Why relevant: The methodology our hypothesis relies on — activation/path patching,
      denoising vs noising, metric choice (logit diff vs KL), and the **backup-behavior
      caveat** that ablation effects can be masked by self-repair.

17. **[Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](2301.04709_Geiger2023_causal_abstraction_mech_interp.pdf)**
    - Authors: A. Geiger, D. Ibeling, A. Zur, M. Chaudhary, S. Chauhan, J. Huang, et al.
    - Year: 2023 — arXiv:2301.04709
    - Why relevant: Formal definition of causal abstraction, interchange interventions,
      and interchange-intervention accuracy — rigorous language for "a pathway realizes
      an aligned-behavior variable," and for redundancy as multiple-realizability.
