# Computational Tools

Tooling assembled for the **Neural Redundancy Framework** research. Everything runs
in the project `.venv` (managed by `uv`; see `../pyproject.toml`). No external
repositories were required — the research is theory/modeling-driven, so the tools
are general-purpose information-theory, symbolic, and numerical libraries plus two
local helper scripts.

## Python packages (in `.venv`)

| Package | Purpose | How it serves the research |
|---------|---------|----------------------------|
| `dit` | Discrete information theory | Partial Information Decomposition (Williams–Beer `I_min`, MMI, BROJA); compute redundancy/synergy/unique atoms over pathway variables |
| `numpy`, `scipy` | Numerical computation | Activation algebra, covariance/projection operators, optimization, statistics |
| `sympy` | Symbolic computation | Symbolic verification of compensatory-effect / direct-effect algebra (e.g. the Hydra `CE`, LayerNorm `Δlogit = (S/S'−1)logit + (S/S')ΔDE`) |
| `networkx` | Graph theory | Neutral-network / "networked buffering" (Whitacre–Bender) and pathway-graph robustness measures |
| `requests`, `httpx` | HTTP | Paper download (`download_papers.py`), arXiv API |
| `pypdf` | PDF handling | Chunking PDFs for reading (`.claude/skills/paper-finder/scripts/pdf_chunker.py`) |

## Local helper scripts

### `search_arxiv.py`
Queries the arXiv API and prints relevance-ranked title/authors/abstract.
Usage: `python code/search_arxiv.py "query string" [max_results]`.
Used for the literature search (paper-finder service was unavailable).

### `download_papers.py`
Downloads the curated arXiv PDF set into `../papers/` with descriptive filenames.
Idempotent (skips existing valid PDFs). Edit the `PAPERS` list to add more.

### `pid_redundancy_demo.py`
**Computational scaffold for the framework's core formalism.** Builds joint
distributions `P(R1,R2,S)` over two pathways `R1,R2` and an aligned-behavior target
`S`, then computes the Williams–Beer PID. Demonstrates the three regimes the
proof-construction phase will reason about:

- **Fully redundant** (`R1=R2=S`): redundancy atom `π{0}{1}=1.0`, synergy `0` →
  robust to single-pathway ablation (compensation possible).
- **Synergistic** (`S=R1 XOR R2`): synergy atom `π{0:1}=1.0`, redundancy `0` →
  single point of failure, no compensation.
- **Partially redundant / degenerate** (noisy agreement, ε=0.1): redundancy atom
  `≈0.531` → partial compensation (the Whitacre–Bender "degeneracy" regime).

Run: `python code/pid_redundancy_demo.py` (verified working — see output in
`../resources.md`).

## Notes on heavier tooling (deferred, optional for proof phase)

The hypothesis is *detectable via ablation studies* on real LLMs. If the
proof/experiment phase wants to instantiate the formalism empirically, the natural
additions are `transformer_lens` (activation/path patching on GPT-2/Pythia),
`sae_lens` (sparse-autoencoder latents for the Joad-style core/tail analysis), and
`nnsight`. These are **not** installed here to keep the resource-finding environment
light; install with `uv add transformer_lens sae_lens` when needed. SageMath was not
required (no heavy symbolic-geometry computation in scope).
