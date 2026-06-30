"""Figures: (1) percolation survival Theta(q)=(1-q^w)^d sharpening with width w;
(2) random-vs-targeted critical-count separation. Saves to figures/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
q = np.linspace(0, 1, 400)
d = 8

# Fig 1: phase transition sharpening
fig, ax = plt.subplots(figsize=(7, 5))
for w in [1, 2, 4, 8, 16, 32]:
    ax.plot(q, (1 - q ** w) ** d, label=f"w={w}")
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set_xlabel("random deletion probability  q")
ax.set_ylabel(r"survival probability  $\Theta(q)=(1-q^w)^d$")
ax.set_title(f"Percolation phase transition in B(w,d), d={d}\n(sharpens as bundle width w grows)")
ax.legend(title="bundle width")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figures/fig1_phase_transition.png", dpi=130)

# Fig 2: random vs targeted critical count
ws = np.arange(1, 33)
qc = (1 - 2 ** (-1.0 / d)) ** (1.0 / ws)
rand_count = qc * ws * d
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ws, rand_count, "o-", label="random critical count  $q_c\\,wd$")
ax.plot(ws, ws, "s--", label="targeted critical count  $R(G)=w$")
ax.set_xlabel("bundle width  w  (= redundancy R(G))")
ax.set_ylabel("vertices that must be deleted to misalign")
ax.set_title(f"Fragile to attack, robust to noise (d={d})\nrandom requires far more deletions than targeted")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figures/fig2_random_vs_targeted.png", dpi=130)
print("Saved figures/fig1_phase_transition.png and figures/fig2_random_vs_targeted.png")
