#!/usr/bin/env python3
"""
Computational scaffold for the Neural Redundancy framework.

Demonstrates the information-theoretic primitives the proof-construction phase
will use to formalize "redundant computational pathways preserve aligned behavior":

  * Partial Information Decomposition (PID, Williams & Beer 2010) of how much
    information two pathways (R1,R2) carry about an aligned-behavior target S,
    into Unique / Redundant / Synergistic atoms.
  * The redundancy measure as a lower bound on each pathway's information
    (I_min <= I(S;Ri)) — the formal sense in which pathways can substitute.

Run:  python code/pid_redundancy_demo.py
Requires: dit, numpy  (installed in the project .venv).
"""
import numpy as np
import dit
from dit.pid import PID_WB   # Williams & Beer I_min PID


def pid_of_distribution(dist, name):
    """Print the Williams-Beer PID of a joint distribution P(R1,R2,S).
    Convention: first two variables are sources (pathways), last is target S."""
    pid = PID_WB(dist, [[0], [1]], [2])
    print(f"\n=== {name} ===")
    print(pid)
    return pid


def build(probs):
    """probs: dict mapping 'r1r2s' string outcomes -> probability."""
    outcomes = list(probs.keys())
    pmf = np.array([probs[o] for o in outcomes], dtype=float)
    pmf = pmf / pmf.sum()
    return dit.Distribution(outcomes, pmf)


def demo_redundant():
    # Two pathways that REDUNDANTLY encode the aligned behavior S:
    # R1 = R2 = S  (each pathway alone determines aligned output).
    # Ablating either one leaves the other fully informative -> compensation.
    d = build({'000': 0.5, '111': 0.5})
    pid_of_distribution(d, "Fully redundant pathways (R1=R2=S): robust to single ablation")


def demo_unique():
    # Two pathways each carrying DISTINCT (unique) info, no redundancy:
    # S = (R1, R2) jointly; neither alone suffices -> single ablation destroys behavior.
    d = build({'000': 0.25, '011': 0.25, '101': 0.25, '110': 0.25})  # S = R1 XOR R2
    pid_of_distribution(d, "Synergistic pathways (S=R1 XOR R2): fragile single point of failure")


def demo_partial():
    # Degenerate / partially-redundant pathways (Whitacre & Bender 2009 sense):
    # S mostly agrees with each pathway but with noise -> partial compensation.
    eps = 0.1
    d = build({
        '000': 0.5 * (1 - eps), '001': 0.5 * eps,
        '111': 0.5 * (1 - eps), '110': 0.5 * eps,
    })
    pid_of_distribution(d, f"Partially redundant / degenerate pathways (eps={eps})")


if __name__ == "__main__":
    print("Neural Redundancy framework — information-theoretic primitives (Williams-Beer PID)")
    demo_redundant()
    demo_unique()
    demo_partial()
    print("\nInterpretation:")
    print("  Redundancy atom R{0}{1}  = alignment info available in BOTH pathways")
    print("                              = capacity for compensation under single ablation.")
    print("  Synergy atom    R{0:1}   = alignment info available only JOINTLY")
    print("                              = single-point-of-failure / non-compensable.")
    print("  Unique atoms    R{0},R{1}= pathway-specific alignment info.")
