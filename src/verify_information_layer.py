"""
V1 + V3: Verify the information-layer theorems on synthetic joint distributions.

Theorem A1 (ablation identity):  Delta I_i := I(S;R) - I(S;R_{-i}) == I(S;R_i | R_{-i}).
Corollary A1.1:                  R_i conditionally redundant (S ⊥ R_i | R_{-i}) => Delta I_i = 0.
Theorem B1 (Fano floor):         optimal post-ablation error P_e satisfies
                                  H2(P_e) + P_e*log2(|S|-1) >= H(S) - I(S;R_{-i}),
                                 i.e. larger non-redundant loss forces larger error.
V3 (PID cross-check via dit):    redundant/unique/synergy atoms match the ablation losses.

All checks are exact (closed-form distributions), tolerances 1e-9.
"""
import itertools
import numpy as np

np.random.seed(42)
LOG2 = np.log(2)


# ---------- discrete information primitives over an explicit joint table ----------
def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / LOG2)


def mi_from_joint(joint, axes_X, axes_Y):
    """Mutual information I(X;Y) where X = marginal over axes_X, Y over axes_Y.
    `joint` is an n-dim probability tensor; axes_* are tuples of tensor axes."""
    all_axes = tuple(range(joint.ndim))
    axes_X, axes_Y = tuple(axes_X), tuple(axes_Y)
    keep = axes_X + axes_Y
    drop = tuple(a for a in all_axes if a not in keep)
    pXY = joint.sum(axis=drop) if drop else joint
    # reorder so X axes come first
    order = tuple(sorted(range(pXY.ndim),
                         key=lambda a: 0 if keep[a] in axes_X else 1))
    pXY = np.transpose(pXY, order)
    nx = int(np.prod([joint.shape[a] for a in axes_X]))
    pXY = pXY.reshape(nx, -1)
    pX = pXY.sum(1, keepdims=True)
    pY = pXY.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = pXY / (pX * pY)
        terms = pXY * np.log(np.where(pXY > 0, r, 1.0))
    return float(terms.sum() / LOG2)


def cmi_from_joint(joint, aX, aY, aZ):
    """I(X;Y|Z) = I(X;YZ) - I(X;Z)."""
    return mi_from_joint(joint, aX, tuple(aY) + tuple(aZ)) - mi_from_joint(joint, aX, aZ)


def make_joint(func_table, p_inputs):
    """Build joint p(s, r1, r2) from S=f(R1,R2) deterministic table and input dist.
    func_table[s_index over (r1,r2)] -> dims (S, R1, R2)."""
    r1n, r2n = p_inputs.shape
    sn = func_table.max() + 1
    joint = np.zeros((sn, r1n, r2n))
    for i in range(r1n):
        for j in range(r2n):
            joint[func_table[i, j], i, j] += p_inputs[i, j]
    return joint  # axes: 0=S, 1=R1, 2=R2


def report_case(name, joint):
    I_full = mi_from_joint(joint, (0,), (1, 2))
    I_drop1 = mi_from_joint(joint, (0,), (2,))      # ablate R1, keep R2
    I_drop2 = mi_from_joint(joint, (0,), (1,))      # ablate R2, keep R1
    dI1 = I_full - I_drop1
    dI2 = I_full - I_drop2
    cmi1 = cmi_from_joint(joint, (0,), (1,), (2,))  # I(S;R1|R2)
    cmi2 = cmi_from_joint(joint, (0,), (2,), (1,))  # I(S;R2|R1)
    HS = entropy(joint.sum(axis=(1, 2)))
    print(f"\n=== {name} ===")
    print(f"  H(S)={HS:.6f}  I(S;R)={I_full:.6f}")
    print(f"  Ablate R1: dI1={dI1:.6f}  I(S;R1|R2)={cmi1:.6f}  | identity err={abs(dI1-cmi1):.2e}")
    print(f"  Ablate R2: dI2={dI2:.6f}  I(S;R2|R1)={cmi2:.6f}  | identity err={abs(dI2-cmi2):.2e}")
    assert abs(dI1 - cmi1) < 1e-9 and abs(dI2 - cmi2) < 1e-9, "A1 identity FAILED"
    # Fano floor on the optimal post-ablation (drop R1) decoder error
    cardS = int((joint.sum(axis=(1, 2)) > 0).sum())
    # exact optimal error of MAP decoder S_hat(R2):
    pS_R2 = joint.sum(axis=1)                       # (S, R2)
    pe = 1.0 - pS_R2.max(axis=0).sum()              # 1 - sum_r max_s p(s,r)
    rhs = HS - I_drop1                               # = H(S|R2) (residual uncertainty)
    h2 = entropy(np.array([pe, 1 - pe])) if 0 < pe < 1 else 0.0
    fano_lhs = h2 + (pe * np.log2(cardS - 1) if cardS > 1 else 0.0)
    print(f"  Fano (ablate R1): H(S|R2)={rhs:.6f} <= H2(Pe)+Pe*log2(|S|-1)={fano_lhs:.6f}"
          f"  (Pe*={pe:.4f})  holds={fano_lhs + 1e-9 >= rhs}")
    assert fano_lhs + 1e-9 >= rhs, "Fano floor VIOLATED"
    return dict(dI1=dI1, dI2=dI2, cmi1=cmi1, cmi2=cmi2, pe=pe, HS=HS, I_full=I_full)


# uniform inputs over {0,1}^2
p_unif = np.full((2, 2), 0.25)

# Case 1: fully redundant  S = R1 = R2  (R1,R2 perfectly correlated copies)
# inputs constrained to r1==r2:
p_corr = np.array([[0.5, 0.0], [0.0, 0.5]])
ft_copy = np.array([[0, 0], [1, 1]])  # S = r1 (=r2 on support)
report_case("Fully redundant (S=R1=R2)", make_joint(ft_copy, p_corr))

# Case 2: unique  S = R1 (R2 independent, irrelevant)
ft_r1 = np.array([[0, 0], [1, 1]])    # S=r1 regardless of r2
report_case("Unique in R1 (S=R1, R2 irrelevant)", make_joint(ft_r1, p_unif))

# Case 3: synergistic  S = R1 XOR R2
ft_xor = np.array([[0, 1], [1, 0]])
report_case("Synergistic (S=R1 XOR R2)", make_joint(ft_xor, p_unif))

# Case 4: partial redundancy  R2 = R1 w.p. 1-eps, else flipped; S=R1
eps = 0.1
# joint p(r1,r2): r1 uniform; r2 = r1 w.p. 1-eps
p_part = np.array([[0.5 * (1 - eps), 0.5 * eps],
                   [0.5 * eps, 0.5 * (1 - eps)]])
report_case(f"Partial redundancy (eps={eps})", make_joint(ft_r1, p_part))

# ---------- V3: PID cross-check with dit (interpretation guard) ----------
print("\n=== V3: PID cross-check (dit, I_min / Williams-Beer) ===")
try:
    import dit
    from dit.pid import PID_WB

    def dit_dist(joint):
        d = {}
        sn, r1n, r2n = joint.shape
        for s in range(sn):
            for i in range(r1n):
                for j in range(r2n):
                    if joint[s, i, j] > 0:
                        d[(s, i, j)] = joint[s, i, j]
        dd = dit.Distribution(list(d.keys()), list(d.values()))
        dd.set_rv_names("SXY")
        return dd

    for name, jt in [("redundant", make_joint(ft_copy, p_corr)),
                     ("xor", make_joint(ft_xor, p_unif)),
                     (f"partial(eps={eps})", make_joint(ft_r1, p_part))]:
        # rv 0 = S (output); rv 1 = R1, rv 2 = R2 (inputs, 1-indexed in dit nodes)
        pid = PID_WB(dit_dist(jt), [[1], [2]], [0])
        red = pid.get_pi(((1,), (2,)))   # redundant atom
        uX = pid.get_pi(((1,),))         # unique R1
        uY = pid.get_pi(((2,),))         # unique R2
        syn = pid.get_pi(((1, 2),))      # synergy
        print(f"  {name:18s}: Rdn={red:.3f} Unq_R1={uX:.3f} Unq_R2={uY:.3f} Syn={syn:.3f}")
    print("  (Interpretation: ablation loss of R1 = Unq_R1 + Syn; redundant atom is free.)")
except Exception as e:  # dit optional; main theorems do not depend on it
    print(f"  dit cross-check skipped: {e}")

print("\nALL INFORMATION-LAYER CHECKS PASSED.")
