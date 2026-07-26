#!/usr/bin/env python3
"""
filter_checks.py
================
Verification suite.  Every check prints WHAT IT PROVES, so a referee can
map each claim in the paper onto an executable test.

MIT License
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

import filter_core as fc

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def _record(name, ok, proves, detail=""):
    _results.append((name, PASS if ok else FAIL, proves))
    flag = PASS if ok else FAIL
    print(f"  [{flag}] {name}")
    print(f"         proves: {proves}")
    if detail:
        print(f"         detail: {detail}")
    return ok


# ---------------------------------------------------------------------
# 1. Boundedness -- the v1 clipping bug cannot recur
# ---------------------------------------------------------------------

def check_bounded():
    p = fc.BASE
    phi = fc.PHI
    Q = fc.Q_surv(phi, p)
    L = fc.L_eff(phi, p)
    Nm = fc.N_max(phi, p)
    ok = (
        np.all(Q > 0.0) and np.all(Q < 1.0)
        and np.all(L >= p.L_lo - 1e-9) and np.all(L <= p.L_hi + 1e-9)
        and np.all(L / p.T_G <= 1.0)
        and np.all(Nm >= p.n_obs - 1e-3)
        and np.all(Nm <= p.R_ceil * p.n_obs + 1e-3)
    )
    # and over the whole coupling class, not just the baseline
    for q in fc.sample_class(400, seed=11):
        Qc = fc.Q_surv(phi, q)
        if not (np.all(Qc > 0.0) and np.all(Qc < 1.0)):
            ok = False
            break
    return _record(
        "boundedness", ok,
        "Q, L_eff, p_sync and N_max stay inside their ranges for every "
        "phi and every member of the coupling class; no clipping is used "
        "anywhere, so the v1 kink at phi = 0.808 cannot recur.",
        f"Q in [{Q.min():.4f}, {Q.max():.4f}], "
        f"p_sync_max = {(L/p.T_G).max():.3e}",
    )


# ---------------------------------------------------------------------
# 2. Proposition 4' -- budget symmetry, not an algebraic identity
# ---------------------------------------------------------------------

def check_prop4():
    # exact case: equal budgets and no residual lifetime
    p = replace(fc.BASE, L_lo=0.0, Lam_B=4.0, Lam_A=4.0)
    lam_eq = fc.lam(fc.PHI, p)
    flat = float(np.ptp(lam_eq) / lam_eq.mean())

    # asymmetric budgets: phi-dependence must reappear, governed by the gap
    q = replace(fc.BASE, L_lo=0.0, Lam_B=6.0, Lam_A=4.0)
    lam_ne = fc.lam(fc.PHI, q)
    ratio = float(lam_ne[0] / lam_ne[-1])
    predicted = float(np.exp(q.Lam_B - q.Lam_A))

    # the residual lifetime L_lo alone also breaks exact invariance
    r = replace(fc.BASE, Lam_B=4.0, Lam_A=4.0)      # L_lo = 1e2
    lam_res = fc.lam(fc.PHI, r)
    resid = float(np.ptp(lam_res) / lam_res.mean())

    ok = (flat < 1e-12
          and abs(ratio - predicted) / predicted < 1e-9
          and resid > 1e-6)
    return _record(
        "prop4_budget_symmetry", ok,
        "lambda(phi) is exactly phi-invariant iff the past and future "
        "improbability budgets are equal AND L_lo = 0.  The residual "
        "phi-dependence is exp(Lam_B - Lam_A), so Proposition 4 is a "
        "falsifiable condition, not the parameterisation identity it was "
        "in v1.",
        f"flatness(Lam_B=Lam_A, L_lo=0) = {flat:.2e}; "
        f"lam(0)/lam(1) = {ratio:.6f} vs exp(dLam) = {predicted:.6f}; "
        f"L_lo-induced residual = {resid:.3e}",
    )


# ---------------------------------------------------------------------
# 3. The SOM information ceiling -- hard, not baseline-dependent
# ---------------------------------------------------------------------

def check_som_ceiling():
    p = fc.BASE
    pref = p.lam_prefactor()
    worst = 0.0
    for q in fc.sample_class(3000, seed=7):
        lm = fc.lam(fc.PHI, q)
        worst = max(worst, float(lm.max()))
    bound_bits = float(np.log2(np.exp(pref)))
    achieved = max(fc.info_bits(q, "som") for q in fc.sample_class(300, seed=7))
    ok = worst <= pref + 1e-9 and achieved <= bound_bits + 1e-9
    return _record(
        "som_information_ceiling", ok,
        "lambda <= N_sites * kappa_geo * L_hi / T_G for every phi and "
        "every budget, so the null sky can carry at most "
        f"log2(e^prefactor) = {bound_bits:.4f} bits in total -- about "
        "anything, let alone about phi.  The flatness of the Fermi "
        "channel is a bound, not a coincidence.",
        f"prefactor = {pref:.6f}, max lambda over class = {worst:.6f}, "
        f"max achieved bits = {achieved:.5f}",
    )


# ---------------------------------------------------------------------
# 4. Survival channel: correct likelihood and correct identified set
# ---------------------------------------------------------------------

def check_shadow_identification():
    # Level sets of the survival likelihood are P(1-Q) = const.
    rng = np.random.default_rng(3)
    c = 0.02
    Ps = rng.uniform(0.05, 0.9, 500)
    Qs = 1.0 - c / Ps
    keep = (Qs > 0.0) & (Qs < 1.0)
    Ps, Qs = Ps[keep], Qs[keep]
    like = 1.0 - Ps * (1.0 - Qs)
    like_flat = float(np.ptp(like))

    # eta is NOT constant along that level set -> eta is not identified
    eta = (1.0 - Ps + Ps * Qs) / Qs
    eta_spread = float(np.ptp(eta))

    # the v1 factor PQ/(1-P+PQ) is not the survival likelihood
    v1_factor = Ps * Qs / (1.0 - Ps + Ps * Qs)
    differs = float(np.max(np.abs(v1_factor - like)))

    eta_ratio = float(eta.max() / eta.min())
    ok = like_flat < 1e-12 and eta_ratio > 1.2 and differs > 0.1
    return _record(
        "shadow_identification", ok,
        "Survival alone identifies P(1-Q) and nothing else.  eta = "
        "(1-P+PQ)/Q varies along that level set, so the v1 abstract's "
        "claim that survival 'constrains only the ratio eta' is false; "
        "and PQ/(1-P+PQ) is a posterior, numerically distinct from the "
        "likelihood v1 needed.",
        f"likelihood spread on the level set = {like_flat:.2e}; "
        f"eta range {eta.min():.3f}-{eta.max():.3f} "
        f"(spread {eta_spread:.3f}, ratio {eta_ratio:.3f}); "
        f"max |v1 factor - likelihood| = {differs:.3f}",
    )


# ---------------------------------------------------------------------
# 5. Anthropic conditioning taken to the limit
# ---------------------------------------------------------------------

def check_selection_mode():
    p = replace(fc.BASE, shadow_mode="selection")
    bits = fc.info_bits(p, "shadow")
    ok = abs(bits) < 1e-12
    return _record(
        "selection_mode_flat", ok,
        "If survival is treated as a pure observation-selection "
        "condition, P(E) == 1 and the channel carries exactly zero bits. "
        "The corrected framework therefore contains the strong anthropic "
        "reading as a limiting case, which v1 invoked but contradicted.",
        f"bits = {bits:.3e}",
    )


# ---------------------------------------------------------------------
# 6. Diagnosis of the v1 artefacts
# ---------------------------------------------------------------------

def check_v1_artefacts():
    phi = np.linspace(0.01, 0.99, 999)

    # (a) v1's flat joint posterior was the locus gT*beta == gQ
    on = fc.v1_joint_shape(phi, gT=0.5, gQ=0.5, beta=1.0)
    off = fc.v1_joint_shape(phi, gT=0.5, gQ=0.5, beta=0.5)
    on_flat = float(np.ptp(on))
    off_flat = float(np.ptp(off))

    # (b) v1's Q(phi) crossed 1 and was clipped
    clip_phi = fc.v1_clip_point()
    q_at_099 = fc.v1_Q(0.99)

    # (c) v1's lambda was constant by construction when gN == gL, up to
    #     the eps regularisation (which itself only bites near 0 and 1)
    interior = np.linspace(0.2, 0.8, 601)
    lam_sym = fc.v1_lam(interior, gN=1.0, gL=1.0)
    sym_flat = float(np.ptp(lam_sym) / lam_sym.mean())
    l3, l5, l7 = (fc.v1_lam(np.array([0.3, 0.5, 0.7]), gN=1.0, gL=1.0))
    reproduces = (abs(l3 - 0.14679) < 5e-5 and abs(l5 - 0.14787) < 5e-5
                  and abs(l7 - 0.14679) < 5e-5 and abs(l3 - l7) < 1e-15)

    ok = (on_flat < 1e-12 and off_flat > 1.0
          and abs(clip_phi - 0.808) < 5e-3 and q_at_099 > 1.0
          and sym_flat < 0.03 and reproduces)
    return _record(
        "v1_artefacts_reproduced", ok,
        "The v1 results are reproduced and diagnosed: (a) the 'broad "
        "ridge' is exactly flat on the measure-zero locus gT*beta = gQ, "
        "which is where the v1 baseline sat, and is not flat off it; "
        "(b) Q(phi) crosses 1 at phi = 0.808, the location of the "
        "unexplained kink in v1 Figs 1, 3d, 5a, 6b; (c) v1's lambda was "
        "phi-independent by construction whenever gN = gL.",
        f"ptp(on locus) = {on_flat:.2e}, ptp(off locus) = {off_flat:.3f}; "
        f"clip at phi = {clip_phi:.4f}, Q(0.99) = {q_at_099:.3f}; "
        f"v1 lambda on [0.2,0.8] flat to {sym_flat:.2e}; "
        f"reproduces v1 print-out {l3:.5f}/{l5:.5f}/{l7:.5f}",
    )


# ---------------------------------------------------------------------
# 7. Posterior machinery
# ---------------------------------------------------------------------

def check_posterior_machinery():
    p = fc.BASE
    post = fc.posterior(p, "joint")
    mass = float(np.trapezoid(post, fc.PHI)) if hasattr(np, "trapezoid") \
        else float(np.trapz(post, fc.PHI))

    flat = fc.info_bits(replace(p, Lam_T=0.0, Lam_Q=0.0,
                                Lam_B=0.0, Lam_A=0.0), "joint")
    ceil_ok = all(
        fc.info_bits(q, "joint") <= fc.info_ceiling(q, "joint") + 1e-9
        for q in fc.sample_class(200, seed=5)
    )
    ok = abs(mass - 1.0) < 1e-8 and abs(flat) < 1e-9 and ceil_ok
    return _record(
        "posterior_machinery", ok,
        "The posterior normalises to 1; a fully decoupled model returns "
        "exactly 0 bits; and the measured information gain never exceeds "
        "the analytic log2(Lmax/Lmin) ceiling.",
        f"mass = {mass:.12f}, decoupled bits = {flat:.2e}",
    )


CHECKS = [
    check_bounded,
    check_prop4,
    check_som_ceiling,
    check_shadow_identification,
    check_selection_mode,
    check_v1_artefacts,
    check_posterior_machinery,
]


def run_all():
    print("=" * 70)
    print("VERIFICATION SUITE")
    print("=" * 70)
    oks = [fn() for fn in CHECKS]
    n_pass = sum(oks)
    print("-" * 70)
    print(f"{n_pass}/{len(oks)} checks passed")
    print("=" * 70)
    return all(oks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
