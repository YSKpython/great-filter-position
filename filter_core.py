#!/usr/bin/env python3
"""
filter_core.py
==============
Core model for "Where Is the Filter?" (v2).

Yavuz Selim Kilinc - Independent Researcher, Manisa, Turkiye
MIT License

WHAT CHANGED FROM v1 (see REPO_NOTES.md for the full list)
----------------------------------------------------------
1. phi is redefined as the share of the chain's total LOG-improbability
   ("improbability budget") that lies in steps already passed.  The
   couplings are therefore exponential in phi rather than reciprocal
   power laws, and the phi-invariance of lambda becomes a substantive
   condition (Lambda_B = Lambda_A) instead of an algebraic identity.
2. Every coupled quantity is bounded BY CONSTRUCTION.  v1 clipped
   Q(phi) at 1.0, which produced the unexplained kink at phi ~ 0.808
   in v1 Figures 1, 3d, 5a and 6b.  Nothing is clipped here.
3. The survival channel uses the SURVIVAL LIKELIHOOD
       P(E | P,Q) = 1 - P(1-Q),
   not the posterior catastrophe probability PQ/(1-P+PQ) that v1
   multiplied into the joint likelihood.  eta is retained as a
   reported diagnostic only.
4. The birth-rank channel marginalises over N with the power-law prior,
   so alpha and beta both survive into the joint model.
5. Information is reported in bits (KL from the uniform prior), with
   analytic per-channel ceilings, instead of an un-normalised
   "posterior variance" that was in fact E[(phi-0.5)^2].
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import numpy as np

try:                                    # NumPy >= 2.0
    _trapz = np.trapezoid
except AttributeError:                  # NumPy < 2.0
    _trapz = np.trapz


# =====================================================================
# 1. PARAMETERS
# =====================================================================

@dataclass(frozen=True)
class Params:
    """All model parameters.  Units: yr, ly, nats."""

    # --- SOM geometry (inherited from Kilinc 2026a, Table 1) ---------
    R_G: float = 50_000.0          # galactic disk radius (ly)
    T_G: float = 1.3e10            # galactic disk age (yr)
    d_max: float = 1_000.0         # detection horizon (ly)
    p_signal: float = 0.1          # signal-class match
    p_survey: float = 0.01         # survey coverage (optimistic edge)
    p_recog: float = 0.5           # recognition probability
    N_sites: float = 1e10          # habitable sites, i.e. N_ever at phi = 0

    # --- observability lifetime bounds ------------------------------
    L_lo: float = 1e2              # lifetime of a civilisation that fails
    L_hi: float = 1e6              # lifetime of one that passes (SOM L_0)

    # --- improbability budgets (nats) -------------------------------
    Lam_B: float = 3.0             # budget carried by steps behind us
    Lam_A: float = 3.0             # budget carried by steps ahead of us

    # --- demographic channel (TWD) ----------------------------------
    n_obs: float = 1.17e11         # cumulative human births (PRB 2024)
    R_ceil: float = 1e4            # N_max / n_obs at phi = 1
    Lam_T: float = 3.0             # demographic coupling strength
    alpha: float = 1.0             # prior exponent  pi(N) ~ N^-alpha
    beta: float = 1.0              # window exponent, likelihood ~ N^-beta

    # --- survival channel (anthropic shadow) ------------------------
    P_cat: float = 0.01            # apriori catastrophe probability
    Q_lo: float = 0.05             # survival probability at phi = 0
    Q_hi: float = 0.95             # survival probability at phi = 1
    Lam_Q: float = 3.0             # survival coupling strength

    # --- switches ---------------------------------------------------
    shadow_mode: str = "likelihood"   # "likelihood" | "selection"

    def kappa_geo(self) -> float:
        """Geometric/instrumental part of kappa (2.0e-7 at baseline)."""
        return ((self.d_max / self.R_G) ** 2
                * self.p_signal * self.p_survey * self.p_recog)

    def lam_prefactor(self) -> float:
        """N_sites * kappa_geo * L_hi / T_G.  Upper bound on lambda."""
        return self.N_sites * self.kappa_geo() * self.L_hi / self.T_G


BASE = Params()

PHI = np.linspace(0.0, 1.0, 2001)      # no regularisation needed


# =====================================================================
# 2. COUPLINGS  (all bounded by construction)
# =====================================================================

def s_ahead(phi, Lam):
    """Probability of passing the steps that remain: exp(-(1-phi) Lam)."""
    return np.exp(-(1.0 - np.asarray(phi, dtype=float)) * Lam)


def N_ever(phi, p: Params = BASE):
    """Civilisations that ever arise: N_sites * exp(-phi Lam_B).

    Bounded in [N_sites e^-Lam_B, N_sites].
    """
    return p.N_sites * np.exp(-np.asarray(phi, dtype=float) * p.Lam_B)


def L_eff(phi, p: Params = BASE):
    """Effective observability lifetime, bounded in [L_lo, L_hi]."""
    return p.L_lo + (p.L_hi - p.L_lo) * s_ahead(phi, p.Lam_A)


def kappa(phi, p: Params = BASE):
    """Per-civilisation detection probability.  p_sync = L_eff / T_G."""
    return p.kappa_geo() * L_eff(phi, p) / p.T_G


def lam(phi, p: Params = BASE):
    """Expected detection count lambda(phi) = N_ever(phi) kappa(phi)."""
    return N_ever(phi, p) * kappa(phi, p)


def N_max(phi, p: Params = BASE):
    """Ceiling on total human births, bounded in [n_obs, R_ceil n_obs]."""
    return p.n_obs * (1.0 + (p.R_ceil - 1.0) * s_ahead(phi, p.Lam_T))


def Q_surv(phi, p: Params = BASE):
    """Survival probability, bounded in [Q_lo, Q_hi] -- never clipped."""
    return p.Q_lo + (p.Q_hi - p.Q_lo) * s_ahead(phi, p.Lam_Q)


def eta_shadow(phi, p: Params = BASE):
    """Overconfidence factor eta = (1 - P + P Q)/Q.

    DIAGNOSTIC ONLY.  v1 multiplied a closely related quantity into the
    joint likelihood; that was the error corrected in this version.
    """
    Q = Q_surv(phi, p)
    return (1.0 - p.P_cat + p.P_cat * Q) / Q


# =====================================================================
# 3. CHANNEL LIKELIHOODS
# =====================================================================

def L_som(phi, p: Params = BASE):
    """Null-sky likelihood P(D_F = 0 | phi) = exp(-lambda(phi))."""
    return np.exp(-lam(phi, p))


def _expint(c, u0, u1):
    """int_{u0}^{u1} exp(-c u) du, stable through c -> 0."""
    c = np.asarray(c, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    u1 = np.asarray(u1, dtype=float)
    d = u1 - u0
    small = np.abs(c) < 1e-12
    c_safe = np.where(small, 1.0, c)
    val = np.exp(-c_safe * u0) * (1.0 - np.exp(-c_safe * d)) / c_safe
    return np.where(small, d, val)


def L_twd(phi, p: Params = BASE):
    """Birth-rank likelihood, marginalised over N.

        pi(N) ~ N^-alpha on [n_obs, N_max(phi)],   P(n_obs | N) ~ N^-beta
        =>  L(phi) = I(alpha+beta-1) / I(alpha-1)      with I as above.

    Both alpha and beta survive: the numerator sees only sigma =
    alpha + beta, the normaliser sees only alpha.  The TWD sum
    degeneracy therefore does NOT transfer verbatim to the joint model,
    which is stated explicitly in the paper rather than assumed away.
    """
    u0 = np.log(p.n_obs)
    u1 = np.log(N_max(phi, p))
    num = _expint(p.alpha + p.beta - 1.0, u0, u1)
    den = _expint(p.alpha - 1.0, u0, u1)
    return num / den


def L_shadow(phi, p: Params = BASE):
    """Survival likelihood.

    "likelihood" mode:  P(E | P,Q) = (1-P) + P Q = 1 - P(1-Q).
    "selection" mode:   P(E) == 1, i.e. survival is a pure observation
                        selection condition and the channel is exactly
                        uninformative (Cirkovic-Sandberg-Bostrom logic
                        taken to its limit).
    """
    if p.shadow_mode == "selection":
        return np.ones_like(np.asarray(phi, dtype=float))
    if p.shadow_mode != "likelihood":
        raise ValueError(f"unknown shadow_mode: {p.shadow_mode!r}")
    return 1.0 - p.P_cat * (1.0 - Q_surv(phi, p))


def L_joint(phi, p: Params = BASE):
    """Product of the three channel likelihoods."""
    return L_som(phi, p) * L_twd(phi, p) * L_shadow(phi, p)


CHANNELS = {
    "som": L_som,
    "twd": L_twd,
    "shadow": L_shadow,
    "joint": L_joint,
}


# =====================================================================
# 4. POSTERIOR AND INFORMATION METRICS
# =====================================================================

def posterior(p: Params = BASE, channel: str = "joint", grid=PHI):
    """Normalised posterior density over phi under a uniform prior."""
    L = np.asarray(CHANNELS[channel](grid, p), dtype=float)
    L = np.where(np.isfinite(L) & (L > 0.0), L, 0.0)
    Z = _trapz(L, grid)
    if not np.isfinite(Z) or Z <= 0.0:
        return np.ones_like(grid)
    return L / Z


def info_bits(p: Params = BASE, channel: str = "joint", grid=PHI):
    """KL(posterior || uniform prior) in bits.

    The uniform prior on [0,1] has density 1, so this is
    int p log2 p dphi.  Zero for a flat likelihood; 1 bit is roughly
    what it takes to decide 'behind' versus 'ahead'.
    """
    post = posterior(p, channel, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(post > 0.0, post * np.log2(post), 0.0)
    return float(_trapz(integrand, grid))


def info_ceiling(p: Params = BASE, channel: str = "joint", grid=PHI):
    """Analytic ceiling on the information a channel can carry.

    For a likelihood bounded by [Lmin, Lmax] on [0,1], the posterior
    density is bounded by Lmax/Lmin, hence
        KL(post || uniform) <= log2(Lmax / Lmin).
    """
    L = np.asarray(CHANNELS[channel](grid, p), dtype=float)
    L = L[np.isfinite(L) & (L > 0.0)]
    if L.size == 0:
        return np.inf
    return float(np.log2(L.max() / L.min()))


def summarise(p: Params = BASE, channel: str = "joint", grid=PHI):
    """Posterior mean, sd, central 95% interval, Pr(phi>0.5), bits."""
    post = posterior(p, channel, grid)
    mean = float(_trapz(post * grid, grid))
    var = float(_trapz(post * (grid - mean) ** 2, grid))
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (post[1:] + post[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    lo = float(np.interp(0.025, cdf, grid))
    hi = float(np.interp(0.975, cdf, grid))
    pr_behind = float(1.0 - np.interp(0.5, grid, cdf))
    return dict(mean=mean, sd=float(np.sqrt(max(var, 0.0))),
                q025=lo, q975=hi, pr_behind=pr_behind,
                bits=info_bits(p, channel, grid),
                bits_ceiling=info_ceiling(p, channel, grid))


# =====================================================================
# 5. THE COUPLING CLASS  (robust-Bayes / partial identification)
# =====================================================================

CLASS_BOUNDS = {
    "Lam_B": (0.0, 20.0),
    "Lam_A": (0.0, 20.0),
    "Lam_T": (0.0, 10.0),
    "Lam_Q": (0.0, 10.0),
    "alpha": (0.5, 2.0),
    "beta": (0.0, 1.0),
    "P_cat": (1e-3, 0.5),
    "Q_lo": (0.01, 0.30),
    "Q_hi": (0.50, 0.99),
}


def sample_class(n: int, seed: int = 20260726, base: Params = BASE):
    """Draw n members of the coupling class C uniformly on CLASS_BOUNDS.

    Fixed integer seed: reproducible without relying on PYTHONHASHSEED.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        kw = {k: float(rng.uniform(lo, hi))
              for k, (lo, hi) in CLASS_BOUNDS.items()}
        out.append(replace(base, **kw))
    return out


def class_envelope(n: int = 20000, seed: int = 20260726,
                   base: Params = BASE, grid=PHI):
    """Range of information gain and of Pr(phi>0.5) over the class.

    This is the identified-set computation: what the three channels can
    and cannot pin down, taken over every coupling in C rather than at
    one hand-picked baseline.
    """
    bits, pr = [], []
    for p in sample_class(n, seed, base):
        s = summarise(p, "joint", grid)
        bits.append(s["bits"])
        pr.append(s["pr_behind"])
    bits = np.asarray(bits)
    pr = np.asarray(pr)
    k = int(np.argmax(bits))
    return dict(
        n=n,
        bits_min=float(bits.min()), bits_max=float(bits.max()),
        bits_median=float(np.median(bits)),
        bits_q95=float(np.quantile(bits, 0.95)),
        pr_min=float(pr.min()), pr_max=float(pr.max()),
        pr_median=float(np.median(pr)),
        frac_over_1bit=float(np.mean(bits > 1.0)),
        argmax_bits=asdict(sample_class(n, seed, base)[k]),
        bits=bits, pr=pr,
    )


# =====================================================================
# 6. LEGACY v1 PARAMETERISATION  (kept only to document the artefacts)
# =====================================================================

def v1_lam(phi, gN=1.0, gL=1.0, eps=0.01, N0=1e10, L0=1e6,
           kb=2.0e-7, TG=1.3e10):
    """v1's lambda: reciprocal power laws.  Constant iff gN == gL."""
    psi = 1.0 - phi
    Ne = N0 * (psi / (phi + eps)) ** gN
    Le = L0 * (phi / (psi + eps)) ** gL
    return Ne * kb * Le / TG


def v1_joint_shape(phi, gT=0.5, gQ=0.5, beta=1.0, eps=0.01):
    """v1's TWD x shadow product, up to constants, in the small-P limit.

    Proportional to (phi/(psi+eps))^(gQ - gT*beta): identically flat on
    the locus gT*beta == gQ, which is exactly where v1's baseline sat
    (gT = 0.5, beta = 1, gQ = 0.5).  v1's headline "broad ridge" was
    this coincidence, not a structural degeneracy.
    """
    psi = 1.0 - phi
    return (phi / (psi + eps)) ** (gQ - gT * beta)


def v1_Q(phi, Q0=0.5, gQ=0.5, eps=0.01):
    """v1's Q(phi) BEFORE clipping.  Exceeds 1 at phi = 0.808..."""
    psi = 1.0 - phi
    return Q0 * (phi / (psi + eps)) ** gQ


def v1_clip_point(Q0=0.5, gQ=0.5, eps=0.01):
    """phi at which v1's Q(phi) hits 1 and was clipped."""
    r = (1.0 / Q0) ** (1.0 / gQ)          # required phi/(psi+eps)
    return r * (1.0 + eps) / (1.0 + r)
