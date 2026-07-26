#!/usr/bin/env python3
"""
filter_experiments.py
=====================
Tables and figures for "Where Is the Filter?" (v2).

Every number reported in the paper is produced here and written to
results.json / results.txt, so the manuscript can be checked against the
pipeline output line by line.

MIT License
"""

from __future__ import annotations

import json
import os
from dataclasses import replace, asdict, fields

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import filter_core as fc

OUT_FIG = "figures"
os.makedirs(OUT_FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

RESULTS: dict = {}
PHI = fc.PHI


# =====================================================================
# Table 1: baseline parameters
# =====================================================================

def table_1():
    rows = [(f.name, getattr(fc.BASE, f.name)) for f in fields(fc.Params)]
    RESULTS["table1_parameters"] = {k: v for k, v in rows}
    lines = ["Table 1 - baseline parameters", "-" * 46]
    for k, v in rows:
        lines.append(f"  {k:<12s} {v}")
    return "\n".join(lines)


# =====================================================================
# Table 2 / Figure 3: what each channel can carry
# =====================================================================

def table_2():
    rows = []
    for ch, label in [("som", "Fermi null sky (SOM)"),
                      ("twd", "Birth rank (TWD)"),
                      ("shadow", "Survival (shadow)"),
                      ("joint", "All three jointly")]:
        s = fc.summarise(fc.BASE, ch)
        rows.append(dict(channel=ch, label=label, **s))
    RESULTS["table2_channel_information"] = rows

    lines = ["Table 2 - information carried by each channel (baseline)",
             "-" * 82,
             f"  {'channel':<22s}{'bits':>12s}{'ceiling':>12s}"
             f"{'mean':>8s}{'sd':>8s}{'Pr(phi>.5)':>12s}"]
    for r in rows:
        lines.append(f"  {r['label']:<22s}{r['bits']:>12.3e}"
                     f"{r['bits_ceiling']:>12.3e}{r['mean']:>8.3f}"
                     f"{r['sd']:>8.3f}{r['pr_behind']:>12.3f}")

    # the SOM ceiling holds for every budget, not just the baseline
    pref = fc.BASE.lam_prefactor()
    hard = float(np.log2(np.exp(pref)))
    RESULTS["som_hard_ceiling_bits"] = hard
    RESULTS["som_lambda_prefactor"] = pref
    lines.append("")
    lines.append(f"  SOM hard ceiling (all budgets): lambda <= {pref:.6f}"
                 f"  =>  <= {hard:.4f} bits")

    # and the same table with survival read as pure observation selection
    sel = fc.summarise(replace(fc.BASE, shadow_mode="selection"), "joint")
    RESULTS["joint_selection_mode"] = sel
    lines.append(f"  Joint under strong anthropic reading (P(E)=1): "
                 f"{sel['bits']:.4f} bits, Pr(phi>.5) = {sel['pr_behind']:.3f}")
    return "\n".join(lines)


def figure_3():
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    titles = [r"(a) SOM: null sky $e^{-\lambda(\phi)}$",
              r"(b) TWD: birth rank",
              r"(c) Shadow: survival $1-P(1-Q)$",
              r"(d) Joint posterior"]
    for ax, ch, t in zip(axes, ["som", "twd", "shadow", "joint"], titles):
        for bv, ls in [(0.0, ":"), (0.5, "--"), (1.0, "-")]:
            p = replace(fc.BASE, beta=bv)
            L = np.asarray(fc.CHANNELS[ch](PHI, p), dtype=float)
            ax.plot(PHI, L / L.max(), ls, lw=1.8,
                    label=rf"$\beta={bv:g}$")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel("normalised")
        ax.set_title(t)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 3 - the three channels and their product",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig3_channels.png")
    plt.close(fig)


# =====================================================================
# Figure 1: the couplings are bounded
# =====================================================================

def figure_1():
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    p = fc.BASE
    axes[0].semilogy(PHI, fc.N_ever(PHI, p), lw=2)
    axes[0].set_ylabel(r"$N_{\rm ever}(\phi)$")
    axes[0].set_title(r"(a) $N_{\rm sites}e^{-\phi\Lambda_B}$")

    axes[1].semilogy(PHI, fc.L_eff(PHI, p), lw=2)
    axes[1].axhline(p.L_lo, color="k", ls=":", lw=1)
    axes[1].axhline(p.L_hi, color="k", ls=":", lw=1)
    axes[1].set_ylabel(r"$L_{\rm eff}(\phi)$ (yr)")
    axes[1].set_title(r"(b) bounded in $[L_{\rm lo},L_{\rm hi}]$")

    axes[2].semilogy(PHI, fc.N_max(PHI, p), lw=2)
    axes[2].set_ylabel(r"$N_{\max}(\phi)$")
    axes[2].set_title(r"(c) bounded in $[n_{\rm obs},Rn_{\rm obs}]$")

    axes[3].plot(PHI, fc.Q_surv(PHI, p), lw=2, label="v2 (bounded)")
    axes[3].plot(PHI, np.minimum(fc.v1_Q(PHI), 1.0), "--", lw=1.6,
                 label="v1 (clipped)")
    axes[3].axvline(fc.v1_clip_point(), color="crimson", ls=":", lw=1.2,
                    label=r"v1 clip at $\phi=0.808$")
    axes[3].set_ylim(0, 1.08)
    axes[3].set_ylabel(r"$Q(\phi)$")
    axes[3].set_title("(d) survival probability")
    axes[3].legend(fontsize=8, loc="lower right")

    for ax in axes:
        ax.set_xlabel(r"$\phi$")
        ax.set_xlim(0, 1)
    fig.suptitle("Figure 1 - every coupling is bounded by construction",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig1_couplings.png")
    plt.close(fig)


# =====================================================================
# Figure 2 / Proposition 4': budget symmetry
# =====================================================================

def figure_2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for dL, ls in [(-4.0, ":"), (-2.0, "--"), (0.0, "-"),
                   (2.0, "-."), (4.0, (0, (3, 1, 1, 1)))]:
        p = replace(fc.BASE, Lam_B=3.0 + dL, Lam_A=3.0, L_lo=0.0)
        axes[0].semilogy(PHI, fc.lam(PHI, p), linestyle=ls, lw=1.8,
                         label=rf"$\Lambda_B-\Lambda_A={dL:+g}$")
    axes[0].set_xlabel(r"$\phi$")
    axes[0].set_ylabel(r"$\lambda(\phi)$")
    axes[0].set_title(r"(a) $\lambda\propto e^{-\phi(\Lambda_B-\Lambda_A)}$"
                      "\n"r"flat iff $\Lambda_B=\Lambda_A$")
    axes[0].legend(fontsize=8)

    pref = fc.BASE.lam_prefactor()
    for LB, ls in [(0.0, "-"), (3.0, "--"), (8.0, ":")]:
        p = replace(fc.BASE, Lam_B=LB, Lam_A=LB)
        axes[1].plot(PHI, fc.L_som(PHI, p), ls, lw=1.8,
                     label=rf"$\Lambda_B=\Lambda_A={LB:g}$")
    axes[1].axhline(np.exp(-pref), color="crimson", ls=":", lw=1.4,
                    label=rf"floor $e^{{-{pref:.4f}}}$")
    axes[1].set_ylim(0.84, 1.005)
    axes[1].set_xlabel(r"$\phi$")
    axes[1].set_ylabel(r"$P(\mathcal{D}_F=0\mid\phi)$")
    axes[1].set_title("(b) the null-sky likelihood cannot leave\n"
                      rf"$[e^{{-{pref:.4f}}},1]$: at most "
                      rf"{np.log2(np.exp(pref)):.3f} bits")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle(r"Figure 2 - Proposition 4$'$ and the SOM information "
                 "ceiling", fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig2_lambda.png")
    plt.close(fig)

    RESULTS["lambda_symmetric_value"] = float(
        fc.lam(0.5, replace(fc.BASE, L_lo=0.0)))
    RESULTS["lambda_range_baseline"] = [
        float(fc.lam(PHI, fc.BASE).min()), float(fc.lam(PHI, fc.BASE).max())]


# =====================================================================
# Figure 4: the flat posterior is a measure-zero coincidence
# =====================================================================

def figure_4(n=61):
    lt = np.linspace(0.0, 8.0, n)
    lq = np.linspace(0.0, 8.0, n)
    B = np.zeros((n, n))
    PR = np.zeros((n, n))
    for i, q in enumerate(lq):
        for j, t in enumerate(lt):
            p = replace(fc.BASE, Lam_T=t, Lam_Q=q)
            s = fc.summarise(p, "joint")
            B[i, j] = s["bits"]
            PR[i, j] = s["pr_behind"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cs = axes[0].contourf(lt, lq, np.log10(np.maximum(B, 1e-8)),
                          levels=24, cmap="viridis")
    fig.colorbar(cs, ax=axes[0], label=r"$\log_{10}$ bits")
    axes[0].plot(fc.BASE.Lam_T, fc.BASE.Lam_Q, "w*", ms=12,
                 label="baseline")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_xlabel(r"$\Lambda_T$ (demographic)")
    axes[0].set_ylabel(r"$\Lambda_Q$ (survival)")
    axes[0].set_title("(a) information gain about $\\phi$ is negligible\n"
                      "everywhere in coupling space")

    cs2 = axes[1].contourf(lt, lq, PR, levels=np.linspace(0, 1, 21),
                           cmap="RdBu")
    fig.colorbar(cs2, ax=axes[1], label=r"$\Pr(\phi>1/2\mid\mathcal{D})$")
    axes[1].contour(lt, lq, PR, levels=[0.5], colors="k", linewidths=1.6)
    axes[1].set_xlabel(r"$\Lambda_T$ (demographic)")
    axes[1].set_ylabel(r"$\Lambda_Q$ (survival)")
    axes[1].set_title(r"(b) the tilt in $\Pr(\phi>1/2)$ is set by"
                      "\nunidentified couplings, not by the data")

    fig.suptitle("Figure 4 - negligible information across coupling "
                 "space, but a sign that still flips",
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig4_flat_locus.png")
    plt.close(fig)

    frac_flat = float(np.mean(B < 0.01))
    RESULTS["flat_locus"] = dict(
        grid=n, bits_min=float(B.min()), bits_max=float(B.max()),
        frac_cells_under_0p01_bits=frac_flat,
        pr_min=float(PR.min()), pr_max=float(PR.max()),
        pr_spans_half=bool(PR.min() < 0.5 < PR.max()),
    )
    return frac_flat


# =====================================================================
# Table 3 / Figure 6: standardised sensitivity
# =====================================================================

SENS_KEYS = ["Lam_B", "Lam_A", "Lam_T", "Lam_Q",
             "alpha", "beta", "P_cat", "Q_lo", "Q_hi"]


def table_3(rel=0.10):
    base_bits = fc.info_bits(fc.BASE, "joint")
    base_pr = fc.summarise(fc.BASE, "joint")["pr_behind"]
    rows = []
    for k in SENS_KEYS:
        v0 = getattr(fc.BASE, k)
        hi = fc.summarise(replace(fc.BASE, **{k: v0 * (1 + rel)}), "joint")
        lo = fc.summarise(replace(fc.BASE, **{k: v0 * (1 - rel)}), "joint")
        d_bits = abs(hi["bits"] - base_bits) + abs(lo["bits"] - base_bits)
        d_pr = abs(hi["pr_behind"] - base_pr) + abs(lo["pr_behind"] - base_pr)
        rows.append(dict(param=k, baseline=v0, d_bits=d_bits, d_pr=d_pr))
    rows.sort(key=lambda r: -r["d_bits"])
    RESULTS["table3_sensitivity"] = dict(
        rel=rel, base_bits=base_bits, base_pr=base_pr, rows=rows)

    lines = [f"Table 3 - standardised sensitivity (+/-{rel:.0%} relative)",
             "-" * 66,
             f"  baseline: {base_bits:.4f} bits, "
             f"Pr(phi>.5) = {base_pr:.3f}",
             f"  {'rank':<6s}{'parameter':<12s}{'baseline':>10s}"
             f"{'|d bits|':>11s}{'|d Pr|':>10s}"]
    for i, r in enumerate(rows, 1):
        lines.append(f"  {i:<6d}{r['param']:<12s}{r['baseline']:>10.4g}"
                     f"{r['d_bits']:>11.4f}{r['d_pr']:>10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    y = np.arange(len(rows))[::-1]
    axes[0].barh(y, [r["d_bits"] for r in rows], color="steelblue")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["param"] for r in rows], fontsize=9)
    axes[0].set_xlabel(r"$|\Delta|$ information gain (bits)")
    axes[0].set_title(f"(a) impact on bits (+/-{rel:.0%} relative)")
    axes[1].barh(y, [r["d_pr"] for r in rows], color="indianred")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([r["param"] for r in rows], fontsize=9)
    axes[1].set_xlabel(r"$|\Delta|\Pr(\phi>1/2)$")
    axes[1].set_title("(b) impact on the sign of the conclusion")
    fig.suptitle("Figure 6 - standardised sensitivity", fontweight="bold",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig6_sensitivity.png")
    plt.close(fig)
    return "\n".join(lines)


# =====================================================================
# Table 4 / Figure 5: the identified set over the coupling class
# =====================================================================

def table_4(n=20000):
    env = fc.class_envelope(n=n)
    keep = {k: v for k, v in env.items() if k not in ("bits", "pr")}
    RESULTS["table4_identified_set"] = keep

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(env["bits"], bins=60, color="steelblue")
    axes[0].axvline(1.0, color="crimson", ls="--", lw=1.6,
                    label="1 bit = decides behind vs ahead")
    axes[0].set_xlabel("information gain about $\\phi$ (bits)")
    axes[0].set_ylabel("couplings in $\\mathcal{C}$")
    axes[0].set_title(f"(a) median {env['bits_median']:.3f}, "
                      f"max {env['bits_max']:.3f} bits")
    axes[0].legend(fontsize=8)

    axes[1].hist(env["pr"], bins=60, color="indianred")
    axes[1].axvline(0.5, color="k", ls="--", lw=1.4)
    axes[1].set_xlabel(r"$\Pr(\phi>1/2\mid\mathcal{D})$")
    axes[1].set_ylabel("couplings in $\\mathcal{C}$")
    axes[1].set_title(f"(b) identified set "
                      f"[{env['pr_min']:.2f}, {env['pr_max']:.2f}]")
    fig.suptitle(r"Figure 5 - what the three channels can and cannot pin "
                 r"down, over the whole class $\mathcal{C}$",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig5_identified_set.png")
    plt.close(fig)

    lines = [f"Table 4 - identified set over the coupling class "
             f"(n = {n})", "-" * 66,
             f"  information gain : min {env['bits_min']:.4f}  "
             f"median {env['bits_median']:.4f}  "
             f"q95 {env['bits_q95']:.4f}  max {env['bits_max']:.4f} bits",
             f"  fraction above 1 bit : {env['frac_over_1bit']:.4f}",
             f"  Pr(phi>1/2)      : min {env['pr_min']:.4f}  "
             f"median {env['pr_median']:.4f}  max {env['pr_max']:.4f}"]
    return "\n".join(lines)


# =====================================================================
# Driver
# =====================================================================

def table_5():
    """What would break the degeneracy, in bits.

    (a) A single confirmed SETI detection.  With D_F = 1 the Poisson
        likelihood is lambda e^-lambda, and since lambda << 1 this is
        proportional to lambda(phi) itself -- so a detection reads the
        budget gap Lam_B - Lam_A directly, where the null sky cannot.
    (b) The survival channel's ceiling grows with P: the shadow is only
        informative when the apriori catastrophe rate is large.  Note
        this reverses v1's sensitivity ranking, in which P came last --
        an artefact of using the posterior in place of the likelihood.
    """
    det, nul = [], []
    for dL in [0.0, 1.0, 2.0, 4.0, 8.0]:
        p = replace(fc.BASE, Lam_B=3.0 + dL, Lam_A=3.0)
        lm = np.asarray(fc.lam(PHI, p), dtype=float)

        Ld = lm * np.exp(-lm)                       # D_F = 1
        Ld = Ld / (np.trapezoid(Ld, PHI) if hasattr(np, "trapezoid")
                   else np.trapz(Ld, PHI))
        with np.errstate(divide="ignore", invalid="ignore"):
            bd = float(np.trapezoid(np.where(Ld > 0, Ld * np.log2(Ld), 0.0),
                                    PHI))
        det.append(dict(dLam=dL, bits=bd))
        nul.append(dict(dLam=dL, bits=fc.info_bits(p, "som")))

    shadow = []
    for P in [1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.9]:
        p = replace(fc.BASE, P_cat=P)
        shadow.append(dict(P=P,
                           ceiling=fc.info_ceiling(p, "shadow"),
                           bits=fc.info_bits(p, "shadow")))

    RESULTS["table5_degeneracy_breakers"] = dict(
        detection=det, null_sky=nul, shadow_vs_P=shadow)

    lines = ["Table 5 - what would break the degeneracy (bits about phi)",
             "-" * 72,
             f"  {'Lam_B-Lam_A':>12s}{'null sky D_F=0':>18s}"
             f"{'one detection D_F=1':>22s}"]
    for a, b in zip(nul, det):
        lines.append(f"  {a['dLam']:>12.1f}{a['bits']:>18.3e}"
                     f"{b['bits']:>22.4f}")
    lines += ["",
              f"  {'P':>8s}{'shadow ceiling':>18s}{'shadow bits':>15s}"]
    for r in shadow:
        lines.append(f"  {r['P']:>8.3g}{r['ceiling']:>18.4f}"
                     f"{r['bits']:>15.3e}")
    return "\n".join(lines)


def run_all():
    blocks = [table_1()]
    figure_1()
    figure_2()
    blocks.append(table_2())
    figure_3()
    frac = figure_4()
    blocks.append("Figure 4 - fraction of the (Lam_T, Lam_Q) grid within "
                  f"0.01 bits of flat: {frac:.4f}")
    blocks.append(table_3())
    blocks.append(table_4())
    blocks.append(table_5())

    text = "\n\n".join(blocks)
    with open("results.txt", "w") as fh:
        fh.write(text + "\n")
    with open("results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=float)
    print(text)
    print(f"\nFigures written to ./{OUT_FIG}/")
    print("Numbers written to results.txt and results.json")


if __name__ == "__main__":
    run_all()
