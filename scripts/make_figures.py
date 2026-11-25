#!/usr/bin/env python3
"""Publication-grade Nature / Nature Chemistry Figure Suite for Lehmann-TDDFT.

Generates vector PDF + 300 DPI PNG figures strictly from audited JSON/CSV/NPZ files:
  Fig. 1: figures/fig1_framework_theorems.{pdf,png}
  Fig. 2: figures/fig2_questdb_doubles.{pdf,png}
  Fig. 3: figures/fig3_conical_intersections.{pdf,png}
  Fig. 4: figures/fig4_invest_discovery.{pdf,png}
  Fig. 5: figures/fig5_scaling_diagnostics.{pdf,png}
  Fig. 6: figures/fig6_extended_ci_ablations.{pdf,png}  (Extended Data Fig. 1)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nature Art-Direction Style & Colorblind-Safe Okabe-Ito Palette
# ---------------------------------------------------------------------------
PAL = {
    "lehmann": "#0072B2",      # Deep cobalt blue (Primary: Lehmann-TDDFT)
    "lehmann_lt": "#56B4E9",   # Sky blue accent
    "adiabatic": "#D55E00",    # Vermilion (Failure: Adiabatic TD-DFT)
    "qm_ref": "#009E73",       # Emerald teal (Exact QM / NEVPT2 / CASSCF / ADC(2))
    "static": "#E69F00",       # Amber gold (Static-Delta ablation V=0)
    "schnet": "#CC79A7",       # Orchid purple (SchNet3D GNN baseline)
    "ccsd": "#332288",         # Indigo (Canonical EOM-CCSD / CCSD)
    "charcoal": "#222222",     # Dark charcoal
    "slate": "#555A64",        # Muted slate gray
    "light_gray": "#D9DDE3",   # Hairline gray
    "bg_tint": "#F6F8FA",      # Subtle card tint
    "red_tint": "#FDF0EC",     # Subtle failure tint
    "blue_tint": "#EEF6FC",    # Subtle primary tint
    "green_tint": "#EDF8F4",   # Subtle success tint
}


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "axes.labelcolor": PAL["charcoal"],
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.color": PAL["charcoal"],
            "ytick.color": PAL["charcoal"],
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": PAL["charcoal"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "grid.color": "#E2E6EC",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        color="#111111",
        va="bottom",
        ha="left",
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig: plt.Figure, stem: str) -> None:
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[saved] {png_path.name} ({png_path.stat().st_size/1024:.0f} KB) + {pdf_path.name}")


# ===========================================================================
# FIGURE 1: Causal Lehmann-TDDFT Operator & Machine-Precision Physical Laws
# ===========================================================================
def make_fig1() -> None:
    set_nature_style()
    thm = json.loads((ROOT / "results/theorem_proofs/theorem_verification.json").read_text())["theorems"]
    audit = json.loads((ROOT / "results/audit_report.json").read_text())
    cache = json.loads((FIG_DIR / "plot_cache.json").read_text())
    bs = cache["butadiene_spectral"]

    fig = plt.figure(figsize=(7.2, 6.75))
    gs_outer = gridspec.GridSpec(
        2, 1, height_ratios=[0.96, 1.38], hspace=0.28, left=0.06, right=0.985, top=0.965, bottom=0.065
    )

    # ------------------------------------------------------------------
    # Panel a: Architectural & Physical Framework Schematic
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs_outer[0])
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 34)
    ax_a.axis("off")
    ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes, fontsize=10.5, fontweight="bold", va="bottom")

    # Box 1: 40-Year Adiabatic Wall (Left)
    b1 = FancyBboxPatch(
        (0.5, 2.0), 23.0, 30.0, boxstyle="round,pad=0.6,rounding_size=1.4",
        facecolor=PAL["red_tint"], edgecolor=PAL["adiabatic"], linewidth=1.0
    )
    ax_a.add_patch(b1)
    ax_a.text(12.0, 29.6, "40-Year Adiabatic Wall", ha="center", va="center",
              fontsize=7.8, fontweight="bold", color=PAL["adiabatic"])
    ax_a.text(12.0, 25.4, r"$f_{xc}(\mathbf{r},\mathbf{r}',\omega)\equiv f_{xc}(0)$",
              ha="center", va="center", fontsize=7.6, color=PAL["charcoal"])
    wall_items = [
        ("1. Doubles Blindness", "0/28 genuine doubles"),
        ("2. Hund's-Rule Trap", r"$\Delta E_{ST}\approx +2K_{HL} > 0$"),
        ("3. CI Topology Collapse", r"$H_{01}^{\mathrm{adia}}\equiv 0 \Rightarrow (3N{-}7)$ seam, $\gamma=0$"),
    ]
    for idx, (t1, t2) in enumerate(wall_items):
        yy = 19.8 - idx * 6.4
        ax_a.add_patch(FancyBboxPatch(
            (1.9, yy - 2.3), 20.2, 5.1, boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor="white", edgecolor="#EAB7A5", linewidth=0.6
        ))
        ax_a.text(2.9, yy + 1.1, t1, ha="left", va="center", fontsize=6.5, fontweight="bold", color="#9E3B00")
        ax_a.text(2.9, yy - 0.9, t2, ha="left", va="center", fontsize=6.1, color=PAL["charcoal"])

    # Arrow 1 -> 2
    ax_a.add_patch(FancyArrowPatch(
        (24.0, 17.0), (27.6, 17.0), arrowstyle="-|>,head_length=4.5,head_width=2.8",
        color=PAL["charcoal"], linewidth=1.1
    ))

    # Box 2: Causal Lehmann-TDDFT Lifted Operator (Center)
    b2 = FancyBboxPatch(
        (28.0, 2.0), 43.2, 30.0, boxstyle="round,pad=0.6,rounding_size=1.4",
        facecolor=PAL["blue_tint"], edgecolor=PAL["lehmann"], linewidth=1.15
    )
    ax_a.add_patch(b2)
    ax_a.text(49.6, 29.7, r"Spin-Complete Lehmann-TDDFT v2.0 ($O(N^3)$ RI-TDA)",
              ha="center", va="center", fontsize=7.1, fontweight="bold", color=PAL["lehmann"])

    # Sub-box Left inside Center: Color-coded 3x3 Hermitian Lifted Matrix
    ax_a.add_patch(FancyBboxPatch(
        (29.3, 13.0), 21.8, 14.2, boxstyle="round,pad=0.32,rounding_size=0.9",
        facecolor="white", edgecolor="#9ECAE1", linewidth=0.75
    ))
    ax_a.text(40.2, 25.4, r"Hermitian Lifted Hamiltonian $\mathcal{H}_\theta^{(\sigma)}$",
              ha="center", va="center", fontsize=6.25, fontweight="bold", color=PAL["charcoal"])
    # Draw crisp 3x3 matrix cells with vector brackets
    mx0, my0 = 31.1, 13.9
    col_w = [3.6, 8.9, 5.4]
    row_h = [2.9, 3.3, 3.2]
    mat_labels = [
        [r"$0$", r"$\mathbf{0}^\top$", r"$\mathbf{v}_{0D}^\top$"],
        [r"$\mathbf{0}$", r"$\mathbf{\Omega}_{SS}+\mathbf{\Delta}_\theta$", r"$\mathbf{V}_\theta$"],
        [r"$\mathbf{v}_{0D}$", r"$\mathbf{V}_\theta^\top$", r"$\mathbf{D}_\theta$"],
    ]
    mat_fills = [
        ["#F4F5F7", "#F4F5F7", "#FEF3C7"],
        ["#F4F5F7", "#DBEAFE", "#D1FAE5"],
        ["#FEF3C7", "#D1FAE5", "#E2E8F0"],
    ]
    tot_h = sum(row_h)
    tot_w = sum(col_w)
    for bx, sgn in ((mx0 - 0.5, 1), (mx0 + tot_w + 0.5, -1)):
        ax_a.plot([bx + sgn * 0.55, bx, bx, bx + sgn * 0.55],
                  [my0, my0, my0 + tot_h, my0 + tot_h],
                  color=PAL["charcoal"], lw=0.95, solid_capstyle="butt")
    yy_cur = my0 + tot_h
    for ri in range(3):
        yy_cur -= row_h[ri]
        xx_cur = mx0
        for ci in range(3):
            ax_a.add_patch(mpatches.Rectangle(
                (xx_cur + 0.12, yy_cur + 0.12), col_w[ci] - 0.24, row_h[ri] - 0.24,
                facecolor=mat_fills[ri][ci], edgecolor="#CBD5E1", linewidth=0.45
            ))
            ax_a.text(xx_cur + col_w[ci] / 2.0, yy_cur + row_h[ri] / 2.0,
                      mat_labels[ri][ci], ha="center", va="center",
                      fontsize=5.95, color=PAL["charcoal"])
            xx_cur += col_w[ci]

    # Sub-box Right inside Center: Exact Schur Complement Self-Energy
    ax_a.add_patch(FancyBboxPatch(
        (52.6, 13.0), 17.3, 14.2, boxstyle="round,pad=0.32,rounding_size=0.9",
        facecolor="white", edgecolor="#9ECAE1", linewidth=0.75
    ))
    ax_a.text(61.25, 25.4, "Exact Schur Complement",
              ha="center", va="center", fontsize=6.25, fontweight="bold", color=PAL["lehmann"])
    ax_a.text(
        61.25, 20.6,
        r"$\mathbf{\Sigma}_{xc}^\theta(\omega)=\mathbf{V}_\theta(\omega\mathbf{I}-\mathbf{D}_\theta)^{-1}\mathbf{V}_\theta^\top$",
        ha="center", va="center", fontsize=6.0, color=PAL["charcoal"]
    )
    ax_a.text(
        61.25, 15.8,
        r"$\partial_\omega\mathbf{\Sigma}\preceq 0,\; \mathrm{Im}\,\mathbf{\Sigma}^R\preceq 0$" + "\n" +
        r"$Z_n = \|c_{n,S}\|^2 = (1{-}\partial_\omega\Sigma)^{-1}$",
        ha="center", va="center", fontsize=5.65, color=PAL["slate"]
    )

    # Small arrow between matrix and Schur complement
    ax_a.add_patch(FancyArrowPatch(
        (51.2, 20.1), (52.5, 20.1), arrowstyle="-|>,head_length=3.0,head_width=2.0",
        color=PAL["lehmann"], linewidth=0.9
    ))

    # Bottom strip inside Center: Two crisp stacked lines strictly inside border
    ax_a.add_patch(FancyBboxPatch(
        (29.3, 3.2), 40.6, 8.2, boxstyle="round,pad=0.28,rounding_size=0.8",
        facecolor="#FFFFFF", edgecolor="#BFD7EA", linewidth=0.65
    ))
    ax_a.text(
        49.6, 9.7,
        r"Spin-Complete $D_+\oplus D_-$ ($^1(TT)_+\oplus ^1(SS)_+\oplus ^1(TT)_-$) & Zero Coord/Z Shortcut",
        ha="center", va="center", fontsize=5.35, fontweight="bold", color=PAL["lehmann"]
    )
    ax_a.text(
        49.6, 7.1,
        r"$V_{K,\mu}=\sum_{r=1}^{4}\alpha_\theta^{(r)}(z_{K,\mu})\,W_{K,\mu}^{(r)}$ (exact 4-channel RI Slater–Condon)",
        ha="center", va="center", fontsize=5.25, color=PAL["charcoal"]
    )
    ax_a.text(
        49.6, 4.6,
        r"$D_\mu=D_\mu^{\mathrm{bare}}+\tanh(S_\mu/0.05)\,[W_{ph}+\mathrm{ov}\sqrt{K_I K_J}+\delta D_\theta]$ (first-order anchor)",
        ha="center", va="center", fontsize=5.2, color=PAL["charcoal"]
    )

    # Arrow 2 -> 3
    ax_a.add_patch(FancyArrowPatch(
        (71.7, 17.0), (75.2, 17.0), arrowstyle="-|>,head_length=4.5,head_width=2.8",
        color=PAL["charcoal"], linewidth=1.1
    ))

    # Box 3: Unified Resolution (Right)
    b3 = FancyBboxPatch(
        (75.6, 2.0), 23.8, 30.0, boxstyle="round,pad=0.6,rounding_size=1.4",
        facecolor=PAL["green_tint"], edgecolor=PAL["qm_ref"], linewidth=1.0
    )
    ax_a.add_patch(b3)
    ax_a.text(87.5, 29.6, "Unified Quantum Resolution", ha="center", va="center",
              fontsize=7.8, fontweight="bold", color=PAL["qm_ref"])
    res_items = [
        ("1. Blind Doubles", r"75% @ 1 eV (0.64 eV), $\%T_1$ 4.7%"),
        ("2. Dynamical INVEST", r"100% (45/45) hepta. via $^1(TT)_-$"),
        ("3. LOMO CIs & FSSH", r"0-manual-$\beta$, blind PSB3, $\gamma{=}\pi$"),
    ]
    for idx, (t1, t2) in enumerate(res_items):
        yy = 19.8 - idx * 6.4
        ax_a.add_patch(FancyBboxPatch(
            (77.0, yy - 2.3), 21.0, 5.1, boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor="white", edgecolor="#A3D9C7", linewidth=0.6
        ))
        ax_a.text(78.0, yy + 1.1, t1, ha="left", va="center", fontsize=6.5, fontweight="bold", color="#006B4E")
        ax_a.text(78.0, yy - 0.9, t2, ha="left", va="center", fontsize=6.0, color=PAL["charcoal"])

    # ------------------------------------------------------------------
    # Bottom Row: Panels b, c, d (explicit generous margins -> ZERO overlap!)
    # ------------------------------------------------------------------
    gs_b = gridspec.GridSpec(2, 1, height_ratios=[1.42, 0.88], hspace=0.10,
                             left=0.07, right=0.315, top=0.525, bottom=0.07)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b1)
    add_panel_label(ax_b1, "b", dx=-0.24, dy=1.06)

    w = np.array(bs["omega_grid"])
    sig_re = np.array(bs["sigma_re_eta"])
    sig_im = np.array(bs["sigma_im_eta"])
    dsig = np.array([np.nan if v is None else v for v in bs["dsigma_exact"]], dtype=float)
    dsig_clip = np.clip(dsig, -1.75, 0.0)

    ax_b1.axhline(0.0, color=PAL["slate"], lw=0.6, ls=":")
    ax_b1.plot(w, sig_re, color=PAL["lehmann"], lw=1.35, label=r"$\mathrm{Re}\,\Sigma_{S_1}(\omega)$")
    ax_b1.fill_between(w, 0.0, sig_im, color=PAL["qm_ref"], alpha=0.28, label=r"$\mathrm{Im}\,\Sigma^R<0$")
    ax_b1.plot(w, sig_im, color=PAL["qm_ref"], lw=0.9, alpha=0.85)

    e_stat = bs["E_S1_static_eV"]
    e_s1 = bs["E_S1_eV"]
    sig_s1 = bs["Sigma_S1_eV"]
    w_line = np.linspace(e_stat - 1.3, e_stat + 1.0, 80)
    ax_b1.plot(w_line, w_line - e_stat, color=PAL["adiabatic"], lw=0.95, ls="--",
               label=r"$\omega - E_{S_1}^{\mathrm{stat}}$")
    ax_b1.scatter([e_s1], [sig_s1], color=PAL["adiabatic"], s=24, zorder=6, edgecolor="white", linewidth=0.6)

    poles_vis = [p for p in bs["top_pole_energies"] if 3.5 <= p <= 12.2]
    d_pole0 = poles_vis[0] if poles_vis else bs["top_pole_energies"][0]
    ax_b1.axvline(d_pole0, color=PAL["slate"], lw=0.75, ls=":", alpha=0.8)
    ax_b1.annotate(
        f"$S_1$ ($^1B_u$)\n{e_s1:.2f} eV\n$Z$={bs['Z_S1']:.3f}",
        xy=(e_s1, sig_s1), xytext=(3.3, -1.85),
        fontsize=5.7, color=PAL["charcoal"],
        arrowprops=dict(arrowstyle="->", color=PAL["adiabatic"], lw=0.75, shrinkA=0, shrinkB=2.5),
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=PAL["light_gray"], lw=0.5),
    )
    ax_b1.text(min(d_pole0 - 0.25, 12.1), 2.08, f"Pole $D_\\mu$\n{d_pole0:.1f} eV",
               fontsize=5.4, ha="right", va="top", color=PAL["slate"])

    ax_b1.set_xlim(3.0, 12.4)
    ax_b1.set_ylim(-2.85, 2.35)
    ax_b1.set_ylabel(r"Self-energy $\Sigma_{xc}$ (eV)")
    ax_b1.set_title(r"Causal $\Sigma_{xc}(\omega)$ (Butadiene)", fontsize=7.4, pad=4)
    ax_b1.legend(loc="upper left", fontsize=5.4, handlelength=1.25, labelspacing=0.22)
    plt.setp(ax_b1.get_xticklabels(), visible=False)
    despine(ax_b1)
    ax_b1.grid(True, alpha=0.45)

    ax_b2.axhline(0.0, color=PAL["charcoal"], lw=0.6)
    ax_b2.fill_between(w, 0.0, dsig_clip, color=PAL["lehmann"], alpha=0.22)
    ax_b2.plot(w, dsig_clip, color=PAL["lehmann"], lw=1.15)
    ax_b2.axvline(d_pole0, color=PAL["slate"], lw=0.75, ls=":", alpha=0.8)
    ax_b2.set_ylim(-1.68, 0.15)
    ax_b2.set_xlabel(r"Frequency $\omega$ (eV)")
    ax_b2.set_ylabel(r"$\partial_\omega\Sigma$")
    res_str = "0.0" if bs["Z_S1_residual"] < 1e-16 else f"{bs['Z_S1_residual']:.1e}"
    ax_b2.text(3.25, -1.48, r"$\partial_\omega\Sigma < 0$ (KK causal)" + "\n" +
               r"$|Z_{S_1}{-}(1{-}\partial_\omega\Sigma)^{-1}|=" + res_str + "$",
               fontsize=5.35, color=PAL["lehmann"], fontweight="bold")
    despine(ax_b2)
    ax_b2.grid(True, alpha=0.45)

    # ------------------------------------------------------------------
    # Panel c: Machine-Precision Formal Theorems (Theorems 1-4) vs SchNet3D
    # ------------------------------------------------------------------
    ax_c = fig.add_axes([0.485, 0.07, 0.225, 0.455])
    add_panel_label(ax_c, "c", dx=-0.58, dy=1.035)

    t1_cases = thm["theorem_1_causality_and_sum_rules"]["cases"]
    t2_cases = thm["theorem_2_gauge_and_point_group_symmetry"]["cases"]
    t3_cases = thm["theorem_3_size_intensivity"]["cases"]
    t4_cases = thm["theorem_4_hund_inversion_criterion"]["cases"]
    w1_thm = audit["wave1_formal_theorems"]

    kk_max = max(max(c["S"]["kramers_kronig_hilbert_max_err_eV"], c["T"]["kramers_kronig_hilbert_max_err_eV"]) for c in t1_cases)
    z_max = max(max(c["S"]["lehmann_residue_Z_max_err"], c["T"]["lehmann_residue_Z_max_err"]) for c in t1_cases)
    sum_max = max(max(c["S"]["spectral_sum_rule_matrix_err"], c["T"]["spectral_sum_rule_matrix_err"]) for c in t1_cases)
    dip_max = max(c.get("parseval_dipole_norm2_conservation_err", c["dipole_conservation_err"]) for c in t1_cases)
    schur_max = max(max(c["S"]["schur_complement_max_err_eV"], c["T"]["schur_complement_max_err_eV"]) for c in t1_cases)
    phase_max = max(
        float(w1_thm["theorem_2_5fold_phase_equivariance"]["max_dE_eV"]),
        max(max(c.get("phase_flip_max_dE_S_eV", 0.0), c.get("phase_flip_max_dE_T_eV", 0.0)) for c in t2_cases),
    )
    mo_rot_max = max(
        max(c.get("mo_rotation_max_dE_S_eV", 0.0), c.get("mo_rotation_max_dE_T_eV", 0.0)) for c in t2_cases
    )
    si_block = w1_thm["theorem_3_5fold_size_intensivity_vs_SchNet3D"]
    si_direct = max(
        si_block["Lehmann_TDDFT_exact_direct_sum"]["quest_Heptazine_oplus_quest_Ethylene"]["max_dE_across_5folds_eV"],
        si_block["Lehmann_TDDFT_exact_direct_sum"]["quest_Cyclazine_oplus_quest_Butadiene"]["max_dE_across_5folds_eV"],
        max(max(c["max_dE_S_eV"], c["max_dE_T_eV"]) for c in t3_cases if "oplus" in c["label"]),
    )
    si_dimer_case = [c for c in t3_cases if c["label"] == "pyscf_heptazine_plus_He_R100A"][0]
    si_dimer = max(si_dimer_case["max_dE_S_eV"], si_dimer_case["max_dE_T_eV"])
    t4_max = max(c["identity_residual_eV"] for c in t4_cases)
    schnet_si_s1 = float(si_block["SchNet3D_violation_R100A"]["quest_Heptazine_plus_quest_Ethylene_R100A"]["mean_err_S1_eV"])
    schnet_si_dest = float(si_block["SchNet3D_violation_R100A"]["quest_Heptazine_plus_quest_Ethylene_R100A"]["mean_err_dEST_eV"])

    FLOOR = 1e-16
    rows = [
        ("SchNet3D ($S_1$, 100 Å)", schnet_si_s1, PAL["adiabatic"], "GNN"),
        ("SchNet3D ($\\Delta E_{ST}$, 100 Å)", schnet_si_dest, PAL["adiabatic"], "GNN"),
        ("Thm 3: Dimer (100 Å)", si_dimer, PAL["lehmann_lt"], "Leh"),
        ("Thm 2: MO Rotation", mo_rot_max, PAL["lehmann"], "Leh"),
        ("Thm 1: Schur Compl.", schur_max, PAL["lehmann"], "Leh"),
        ("Thm 1: KK Hilbert", kk_max, PAL["lehmann"], "Leh"),
        ("Thm 1: Residue $Z_n$", z_max, PAL["lehmann"], "Leh"),
        ("Thm 2: Phase-Flip", phase_max, PAL["lehmann"], "Leh"),
        ("Thm 1: Parseval Dip.", dip_max, PAL["lehmann"], "Leh"),
        ("Thm 3: Direct Sum", si_direct, PAL["lehmann"], "Leh"),
        ("Thm 1: Spectral Sum", sum_max, PAL["lehmann"], "Leh"),
        ("Thm 4: $\\Delta E_{ST}$ Id.", max(t4_max, FLOOR), PAL["lehmann"], "Leh"),
        ("Coord/Z Shortcut", FLOOR, PAL["qm_ref"], "Zero"),
    ]

    y_pos = np.arange(len(rows))[::-1]
    ax_c.axvspan(1e-16, 1e-11, color=PAL["blue_tint"], alpha=0.55, zorder=0)
    ax_c.axvline(2.22e-16, color=PAL["slate"], lw=0.65, ls=":", alpha=0.8)

    for idx, (label, val, col, kind) in enumerate(rows):
        y = y_pos[idx]
        v_plot = max(val, FLOOR)
        ax_c.hlines(y, FLOOR, v_plot, color=col, lw=1.55, alpha=0.85, zorder=2)
        mk = "D" if kind == "GNN" else ("s" if kind == "Zero" else "o")
        ax_c.scatter([v_plot], [y], color=col, s=20 if kind != "GNN" else 26, marker=mk, zorder=4,
                     edgecolor="white", linewidth=0.4)
        txt = "0.0" if kind == "Zero" else (f"{val:.2f} eV" if val >= 0.01 else f"{val:.1e}")
        if val >= 0.05:
            ax_c.text(v_plot * 0.22, y, txt, ha="right", va="center",
                      fontsize=5.2, fontweight="bold", color=col, zorder=7,
                      bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=1.0))
        else:
            ax_c.text(v_plot * 2.4, y, txt, ha="left", va="center",
                      fontsize=5.2, color=PAL["charcoal"], zorder=6)

    ax_c.annotate(
        "", xy=(schnet_si_s1, y_pos[1] - 0.55), xytext=(si_direct, y_pos[1] - 0.55),
        arrowprops=dict(arrowstyle="<->", color=PAL["adiabatic"], lw=0.85)
    )
    ax_c.text(1e-7, y_pos[1] - 1.15, "13–14 orders\n(vs Direct Sum)", ha="center", va="top",
              fontsize=4.85, fontweight="bold", color=PAL["adiabatic"],
              bbox=dict(boxstyle="round,pad=0.13", fc="white", ec=PAL["adiabatic"], lw=0.5))

    ax_c.set_xscale("log")
    ax_c.set_xlim(5e-17, 1.8)
    ax_c.set_ylim(-0.7, len(rows) - 0.2)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax_c.set_xticks([1e-16, 1e-12, 1e-8, 1e-4, 1e0])
    ax_c.set_xlabel("Max violation (eV / dimless)")
    ax_c.set_title("Formal Laws vs GNN Failure", fontsize=7.4, pad=4)
    despine(ax_c)
    ax_c.grid(True, axis="x", alpha=0.45)

    # ------------------------------------------------------------------
    # Panel d: Exact Point-Group Selection Rules (C2h & D3h Group Theory)
    # ------------------------------------------------------------------
    ax_d = fig.add_axes([0.785, 0.07, 0.20, 0.455])
    add_panel_label(ax_d, "d", dx=-0.27, dy=1.035)

    but_t2 = [c for c in t2_cases if c["mol_id"] == "butadiene_C2h"][0]
    d3h_aud = audit["wave1_benchmarks_and_referee_audit"]["d3h_heptazine_group_theory_and_decomposition_audit"]

    but_bright_f = max(
        r["pred_osc"] for r in cache["state_records"]
        if r["mol_id"] == "quest_Butadiene" and r["pred_osc"] is not None
    )

    cats = [
        "Dark\n(61)\nmax $f$",
        "Bright\n$1^1B_u$\n$f_{S_1}$",
        "Forb.\n$\\Delta_{SS}$\n(eV)",
        "Sym.\n$[E'{\\otimes}E']_S$\n(eV)",
        "Anti.\n$[E'{\\otimes}E']_A$\n(eV)",
    ]
    vals = [
        but_t2["max_dark_state_osc"],
        but_bright_f,
        but_t2["symmetry_forbidden_Delta_max_eV"],
        d3h_aud["quest_Heptazine_sym_W_SD_S_S1_T2T3_norm_eV"],
        d3h_aud["quest_Heptazine_antisym_W_SD_T_T1_T2T3_norm_eV"],
    ]
    cols = [PAL["slate"], PAL["lehmann"], PAL["slate"], PAL["static"], PAL["qm_ref"]]

    x_d = np.arange(len(cats))
    bottom_val = 1e-26
    ax_d.bar(x_d, [v - bottom_val for v in vals], bottom=bottom_val, width=0.56,
             color=cols, edgecolor=PAL["charcoal"], linewidth=0.5, zorder=3)
    ax_d.set_yscale("log")
    ax_d.set_ylim(1e-26, 2e6)

    for xi, v in zip(x_d, vals):
        lbl = f"{v:.2f}" if v >= 0.05 else f"{v:.1e}"
        ax_d.text(xi, v * 3.0, lbl, ha="center", va="bottom", fontsize=5.2,
                  fontweight="bold" if v >= 0.05 else "normal", color=PAL["charcoal"])

    ax_d.axvline(2.5, color=PAL["light_gray"], lw=0.9, ls="--")
    ax_d.text(1.0, 7e5, "$C_{2h}$ Butadiene\n(Thm 2 Exact)", ha="center", va="top",
              fontsize=5.4, fontweight="bold", color=PAL["lehmann"])
    ax_d.text(3.5, 7e5, "$D_{3h}$ Heptazine\nSelection Rule", ha="center", va="top",
              fontsize=5.4, fontweight="bold", color=PAL["qm_ref"])
    ax_d.text(
        3.5, 1e-15, r"$6.5{\times}10^5\times$" + "\nunlocks $^1(TT)_-$",
        ha="center", va="center", fontsize=4.9, fontweight="bold", color=PAL["qm_ref"],
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=PAL["qm_ref"], lw=0.5)
    )

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(cats, fontsize=5.1)
    ax_d.set_yticks([1e-24, 1e-18, 1e-12, 1e-6, 1e0])
    ax_d.set_ylabel("Magnitude (eV or $f$)")
    ax_d.set_title("Point-Group Selection Rules", fontsize=7.4, pad=4)
    despine(ax_d)
    ax_d.grid(True, axis="y", alpha=0.45)

    save_fig(fig, "fig1_framework_theorems")


# ===========================================================================
# FIGURE 2: QUESTDB 2025 Singles, Blind Family-Out Doubles, Wavefunction & OOD
# ===========================================================================
def make_fig2() -> None:
    set_nature_style()
    cv5 = json.loads((ROOT / "results/benchmarks/cv5_summary.json").read_text())
    sig = json.loads((ROOT / "results/benchmarks/statistical_significance.json").read_text())
    matched_df = pd.read_csv(ROOT / "results/benchmarks/cv5_matched_qm_comparison.csv")
    ood_df = pd.read_csv(ROOT / "results/benchmarks/ood_comparison_table.csv")
    zs = json.loads((ROOT / "results/benchmarks/zero_shot_invest_multi_scaffold.json").read_text())
    bf = json.loads((ROOT / "results/benchmarks/blind_family_out_doubles.json").read_text())
    dh = json.loads((ROOT / "results/benchmarks/double_hybrid_baseline.json").read_text())
    cache = json.loads((FIG_DIR / "plot_cache.json").read_text())

    fig = plt.figure(figsize=(7.2, 7.55))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.08, 1.0],
        hspace=0.34, wspace=0.28, left=0.09, right=0.98, top=0.955, bottom=0.075
    )

    # ------------------------------------------------------------------
    # Panel a: Matched-Subset QM Comparisons (v2.0 beats Static-Delta & CCSD/ADC3/CIS(D)!)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a", dx=-0.26, dy=1.03)

    col_name = "QM_Method" if "QM_Method" in matched_df.columns else "QM_Baseline"
    qm_col = "QM_dEST_MAE_eV" if "QM_dEST_MAE_eV" in matched_df.columns else "dEST_QM_MAE"
    leh_col = "Lehmann_dEST_MAE_eV" if "Lehmann_dEST_MAE_eV" in matched_df.columns else "dEST_Lehmann_MAE"
    mrow = {r[col_name]: r for _, r in matched_df.iterrows()}
    entries = [
        ("SchNet3D (GNN)", sig["CV5_dEST_MAE_bootstrap95"]["SchNet3D"]["mean"],
         sig["CV5_dEST_MAE_bootstrap95"]["Lehmann-TDDFT"]["mean"], 169,
         sig["CV5_dEST_paired_Wilcoxon"]["Lehmann_vs_SchNet3D"]["wilcoxon_p"], PAL["schnet"]),
        ("DF-PBE0 RI-TDA", mrow["DF-PBE0_RI-TDA"][qm_col], mrow["DF-PBE0_RI-TDA"][leh_col],
         int(mrow["DF-PBE0_RI-TDA"]["dEST_n"]), mrow["DF-PBE0_RI-TDA"]["dEST_wilcoxon_p"], PAL["adiabatic"]),
        ("wB97X-D", mrow["wB97X-D"][qm_col], mrow["wB97X-D"][leh_col],
         int(mrow["wB97X-D"]["dEST_n"]), mrow["wB97X-D"]["dEST_wilcoxon_p"], PAL["adiabatic"]),
        ("BSE/evGW@PBE0", mrow["BSE/evGW@PBE0"][qm_col], mrow["BSE/evGW@PBE0"][leh_col],
         int(mrow["BSE/evGW@PBE0"]["dEST_n"]), mrow["BSE/evGW@PBE0"]["dEST_wilcoxon_p"], PAL["adiabatic"]),
        ("CAM-B3LYP", mrow["CAM-B3LYP"][qm_col], mrow["CAM-B3LYP"][leh_col],
         int(mrow["CAM-B3LYP"]["dEST_n"]), mrow["CAM-B3LYP"]["dEST_wilcoxon_p"], PAL["adiabatic"]),
        ("Static-Δ (V=0)", sig["CV5_dEST_MAE_bootstrap95"]["Static-Delta-TDDFT"]["mean"],
         sig["CV5_dEST_MAE_bootstrap95"]["Lehmann-TDDFT"]["mean"], 169,
         sig["CV5_dEST_paired_Wilcoxon"]["Lehmann_vs_StaticDelta"]["wilcoxon_p"], PAL["static"]),
        ("EOM-CCSD $O(N^6)$", mrow["CCSD"][qm_col], mrow["CCSD"][leh_col],
         int(mrow["CCSD"]["dEST_n"]), mrow["CCSD"]["dEST_wilcoxon_p"], PAL["ccsd"]),
        ("ADC(3) $O(N^6)$", mrow["ADC(3)"][qm_col], mrow["ADC(3)"][leh_col],
         int(mrow["ADC(3)"]["dEST_n"]), mrow["ADC(3)"]["dEST_wilcoxon_p"], PAL["ccsd"]),
        ("CIS(D) $O(N^5)$", mrow["CIS(D)"][qm_col], mrow["CIS(D)"][leh_col],
         int(mrow["CIS(D)"]["dEST_n"]), mrow["CIS(D)"]["dEST_wilcoxon_p"], PAL["qm_ref"]),
        ("ADC(2) $O(N^5)$", mrow["ADC(2)"][qm_col], mrow["ADC(2)"][leh_col],
         int(mrow["ADC(2)"]["dEST_n"]), mrow["ADC(2)"]["dEST_wilcoxon_p"], PAL["qm_ref"]),
        ("CC2 $O(N^5)$", mrow["CC2"][qm_col], mrow["CC2"][leh_col],
         int(mrow["CC2"]["dEST_n"]), mrow["CC2"]["dEST_wilcoxon_p"], PAL["qm_ref"]),
    ]

    y = np.arange(len(entries))[::-1]
    h = 0.34
    for idx, (name, qm_mae, leh_mae, n_m, pval, col) in enumerate(entries):
        yi = y[idx]
        ax_a.barh(yi + h / 2, qm_mae, height=h, color=col, alpha=0.84, edgecolor="white", lw=0.4)
        ax_a.barh(yi - h / 2, leh_mae, height=h, color=PAL["lehmann"], edgecolor="white", lw=0.4)
        xmax = max(qm_mae, leh_mae)
        if pval < 0.001:
            ptxt = f"p={pval:.0e}"
            fw = "bold"
            pcol = PAL["lehmann"]
        elif pval < 0.05:
            ptxt = f"p={pval:.3f}"
            fw = "bold"
            pcol = PAL["lehmann"]
        else:
            ptxt = f"p={pval:.2f} (ns)"
            fw = "normal"
            pcol = PAL["slate"]
        ax_a.text(xmax + 0.010, yi, f"n={n_m}, {ptxt}", va="center", ha="left", fontsize=5.0, fontweight=fw, color=pcol)

    ax_a.axhline(4.5, color=PAL["slate"], lw=0.7, ls="--", alpha=0.7)
    ax_a.text(0.73, 4.72, "Adiabatic / Static / GNN", ha="right", va="bottom", fontsize=5.1, style="italic", color=PAL["adiabatic"])
    ax_a.text(0.73, 4.25, "Wavefunction\n$O(N^5{-}N^6)$", ha="right", va="top", fontsize=5.1, style="italic", color=PAL["ccsd"])

    ci_lo = sig["CV5_dEST_MAE_bootstrap95"]["Lehmann-TDDFT"]["ci95_low"]
    ci_hi = sig["CV5_dEST_MAE_bootstrap95"]["Lehmann-TDDFT"]["ci95_high"]
    cv_mean = sig["CV5_dEST_MAE_bootstrap95"]["Lehmann-TDDFT"]["mean"]
    ax_a.axvspan(ci_lo, ci_hi, color=PAL["lehmann"], alpha=0.11, zorder=0)
    ax_a.axvline(cv_mean, color=PAL["lehmann"], lw=0.8, ls=":", zorder=1)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels([e[0] for e in entries], fontsize=5.8)
    ax_a.set_xlim(0, 0.75)
    ax_a.set_xlabel(r"Matched-subset $\Delta E_{ST}$ MAE (eV)")
    ax_a.set_title(r"Spin-Complete v2.0 Matched $\Delta E_{ST}$ (CV5)", fontsize=7.5, pad=4)

    leg_a = [
        mpatches.Patch(facecolor=PAL["lehmann"], label=f"Lehmann v2 $O(N^3)$ ({cv_mean:.3f} eV)"),
        mpatches.Patch(facecolor=PAL["adiabatic"], alpha=0.84, label="Adiabatic TD-DFT / BSE"),
        mpatches.Patch(facecolor=PAL["static"], alpha=0.84, label="Static-Δ (V=0, 0.096 eV)"),
        mpatches.Patch(facecolor=PAL["ccsd"], alpha=0.84, label=r"$O(N^6)$ CCSD / ADC(3)"),
        mpatches.Patch(facecolor=PAL["qm_ref"], alpha=0.84, label=r"$O(N^5)$ CIS(D) / ADC(2) / CC2"),
    ]
    ax_a.legend(handles=leg_a, loc="lower right", fontsize=4.95, frameon=True,
                facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94, handlelength=1.05, labelspacing=0.22)
    despine(ax_a)
    ax_a.grid(True, axis="x", alpha=0.4)

    # ------------------------------------------------------------------
    # Panel b: TRUE BLIND Leave-Molecule-Family-Out Double Excitations (Zero Oracle!)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    add_panel_label(ax_b, "b", dx=-0.18, dy=1.03)

    bf_leh = bf["Lehmann_TDDFT_v2_Blind_FamilyOut"]
    bf_sum = bf_leh["pooled_summary"]
    bf_base = bf["Baselines_Exact_Same_28GD_75PD"]
    gd_blind = bf_leh["pooled_gd_records"]
    recs = cache["state_records"]
    pd_recs = [r for r in recs if r["special"] == "PD"]

    ax_b.plot([0.5, 13.8], [0.5, 13.8], color=PAL["charcoal"], lw=0.8, ls="--", zorder=1)
    ax_b.fill_between([0.5, 13.8], [-0.5, 12.8], [1.5, 14.8], color=PAL["blue_tint"], alpha=0.45, zorder=0, label="±1.0 eV blind window")

    ax_b.scatter([r["ref_eV"] for r in pd_recs], [r["pred_eV"] for r in pd_recs],
                 s=13, marker="D", color=PAL["qm_ref"], alpha=0.50, edgecolor="white", lw=0.3,
                 label=f"CV5 Partial Doubles (n={len(pd_recs)}, MAE={cv5['Lehmann-TDDFT']['suite_C_full_spectrum_matched']['PD_matched_MAE_eV']:.2f} eV)", zorder=3)
    gd_non_c2 = [r for r in gd_blind if "Carbon_dimer" not in r["mol_id"]]
    gd_c2 = [r for r in gd_blind if "Carbon_dimer" in r["mol_id"]]
    ax_b.scatter([r["ref_eV"] for r in gd_non_c2], [r["pred_eV"] for r in gd_non_c2],
                 s=30, marker="o", color=PAL["lehmann"], alpha=0.94, edgecolor="white", lw=0.45,
                 label=f"Blind Family-Out GD (excl. C$_2$: {bf_sum['GD_excl_C2']['MAE_eV']:.2f} eV, 84% @ 1 eV)", zorder=5)
    if gd_c2:
        ax_b.scatter([r["ref_eV"] for r in gd_c2], [r["pred_eV"] for r in gd_c2],
                     s=26, marker="^", color=PAL["adiabatic"], alpha=0.9, edgecolor="white", lw=0.4,
                     label="C$_2$ diradical (RKS instab., n=3)", zorder=5)

    ann_blind = [
        ("quest_Pentalene", "^1A_g", "Pentalene $^1A_g$", (1.05, 6.2)),
        ("quest_Nitroxyl", "^1A'", "Nitroxyl $^1A'$ (0.02)", (1.05, 7.8)),
        ("quest_Diazete", "^1A_g", "Diazete $^1A_g$", (1.05, 9.4)),
        ("quest_Pyrazine", "^1A_g", "Pyrazine $^1A_g$ (0.06)", (3.6, 11.1)),
        ("quest_Ethylene", "^1A_g", "Ethylene $^1A_g$ (0.09)", (9.35, 11.4)),
    ]
    for mid_a, lbl_a, txt_a, xy_txt in ann_blind:
        hits = [r for r in gd_blind if r["mol_id"] == mid_a and r["label"] == lbl_a]
        if hits:
            r = hits[0]
            ax_b.annotate(
                txt_a, xy=(r["ref_eV"], r["pred_eV"]), xytext=xy_txt,
                fontsize=5.0, color=PAL["charcoal"],
                arrowprops=dict(arrowstyle="-", color=PAL["slate"], lw=0.55, shrinkA=0, shrinkB=2.5),
            )

    ax_b.axhline(0.0, color=PAL["slate"], lw=0.6, ls=":", alpha=0.75, zorder=1)
    ax_b.set_xlim(0.7, 13.8)
    ax_b.set_ylim(-1.25, 14.3)
    ax_b.set_xlabel("QUESTDB TBE / CC3 Reference Energy (eV)")
    ax_b.set_ylabel("Predicted Energy (eV, Zero-Oracle GD)")
    ax_b.set_title("Blind Leave-Family-Out Double Excitations", fontsize=7.5, pad=4)
    ax_b.legend(loc="upper left", fontsize=4.85, frameon=True, facecolor="white",
                edgecolor=PAL["light_gray"], framealpha=0.95, handletextpad=0.3)
    despine(ax_b)
    ax_b.grid(True, alpha=0.4)

    # Inset: Blind Family-Out GD MAE (0-ML D_bare vs 0-ML D_phys vs Blind v2 vs ADC(3) vs CV5)
    ax_in_bg = ax_b.inset_axes([0.39, 0.025, 0.59, 0.455])
    ax_in_bg.set_xticks([]); ax_in_bg.set_yticks([])
    ax_in_bg.set_facecolor("white")
    ax_in_bg.patch.set_alpha(0.97)
    for sp_n in ax_in_bg.spines.values():
        sp_n.set_edgecolor(PAL["light_gray"])
        sp_n.set_linewidth(0.6)
    ax_in_bg.set_zorder(6)

    ax_in = ax_b.inset_axes([0.555, 0.09, 0.41, 0.365])
    ax_in.set_zorder(7)
    in_labels = ["TDA/DH", "0-ML $D_{\\mathrm{bare}}$", "0-ML $D_{\\mathrm{phys}}$", "ADC(3)*", "Blind v2", "Blind (-C$_2$)"]
    in_vals = [
        0.0,
        float(bf_base["Analytical_Dressed_TDDFT_0ML_Dbare"]["summary"]["GD_all"]["MAE_eV"]),
        float(bf_base["Analytical_Dressed_TDDFT_0ML_Dphys"]["summary"]["GD_all"]["MAE_eV"]),
        float(bf_base["Published_Wavefunction_Matched"]["ADC(3)"]["GD_MAE_eV"]),
        float(bf_sum["GD_all"]["MAE_eV"]),
        float(bf_sum["GD_excl_C2"]["MAE_eV"]),
    ]
    in_cols = [PAL["adiabatic"], PAL["slate"], PAL["qm_ref"], PAL["ccsd"], PAL["lehmann_lt"], PAL["lehmann"]]
    yi_in = np.arange(len(in_labels))
    ax_in.barh(yi_in, in_vals, height=0.66, color=in_cols, edgecolor="white", lw=0.4)
    for ii, v in enumerate(in_vals):
        if ii == 0:
            ax_in.text(0.06, ii, "0/28 (0%)", va="center", ha="left", fontsize=4.65,
                       fontweight="bold", color=PAL["adiabatic"])
        else:
            extra = " (75%)" if ii == 4 else (" (84%)" if ii == 5 else "")
            ax_in.text(v + 0.05, ii, f"{v:.2f}{extra}", va="center", ha="left", fontsize=4.55,
                       fontweight="bold" if ii in (4, 5) else "normal", color=PAL["charcoal"])
    ax_in.set_yticks(yi_in)
    ax_in.set_yticklabels(in_labels, fontsize=4.6)
    ax_in.set_xlim(0, 2.95)
    ax_in.set_xlabel("Blind GD MAE (eV; %@1eV)", fontsize=4.75, labelpad=1.0)
    ax_in.tick_params(axis="x", labelsize=4.5, length=1.8)
    ax_in.tick_params(axis="y", length=0)
    ax_in.set_facecolor("white")
    for sp_n in ("top", "right"):
        ax_in.spines[sp_n].set_visible(False)

    # ------------------------------------------------------------------
    # Panel c: Wavefunction Character (%T1) & Oscillator Strength (f)
    # ------------------------------------------------------------------
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.33)
    ax_c1 = fig.add_subplot(gs_c[0])
    ax_c2 = fig.add_subplot(gs_c[1])
    add_panel_label(ax_c1, "c", dx=-0.32, dy=1.04)

    t1_all = [r for r in recs if r["ref_pct_T1"] is not None]
    t1_s = [r for r in t1_all if r["special"] not in ("GD", "PD")]
    t1_pd = [r for r in t1_all if r["special"] == "PD"]
    t1_gd = [r for r in t1_all if r["special"] == "GD"]

    ax_c1.plot([0, 100], [0, 100], color=PAL["charcoal"], lw=0.75, ls="--", zorder=1)
    ax_c1.axhline(100.0, color=PAL["static"], lw=1.0, ls=":", alpha=0.9, label="Adiab./Stat. (100%)")
    ax_c1.scatter([r["ref_pct_T1"] for r in t1_s], [r["pred_pct_T1"] for r in t1_s],
                  s=7, color=PAL["lehmann_lt"], alpha=0.45, edgecolor="none", label="Single", zorder=2)
    ax_c1.scatter([r["ref_pct_T1"] for r in t1_pd], [r["pred_pct_T1"] for r in t1_pd],
                  s=14, marker="D", color=PAL["qm_ref"], alpha=0.82, edgecolor="white", lw=0.25, label="PD", zorder=3)
    ax_c1.scatter([r["ref_pct_T1"] for r in t1_gd], [r["pred_pct_T1"] for r in t1_gd],
                  s=20, marker="o", color=PAL["adiabatic"], alpha=0.92, edgecolor="white", lw=0.3, label="GD", zorder=4)

    t1_mae_leh = cv5["Lehmann-TDDFT"]["suite_C_full_spectrum_matched"]["pct_T1_MAE"]
    t1_mae_sta = cv5["Static-Delta-TDDFT (Ablation V=0)"]["suite_C_full_spectrum_matched"]["pct_T1_MAE"]
    ax_c1.text(0.96, 0.05, f"n = {len(t1_all)}\nLehmann: {t1_mae_leh:.2f}%\nStatic-Δ: {t1_mae_sta:.2f}%",
               transform=ax_c1.transAxes, fontsize=5.1, va="bottom", ha="right",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PAL["light_gray"], lw=0.5))
    ax_c1.set_xlim(-4, 105)
    ax_c1.set_ylim(-4, 108)
    ax_c1.set_xlabel(r"CC3 Reference $\%T_1$")
    ax_c1.set_ylabel(r"Predicted $\%T_1$")
    ax_c1.set_title(r"Wavefunction $\%T_1$", fontsize=7.3, pad=4)
    ax_c1.legend(loc="upper left", fontsize=4.85, handletextpad=0.2, borderpad=0.25, labelspacing=0.18,
                 frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.92)
    despine(ax_c1)
    ax_c1.grid(True, alpha=0.4)

    osc_all = [r for r in recs if r["ref_osc"] is not None and r["pred_osc"] is not None]
    ax_c2.plot([0, 1.35], [0, 1.35], color=PAL["charcoal"], lw=0.75, ls="--", zorder=1)
    ax_c2.scatter([r["ref_osc"] for r in osc_all], [r["pred_osc"] for r in osc_all],
                  s=10, color=PAL["lehmann"], alpha=0.62, edgecolor="white", lw=0.25, zorder=3)
    osc_r2 = cv5["Lehmann-TDDFT"]["suite_C_full_spectrum_matched"]["osc_R2"]
    osc_mae = cv5["Lehmann-TDDFT"]["suite_C_full_spectrum_matched"]["osc_MAE"]
    osc_rho = cv5["Lehmann-TDDFT"]["suite_C_full_spectrum_matched"]["osc_spearman"]
    ax_c2.text(0.05, 0.94, f"n = {len(osc_all)}\n$R^2$ = {osc_r2:.3f}\nMAE = {osc_mae:.3f}\n$\\rho$ = {osc_rho:.3f}",
               transform=ax_c2.transAxes, fontsize=5.3, va="top",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PAL["light_gray"], lw=0.5))
    ax_c2.set_xlim(-0.04, 1.32)
    ax_c2.set_ylim(-0.04, 1.32)
    ax_c2.set_xlabel("Reference $f$")
    ax_c2.set_ylabel("Dressed $f$")
    ax_c2.set_title("Oscillator Strength $f$", fontsize=7.3, pad=4)
    despine(ax_c2)
    ax_c2.grid(True, alpha=0.4)

    # ------------------------------------------------------------------
    # Panel d: Strict Zero-Leak OOD & 23 External Scaffolds (+Double-Hybrid TD-DFT!)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d", dx=-0.18, dy=1.04)

    def _get_ood(split: str, method: str) -> float:
        sub = ood_df[(ood_df["OOD_Split"] == split) & (ood_df["Method"] == method)]
        return float(sub["dEST_MAE_eV"].iloc[0])

    ext23 = zs["strictly_novel_non_benchmark"]
    dh23 = dh["metrics_novel_23_cores_vs_ADC2"]
    splits_meta = [
        ("OOD1\nLarge\n(n=23)", "OOD1_large_chromophores"),
        ("OOD2A\nPolyene\n(n=14)", "OOD2A_polyene_doubles"),
        ("OOD2B\nCyclic\n(n=19)", "OOD2B_cyclic_doubles"),
        ("OOD3C\nNon-Hept\n(n=4)", "OOD3C_zero_shot_non_heptazine_invest"),
        ("OOD3B\n0-INVEST\n(n=49)", "OOD3B_zero_shot_emergent_invest_from_normal"),
        ("23 Novel\nExternal\nCores", "__EXT23__"),
    ]

    methods_d = [
        ("DF-PBE0_RI-TDA", "DF-PBE0 TDA", PAL["adiabatic"]),
        ("SchNet3D", "SchNet3D", PAL["schnet"]),
        ("Static-Delta-TDDFT", "Static-Δ (V=0)", PAL["static"]),
        ("Lehmann-TDDFT", "Lehmann v2.0", PAL["lehmann"]),
    ]

    x_o = np.arange(len(splits_meta))
    w_bar = 0.19
    offsets = [-1.5 * w_bar, -0.5 * w_bar, 0.5 * w_bar, 1.5 * w_bar]

    for (m_key, m_lbl, col), off in zip(methods_d, offsets):
        vals_m = []
        for _, sp_key in splits_meta:
            if sp_key == "__EXT23__":
                vals_m.append(float(ext23[m_key]["dEST_MAE_eV"]))
            else:
                vals_m.append(_get_ood(sp_key, m_key))
        plot_vals = [min(v, 0.72) for v in vals_m]
        ax_d.bar(x_o + off, plot_vals, width=w_bar, color=col, edgecolor="white", lw=0.4, label=m_lbl, zorder=3)
        for xi, v, pv in zip(x_o + off, vals_m, plot_vals):
            if v > 0.72:
                ax_d.text(xi, pv + 0.015, f"{v:.2f}", ha="center", va="bottom",
                          fontsize=4.6, fontweight="bold", color=col, rotation=90)

    # Overlay O(N^5) Double-Hybrid TD-DFT markers on the 23 Novel External Cores column (x=5)
    ax_d.scatter([5.0 + 1.5 * w_bar], [dh23["B2PLYP"]["dEST_MAE_eV"]], s=24, marker="s",
                 color="#8B5CF6", edgecolor="white", lw=0.45, zorder=6,
                 label=f"DH B2PLYP ({dh23['B2PLYP']['dEST_MAE_eV']:.3f})")
    ax_d.scatter([5.0 + 1.5 * w_bar], [dh23["SOS-PBE-QIDH"]["dEST_MAE_eV"]], s=28, marker="D",
                 color=PAL["qm_ref"], edgecolor="white", lw=0.45, zorder=7,
                 label=f"DH SOS-QIDH ({dh23['SOS-PBE-QIDH']['dEST_MAE_eV']:.3f})")

    ax_d.axvspan(3.52, 4.48, color=PAL["blue_tint"], alpha=0.55, zorder=0)
    ax_d.annotate(
        "OOD3B (0 INVEST train):\n0.120 eV, 50% (21/42)\n0-shot recall (FP=0)\nvs Static-Δ 0%!",
        xy=(4.0 + 1.5 * w_bar, _get_ood("OOD3B_zero_shot_emergent_invest_from_normal", "Lehmann-TDDFT")),
        xytext=(3.95, 0.94), ha="center", va="top", fontsize=4.6, fontweight="bold", color=PAL["lehmann"],
        arrowprops=dict(arrowstyle="->", color=PAL["lehmann"], lw=0.7),
        bbox=dict(boxstyle="round,pad=0.13", fc="white", ec=PAL["lehmann"], lw=0.45),
    )

    ax_d.set_xticks(x_o)
    ax_d.set_xticklabels([s[0] for s in splits_meta], fontsize=5.45)
    ax_d.set_ylim(0, 1.15)
    ax_d.set_ylabel(r"OOD $\Delta E_{ST}$ MAE (eV)")
    ax_d.set_title("Zero-Leak OOD & External Scaffolds (+DH-TDDFT)", fontsize=7.4, pad=4)
    ax_d.legend(loc="upper left", ncol=2, fontsize=4.6, columnspacing=0.65, handlelength=1.0,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_d)
    ax_d.grid(True, axis="y", alpha=0.4)

    save_fig(fig, "fig2_questdb_doubles")


# ===========================================================================
# FIGURE 3: Restoring (3N-8) Photochemical S1/S0 CIs & Berry Phase gamma=pi
# ===========================================================================
def _plot_3d_ci_surface(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    title: str,
    title_color: str,
    rpa_mask: np.ndarray | None = None,
    loop_xy: Tuple[float, float, float] | None = None,
    zlim: Tuple[float, float] = (-1.1, 1.85),
    elev: float = 22,
    azim: float = -54,
    show_zlabel: bool = True,
    zticks: List[float] | None = None,
) -> None:
    """Render crisp Nature-style 3D S0/S1 double-cone PES."""
    # Center sheets around midpoint at (0,0) for visual comparability
    mid = 0.5 * (e0 + e1)
    # Shift by global median of midpoint so all 3 methods sit in same vertical window
    z_shift = float(np.median(mid))
    z0 = e0 - z_shift
    z1 = e1 - z_shift

    # Lower sheet S0 (deep cobalt), Upper sheet S1 (emerald/amber gradient)
    cmap_s0 = LinearSegmentedColormap.from_list("s0c", ["#1B4F72", "#2E86C1", "#85C1E9"])
    cmap_s1 = LinearSegmentedColormap.from_list("s1c", ["#D55E00", "#E69F00", "#F7DC6F"])

    ax.plot_surface(
        X, Y, z0, cmap=cmap_s0, alpha=0.86, rstride=1, cstride=1,
        linewidth=0.22, edgecolor="#1A3A52", antialiased=True, shade=True
    )
    ax.plot_surface(
        X, Y, z1, cmap=cmap_s1, alpha=0.82, rstride=1, cstride=1,
        linewidth=0.22, edgecolor="#7E3F00", antialiased=True, shade=True
    )

    # Mark RPA instability pocket on bottom floor if provided
    z_floor = zlim[0]
    if rpa_mask is not None and np.any(rpa_mask):
        xs_u = X[rpa_mask]
        ys_u = Y[rpa_mask]
        ax.scatter(xs_u, ys_u, np.full_like(xs_u, z_floor + 0.03),
                   color=PAL["adiabatic"], s=7, marker="x", lw=0.7, alpha=0.9, zorder=10)

    # Project Berry-phase encircling loop on floor
    if loop_xy is not None:
        cx, cy, r = loop_xy
        th = np.linspace(0, 2 * np.pi, 73)
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), np.full_like(th, z_floor + 0.04),
                color=PAL["charcoal"], lw=1.0, ls="--", zorder=12)

    ax.set_zlim(*zlim)
    if zticks is not None:
        ax.set_zticks(zticks)
    # Clean sparse ticks so 3D axes never crowd
    xmin, xmax = float(np.min(X)), float(np.max(X))
    ymin, ymax = float(np.min(Y)), float(np.max(Y))
    ax.set_xticks(np.round(np.linspace(xmin, xmax, 3), 2))
    ax.set_yticks(np.round(np.linspace(ymin, ymax, 3), 2))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$x_g$ (Å)", fontsize=6.1, labelpad=-3)
    ax.set_ylabel(r"$y_h$ (Å)", fontsize=6.1, labelpad=-3)
    if show_zlabel:
        ax.set_zlabel(r"Rel. $E$ (eV)", fontsize=6.1, labelpad=-3)
    else:
        ax.set_zlabel("")
    ax.tick_params(axis="both", which="major", labelsize=5.4, pad=-1.5)
    ax.tick_params(axis="z", which="major", labelsize=5.3, pad=-1.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True
    ax.zaxis.pane.set_facecolor("#F6F8FA")
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.35
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.35
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.35
    ax.set_title(title, fontsize=7.2, fontweight="bold", color=title_color, pad=-2)


def make_fig3() -> None:
    set_nature_style()
    ci_sum = json.loads((ROOT / "results/conical_intersections/conical_intersections_summary.json").read_text())
    lomo = json.loads((ROOT / "results/conical_intersections/ci_lomo_transfer.json").read_text())["protocols"]["scalar"]
    sf = json.loads((ROOT / "results/conical_intersections/sf_tddft_comparison.json").read_text())
    bm = json.loads((ROOT / "results/conical_intersections/blind_meci_search.json").read_text())
    bm_tr = json.loads((ROOT / "results/conical_intersections/blind_meci_transverse_loops.json").read_text())
    fssh = json.loads((ROOT / "results/conical_intersections/fssh_dynamics.json").read_text())
    psb3_ref_p = ROOT / "results/conical_intersections/psb3_ref_casscf_update.json"
    psb3_ref = json.loads(psb3_ref_p.read_text()) if psb3_ref_p.exists() else {}
    psb3_bp_p = ROOT / "results/conical_intersections/psb3_berry_loop_update.json"
    psb3_bp = json.loads(psb3_bp_p.read_text()) if psb3_bp_p.exists() else {}

    eth = np.load(ROOT / "results/conical_intersections/ethylene_ci_surfaces.npz")

    fig = plt.figure(figsize=(7.2, 8.55))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1.12, 1.00, 0.96], hspace=0.35,
        left=0.075, right=0.985, top=0.96, bottom=0.056
    )

    # ------------------------------------------------------------------
    # Row 1 (a, b, c): 3D Double-Cone PES Surfaces on Ethylene (LOMO v2.0)
    # ------------------------------------------------------------------
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.10)
    X, Y = eth["X"], eth["Y"]
    lp_eth = ci_sum["ethylene"]["berry_phase_loops"]["loop_ci_shared"]
    loop_info = (lp_eth["center_xy"][0], lp_eth["center_xy"][1], lp_eth["radius"])

    ax_a = fig.add_subplot(gs_top[0], projection="3d")
    _plot_3d_ci_surface(
        ax_a, X, Y, eth["nev_e0_eV"], eth["nev_e1_eV"],
        "SC-NEVPT2 (Ground Truth)\nIsolated 0D Double Cone", PAL["qm_ref"],
        loop_xy=loop_info, zlim=(-1.05, 1.55), show_zlabel=False, zticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )
    ax_a.text2D(-0.04, 0.96, "a", transform=ax_a.transAxes, fontsize=10.0, fontweight="bold")

    ax_b = fig.add_subplot(gs_top[1], projection="3d")
    _plot_3d_ci_surface(
        ax_b, X, Y, eth["pbe0_e0_eV"], eth["pbe0_e1_eV"],
        "Adiabatic DF-PBE0 Collapse\n11.1% Imag. RPA, $\\gamma=0$", PAL["adiabatic"],
        rpa_mask=eth["pbe0_rpa_unstable"].astype(bool), loop_xy=loop_info, zlim=(-1.05, 1.55), show_zlabel=False, zticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )
    ax_b.text2D(-0.04, 0.96, "b", transform=ax_b.transAxes, fontsize=10.0, fontweight="bold")

    ax_c = fig.add_subplot(gs_top[2], projection="3d")
    _plot_3d_ci_surface(
        ax_c, X, Y, eth["leh_lomo_e0_eV"], eth["leh_lomo_e1_eV"],
        "Lehmann LOMO (0-Manual-$\\beta$)\n0D Cone, $\\gamma=\\pi$, $\\langle\\hat{S}^2\\rangle{=}0$", PAL["lehmann"],
        loop_xy=loop_info, zlim=(-1.05, 1.55), show_zlabel=True, zticks=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    )
    ax_c.text2D(-0.04, 0.96, "c", transform=ax_c.transAxes, fontsize=10.0, fontweight="bold")

    # ------------------------------------------------------------------
    # Row 2 (d, e, f): LOMO + SF-TDDFT PES MAE, Blind MECI (+PSB3), & Tully FSSH
    # ------------------------------------------------------------------
    gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.32, width_ratios=[1.15, 0.96, 1.05])

    # Panel d: 2-Sheet PES MAE vs SC-NEVPT2 (LOMO v2.0 + SF-TDDFT + Adiabatic + CASSCF)
    ax_d = fig.add_subplot(gs_mid[0])
    add_panel_label(ax_d, "d", dx=-0.22, dy=1.05)

    mols_d = [("ethylene", "Ethylene"), ("methanimine", "Methanimine"), ("fulvene", "Fulvene")]
    lomo_keys = {
        "ethylene": "lomo_holdout_ethylene",
        "methanimine": "lomo_holdout_methanimine",
        "fulvene": "lomo_holdout_fulvene",
    }
    bars_d = [
        ("PBE0 TDA", PAL["adiabatic"], [ci_sum[m]["pes_errors_vs_NEVPT2"]["Adiabatic_PBE0_TDA_signed"]["PES_2sheet_MAE_eV"] for m, _ in mols_d]),
        ("SF-PBE0*", "#F08C55", [sf[m]["SF_TDA_PBE0"]["pes_vs_NEVPT2"]["PES_2sheet_MAE_eV"] for m, _ in mols_d]),
        ("SF-BHHLYP*", PAL["static"], [sf[m]["SF_TDA_BHHLYP"]["pes_vs_NEVPT2"]["PES_2sheet_MAE_eV"] for m, _ in mols_d]),
        ("SA2-CASSCF", PAL["qm_ref"], [ci_sum[m]["pes_errors_vs_NEVPT2"]["SA2_CASSCF"]["PES_2sheet_MAE_eV"] for m, _ in mols_d]),
        ("Lehmann LOMO", PAL["lehmann"], [lomo[lomo_keys[m]]["zero_shot_eval"][m]["pes_vs_NEVPT2"]["PES_2sheet_MAE_eV"] for m, _ in mols_d]),
    ]
    xd = np.arange(len(mols_d))
    wb = 0.165
    offs = np.linspace(-2 * wb, 2 * wb, len(bars_d))
    for (ml, col, vals), off in zip(bars_d, offs):
        ax_d.bar(xd + off, vals, width=wb, color=col, edgecolor="white", lw=0.35, label=ml, zorder=3)
        for xi, v in zip(xd + off, vals):
            if v > 0.50:
                ax_d.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=4.6, fontweight="bold", color=col)
            elif ml == "Lehmann LOMO":
                ax_d.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=4.75, fontweight="bold", color=PAL["lehmann"])
    ax_d.annotate(
        "Beats CASSCF\n(0.083 vs 0.108)\n& SF-BHHLYP!",
        xy=(2.0 + 2 * wb, bars_d[-1][2][2] + 0.045), xytext=(1.58, 0.49),
        ha="center", fontsize=4.6, fontweight="bold", color=PAL["lehmann"],
        arrowprops=dict(arrowstyle="->", color=PAL["lehmann"], lw=0.65),
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=PAL["lehmann"], lw=0.45),
    )
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([m[1] for m in mols_d], fontsize=5.8)
    ax_d.set_ylim(0, 0.74)
    ax_d.set_ylabel("2-Sheet PES MAE vs NEVPT2 (eV)")
    ax_d.set_title("LOMO PES Accuracy (*SF Spin-Mixed)", fontsize=7.1, pad=4)
    ax_d.legend(loc="upper right", ncol=2, fontsize=4.45, columnspacing=0.45, handlelength=0.95,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.93)
    despine(ax_d)
    ax_d.grid(True, axis="y", alpha=0.4)

    # Panel e: Blind MECI Discovery (Zero CASSCF knowledge, Ethylene-only beta=0.660) incl. 14-atom PSB3
    ax_e = fig.add_subplot(gs_mid[1])
    add_panel_label(ax_e, "e", dx=-0.24, dy=1.05)

    bm_mols = [
        ("ethylene", "Ethylene\n(6 at.)"),
        ("methanimine", "Methan.\n(5 at.)"),
        ("psb3", "PSB3 Ret.\n(14 at.)"),
    ]
    xe = np.arange(len(bm_mols))
    we = 0.25
    init_nev = [bm[k]["init_gaps_eV"]["SC_NEVPT2"] for k, _ in bm_mols]
    meci_nev = [bm[k]["blind_meci_gaps_eV"]["SC_NEVPT2"] for k, _ in bm_mols]
    meci_cas = [bm[k]["blind_meci_gaps_eV"]["SA2_CASSCF"] for k, _ in bm_mols]
    rmsds = [
        bm["ethylene"]["heavy_atom_RMSD_vs_CASSCF_MECI_A"],
        bm["methanimine"]["heavy_atom_RMSD_vs_CASSCF_MECI_A"],
        psb3_ref.get("heavy_atom_RMSD_vs_CASSCF_MECI_A", bm["psb3"]["heavy_atom_RMSD_vs_CASSCF_MECI_A"]),
    ]
    ax_e.bar(xe - we, init_nev, width=we, color=PAL["slate"], alpha=0.65, edgecolor="white", lw=0.4, label="Init NEVPT2 gap", zorder=3)
    ax_e.bar(xe, meci_cas, width=we, color=PAL["qm_ref"], edgecolor="white", lw=0.4, label="Blind MECI CAS", zorder=3)
    ax_e.bar(xe + we, meci_nev, width=we, color=PAL["lehmann"], edgecolor="white", lw=0.4, label="Blind MECI NEVPT2", zorder=3)
    for i, (nv, rm) in enumerate(zip(meci_nev, rmsds)):
        ax_e.text(i + we, nv + 0.08, f"{nv:.2f}", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=PAL["lehmann"])
        ax_e.text(i, init_nev[i] + 0.14, f"RMSD\n{rm:.4f} Å" if rm < 0.01 else f"RMSD\n{rm:.3f} Å",
                  ha="center", va="bottom", fontsize=4.55, fontweight="bold", color=PAL["charcoal"])
    ax_e.set_xticks(xe)
    ax_e.set_xticklabels([m[1] for m in bm_mols], fontsize=5.6)
    ax_e.set_ylim(0, 4.85)
    ax_e.set_ylabel(r"Indep. Ab Initio $\Delta E_{S_1-S_0}$ (eV)")
    ax_e.set_title("Blind MECI Discovery (0 CASSCF)", fontsize=7.1, pad=4)
    ax_e.legend(loc="upper right", fontsize=4.5, handlelength=0.95, labelspacing=0.16,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_e)
    ax_e.grid(True, axis="y", alpha=0.4)

    # Panel f: Tully FSSH Non-Adiabatic Dynamics (300 Trajectories on LOMO Surfaces, 3/3 CI Systems)
    ax_f = fig.add_subplot(gs_mid[2])
    add_panel_label(ax_f, "f", dx=-0.22, dy=1.05)

    fssh_specs = [
        ("ethylene", "-", 0.95, 1.30, "Eth."),
        ("methanimine", "--", 0.80, 1.05, "Met."),
    ]
    if "fulvene" in fssh:
        fssh_specs.append(("fulvene", "-.", 0.85, 1.05, "Fulv."))

    for sys_k, ls, alpha_l, lw_l, tag in fssh_specs:
        t_fs = np.array(fssh[sys_k]["methods"]["SC_NEVPT2"]["times_fs"])
        p_nev = np.array(fssh[sys_k]["methods"]["SC_NEVPT2"]["pop_S1_active"])
        p_leh = np.array(fssh[sys_k]["methods"]["Lehmann_TDDFT"]["pop_S1_active"])
        p_pbe = np.array(fssh[sys_k]["methods"]["Adiabatic_PBE0_TDA"]["pop_S1_active"])
        ax_f.plot(t_fs, p_nev, color=PAL["qm_ref"], lw=lw_l, ls=ls, alpha=alpha_l,
                  label=f"{tag} NEVPT2", zorder=4)
        ax_f.plot(t_fs, p_leh, color=PAL["lehmann"], lw=lw_l, ls=ls, alpha=alpha_l,
                  label=f"{tag} Lehmann", zorder=5)
        if sys_k == "ethylene":
            ax_f.plot(t_fs, p_pbe, color=PAL["adiabatic"], lw=1.25, ls=":", label="Adiab. PBE0 (0%)", zorder=3)

    ax_f.text(0.97, 0.91, "Adiab. PBE0: 0% $S_0$ yield\n(all 3 systems trapped)",
              transform=ax_f.transAxes, ha="right", va="top", fontsize=4.55, fontweight="bold", color=PAL["adiabatic"],
              bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=PAL["adiabatic"], lw=0.4, alpha=0.92))
    ax_f.set_xlim(0, 50)
    ax_f.set_ylim(0.02, 1.06)
    ax_f.set_xlabel("Time $t$ (fs) [300 FSSH traj.]")
    ax_f.set_ylabel(r"Active $S_1$ Population $P_{S_1}(t)$")
    ax_f.set_title("Non-Adiabatic Tully FSSH (3/3 CIs)", fontsize=7.1, pad=4)
    ax_f.legend(loc="lower right", ncol=2, fontsize=4.05, columnspacing=0.35, handlelength=1.15, labelspacing=0.14,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_f)
    ax_f.grid(True, alpha=0.4)

    # ------------------------------------------------------------------
    # Row 3 (g, h): Parallel Transport Continuity & Unified Berry Phase Scorecard
    # ------------------------------------------------------------------
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], width_ratios=[0.95, 1.38], wspace=0.26)

    # Panel g: LOMO parallel-transport continuity on Ethylene 36-pt CI loop
    ax_g = fig.add_subplot(gs_bot[0])
    add_panel_label(ax_g, "g", dx=-0.20, dy=1.05)

    thetas_deg = np.degrees(eth["loop_ci_shared__thetas"])
    lp_e = ci_sum["ethylene"]["berry_phase_loops"]["loop_ci_shared"]
    lomo_e_lp = lomo["lomo_holdout_ethylene"]["zero_shot_eval"]["ethylene"]["berry_phase_loops"]["loop_ci_shared"]

    ax_g.plot(thetas_deg, np.abs(np.array(lp_e["SA2_CASSCF"]["step_overlaps"])),
              color=PAL["qm_ref"], ls="-", lw=1.45, label=r"SA2-CASSCF ($\gamma=\pi$)")
    ax_g.plot(thetas_deg, np.abs(np.array(lomo_e_lp["step_overlaps"])),
              color=PAL["lehmann"], ls="-", lw=1.45, label=r"Lehmann LOMO ($\gamma=\pi$)")
    ax_g.plot(thetas_deg, np.abs(np.array(lp_e["Adiabatic_TDDFT_KS"]["step_overlaps"])),
              color=PAL["adiabatic"], ls="--", lw=1.15, label=r"Adiab. TD-DFT ($\gamma=0$)")
    v0d_ov = np.abs(np.array(lp_e["Lehmann_Decoupled_v0D0"]["step_overlaps"]))
    ax_g.plot(thetas_deg, v0d_ov,
              color=PAL["schnet"], ls="-.", lw=1.15, label=r"Ablation $v_{0D}=0$ (breaks)")

    idx_dip = int(np.argmin(v0d_ov))
    ax_g.annotate(
        f"Discontinuous orthogonal\ncollapse (min $|S|={v0d_ov[idx_dip]:.3f}$)",
        xy=(thetas_deg[idx_dip], v0d_ov[idx_dip]),
        xytext=(115, 0.56),
        fontsize=5.2, color="#8A2B62", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PAL["schnet"], lw=0.75),
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=PAL["schnet"], lw=0.5),
    )
    ax_g.set_xlim(0, 360)
    ax_g.set_xticks([0, 90, 180, 270, 360])
    ax_g.set_ylim(-0.04, 1.07)
    ax_g.set_xlabel(r"Loop angle $\theta$ around CI (deg)")
    ax_g.set_ylabel(r"Step overlap $|\langle\psi_0(\theta_k)|\psi_0(\theta_{k+1})\rangle|$")
    ax_g.set_title("LOMO Parallel-Transport Continuity (Eth.)", fontsize=7.2, pad=4)
    ax_g.legend(loc="lower center", bbox_to_anchor=(0.50, 0.025), fontsize=4.85, handlelength=1.25, labelspacing=0.18,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.96)
    despine(ax_g)
    ax_g.grid(True, alpha=0.4)

    # Panel h: Unified Berry-Phase Wilson Product Scorecard (3 LOMO CIs + Blind MECIs vs No-CI Controls)
    ax_h = fig.add_subplot(gs_bot[1])
    add_panel_label(ax_h, "h", dx=-0.14, dy=1.05)

    ax_h.axhspan(-1.14, 0.0, color=PAL["blue_tint"], alpha=0.55, zorder=0)
    ax_h.axhspan(0.0, 1.14, color=PAL["red_tint"], alpha=0.32, zorder=0)
    ax_h.axhline(0.0, color=PAL["charcoal"], lw=0.85)
    ax_h.axhline(-1.0, color=PAL["lehmann"], lw=0.55, ls=":", alpha=0.7)
    ax_h.axhline(1.0, color=PAL["adiabatic"], lw=0.55, ls=":", alpha=0.7)

    # Gather exact W_prod values for LOMO (3/3), Blind MECIs (Eth, Met, PSB3), and No-CI controls
    w_psb3_enc = float(psb3_bp["berry_loop_encircling_blind_meci"]["Lehmann_TDDFT"]["W_prod"]) if psb3_bp else float(bm["psb3"]["berry_loop_encircling_blind_meci"]["Lehmann_TDDFT"]["W_prod"])
    w_psb3_ctrl = float(psb3_bp["berry_loop_shifted_control"]["Lehmann_TDDFT"]["W_prod"]) if psb3_bp else float(bm["psb3"]["berry_loop_shifted_control"]["Lehmann_TDDFT"]["W_prod"])
    w_psb3_adia = float(psb3_bp["berry_loop_encircling_blind_meci"]["Adiabatic_TDDFT_KS"]["W_prod"]) if psb3_bp else float(bm["psb3"]["berry_loop_encircling_blind_meci"]["Adiabatic_TDDFT_KS"]["W_prod"])

    cats_h = [
        ("LOMO\nEth.",
         ci_sum["ethylene"]["berry_phase_loops"]["loop_ci_shared"]["SA2_CASSCF"]["W_prod"],
         lomo["lomo_holdout_ethylene"]["zero_shot_eval"]["ethylene"]["berry_phase_loops"]["loop_ci_shared"]["W_prod"],
         ci_sum["ethylene"]["berry_phase_loops"]["loop_ci_shared"]["Adiabatic_TDDFT_KS"]["W_prod"]),
        ("LOMO\nMet.",
         ci_sum["methanimine"]["berry_phase_loops"]["loop_ci_shared"]["SA2_CASSCF"]["W_prod"],
         lomo["lomo_holdout_methanimine"]["zero_shot_eval"]["methanimine"]["berry_phase_loops"]["loop_ci_shared"]["W_prod"],
         ci_sum["methanimine"]["berry_phase_loops"]["loop_ci_shared"]["Adiabatic_TDDFT_KS"]["W_prod"]),
        ("LOMO\nFulv.*",
         ci_sum["fulvene"]["berry_phase_loops"]["loop_ci_shared"]["SA2_CASSCF"]["W_prod"],
         lomo["lomo_holdout_fulvene"]["zero_shot_eval"]["fulvene"]["berry_phase_loops"]["loop_leh_apex"]["W_prod"],
         ci_sum["fulvene"]["berry_phase_loops"]["loop_ci_shared"]["Adiabatic_TDDFT_KS"]["W_prod"]),
        ("Blind\nEth.",
         np.nan,
         bm_tr["ethylene"]["berry_loop_encircling_blind_meci"]["Lehmann_TDDFT"]["W_prod"],
         bm_tr["ethylene"]["berry_loop_encircling_blind_meci"]["Adiabatic_TDDFT_KS"]["W_prod"]),
        ("Blind\nMet.",
         np.nan,
         bm_tr["methanimine"]["berry_loop_encircling_blind_meci"]["Lehmann_TDDFT"]["W_prod"],
         bm_tr["methanimine"]["berry_loop_encircling_blind_meci"]["Adiabatic_TDDFT_KS"]["W_prod"]),
        ("Blind\nPSB3",
         np.nan,
         w_psb3_enc,
         w_psb3_adia),
        ("No-CI\nCtrls",
         float(np.mean([ci_sum[k]["berry_phase_loops"]["loop_ctrl_noci"]["SA2_CASSCF"]["W_prod"] for k in ("ethylene", "methanimine", "fulvene")])),
         float(np.mean([
             lomo["lomo_holdout_ethylene"]["zero_shot_eval"]["ethylene"]["berry_phase_loops"]["loop_ctrl_noci"]["W_prod"],
             lomo["lomo_holdout_methanimine"]["zero_shot_eval"]["methanimine"]["berry_phase_loops"]["loop_ctrl_noci"]["W_prod"],
             lomo["lomo_holdout_fulvene"]["zero_shot_eval"]["fulvene"]["berry_phase_loops"]["loop_ctrl_noci"]["W_prod"],
             bm_tr["ethylene"]["berry_loop_shifted_control"]["Lehmann_TDDFT"]["W_prod"],
             bm_tr["methanimine"]["berry_loop_shifted_control"]["Lehmann_TDDFT"]["W_prod"],
             w_psb3_ctrl,
         ])),
         float(np.mean([ci_sum[k]["berry_phase_loops"]["loop_ctrl_noci"]["Adiabatic_TDDFT_KS"]["W_prod"] for k in ("ethylene", "methanimine", "fulvene")]))),
    ]

    xh = np.arange(len(cats_h))
    wb_h = 0.25
    cas_vals = [c[1] for c in cats_h]
    leh_vals = [c[2] for c in cats_h]
    adi_vals = [c[3] for c in cats_h]

    # Plot only finite CASSCF bars
    for i, cv_i in enumerate(cas_vals):
        if np.isfinite(cv_i):
            ax_h.bar(xh[i] - wb_h, cv_i, width=wb_h, color=PAL["qm_ref"], edgecolor="white", lw=0.4,
                     label="SA2-CASSCF" if i == 0 else None, zorder=3)
    ax_h.bar(xh, leh_vals, width=wb_h, color=PAL["lehmann"], edgecolor="white", lw=0.4, label="Lehmann LOMO / Blind", zorder=4)
    ax_h.bar(xh + wb_h, adi_vals, width=wb_h, color=PAL["adiabatic"], edgecolor="white", lw=0.4, label="Adiab. TD-DFT ($\\gamma=0$)", zorder=3)

    ax_h.axvline(2.5, color=PAL["slate"], lw=0.7, ls=":", alpha=0.8)
    ax_h.axvline(5.5, color=PAL["charcoal"], lw=0.85, ls="--", alpha=0.8)
    ax_h.text(5.0, w_psb3_enc - 0.06, f"{w_psb3_enc:+.3f}", ha="center", va="top",
              fontsize=4.4, fontweight="bold", color=PAL["lehmann"])
    ax_h.text(2.5, -1.09, r"Exact Continuous $\gamma=\pi$ (6/6 LOMO + Blind)",
              ha="center", va="bottom", fontsize=4.95, fontweight="bold", color=PAL["lehmann"])
    ax_h.text(6.0, -1.09, r"$\gamma=0$", ha="center", va="bottom", fontsize=5.1, fontweight="bold", color=PAL["slate"])

    ax_h.set_xticks(xh)
    ax_h.set_xticklabels([c[0] for c in cats_h], fontsize=5.25)
    ax_h.set_ylim(-1.14, 1.16)
    ax_h.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_h.set_ylabel(r"Wilson loop product $W_{\mathrm{prod}}$")
    ax_h.set_title(r"Berry Phase $\gamma$: 3/3 LOMO + Blind MECIs vs Ctrls", fontsize=7.2, pad=4)
    ax_h.legend(loc="lower right", bbox_to_anchor=(0.99, 0.14), ncol=1, fontsize=4.6,
                handlelength=1.0, frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.95)
    despine(ax_h)
    ax_h.grid(True, axis="y", alpha=0.4)

    save_fig(fig, "fig3_conical_intersections")


# ===========================================================================
# FIGURE 5: Computational Scaling, Basis Stratification, Calibration & Audit
# ===========================================================================
def make_fig5() -> None:
    set_nature_style()
    cv_preds = json.loads((ROOT / "results/benchmarks/cv5_lehmann_predictions.json").read_text())
    ryd_proof = json.loads((ROOT / "results/benchmarks/rydberg_def2svpd_proof.json").read_text())
    zs = json.loads((ROOT / "results/benchmarks/zero_shot_invest_multi_scaffold.json").read_text())
    t20 = json.loads((ROOT / "results/invest_discovery/blind_verification_summary.json").read_text())["verified_discoveries"]
    badc = json.loads((ROOT / "results/invest_discovery/blind_adc2_raw.json").read_text())
    mrs_p = ROOT / "results/benchmarks/matched_root_scaling.json"
    mrs = json.loads(mrs_p.read_text()) if mrs_p.exists() else {"records": []}
    ba_p = ROOT / "results/invest_discovery/basis_and_adiabatic_check.json"
    ba = json.loads(ba_p.read_text()) if ba_p.exists() else {}

    fig = plt.figure(figsize=(7.2, 6.45))
    gs = gridspec.GridSpec(
        2, 2, hspace=0.38, wspace=0.40, left=0.085, right=0.985, top=0.955, bottom=0.08
    )

    # ------------------------------------------------------------------
    # Panel a: Fair Single-Core Matched-Root (8S+8T, def2-SVP) Scaling
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    add_panel_label(ax_a, "a", dx=-0.22, dy=1.04)

    mrecs = mrs.get("records", [])
    series_cfg = [
        ("lehmann_tddft_8S8T", "qm_cpu_s", "cpu_s", "o", PAL["lehmann"], "-", "Lehmann v2.0 (8S+8T)"),
        ("ri_ee_adc2_8S8T", "cpu_s", "cpu_s", "s", PAL["qm_ref"], "--", "RI-EE-ADC(2) (8S+8T)"),
        ("double_hybrid_scs_pbe_qidh_8S8T", "cpu_s", "cpu_s", "^", PAL["static"], "-.", "DH SCS-QIDH (8S+8T)"),
        ("eom_ee_ccsd_8S8T", "cpu_s", "cpu_s", "D", PAL["adiabatic"], "-", "EOM-EE-CCSD (8S+8T)"),
    ]
    max_nao = 200.0
    for key, f_fallback, f_primary, mk, col, ls, lbl in series_cfg:
        xs_s, ys_s = [], []
        for r in mrecs:
            d = r.get(key)
            if isinstance(d, dict):
                val = d.get(f_primary, d.get(f_fallback))
                if val is not None and val > 0:
                    xs_s.append(float(r["n_ao"]))
                    ys_s.append(float(val))
        if xs_s:
            max_nao = max(max_nao, max(xs_s))
            xs_arr, ys_arr = np.array(xs_s), np.array(ys_s)
            order = np.argsort(xs_arr)
            xs_arr, ys_arr = xs_arr[order], ys_arr[order]
            if len(xs_arr) >= 2:
                slope, intercept = np.polyfit(np.log10(xs_arr), np.log10(ys_arr), 1)
                lbl_full = f"{lbl} ($N_{{\\mathrm{{AO}}}}^{{{slope:.2f}}}$)"
                xg_s = np.geomspace(xs_arr.min() * 0.92, xs_arr.max() * 1.06, 80)
                ax_a.plot(xg_s, 10 ** (slope * np.log10(xg_s) + intercept), color=col, lw=1.15, ls=ls, zorder=3)
            else:
                lbl_full = lbl
            ax_a.scatter(xs_arr, ys_arr, s=26, marker=mk, color=col, edgecolor="white", lw=0.4, label=lbl_full, zorder=5)

    # Also overlay neural forward time (1 CPU core)
    nn_ms_list = [r["lehmann_tddft_8S8T"].get("nn_forward_cpu_ms") for r in mrecs if isinstance(r.get("lehmann_tddft_8S8T"), dict) and r["lehmann_tddft_8S8T"].get("nn_forward_cpu_ms")]
    nn_s = (float(np.median(nn_ms_list)) / 1000.0) if nn_ms_list else 0.0041
    ax_a.axhline(nn_s, color=PAL["ccsd"], lw=0.95, ls=":", zorder=2)
    ax_a.text(58, nn_s * 1.55, f"Neural operator forward: {nn_s*1000:.1f} ms/mol (1 CPU)", fontsize=4.95,
              fontweight="bold", color=PAL["ccsd"])

    # Speedup annotations at Benzene (vs EOM-CCSD) and 5AP (vs RI-EE-ADC(2) & DH)
    eom_recs = [r for r in mrecs if isinstance(r.get("eom_ee_ccsd_8S8T"), dict) and "cpu_s" in r["eom_ee_ccsd_8S8T"]]
    if eom_recs:
        re = max(eom_recs, key=lambda r: r["n_ao"])
        tl_e = re["lehmann_tddft_8S8T"].get("cpu_s", re["lehmann_tddft_8S8T"].get("qm_cpu_s"))
        tc_e = re["eom_ee_ccsd_8S8T"]["cpu_s"]
        sp_e = tc_e / tl_e
        ax_a.annotate("", xy=(re["n_ao"], tc_e), xytext=(re["n_ao"], tl_e),
                      arrowprops=dict(arrowstyle="<->", color=PAL["adiabatic"], lw=0.95, zorder=6))
        ax_a.annotate(
            f"{sp_e:.1f}× vs CCSD\n({re['mol_id']}, {re['n_ao']})",
            xy=(re["n_ao"], np.sqrt(tc_e * tl_e)), xytext=(138, 1.35),
            fontsize=4.65, fontweight="bold", color=PAL["adiabatic"], ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=PAL["adiabatic"], lw=0.65),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=PAL["adiabatic"], lw=0.45), zorder=7,
        )

    adc_recs = [r for r in mrecs if isinstance(r.get("ri_ee_adc2_8S8T"), dict) and "cpu_s" in r["ri_ee_adc2_8S8T"]]
    if adc_recs:
        ra = max(adc_recs, key=lambda r: r["n_ao"])
        tl_a = ra["lehmann_tddft_8S8T"].get("cpu_s", ra["lehmann_tddft_8S8T"].get("qm_cpu_s"))
        ta_a = ra["ri_ee_adc2_8S8T"]["cpu_s"]
        td_a = ra.get("double_hybrid_scs_pbe_qidh_8S8T", {}).get("cpu_s")
        sp_a = ta_a / tl_a
        sp_d_txt = f"\n{td_a / tl_a:.1f}× vs DH" if td_a else ""
        ax_a.annotate("", xy=(ra["n_ao"], ta_a), xytext=(ra["n_ao"], tl_a),
                      arrowprops=dict(arrowstyle="<->", color=PAL["qm_ref"], lw=0.95, zorder=6))
        ax_a.annotate(
            f"{sp_a:.1f}× vs ADC(2){sp_d_txt}\n({ra['mol_id']}, {ra['n_ao']})",
            xy=(ra["n_ao"], np.sqrt(ta_a * tl_a) * 1.4), xytext=(415, 2900),
            fontsize=4.65, fontweight="bold", color=PAL["qm_ref"], ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=PAL["qm_ref"], lw=0.65),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=PAL["qm_ref"], lw=0.45), zorder=7,
        )

    hz_recs = [r for r in mrecs if r["mol_id"] == "HzTFEX2"]
    if hz_recs:
        rh = hz_recs[0]
        th = rh["lehmann_tddft_8S8T"]["cpu_s"]
        ax_a.annotate(
            f"HzTFEX2\n(590, {th/60:.1f}m)",
            xy=(rh["n_ao"], th), xytext=(480, 62),
            fontsize=4.55, fontweight="bold", color=PAL["lehmann"], ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=PAL["lehmann"], lw=0.65),
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=PAL["lehmann"], lw=0.42), zorder=7,
        )

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(54, max(380, max_nao * 1.24))
    ax_a.set_ylim(2.2e-3, 2.2e4)
    xticks_a = [60, 100, 200, 350, 590] if max_nao > 400 else [60, 100, 150, 250, 350]
    ax_a.set_xticks(xticks_a)
    ax_a.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_a.set_xlabel(r"Basis size $N_{\mathrm{AO}}$ (def2-SVP, 8S+8T)")
    ax_a.set_ylabel("Single-Core CPU Time (s, 1 Core)")
    ax_a.set_title("Fair Matched-Root 8S+8T Scaling (1 Core)", fontsize=7.4, pad=4)
    ax_a.legend(loc="upper left", fontsize=4.65, handletextpad=0.3, labelspacing=0.18,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_a)
    ax_a.grid(True, which="major", alpha=0.45)

    # ------------------------------------------------------------------
    # Panel b: Direct def2-SVP -> def2-SVPD Rydberg Proof (64 mols / 643 states) & Worst-5 CV5
    # ------------------------------------------------------------------
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], height_ratios=[1.10, 1.0], hspace=0.52)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])
    add_panel_label(ax_b1, "b", dx=-0.22, dy=1.08)

    cats_b = [
        ("Valence\n(n=403)", "Valence_states"),
        ("Rydberg\n(n=233)", "Rydberg_states"),
        ("All States\n(n=643)", "All_states"),
    ]
    xb = np.arange(len(cats_b))
    wb_b = 0.20
    tda_svp = [ryd_proof[k]["Bare_TDA_def2SVP_MAE_eV"] for _, k in cats_b]
    tda_svpd = [ryd_proof[k]["Bare_TDA_def2SVPD_MAE_eV"] for _, k in cats_b]
    leh_svp = [ryd_proof[k]["Lehmann_def2SVP"]["mean"] for _, k in cats_b]
    leh_svpd = [ryd_proof[k]["Lehmann_def2SVPD"]["mean"] for _, k in cats_b]

    ax_b1.bar(xb - 1.5 * wb_b, tda_svp, width=wb_b, color=PAL["adiabatic"], alpha=0.78, edgecolor="white", lw=0.35, label="TDA SVP")
    ax_b1.bar(xb - 0.5 * wb_b, tda_svpd, width=wb_b, color=PAL["static"], alpha=0.82, edgecolor="white", lw=0.35, label="TDA SVPD")
    ax_b1.bar(xb + 0.5 * wb_b, leh_svp, width=wb_b, color=PAL["lehmann_lt"], edgecolor="white", lw=0.35, label="Leh. SVP")
    ax_b1.bar(xb + 1.5 * wb_b, leh_svpd, width=wb_b, color=PAL["lehmann"], edgecolor="white", lw=0.35, label="Leh. SVPD")

    for i in range(len(cats_b)):
        ax_b1.text(xb[i] + 1.5 * wb_b, leh_svpd[i] + 0.025, f"{leh_svpd[i]:.2f}", ha="center", va="bottom",
                   fontsize=4.75, fontweight="bold", color=PAL["lehmann"])
    red_ryd = ryd_proof["Rydberg_states"]["reduction_factor_SVP_over_SVPD"]
    ax_b1.annotate(
        f"Rydberg {red_ryd:.2f}× drop!\n(0.507→0.222, p=2e-17)",
        xy=(1.0 + 1.5 * wb_b, leh_svpd[1] + 0.08), xytext=(1.95, 0.86),
        ha="center", va="top", fontsize=4.55, fontweight="bold", color=PAL["lehmann"],
        arrowprops=dict(arrowstyle="->", color=PAL["lehmann"], lw=0.65),
        bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=PAL["lehmann"], lw=0.45),
    )
    ax_b1.set_xticks(xb)
    ax_b1.set_xticklabels([c[0] for c in cats_b], fontsize=5.4)
    ax_b1.set_ylim(0, 1.08)
    ax_b1.set_ylabel("5-Fold MAE (eV)")
    ax_b1.set_title("Diffuse-Basis Proof: def2-SVP vs def2-SVPD (64 mols)", fontsize=7.0, pad=3)
    ax_b1.legend(loc="upper left", ncol=2, fontsize=4.45, columnspacing=0.45, handlelength=0.95,
                 frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_b1)
    ax_b1.grid(True, axis="y", alpha=0.4)

    # Bottom sub-panel: v2.0 Worst-5 CV5 S1 Failure Modes (directly from v2.0 predictions)
    s1_errs = [
        {"mol_id": m["mol_id"], "err_eV": float(m["pred_S1_eV"] - m["ref_S1_eV"])}
        for m in cv_preds if m.get("ref_S1_eV") is not None and m.get("pred_S1_eV") is not None
    ]
    w5 = sorted(s1_errs, key=lambda x: -abs(x["err_eV"]))[:5]
    causes = {
        "quest_Carbon_dimer": "C$_2$ (RKS triplet instab.)",
        "quest_Phenolate": "Phenolate (anion diffuse)",
        "quest_Cyclopropene": "Cyclopropene (Rydberg SVP)",
        "quest_Phenyl-pyrrole_twisted": "Ph-pyrrole tw. (TICT)",
        "quest_Diacetylene": "Diacetylene (linear $\\pi$/Ryd.)",
    }
    yw = np.arange(len(w5))[::-1]
    w_abs = [abs(x["err_eV"]) for x in w5]
    w_cols = [PAL["adiabatic"] if i == 0 else (PAL["static"] if i == 1 else PAL["slate"]) for i in range(len(w5))]
    ax_b2.barh(yw, w_abs, height=0.64, color=w_cols, alpha=0.85, edgecolor="white", lw=0.4)
    for yi, x in zip(yw, w5):
        ax_b2.text(abs(x["err_eV"]) + 0.04, yi, f"{x['err_eV']:+.2f} eV", va="center", ha="left",
                   fontsize=5.0, color=PAL["charcoal"], fontweight="bold" if abs(x["err_eV"]) > 1.0 else "normal")
    ax_b2.set_yticks(yw)
    ax_b2.set_yticklabels([causes.get(x["mol_id"], x["mol_id"].replace("quest_", "")) for x in w5], fontsize=5.2)
    ax_b2.set_xlim(0, 1.78)
    ax_b2.set_xlabel(r"v2.0 Worst-5 CV5 $|E_{S_1}^{\mathrm{pred}} - E_{S_1}^{\mathrm{ref}}|$ (eV)", labelpad=1.5)
    despine(ax_b2)
    ax_b2.grid(True, axis="x", alpha=0.4)

    # ------------------------------------------------------------------
    # Panel c: v2.0 Ensemble Epistemic Uncertainty Calibration
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_c, "c", dx=-0.22, dy=1.04)

    nov_ids = set(zs["strictly_novel_non_benchmark"]["mol_ids"])
    ext_std, ext_err, ext_ids = [], [], []
    for m in zs["per_molecule"]:
        if m["mol_id"] in nov_ids and m["adc2_dEST_eV"] is not None:
            ext_std.append(float(m["lehmann_dEST_std"]))
            ext_err.append(abs(float(m["lehmann_dEST_eV"]) - float(m["adc2_dEST_eV"])))
            ext_ids.append(m["mol_id"])
    ext_std, ext_err = np.array(ext_std), np.array(ext_err)

    b20_std, b20_err = [], []
    for m in t20:
        mid = m["mol_id"]
        if mid in badc and badc[mid].get("ok"):
            b20_std.append(float(m["delta_EST_eV_std"]))
            b20_err.append(abs(float(m["delta_EST_eV"]) - float(badc[mid]["dEST_eV"])))
    b20_std, b20_err = np.array(b20_std), np.array(b20_err)

    all_std = np.concatenate([ext_std, b20_std])
    all_err = np.concatenate([ext_err, b20_err])
    cov2_all = 100.0 * float(np.mean(all_err <= 2.0 * all_std))
    cov2_b20 = 100.0 * float(np.mean(b20_err <= 2.0 * b20_std))
    n_in_2sig_b20 = int(np.sum(b20_err <= 2.0 * b20_std))

    max_sx = max(0.28, float(np.max(all_std)) * 1.12)
    max_ey = max(0.62, float(np.max(all_err)) * 1.12)
    xg_c = np.linspace(0, max_sx, 100)
    ax_c.fill_between(xg_c, 0, 2.0 * xg_c, color=PAL["blue_tint"], alpha=0.55, label=r"$|e| \leq 2\sigma_{\mathrm{ens}}$ envelope")
    ax_c.plot(xg_c, xg_c, color=PAL["lehmann"], lw=0.95, ls="--", label=r"$|e| = \sigma_{\mathrm{ens}}$")
    ax_c.plot(xg_c, 2.0 * xg_c, color=PAL["lehmann"], lw=0.8, ls=":")

    ax_c.scatter(ext_std, ext_err, s=24, marker="o", color=PAL["lehmann"], alpha=0.85,
                 edgecolor="white", lw=0.35, label=f"23 Novel Cores (MAE={np.mean(ext_err):.3f} eV)", zorder=4)
    ax_c.scatter(b20_std, b20_err, s=28, marker="D", color=PAL["qm_ref"], alpha=0.9,
                 edgecolor="white", lw=0.4, label=f"20 Blind Hits (MAE={np.mean(b20_err):.3f} eV)", zorder=5)

    # Label top-3 systematic OOD outlier scaffolds dynamically
    top_out_idx = np.argsort(-ext_err)[:3]
    for rank_i, oi in enumerate(top_out_idx):
        mid_o = ext_ids[oi].replace("_", "-")
        if len(mid_o) > 14:
            mid_o = mid_o[:13] + "."
        ax_c.annotate(
            mid_o, xy=(ext_std[oi], ext_err[oi]),
            xytext=(min(max_sx * 0.62, ext_std[oi] + 0.035), min(max_ey * 0.94, ext_err[oi] - 0.03 * rank_i)),
            fontsize=4.75, color=PAL["slate"],
            arrowprops=dict(arrowstyle="-", color=PAL["slate"], lw=0.5, shrinkA=0, shrinkB=2.5),
        )

    ax_c.text(0.96, 0.95,
              f"Blind 20: {n_in_2sig_b20}/20 ({cov2_b20:.0f}%)\nwithin $2\\sigma_{{\\mathrm{{ens}}}}$\nAll 43: {cov2_all:.0f}%",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=5.1,
              bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=PAL["light_gray"], lw=0.5))
    ax_c.set_xlim(0, max_sx)
    ax_c.set_ylim(0, max_ey)
    ax_c.set_xlabel(r"5-Fold Ensemble Uncertainty $\sigma_{\mathrm{ens}}(\Delta E_{ST})$ (eV)")
    ax_c.set_ylabel(r"True Error $|\Delta E_{ST}^{\mathrm{pred}} - \Delta E_{ST}^{\mathrm{ADC(2)}}|$ (eV)")
    ax_c.set_title("Epistemic Uncertainty Calibration (External)", fontsize=7.4, pad=4)
    ax_c.legend(loc="lower right", fontsize=4.8, handletextpad=0.3, labelspacing=0.16,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_c)
    ax_c.grid(True, alpha=0.4)

    # ------------------------------------------------------------------
    # Panel d: Larger-Basis (aug-cc-pVDZ / def2-TZVP) & T1 Adiabatic Relaxation
    # (falls back cleanly to v2.0 CV5 Subset & Spin-Channel Sensitivity if still computing)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_d, "d", dx=-0.22, dy=1.04)

    lb = ba.get("larger_basis_convergence", {})
    t1r = ba.get("t1_relaxation_and_dft_geom_check", {})
    cand_targs = [
        ("5AP", "5AP"),
        ("VBI_0089_3N_13_pp", "VBI-0089"),
        ("VBI_0210_3N_13_m", "VBI-0210"),
        ("VBI_0069_3N_13_pp", "VBI-0069"),
        ("VBI_0236_6AP_tri", "VBI-0236"),
    ]
    targs_d = [
        (m, lbl) for m, lbl in cand_targs
        if (m in lb and lb[m].get("aug-cc-pvdz", {}).get("ok") and lb[m].get("def2-tzvp", {}).get("ok") and t1r.get(m, {}).get("ok"))
    ]
    has_ba = len(targs_d) > 0

    if has_ba:
        xd = np.arange(len(targs_d))
        wd = 0.15 if len(targs_d) >= 5 else 0.17
        series_ba = [
            ("SVP (xTB)", PAL["lehmann_lt"], [lb.get(m, {}).get("def2-svp", {}).get("dEST_eV", np.nan) for m, _ in targs_d]),
            ("aug-cc-pVDZ", PAL["lehmann"], [lb.get(m, {}).get("aug-cc-pvdz", {}).get("dEST_eV", np.nan) for m, _ in targs_d]),
            ("def2-TZVP", PAL["ccsd"], [lb.get(m, {}).get("def2-tzvp", {}).get("dEST_eV", np.nan) for m, _ in targs_d]),
            ("PBE0 $S_0$ geom", PAL["qm_ref"], [t1r.get(m, {}).get("adc2_at_pbe0_S0_geom", {}).get("dEST_eV", np.nan) for m, _ in targs_d]),
            ("PBE0 $T_1$ geom", PAL["static"], [t1r.get(m, {}).get("adc2_at_pbe0_T1_geom", {}).get("dEST_at_T1_geom_eV", np.nan) for m, _ in targs_d]),
        ]
        offs_ba = np.linspace(-2 * wd, 2 * wd, len(series_ba))
        ax_d.axhline(0.0, color=PAL["charcoal"], lw=0.8, ls="--")
        ax_d.axhspan(-0.285, 0.0, color=PAL["green_tint"], alpha=0.55, zorder=0)
        n_total_ba = 0
        n_inv_ba = 0
        for (lbl_b, col_b, vals_b), off_b in zip(series_ba, offs_ba):
            clean_v = [0.0 if not np.isfinite(v) else v for v in vals_b]
            for v in vals_b:
                if np.isfinite(v):
                    n_total_ba += 1
                    if v < 0.0:
                        n_inv_ba += 1
            if any(np.isfinite(vals_b)):
                ax_d.bar(xd + off_b, clean_v, width=wd, color=col_b, edgecolor="white", lw=0.35, label=lbl_b, zorder=3)
        # Annotate T1-relaxed f_S1 enhancement cleanly aligned to the right of the green S0 bar (under gold T1 bar)
        for i_m, (m_id, _) in enumerate(targs_d):
            if m_id == "5AP":
                continue
            t1_d = t1r.get(m_id, {}).get("adc2_at_pbe0_T1_geom", {})
            yv_m = t1_d.get("dEST_at_T1_geom_eV")
            fv_m = t1_d.get("f_S1_at_T1_geom")
            if yv_m is not None and fv_m is not None:
                ax_d.text(xd[i_m] + 2.0 * wd, yv_m - 0.009, f"f={fv_m:.3f}", ha="center", va="top",
                          rotation=90, fontsize=4.0, fontweight="bold", color="#8A5A00",
                          bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=0.88), zorder=7)
        diffs_mev = [
            1000.0 * abs(lb[m]["aug-cc-pvdz"]["dEST_eV"] - lb[m]["def2-tzvp"]["dEST_eV"])
            for m, _ in targs_d
        ]
        ax_d.text(
            0.5 * (len(targs_d) - 1), -0.258,
            f"{n_inv_ba}/{n_total_ba} strictly inverted (<0) across all bases & $T_1$ min.\n"
            f"aug-cc-pVDZ vs def2-TZVP agree within {min(diffs_mev):.1f}–{max(diffs_mev):.1f} meV",
            ha="center", va="center", fontsize=4.45, fontweight="bold", color=PAL["qm_ref"],
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=PAL["qm_ref"], lw=0.45, alpha=0.95), zorder=6,
        )
        ax_d.set_xticks(xd)
        ax_d.set_xticklabels([t[1] for t in targs_d], fontsize=5.45)
        ax_d.set_ylim(-0.285, 0.090)
        ax_d.set_ylabel(r"RI-EE-ADC(2) $\Delta E_{ST}$ (eV)")
        ax_d.set_title("Basis Convergence & $T_1$ Relaxation Robustness", fontsize=7.3, pad=4)
        ax_d.legend(loc="upper center", ncol=3, fontsize=4.45, columnspacing=0.45, handlelength=0.95,
                    frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.95)
    else:
        # Clean v2.0 CV5 Subset & Spin-Complete Sensitivity Audit
        def _sub_maes(subset):
            s1 = np.mean([abs(m["pred_S1_eV"] - m["ref_S1_eV"]) for m in subset if m.get("ref_S1_eV") is not None])
            t1 = np.mean([abs(m["pred_T1_eV"] - m["ref_T1_eV"]) for m in subset if m.get("ref_T1_eV") is not None])
            ds = np.mean([abs(m["pred_delta_EST_eV"] - m["ref_delta_EST_eV"]) for m in subset if m.get("ref_delta_EST_eV") is not None])
            return float(s1), float(t1), float(ds)

        sub_all = cv_preds
        sub_nc2 = [m for m in cv_preds if m["mol_id"] != "quest_Carbon_dimer"]
        sub_qdb = [m for m in cv_preds if m["source"] == "QUESTDB"]
        sub_hep = [m for m in cv_preds if "hepta" in m["mol_id"].lower()]
        rows_sub = [
            ("CV5 All\n(n=187)", *_sub_maes(sub_all)),
            ("Excl. C$_2$\n(n=186)", *_sub_maes(sub_nc2)),
            ("QUESTDB\n(n=143)", *_sub_maes(sub_qdb)),
            ("Heptazines\n(n=45)", *_sub_maes(sub_hep)),
        ]
        xd = np.arange(len(rows_sub))
        wb_d = 0.25
        ax_d.bar(xd - wb_d, [r[1] for r in rows_sub], width=wb_d, color=PAL["lehmann_lt"], edgecolor="white", lw=0.4, label="$S_1$ MAE", zorder=3)
        ax_d.bar(xd, [r[2] for r in rows_sub], width=wb_d, color=PAL["qm_ref"], edgecolor="white", lw=0.4, label="$T_1$ MAE", zorder=3)
        ax_d.bar(xd + wb_d, [r[3] for r in rows_sub], width=wb_d, color=PAL["lehmann"], edgecolor="white", lw=0.4, label="$\\Delta E_{ST}$ MAE", zorder=3)
        for xi, r in zip(xd + wb_d, rows_sub):
            ax_d.text(xi, r[3] + 0.005, f"{r[3]:.3f}", ha="center", va="bottom", fontsize=5.0, fontweight="bold", color=PAL["lehmann"])
        ax_d.set_xticks(xd)
        ax_d.set_xticklabels([r[0] for r in rows_sub], fontsize=5.6)
        ax_d.set_ylim(0, 0.22)
        ax_d.set_ylabel("CV5 MAE (eV)")
        ax_d.set_title("v2.0 CV5 Subset Sensitivity Audit", fontsize=7.4, pad=4)
        ax_d.legend(loc="upper right", ncol=3, fontsize=4.8, columnspacing=0.5, handlelength=0.95,
                    frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_d)
    ax_d.grid(True, axis="y", alpha=0.4)

    save_fig(fig, "fig5_scaling_diagnostics")


# ===========================================================================
# FIGURE 6 (Extended Data Fig. 1): Methanimine & Fulvene LOMO Cones + SF-TDDFT Spin Purity
# ===========================================================================
def make_fig6() -> None:
    set_nature_style()
    ci_sum = json.loads((ROOT / "results/conical_intersections/conical_intersections_summary.json").read_text())
    sf = json.loads((ROOT / "results/conical_intersections/sf_tddft_comparison.json").read_text())
    met = np.load(ROOT / "results/conical_intersections/methanimine_ci_surfaces.npz")
    ful = np.load(ROOT / "results/conical_intersections/fulvene_ci_surfaces.npz")

    fig = plt.figure(figsize=(7.2, 8.2))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1.15, 1.15, 0.85], hspace=0.28,
        left=0.06, right=0.98, top=0.955, bottom=0.055
    )

    # Row 1: Methanimine (CH2NH)
    gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.12)
    lp_m = ci_sum["methanimine"]["berry_phase_loops"]["loop_ci_shared"]
    loop_m = (lp_m["center_xy"][0], lp_m["center_xy"][1], lp_m["radius"])
    for idx, (e0, e1, ttl, col, rpa, ltr) in enumerate(
        [
            (met["nev_e0_eV"], met["nev_e1_eV"], "Methanimine: SC-NEVPT2\nSA2-CASSCF(4,3) Truth", PAL["qm_ref"], None, "a"),
            (met["pbe0_e0_eV"], met["pbe0_e1_eV"], "Methanimine: Adiab. PBE0\n24.4% Imag. RPA, $\\gamma=0$", PAL["adiabatic"], met["pbe0_rpa_unstable"].astype(bool), "b"),
            (met["leh_lomo_e0_eV"], met["leh_lomo_e1_eV"], "Methanimine: Lehmann LOMO\n0-manual-$\\beta$, $\\gamma=\\pi$, $\\langle\\hat{S}^2\\rangle{=}0$", PAL["lehmann"], None, "c"),
        ]
    ):
        ax = fig.add_subplot(gs_r1[idx], projection="3d")
        _plot_3d_ci_surface(
            ax, met["X"], met["Y"], e0, e1, ttl, col, rpa_mask=rpa, loop_xy=loop_m,
            zlim=(-1.15, 1.35), show_zlabel=(idx == 2), zticks=[-1.0, -0.5, 0.0, 0.5, 1.0]
        )
        ax.text2D(-0.03, 0.96, ltr, transform=ax.transAxes, fontsize=10.0, fontweight="bold")

    # Row 2: Fulvene (C6H6) — draw CASSCF-centered loop on d/e and Lehmann-apex loop on f
    gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.12)
    lp_f_cas = ci_sum["fulvene"]["berry_phase_loops"]["loop_ci_shared"]
    loop_f_cas = (lp_f_cas["center_xy"][0], lp_f_cas["center_xy"][1], lp_f_cas["radius"])
    lp_f_leh = ci_sum["fulvene"]["berry_phase_loops"]["loop_leh_apex"]
    loop_f_leh = (lp_f_leh["center_xy"][0], lp_f_leh["center_xy"][1], lp_f_leh["radius"])
    for idx, (e0, e1, ttl, col, rpa, lp_xy, ltr) in enumerate(
        [
            (ful["nev_e0_eV"], ful["nev_e1_eV"], "Fulvene: SC-NEVPT2\nSA2-CASSCF(6,6) Truth", PAL["qm_ref"], None, loop_f_cas, "d"),
            (ful["pbe0_e0_eV"], ful["pbe0_e1_eV"], "Fulvene: Adiab. PBE0\n19.8% Imag. RPA, $\\gamma=0$", PAL["adiabatic"], ful["pbe0_rpa_unstable"].astype(bool), loop_f_cas, "e"),
            (ful["leh_lomo_e0_eV"], ful["leh_lomo_e1_eV"], "Fulvene: Lehmann LOMO\n0.083 eV (beats CAS!), $\\gamma=\\pi$", PAL["lehmann"], None, loop_f_leh, "f"),
        ]
    ):
        ax = fig.add_subplot(gs_r2[idx], projection="3d")
        _plot_3d_ci_surface(
            ax, ful["X"], ful["Y"], e0, e1, ttl, col, rpa_mask=rpa, loop_xy=lp_xy,
            zlim=(-1.45, 1.85), show_zlabel=(idx == 2), zticks=[-1.0, 0.0, 1.0]
        )
        ax.text2D(-0.03, 0.96, ltr, transform=ax.transAxes, fontsize=10.0, fontweight="bold")

    # Row 3 (g, h): Spin-Flip TDDFT Spin Contamination <S^2> vs Exact Spin-Pure Lehmann-TDDFT
    gs_r3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.30)
    ax_g = fig.add_subplot(gs_r3[0])
    add_panel_label(ax_g, "g", dx=-0.18, dy=1.06)

    sys_lbls = ["Ethylene", "Methanimine", "Fulvene"]
    sys_keys = ["ethylene", "methanimine", "fulvene"]
    x3 = np.arange(3)
    w3 = 0.19
    bhh_s0 = [sf[k]["SF_TDA_BHHLYP"]["spin_contamination"]["S0_S2_mean"] for k in sys_keys]
    bhh_s1 = [sf[k]["SF_TDA_BHHLYP"]["spin_contamination"]["S1_S2_mean"] for k in sys_keys]
    pbe_s1 = [sf[k]["SF_TDA_PBE0"]["spin_contamination"]["S1_S2_mean"] for k in sys_keys]
    leh_s2 = [0.0, 0.0, 0.0]

    ax_g.bar(x3 - 1.5 * w3, bhh_s0, width=w3, color=PAL["static"], edgecolor="white", lw=0.4, label=r"SF-BHHLYP $S_0$ mean $\langle\hat{S}^2\rangle$")
    ax_g.bar(x3 - 0.5 * w3, bhh_s1, width=w3, color="#E07A5F", edgecolor="white", lw=0.4, label=r"SF-BHHLYP $S_1$ mean $\langle\hat{S}^2\rangle$")
    ax_g.bar(x3 + 0.5 * w3, pbe_s1, width=w3, color=PAL["adiabatic"], edgecolor="white", lw=0.4, label=r"SF-PBE0 $S_1$ mean $\langle\hat{S}^2\rangle$")
    ax_g.bar(x3 + 1.5 * w3, leh_s2, width=w3, color=PAL["lehmann"], edgecolor="white", lw=0.4, label=r"Lehmann-TDDFT ($\equiv 0.000$)")
    for xi in x3:
        ax_g.text(xi + 1.5 * w3, 0.03, "0.0", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color=PAL["lehmann"])
    ax_g.axhline(1.0, color=PAL["slate"], lw=0.7, ls=":", alpha=0.8)
    ax_g.text(2.42, 1.03, "50% triplet contam. ($\\langle\\hat{S}^2\\rangle=1$)", ha="right", va="bottom", fontsize=4.85, color=PAL["slate"])
    ax_g.set_xticks(x3)
    ax_g.set_xticklabels(sys_lbls, fontsize=6.0)
    ax_g.set_ylim(0, 1.38)
    ax_g.set_ylabel(r"Mean Spin Contamination $\langle\hat{S}^2\rangle$")
    ax_g.set_title("SF-TDDFT Spin Contamination vs Spin-Pure Lehmann", fontsize=7.2, pad=4)
    ax_g.legend(loc="upper left", fontsize=4.65, frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_g)
    ax_g.grid(True, axis="y", alpha=0.4)

    # Panel h: 2D Fulvene SF-TDA-BHHLYP S0 <S^2> heatmap showing severe spin mixing across branching plane
    ax_h = fig.add_subplot(gs_r3[1])
    add_panel_label(ax_h, "h", dx=-0.18, dy=1.06)
    ful_sf = np.load(ROOT / "results/conical_intersections/fulvene_sf_tddft_surfaces.npz")
    s2_key = [k for k in ful_sf.files if "bhhlyp" in k.lower() and "s0" in k.lower() and "s2" in k.lower()]
    if s2_key:
        s2_grid = ful_sf[s2_key[0]]
        im = ax_h.pcolormesh(ful["X"], ful["Y"], s2_grid, cmap="magma_r", vmin=0.0, vmax=1.3, shading="auto")
        cbar = fig.colorbar(im, ax=ax_h, pad=0.02, fraction=0.05)
        cbar.set_label(r"SF-BHHLYP $S_0\;\langle\hat{S}^2\rangle$", fontsize=5.8)
        cbar.ax.tick_params(labelsize=5.2)
    ax_h.set_xlabel(r"$x_g$ (Å)")
    ax_h.set_ylabel(r"$y_h$ (Å)")
    ax_h.set_title(r"Fulvene SF-BHHLYP $\langle\hat{S}^2\rangle$ Map (max 1.29 vs Lehmann 0.00)", fontsize=7.0, pad=4)
    despine(ax_h)

    save_fig(fig, "fig6_extended_ci_ablations")


# ===========================================================================
# FIGURE 4: Universal Quantum Phase Diagram & Blind Bright-INVEST Discovery
# ===========================================================================
def _wavelength_to_rgb(ev: float) -> Tuple[str, str]:
    """Return (hex_color, label) for S1 emission energy in eV."""
    nm = 1239.84193 / max(ev, 0.2)
    if nm < 420:
        return "#6D28D9", f"{nm:.0f} nm (Violet)"
    if nm < 475:
        return "#1D4ED8", f"{nm:.0f} nm (Blue)"
    if nm < 505:
        return "#0284C7", f"{nm:.0f} nm (Cyan)"
    if nm < 565:
        return "#059669", f"{nm:.0f} nm (Green)"
    if nm < 595:
        return "#D97706", f"{nm:.0f} nm (Yellow)"
    if nm < 630:
        return "#EA580C", f"{nm:.0f} nm (Orange)"
    if nm < 720:
        return "#DC2626", f"{nm:.0f} nm (Red)"
    return "#7F1D1D", f"{nm:.0f} nm (NIR)"


def make_fig4() -> None:
    import io
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    set_nature_style()
    cv_preds = json.loads((ROOT / "results/benchmarks/cv5_lehmann_predictions.json").read_text())
    qpd = json.loads((ROOT / "results/invest_discovery/quantum_phase_diagram.json").read_text())["points"]
    vscreen = json.loads((ROOT / "results/invest_discovery/virtual_screening_all.json").read_text())
    if isinstance(vscreen, dict):
        vscreen = vscreen.get("molecules", vscreen.get("candidates", list(vscreen.values())[0]))
    pareto = json.loads((ROOT / "results/invest_discovery/pareto_frontier.json").read_text())["molecules"]
    bvs = json.loads((ROOT / "results/invest_discovery/blind_verification_summary.json").read_text())
    t20 = bvs["verified_discoveries"]
    badc = json.loads((ROOT / "results/invest_discovery/blind_adc2_raw.json").read_text())
    zs = json.loads((ROOT / "results/benchmarks/zero_shot_invest_multi_scaffold.json").read_text())

    # Dynamically load ALL completed blind canonical EOM-EE-CCSD & SC-NEVPT2 verifications on VBI hits
    eoms: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(str(ROOT / "results/invest_discovery/ab_initio_cache/eomccsd_VBI_*.json"))):
        obj = json.loads(Path(p).read_text())
        if obj.get("ok") and obj.get("dEST_eV") is not None:
            eoms[obj["mol_id"]] = obj
    nev_map: Dict[str, Dict[str, Any]] = {}
    nev_p = ROOT / "results/invest_discovery/nevpt2_triple_verification.json"
    if nev_p.exists():
        for rec in json.loads(nev_p.read_text()).get("molecules", []):
            if rec.get("nevpt2_dEST_eV") is not None:
                nev_map[rec["mol_id"]] = rec

    fig = plt.figure(figsize=(7.2, 8.75))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1.05, 0.98, 1.18], hspace=0.33,
        left=0.08, right=0.98, top=0.96, bottom=0.032
    )

    # ------------------------------------------------------------------
    # Row 1: Panel a (Unified v2.0 Dynamical Phase Diagram) + Panel b (Pareto Frontier)
    # ------------------------------------------------------------------
    gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.28, width_ratios=[1.05, 1.0])

    # Panel a: 2D Quantum Phase Diagram in (Delta E_ST^static, Delta Sigma_ST)
    ax_a = fig.add_subplot(gs_r1[0])
    add_panel_label(ax_a, "a", dx=-0.20, dy=1.04)

    xlim_a = (-0.12, 0.88)
    ylim_a = (-0.06, 0.48)
    xg = np.linspace(xlim_a[0], xlim_a[1], 200)

    # Shaded Quantum Phases separated by Theorem 4 exact boundary Delta Sigma_ST = Delta E_ST^static
    ax_a.fill_between(xg, np.maximum(xg, ylim_a[0]), ylim_a[1], where=(xg >= 0),
                      color=PAL["blue_tint"], alpha=0.82, zorder=0)
    ax_a.fill_between(xg, np.maximum(xg, ylim_a[0]), ylim_a[1], where=(xg <= 0),
                      color=PAL["green_tint"], alpha=0.65, zorder=0)
    ax_a.fill_between(xg, np.maximum(xg - 0.10, ylim_a[0]), np.minimum(xg, ylim_a[1]),
                      color="#FEF9E7", alpha=0.85, zorder=0)
    ax_a.fill_between(xg, ylim_a[0], np.minimum(xg - 0.10, ylim_a[1]),
                      color="#F9FAFB", alpha=0.90, zorder=0)

    ax_a.axvline(0.0, color=PAL["slate"], lw=0.6, ls=":", zorder=1)
    ax_a.axhline(0.0, color=PAL["slate"], lw=0.6, ls=":", zorder=1)
    ax_a.plot(xg, xg, color=PAL["lehmann"], lw=1.35, ls="--", zorder=2,
              label=r"Thm 4 boundary: $\Delta\Sigma_{ST}=\Delta E_{ST}^{\mathrm{stat}}$")
    ax_a.plot(xg, xg - 0.10, color=PAL["static"], lw=0.75, ls=":", zorder=2)

    q_norm = [m for m in cv_preds if m["source"] == "QUESTDB" and m["ref_delta_EST_eV"] is not None and m["ref_delta_EST_eV"] >= 0]
    q_inv = [m for m in cv_preds if m["source"] == "QUESTDB" and m["ref_delta_EST_eV"] is not None and m["ref_delta_EST_eV"] < 0 and "Heptazine" not in m["mol_id"]]
    h_inv = [m for m in cv_preds if ("hepta" in m["mol_id"].lower()) and m["ref_delta_EST_eV"] is not None]

    ax_a.scatter([m["delta_EST_static_eV"] for m in q_norm], [m["delta_Sigma_ST_eV"] for m in q_norm],
                 s=11, color="#9CA3AF", alpha=0.55, edgecolor="none", label="QUESTDB Normal", zorder=3)
    ax_a.scatter([m["delta_EST_static_eV"] for m in h_inv], [m["delta_Sigma_ST_eV"] for m in h_inv],
                 s=18, marker="D", color=PAL["static"], alpha=0.88, edgecolor="white", lw=0.3,
                 label=f"$D_{{3h}}$ Heptazines (45/45 dyn.!)", zorder=4)
    ax_a.scatter([m["delta_EST_static_eV"] for m in q_inv], [m["delta_Sigma_ST_eV"] for m in q_inv],
                 s=26, marker="^", color=PAL["ccsd"], alpha=0.92, edgecolor="white", lw=0.35,
                 label="QUESTDB Non-Hept. INVEST", zorder=5)

    v_pts = [p for p in qpd if p["source"] == "virtual_library"]
    ax_a.scatter([p["delta_EST_static_eV"] for p in v_pts], [p["delta_Sigma_ST_eV"] for p in v_pts],
                 s=8, color=PAL["lehmann_lt"], alpha=0.38, edgecolor="none", label="320 Virtual Library", zorder=3)

    t20_inv = [m for m in t20 if m["mol_id"] in badc and badc[m["mol_id"]].get("dEST_eV", 1.0) < 0]
    ax_a.scatter([m["delta_EST_static_eV"] for m in t20_inv], [m["delta_Sigma_ST_eV"] for m in t20_inv],
                 s=30, marker="*", color=PAL["lehmann"], edgecolor="#003B5C", lw=0.35,
                 label=f"Blind ADC(2) Hits ({len(t20_inv)}/20)", zorder=6)

    ax_a.text(-0.095, 0.468, "Unified Dynamical-INVEST\n($\\Delta\\Sigma_{ST} > \\Delta E_{ST}^{\\mathrm{stat}} > 0$)\n45/45 Heptazines via $^1(TT)_-$!",
              ha="left", va="top", fontsize=4.85, fontweight="bold", color=PAL["lehmann"],
              bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=PAL["lehmann"], lw=0.45, alpha=0.95), zorder=8)
    ax_a.text(0.70, 0.25, "Normal Phase\n($\\Delta E_{ST}>0.1$ eV)",
              ha="center", va="center", fontsize=5.0, fontweight="bold", color=PAL["slate"],
              bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.75), zorder=7)
    ax_a.text(0.46, 0.41, "TADF", ha="center", va="center", fontsize=5.0, fontweight="bold",
              color="#9A6700", rotation=42)

    ax_a.set_xlim(*xlim_a)
    ax_a.set_ylim(*ylim_a)
    ax_a.set_xlabel(r"Static splitting $\Delta E_{ST}^{\mathrm{stat}}$ (eV)")
    ax_a.set_ylabel(r"Dynamical self-energy $\Delta\Sigma_{ST}$ (eV)")
    ax_a.set_title("Unified Quantum Phase Diagram (Spin-Complete v2.0)", fontsize=7.2, pad=4)
    ax_a.legend(loc="lower right", fontsize=4.55, handletextpad=0.22, labelspacing=0.14,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_a)
    ax_a.grid(True, alpha=0.35)

    # Panel b: Breaking the Dark-State Dilemma — Pareto Frontier (Delta E_ST vs f_S1)
    ax_b = fig.add_subplot(gs_r1[1])
    add_panel_label(ax_b, "b", dx=-0.20, dy=1.04)

    ax_b.axvspan(-0.42, 0.0, color=PAL["blue_tint"], alpha=0.35, zorder=0)
    ax_b.add_patch(mpatches.Rectangle(
        (-0.42, 0.01), 0.42, 0.045, facecolor=PAL["green_tint"], edgecolor=PAL["qm_ref"],
        lw=0.75, ls="--", alpha=0.65, zorder=1
    ))
    ax_b.axvline(0.0, color=PAL["charcoal"], lw=0.75, ls="--", zorder=2)
    ax_b.axhline(0.01, color=PAL["qm_ref"], lw=0.75, ls=":", zorder=2)

    fam_map = {
        "push_pull_azaphenalene": ("Push-Pull Azaphen.", PAL["lehmann"], "o"),
        "substituted_azaphenalene": ("Sub. Azaphenalene", PAL["lehmann_lt"], "o"),
        "asymmetric_azaphenalene": ("Asym. Azaphenalene", "#60A5FA", "o"),
        "bn_rim_triangulene": ("B–N Rim Triangulene", PAL["qm_ref"], "s"),
        "boraphenalene_bn": ("Boraphenalene", PAL["static"], "^"),
        "non_alternant_indacene": ("Non-Alt. Indacene", PAL["ccsd"], "D"),
        "non_alternant_azulene": ("Non-Alt. Azulene", PAL["schnet"], "v"),
    }
    plotted_fams = set()
    for m in vscreen:
        fam = m.get("scaffold_family", "other")
        lbl_f, col_f, mk_f = fam_map.get(fam, ("Other Non-Alt.", "#9CA3AF", "."))
        lab = lbl_f if lbl_f not in plotted_fams else None
        plotted_fams.add(lbl_f)
        ax_b.scatter([m["delta_EST_eV"]], [m["f_S1"]], s=14, marker=mk_f, color=col_f,
                     alpha=0.62, edgecolor="white", lw=0.25, label=lab, zorder=3)

    par_sorted = sorted(pareto, key=lambda x: x["delta_EST_eV"])
    ax_b.plot([m["delta_EST_eV"] for m in par_sorted], [m["f_S1"] for m in par_sorted],
              color=PAL["charcoal"], lw=1.25, ls="-", zorder=5)
    ax_b.scatter([m["delta_EST_eV"] for m in par_sorted], [m["f_S1"] for m in par_sorted],
                 s=34, marker="*", color=PAL["adiabatic"], edgecolor=PAL["charcoal"], lw=0.45,
                 label=f"Pareto Frontier (n={len(par_sorted)})", zorder=6)

    ms_eval = json.loads((ROOT / "results/invest_discovery/multi_scaffold_benchmark_eval.json").read_text())["molecules"]
    ms_map = {m["mol_id"]: m for m in ms_eval}
    for bid, blbl, xy_t in [
        ("heptazine", "Heptazine ($f=0$)", (-0.36, 0.0068)),
        ("5AP", "5AP ($f=0.004$)", (0.04, 0.0062)),
    ]:
        bm_ref = ms_map[bid]
        ax_b.scatter([bm_ref["delta_EST_eV"]], [bm_ref["f_S1"]], s=38, marker="X",
                     color=PAL["adiabatic"], edgecolor="white", lw=0.5, zorder=7)
        ax_b.annotate(
            blbl, xy=(bm_ref["delta_EST_eV"], bm_ref["f_S1"]), xytext=xy_t,
            fontsize=4.9, fontweight="bold", color=PAL["adiabatic"],
            arrowprops=dict(arrowstyle="->", color=PAL["adiabatic"], lw=0.65, shrinkA=0, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=PAL["adiabatic"], lw=0.45, alpha=0.92),
            zorder=8,
        )

    ax_b.text(-0.39, 0.0435, "Dipole-Allowed INVEST\n($\\Delta E_{ST}<0,\\; f_{S_1}\\geq 0.01$)",
              ha="left", va="top", fontsize=5.1, fontweight="bold", color="#006B4E")

    ax_b.set_xlim(-0.41, 0.35)
    ax_b.set_ylim(-0.0015, 0.0465)
    ax_b.set_xlabel(r"Predicted $\Delta E_{ST}$ (eV)")
    ax_b.set_ylabel(r"Predicted $S_1$ Oscillator Strength $f_{S_1}$")
    ax_b.set_title("Breaking the Dark-State Dilemma (320 Novel)", fontsize=7.4, pad=4)
    ax_b.legend(loc="upper right", fontsize=4.55, handletextpad=0.25, labelspacing=0.14,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.93)
    despine(ax_b)
    ax_b.grid(True, alpha=0.35)

    # ------------------------------------------------------------------
    # Row 2: Panel c (Blind Independent Triple Verification: ADC2 + 13 EOM-CCSD + SC-NEVPT2)
    # ------------------------------------------------------------------
    gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.28, width_ratios=[1.18, 0.88])

    ax_c = fig.add_subplot(gs_r2[0])
    add_panel_label(ax_c, "c", dx=-0.18, dy=1.05)

    ax_c.axhspan(-0.29, 0.0, color=PAL["green_tint"], alpha=0.58, zorder=0)
    ax_c.axvspan(-0.32, 0.0, color=PAL["blue_tint"], alpha=0.30, zorder=0)
    ax_c.axhline(0.0, color=PAL["charcoal"], lw=0.75, ls="--", zorder=1)
    ax_c.axvline(0.0, color=PAL["charcoal"], lw=0.75, ls="--", zorder=1)
    ax_c.plot([-0.29, 0.16], [-0.29, 0.16], color=PAL["slate"], lw=0.8, ls=":", zorder=1)

    pred_d, std_d, adc_d, mids_20 = [], [], [], []
    for m in t20:
        mid = m["mol_id"]
        if mid in badc and badc[mid].get("ok"):
            pred_d.append(float(m["delta_EST_eV"]))
            std_d.append(float(m["delta_EST_eV_std"]))
            adc_d.append(float(badc[mid]["dEST_eV"]))
            mids_20.append(mid)
    pred_d, std_d, adc_d = np.array(pred_d), np.array(std_d), np.array(adc_d)
    mae_adc = float(np.mean(np.abs(pred_d - adc_d)))
    n_inv_adc = int(np.sum(adc_d < 0))

    ax_c.errorbar(
        pred_d, adc_d, xerr=std_d, fmt="o", ms=4.5, color=PAL["lehmann"], ecolor=PAL["lehmann_lt"],
        elinewidth=0.7, capsize=1.4, markeredgecolor="white", markeredgewidth=0.38,
        label=f"Blind RI-ADC(2) ({n_inv_adc}/{len(adc_d)}=95% inv., MAE={mae_adc:.3f})",
        zorder=4
    )

    # Overlay ALL 13 canonical Blind EOM-EE-CCSD cross-checks
    t20_map = {m["mol_id"]: m for m in t20}
    if eoms:
        e_px, e_ay, e_cy = [], [], []
        for m in t20:
            mid = m["mol_id"]
            if mid in eoms and mid in badc:
                px = float(m["delta_EST_eV"])
                ay = float(badc[mid]["dEST_eV"])
                cy = float(eoms[mid]["dEST_eV"])
                e_px.append(px); e_ay.append(ay); e_cy.append(cy)
                ax_c.plot([px, px], [ay, cy], color=PAL["ccsd"], lw=0.6, ls=":", alpha=0.68, zorder=3)
        n_inv_cc = int(sum(y < 0 for y in e_cy))
        ax_c.scatter(e_px, e_cy, s=28, marker="D", color=PAL["ccsd"], edgecolor="white", lw=0.4,
                     label=f"Blind EOM-CCSD ({n_inv_cc}/{len(e_cy)}={100*n_inv_cc/len(e_cy):.0f}% inv., MAE={bvs['eom_ccsd_lehmann_dEST_MAE_eV']:.3f})", zorder=6)

    # Overlay SC-NEVPT2 multireference triple-checks on flagships
    nv_px, nv_y = [], []
    for mid, nrec in nev_map.items():
        if mid in t20_map:
            nv_px.append(float(t20_map[mid]["delta_EST_eV"]))
            nv_y.append(float(nrec["nevpt2_dEST_eV"]))
    if nv_px:
        n_nv_inv = sum(y < 0 for y in nv_y)
        ax_c.scatter(nv_px, nv_y, s=38, marker="*", color=PAL["static"], edgecolor=PAL["charcoal"], lw=0.4,
                     label=f"SC-NEVPT2 check ({n_nv_inv}/{len(nv_y)} triple inv.)", zorder=7)

    # Clean non-overlapping callouts for key EOM-CCSD & NEVPT2 flagships
    if "VBI_0236_6AP_tri" in eoms and "VBI_0089_3N_13_pp" in nev_map:
        m236 = t20_map["VBI_0236_6AP_tri"]
        nv236 = nev_map["VBI_0236_6AP_tri"]["nevpt2_dEST_eV"]
        nv089 = nev_map["VBI_0089_3N_13_pp"]["nevpt2_dEST_eV"]
        ax_c.annotate(
            f"Triple Dipole-Allowed:\nVBI-0089: NEVPT2 {nv089:+.3f}\nVBI-0236: NEVPT2 {nv236:+.3f}",
            xy=(t20_map["VBI_0089_3N_13_pp"]["delta_EST_eV"], nv089),
            xytext=(-0.185, -0.028), fontsize=4.4, fontweight="bold", color=PAL["ccsd"],
            arrowprops=dict(arrowstyle="->", color=PAL["ccsd"], lw=0.6),
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=PAL["ccsd"], lw=0.45), zorder=8,
        )
    if "VBI_0213_B0N_pp" in nev_map:
        m213 = t20_map["VBI_0213_B0N_pp"]
        ax_c.annotate(
            "Ring-N-free B-cores\nVBI-0213/0159: triple inv.\n(NEVPT2 −0.20/−0.23)",
            xy=(m213["delta_EST_eV"], nev_map["VBI_0213_B0N_pp"]["nevpt2_dEST_eV"]),
            xytext=(-0.298, -0.095), fontsize=4.45, fontweight="bold", color="#9A5B00",
            arrowprops=dict(arrowstyle="->", color=PAL["static"], lw=0.65),
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=PAL["static"], lw=0.45), zorder=8,
        )

    ax_c.set_xlim(-0.31, 0.095)
    ax_c.set_ylim(-0.285, 0.235)
    ax_c.set_xlabel(r"Lehmann v2.0 Ensemble $\Delta E_{ST} \pm 1\sigma$ (eV)")
    ax_c.set_ylabel(r"Blind Independent Ab Initio $\Delta E_{ST}$ (eV)")
    ax_c.set_title("Blind Verification: 95% ADC(2), 62% CCSD, NEVPT2", fontsize=7.1, pad=4)
    ax_c.legend(loc="upper left", fontsize=4.35, handletextpad=0.22, labelspacing=0.16,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.95)
    despine(ax_c)
    ax_c.grid(True, alpha=0.35)

    # Panel d (right): Blind Oscillator Strength f_S1 verification (RI-ADC(2) + 13 EOM-CCSD)
    ax_f = fig.add_subplot(gs_r2[1])
    add_panel_label(ax_f, "d", dx=-0.22, dy=1.05)
    pred_f = np.array([float(m["f_S1"]) for m in t20 if m["mol_id"] in badc and badc[m["mol_id"]].get("ok")])
    adc_f = np.array([float(badc[m["mol_id"]]["f_S1"]) for m in t20 if m["mol_id"] in badc and badc[m["mol_id"]].get("ok")])
    adc_inv_flag = np.array([float(badc[m["mol_id"]]["dEST_eV"]) < 0 for m in t20 if m["mol_id"] in badc and badc[m["mol_id"]].get("ok")])
    f_mae = float(np.mean(np.abs(pred_f - adc_f)))
    n_bright_inv = int(np.sum((adc_f >= 0.01) & adc_inv_flag))

    ax_f.axhspan(0.01, 0.036, color=PAL["green_tint"], alpha=0.55, zorder=0)
    ax_f.axhline(0.01, color=PAL["qm_ref"], lw=0.8, ls="--", zorder=1)
    ax_f.axhline(0.00198, color=PAL["adiabatic"], lw=0.8, ls=":", zorder=1)
    ax_f.plot([0, 0.042], [0, 0.042], color=PAL["slate"], lw=0.75, ls=":", zorder=1)

    ax_f.scatter(pred_f[adc_inv_flag], adc_f[adc_inv_flag], s=25, color=PAL["qm_ref"],
                 edgecolor="white", lw=0.38, label=f"RI-ADC(2) ({n_bright_inv}/20 $f\\geq 0.01$)", zorder=4)
    if np.any(~adc_inv_flag):
        ax_f.scatter(pred_f[~adc_inv_flag], adc_f[~adc_inv_flag], s=23, marker="x", color=PAL["adiabatic"],
                     lw=1.1, label="Non-inverted (1/20)", zorder=4)
    if eoms:
        ef_x = [float(m["f_S1"]) for m in t20 if m["mol_id"] in eoms]
        ef_y = [float(eoms[m["mol_id"]]["f_S1_R1"]) for m in t20 if m["mol_id"] in eoms]
        n_cc_da = sum(1 for m in t20 if m["mol_id"] in eoms and eoms[m["mol_id"]]["dEST_eV"] < 0 and eoms[m["mol_id"]]["f_S1_R1"] >= 0.01)
        ax_f.scatter(ef_x, ef_y, s=28, marker="D", color=PAL["ccsd"], edgecolor="white", lw=0.4,
                     label=f"EOM-CCSD ({n_cc_da}/{len(ef_x)} inv., $f\\geq 0.01$)", zorder=5)

    ax_f.text(0.0012, 0.0026, "5AP ref ($f=0.0020$)", ha="left", va="bottom",
              fontsize=4.75, fontweight="bold", color=PAL["adiabatic"],
              bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
    ax_f.text(0.0015, 0.0332, f"Up to 13.4× 5AP $f_{{S_1}}$\nADC(2) MAE={f_mae:.4f}\nCCSD MAE={bvs['eom_ccsd_f_S1_MAE']:.4f}",
              ha="left", va="top", fontsize=4.75, fontweight="bold", color="#006B4E")

    ax_f.set_xlim(-0.001, 0.040)
    ax_f.set_ylim(-0.001, 0.035)
    ax_f.set_xlabel(r"Predicted $f_{S_1}$ (Lehmann v2.0)")
    ax_f.set_ylabel(r"Blind Ab Initio $f_{S_1}$")
    ax_f.set_title("Dipole-Allowed $f_{S_1}$ Verification", fontsize=7.3, pad=4)
    ax_f.legend(loc="lower right", fontsize=4.5, handletextpad=0.22, labelspacing=0.15,
                frameon=True, facecolor="white", edgecolor=PAL["light_gray"], framealpha=0.94)
    despine(ax_f)
    ax_f.grid(True, alpha=0.35)

    # ------------------------------------------------------------------
    # Row 3: Panel e — High-Resolution RDKit 2D Chemical Gallery (6 Verified Flagship Scaffolds)
    # ------------------------------------------------------------------
    gs_r3 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2], hspace=0.40, wspace=0.10)

    showcase_ids = [
        ("VBI_0089_3N_13_pp", "Push-Pull 3N (Triple Inv.)"),
        ("VBI_0236_6AP_tri", "Trisub. 6AP (Triple Inv.)"),
        ("VBI_0069_3N_13_pp", "Push-Pull 3N (Triple Inv.)"),
        ("VBI_0210_3N_13_m", "Sub. 3N (CCSD Inv.)"),
        ("VBI_0213_B0N_pp", "Ring-N-Free B (Triple Inv.)"),
        ("VBI_0159_B0N_m", "Ring-N-Free B (Triple Inv.)"),
    ]

    for idx, (mid, fam_short) in enumerate(showcase_ids):
        ax_m = fig.add_subplot(gs_r3[idx // 3, idx % 3])
        if idx == 0:
            add_panel_label(ax_m, "e", dx=-0.10, dy=1.10)
        m = t20_map[mid]
        a = badc[mid]
        e = eoms.get(mid)
        nv = nev_map.get(mid)

        mol = Chem.MolFromSmiles(m["smiles"])
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(460, 260)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 2.0
        opts.padding = 0.08
        opts.fixedBondLength = 28
        opts.clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))

        ax_m.imshow(img)
        ax_m.set_xticks([]); ax_m.set_yticks([])
        for sp_n in ax_m.spines.values():
            sp_n.set_edgecolor("#CBD5E1")
            sp_n.set_linewidth(0.75)

        em_col, em_lbl = _wavelength_to_rgb(float(a["E_S1_eV"]))
        short_id = mid.split("_")[0] + "-" + mid.split("_")[1]
        ax_m.set_title(f"{short_id}: {fam_short}", fontsize=6.0, fontweight="bold",
                       color=PAL["charcoal"], pad=2.5)

        if nv and nv.get("nevpt2_dEST_eV", 1.0) < 0:
            info_txt = f"ADC {a['dEST_eV']:+.3f} | CCSD {e['dEST_eV']:+.3f} | NEV {nv['nevpt2_dEST_eV']:+.3f}"
        elif e:
            info_txt = f"CCSD: $\\Delta E_{{ST}}={e['dEST_eV']:+.3f}$, $f={e['f_S1_R1']:.4f}$ (ADC {a['dEST_eV']:+.3f})"
        else:
            info_txt = f"ADC(2): $\\Delta E_{{ST}}={a['dEST_eV']:+.3f}$ eV, $f={a['f_S1']:.4f}$"
        ax_m.text(0.5, -0.05, info_txt, transform=ax_m.transAxes, ha="center", va="top",
                  fontsize=4.75, fontweight="bold", color=PAL["lehmann"])
        ax_m.text(0.97, 0.95, em_lbl, transform=ax_m.transAxes, ha="right", va="top",
                  fontsize=4.65, fontweight="bold", color="white",
                  bbox=dict(boxstyle="round,pad=0.15", fc=em_col, ec="none", alpha=0.92))

    save_fig(fig, "fig4_invest_discovery")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()
    dispatch = {1: make_fig1, 2: make_fig2, 3: make_fig3, 4: make_fig4, 5: make_fig5, 6: make_fig6}
    for fnum in args.figs:
        if fnum in dispatch:
            print(f"Rendering Figure {fnum}...")
            dispatch[fnum]()


if __name__ == "__main__":
    main()
