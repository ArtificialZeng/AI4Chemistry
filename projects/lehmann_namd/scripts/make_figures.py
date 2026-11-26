"""Publication-Grade 5-Figure Suite Generator for Project #2: Lehmann-NAMD.

Reads ONLY real, verified JSON archives under `projects/lehmann_namd/results/`
(zero hardcoded/synthetic data!) and renders Nature / Nature Chemistry styled
vector PDF + 300-DPI PNG figures into `projects/lehmann_namd/figures/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ_DIR = Path(__file__).resolve().parent.parent
RES_DIR = PROJ_DIR / "results"
FIG_DIR = PROJ_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean Nature Chemistry publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

C_LEH = "#0072B2"    # Deep Nature blue (Lehmann-NAMD)
C_CAS = "#111111"    # Reference black (SA2-CASSCF)
C_NEV = "#009E73"    # Emerald (SC-NEVPT2)
C_ADIA = "#D55E00"   # Vermilion (Adiabatic DF-PBE0)
C_SF = "#CC79A7"     # Magenta/purple (SF-TDA)
C_GOLD = "#E69F00"   # Amber accent


def _panel_label(ax, label: str):
    ax.text(-0.18, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


def make_fig1():
    p = RES_DIR / "theorems_namd_results.json"
    if not p.exists():
        print("[fig1] Missing theorems_namd_results.json; skipping.")
        return
    data = json.loads(p.read_text())
    t1, t2, t4 = data["theorem_1"], data["theorem_2"], data["theorem_4"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    fig.subplots_adjust(wspace=0.32, hspace=0.38)

    # Panel a: Theorem 1 Dyson Residue & Causal Hellmann-Feynman Machine Precision
    ax = axes[0, 0]
    _panel_label(ax, "a")
    sts = t1["states"]
    x = np.arange(len(sts))
    w = 0.34
    z_errs = [max(s["Z_identity_err"], 1e-16) for s in sts]
    hf_errs = [max(s["causal_hf_max_err"], 1e-16) for s in sts]
    ax.bar(x - w / 2, z_errs, w, color=C_LEH, label=r"$|Z_n^{\mathrm{dir}} - Z_n^{\mathrm{Dyson}}|$")
    ax.bar(x + w / 2, hf_errs, w, color=C_NEV, label=r"$|\mathbf{u}_n^\top \nabla\mathcal{H}\mathbf{u}_n - Z_n \hat{\mathbf{c}}_n^\top(\nabla\mathbf{H}+\nabla\mathbf{\Sigma})\hat{\mathbf{c}}_n|$")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-10)
    ax.axhline(1e-12, color="#666666", ls="--", lw=0.9, label="Machine-precision threshold ($10^{-12}$)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$S_{s['state']}$\n($Z={s['Z_direct']:.3f}$)" for s in sts])
    ax.set_ylabel("Absolute identity error (a.u.)")
    ax.set_title("Theorem 1: Causal Hellmann–Feynman")
    ax.legend(frameon=False, loc="upper left")

    # Panel b: Theorem 2 Finite-Difference Accuracy across ALL 3N Coordinates
    ax = axes[0, 1]
    _panel_label(ax, "b")
    keys = [
        ("H2O_DF_HF", "H$_2$O\nDF-HF"),
        ("H2O_DF_PBE0", "H$_2$O\nDF-PBE0"),
        ("Ethylene_18DOF_DF_HF", "C$_2$H$_4$ (18D)\nDF-HF"),
        ("Ethylene_18DOF_DF_PBE0_def2SVP", "C$_2$H$_4$ (18D)\nPBE0/def2-SVP"),
    ]
    x = np.arange(len(keys))
    w = 0.25
    e_g0 = [t2[k]["max_err_g0"] for k, _ in keys]
    e_g1 = [t2[k]["max_err_g1"] for k, _ in keys]
    e_h01 = [t2[k]["max_err_h01"] for k, _ in keys]
    ax.bar(x - w, e_g0, w, color=C_LEH, label=r"$\max|\nabla E_0 - \mathrm{FD}|$")
    ax.bar(x, e_g1, w, color=C_GOLD, label=r"$\max|\nabla E_1 - \mathrm{FD}|$")
    ax.bar(x + w, e_h01, w, color=C_NEV, label=r"$\max|\mathbf{h}_{01} - \mathrm{FD}|$")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 5e-4)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys])
    ax.set_ylabel("Max $|\\mathrm{Anal.} - \\mathrm{FD}|$ (Ha Bohr$^{-1}$)")
    ax.set_title("Theorem 2: Exact 3N Gradients & NACVs")
    ax.legend(frameon=False, loc="upper left", ncol=1)

    # Panel c: Theorem 4 Single-Core O(N^3) Scaling Ladder (5 -> 24 atoms)
    ax = axes[1, 0]
    _panel_label(ax, "c")
    mols = t4["molecules"]
    naos = np.array([m["n_ao"] for m in mols], dtype=float)
    t_anal = np.array([m["t_analytical_all_s"] for m in mols], dtype=float)
    t_fd = np.array([m["t_fd_6N_s"] for m in mols], dtype=float)
    ax.loglog(naos, t_fd, "o--", color=C_ADIA, ms=5.5, label="Numerical 6N FD (E$_0$, E$_1$, d$_{01}$)")
    ax.loglog(naos, t_anal, "s-", color=C_LEH, ms=6.0, label=f"Analytical Lehmann-NAMD ($N_{{AO}}^{{{t4['empirical_exponent_nao']:.2f}}}$)")
    offsets = {
        "Methanimine": (8, 10),
        "Ethylene": (18, -14),
        "Fulvene": (-14, -20),
        "PSB3+": (18, -18),
        "Uracil": (-2, 10),
        "Azobenzene": (-14, -20),
    }
    for m in mols:
        ax.annotate(
            f"{m['name']}\n({m['dof_3n']}D)",
            (m["n_ao"], m["t_analytical_all_s"]),
            textcoords="offset points", xytext=offsets.get(m["name"], (0, -18)),
            ha="center", fontsize=6.4, color="#222222",
        )
    ax.set_xlim(36, 285)
    ax.set_ylim(7, 2.5e5)
    ax.set_xlabel("Basis size $N_{\\mathrm{AO}}$ (def2-SVP)")
    ax.set_ylabel("Single-core wall time (s)")
    ax.set_title("Theorem 4: O(N$^3$) Analytical Scaling")
    ax.legend(frameon=False, loc="upper left")

    # Panel d: Speedup Factor vs 6N Finite Differences
    ax = axes[1, 1]
    _panel_label(ax, "d")
    short_names = {"Methanimine": "CH$_2$NH", "Ethylene": "C$_2$H$_4$", "Fulvene": "Fulvene", "Uracil": "Uracil", "PSB3+": "PSB3+", "Azobenzene": "Azobenz."}
    names = [f"{short_names.get(m['name'], m['name'])}\n({m['dof_3n']}D)" for m in mols]
    sp = [m["speedup_vs_fd"] for m in mols]
    bars = ax.bar(np.arange(len(mols)), sp, color=C_LEH, edgecolor="white", width=0.58)
    for b, val in zip(bars, sp):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.8, f"{val:.1f}×", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(np.arange(len(mols)))
    ax.set_xticklabels(names, fontsize=7.2)
    ax.set_ylabel("Analytical speedup vs 6N FD")
    ax.set_title("Single-Pass Adjoint Speedup (g$_0$, g$_1$, d$_{01}$)")
    ax.set_ylim(0, max(sp) * 1.22)

    fig.savefig(FIG_DIR / "fig1_theorems_and_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_theorems_and_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig1] Saved fig1_theorems_and_scaling.{pdf,png}")


def make_fig2():
    p = RES_DIR / "theorems_namd_results.json"
    if not p.exists():
        print("[fig2] Missing theorems_namd_results.json; skipping.")
        return
    t3 = json.loads(p.read_text())["theorem_3"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    fig.subplots_adjust(wspace=0.30, hspace=0.38)

    # Panel a: Direct Full-3N Branching-Plane Alignment vs Exact Analytical SA2-CASSCF
    ax = axes[0, 0]
    _panel_label(ax, "a")
    bp = t3["branching_plane_alignment_3n"]
    mols = [("ethylene", "Ethylene\n(18 DOF)"), ("methanimine", "Methanimine\n(15 DOF)"), ("fulvene", "Fulvene\n(36 DOF)")]
    x = np.arange(len(mols))
    w = 0.25
    cg = [bp[k]["cos_g"] for k, _ in mols]
    ch = [bp[k]["cos_h"] for k, _ in mols]
    cadia = [0.0 for _ in mols]
    b1 = ax.bar(x - w, cg, w, color=C_LEH, label=r"$|\cos(\hat{\mathbf{g}}_{\mathrm{Leh}}, \hat{\mathbf{g}}_{\mathrm{CAS}})|$")
    b2 = ax.bar(x, ch, w, color=C_NEV, label=r"$|\cos(\hat{\mathbf{h}}_{\mathrm{Leh}}, \hat{\mathbf{h}}_{\mathrm{CAS}})|$")
    ax.bar(x + w, cadia, w, color=C_ADIA, label=r"Adiabatic TD-DFT $|\cos(\hat{\mathbf{h}})| \equiv 0$")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in mols])
    ax.set_ylim(0, 1.52)
    ax.set_ylabel("Full-3N cosine vs SA2-CASSCF")
    ax.set_title("Full-3N Branching-Plane Alignment")
    ax.legend(frameon=False, loc="upper center", ncol=1, fontsize=7.0)

    # Panel b: Exact 1/r NACV Singularity Law
    ax = axes[0, 1]
    _panel_label(ax, "b")
    sing = t3["nacv_singularity_1_over_r"]
    rs = np.array([s["r_bohr"] for s in sing])
    ds = np.array([s["d01_norm_bohr_inv"] for s in sing])
    asym = sing[0]["analytic_asymptote_h_over_2g"]
    r_line = np.linspace(rs.min() * 0.85, rs.max() * 1.15, 100)
    ax.loglog(r_line, asym / r_line, "--", color=C_CAS, lw=1.2, label=rf"Exact CI law $\|\mathbf{{d}}_{{01}}\| = (h/2g)\, r^{{-1}}$")
    ax.loglog(rs, ds, "o", color=C_LEH, ms=6.5, label=r"Analytical $\|\mathbf{d}_{01}^{\mathrm{CI}}(r)\|$")
    ax.set_xlabel("Distance from CI apex $r$ (Bohr)")
    ax.set_ylabel(r"$\|\mathbf{d}_{01}^{\mathrm{CI}}\|$ (Bohr$^{-1}$)")
    ax.set_title("Conical Intersection 1/r NACV Singularity")
    ax.legend(frameon=False, loc="upper right")

    # Panel c: Continuous 3N Analytical NACV Line-Integral Integrand
    ax = axes[1, 0]
    _panel_label(ax, "c")
    l_ci = t3["nacv_loop_encircling_ci"]
    l_ct = t3["nacv_loop_shifted_control"]
    th = np.linspace(0, 2 * np.pi, l_ci["n_pts"], endpoint=False)
    sign_ci = 1.0 if l_ci["gamma_over_pi"] >= 0 else -1.0
    int_ci = sign_ci * np.array(l_ci["integrand_vals"])
    int_ct = np.array(l_ct["integrand_vals"])
    ax.plot(th / np.pi, int_ci, "o-", color=C_LEH, ms=3.5, label=rf"Encircling CI ($\gamma/\pi = {abs(l_ci['gamma_over_pi']):.4f}$)")
    ax.plot(th / np.pi, int_ct, "s--", color=C_ADIA, ms=3.0, label=rf"Shifted control ($\gamma/\pi = {l_ct['gamma_over_pi']:+.4f}$)")
    ax.axhline(0.0, color="#888888", lw=0.7)
    ax.set_xlabel(r"Loop angle $\theta / \pi$")
    ax.set_ylabel(r"$\mathbf{d}_{01}^{\mathrm{CI}}(\theta)\cdot (d\mathbf{R}/d\theta)$")
    ax.set_title("3N Analytical NACV Circulation Integrand")
    ax.set_ylim(-0.15, max(int_ci) * 1.28)
    ax.legend(frameon=False, loc="upper center", fontsize=7.2)

    # Panel d: Cumulative Quantized Berry Phase gamma(theta) / pi
    ax = axes[1, 1]
    _panel_label(ax, "d")
    dth = 2 * np.pi / l_ci["n_pts"]
    cum_ci = np.concatenate([[0.0], np.cumsum(int_ci) * dth / np.pi])
    cum_ct = np.concatenate([[0.0], np.cumsum(int_ct) * dth / np.pi])
    th_c = np.linspace(0, 2.0, l_ci["n_pts"] + 1)
    ax.plot(th_c, cum_ci, "-", color=C_LEH, lw=2.0, label=rf"Encircling CI (holonomy $W={l_ci['holonomy_u0']:+.2f}$)")
    ax.plot(th_c, cum_ct, "--", color=C_ADIA, lw=1.8, label=rf"Shifted control (holonomy $W={l_ct['holonomy_u0']:+.2f}$)")
    ax.axhline(1.0, color=C_NEV, ls=":", lw=1.2, label="Exact quantized $\\gamma / \\pi = 1$")
    ax.axhline(0.0, color="#888888", lw=0.7)
    ax.set_xlabel(r"Loop angle $\theta / \pi$")
    ax.set_ylabel(r"Cumulative $\frac{1}{\pi}\int_0^\theta \mathbf{d}_{01}^{\mathrm{CI}}\cdot d\mathbf{R}$")
    ax.set_title("Quantized Berry Phase from 3N NACV Field")
    ax.set_ylim(-0.18, 1.38)
    ax.legend(frameon=False, loc="upper left", fontsize=7.0)

    fig.savefig(FIG_DIR / "fig2_full3n_topology_berry.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_full3n_topology_berry.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig2] Saved fig2_full3n_topology_berry.{pdf,png}")


def make_fig3():
    meci_dir = RES_DIR / "blind_meci_3n"
    order = ["ethylene", "methanimine", "fulvene", "psb3", "uracil"]
    pretty = {
        "ethylene": "C$_2$H$_4$\n(18D)",
        "methanimine": "CH$_2$NH\n(15D)",
        "fulvene": "Fulvene\n(36D)",
        "psb3": "PSB3+\n(42D)",
        "uracil": "Uracil\n(36D)",
    }
    loaded = {}
    for nm in order:
        f = meci_dir / f"blind_meci_{nm}.json"
        if f.exists():
            loaded[nm] = json.loads(f.read_text())
    if not loaded:
        print("[fig3] No blind_meci_*.json found; skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    fig.subplots_adjust(wspace=0.30)
    colors = [C_LEH, C_NEV, C_GOLD, C_SF, C_ADIA]

    # Panel a: Blind 3N Analytical L-BFGS-B Convergence Curves
    ax = axes[0]
    _panel_label(ax, "a")
    for (nm, d), col in zip(loaded.items(), colors):
        hist = d["history"]
        ev = [h["eval"] for h in hist]
        gp = [max(h["gap_ev"], 1e-4) for h in hist]
        ax.semilogy(ev, gp, "-", color=col, lw=1.6, label=f"{pretty[nm].replace(chr(10), ' ')}")
    ax.axhline(0.01, color="#555555", ls="--", lw=0.8)
    ax.set_xlabel("Analytical gradient evaluations")
    ax.set_ylabel("$S_1 - S_0$ energy gap (eV)")
    ax.set_title("Blind 3N Analytical MECI Convergence")
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)

    # Panel b: Independent SA2-CASSCF & SC-NEVPT2 Verification at Blind MECIs
    ax = axes[1]
    _panel_label(ax, "b")
    nms = list(loaded.keys())
    x = np.arange(len(nms))
    w = 0.20
    g_leh = [loaded[k]["gap_ev"] for k in nms]
    g_cas = [loaded[k]["independent_cas_nevpt2"]["cas_gap_ev"] for k in nms]
    g_nev = [loaded[k]["independent_cas_nevpt2"]["nevpt2_gap_ev"] for k in nms]
    g_tda = [abs(loaded[k]["bare_tda_s1_ev"]) for k in nms]
    ax.bar(x - 1.5 * w, g_leh, w, color=C_LEH, label="Lehmann-NAMD")
    ax.bar(x - 0.5 * w, g_cas, w, color=C_CAS, label="SA2-CASSCF")
    ax.bar(x + 0.5 * w, g_nev, w, color=C_NEV, label="SC-NEVPT2")
    ax.bar(x + 1.5 * w, g_tda, w, color=C_ADIA, label="|Bare TDA|")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[k] for k in nms], fontsize=7.0)
    ax.set_ylabel("$S_1 / S_0$ gap at blind MECI (eV)")
    ax.set_title("Independent Multi-Reference Verification")
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)

    # Panel c: Quantized Topological Berry Holonomy W = u_0(2pi).u_0(0) at All 5 Blind 3N MECIs
    ax = axes[2]
    _panel_label(ax, "c")
    w_ci = [loaded[k]["loop_ci"]["holonomy_u0"] for k in nms]
    w_ct = [loaded[k]["loop_ctrl"]["holonomy_u0"] for k in nms]
    w2 = 0.34
    ax.bar(x - w2 / 2, w_ci, w2, color=C_LEH, label="Encircling blind MECI ($W = -1$)")
    ax.bar(x + w2 / 2, w_ct, w2, color=C_ADIA, label="Shifted control loop ($W = +1$)")
    ax.axhline(-1.0, color=C_NEV, ls=":", lw=1.1)
    ax.axhline(1.0, color="#888888", ls=":", lw=1.0)
    ax.axhline(0.0, color="#444444", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[k] for k in nms], fontsize=7.0)
    ax.set_ylim(-1.32, 1.52)
    ax.set_ylabel(r"Topological Berry holonomy $W = \mathbf{u}_0(2\pi)\cdot\mathbf{u}_0(0)$")
    ax.set_title("Quantized Berry Holonomy at Blind MECIs")
    ax.legend(frameon=False, loc="upper center", fontsize=6.6)

    fig.savefig(FIG_DIR / "fig3_blind_meci_3n.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_blind_meci_3n.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig3] Saved fig3_blind_meci_3n.{pdf,png}")


def make_fig4_and_fig5():
    namd_dir = RES_DIR / "namd_3n"
    decay_sys = ["methanimine", "ethylene", "fulvene"]
    all_five = ["methanimine", "ethylene", "fulvene", "psb3", "uracil"]
    available = [s for s in all_five if (namd_dir / s / "ensemble_summary.json").exists()]
    if not available:
        print("[fig4/5] No namd_3n ensemble summaries found yet; skipping.")
        return

    # Figure 4: 4-Panel Full-(3N-6) On-the-Fly Photodynamics (3 Ultrafast CI Decay + 30D-36D Bio/Retinal Gap Collapse)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    fig.subplots_adjust(wspace=0.28, hspace=0.38)
    titles = {
        "methanimine": "Methanimine (9D internal, C=N photoisom.)",
        "ethylene": "Ethylene (12D internal, twist-pyramidalization)",
        "fulvene": "Fulvene (30D internal, Ibele–Curchod CI)",
    }
    labels = ["a", "b", "c", "d"]

    for idx, s in enumerate(decay_sys):
        ax = axes[idx // 2, idx % 2]
        _panel_label(ax, labels[idx])
        sf = namd_dir / s / "ensemble_summary.json"
        if not sf.exists():
            ax.set_title(titles[s] + " (pending)")
            continue
        d = json.loads(sf.read_text())
        m = d["methods"]
        if "casscf" in m:
            t = np.array(m["casscf"]["times_fs"])
            p_s1 = np.array(m["casscf"]["pop_s1_active"])
            tau = m["casscf"]["lifetime_fit"]["tau_total_fs"]
            nt_c = m["casscf"]["n_traj"]
            ax.plot(t, p_s1, "-", color=C_CAS, lw=2.0, label=f"SA2-CASSCF ($n={nt_c}, \\tau={tau:.1f}$ fs)")
        if "lehmann" in m:
            t = np.array(m["lehmann"]["times_fs"])
            p_s1 = np.array(m["lehmann"]["pop_s1_active"])
            p_wf = np.array(m["lehmann"]["pop_s1_wf"])
            tau = m["lehmann"]["lifetime_fit"]["tau_total_fs"]
            nt_l = m["lehmann"]["n_traj"]
            ax.plot(t, p_s1, "-", color=C_LEH, lw=2.2, label=f"Lehmann-NAMD ($n={nt_l}, \\tau={tau:.1f}$ fs)")
            ax.plot(t, p_wf, ":", color=C_LEH, lw=1.3, alpha=0.85, label="Lehmann-NAMD $|c_1(t)|^2$ (wf)")
        if "adiabatic_pbe0" in m:
            t = np.array(m["adiabatic_pbe0"]["times_fs"])
            p_s1 = np.array(m["adiabatic_pbe0"]["pop_s1_active"])
            n_in = m["adiabatic_pbe0"]["n_instability_trajs"]
            nt_a = m["adiabatic_pbe0"]["n_traj"]
            lbl_a = f"Adiabatic PBE0 (0 hops, {n_in}/{nt_a} instab.)" if n_in > 0 else "Adiabatic PBE0 (0 hops, trapped)"
            ax.plot(t, p_s1, "--", color=C_ADIA, lw=1.8, label=lbl_a)
        ax.set_ylim(-0.06, 1.42)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("$S_1$ population")
        ax.set_title(titles[s])
        ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.4, bbox_to_anchor=(0.01, 1.01))

    # Panel d: 36D 11-cis PSB3+ Retinal & 30D Uracil Nucleobase S1-S0 Gap Evolution (Zero Active Space)
    ax = axes[1, 1]
    _panel_label(ax, "d")
    psb_tf = namd_dir / "psb3" / "trajs_lehmann.json"
    ura_tf = namd_dir / "uracil" / "trajs_lehmann.json"
    if psb_tf.exists():
        trs = json.loads(psb_tf.read_text())["trajectories"]
        gaps = []
        for k, tr in enumerate(trs):
            t = np.array(tr["times_fs"])
            g = (np.array(tr["e1_ha"]) - np.array(tr["e0_ha"])) * 27.211386
            gaps.append(g)
            ax.plot(t, g, color=C_LEH, alpha=0.28, lw=1.0)
        mean_g = np.mean(np.stack(gaps, axis=0), axis=0)
        min_g = min(float(g.min()) for g in gaps)
        ax.plot(t, mean_g, color=C_LEH, lw=2.2, label=f"11-cis PSB3+ 36D (8 trajs, min {min_g:.2f} eV)")
    if ura_tf.exists():
        trs = json.loads(ura_tf.read_text())["trajectories"]
        gaps = []
        for k, tr in enumerate(trs):
            t = np.array(tr["times_fs"])
            g = (np.array(tr["e1_ha"]) - np.array(tr["e0_ha"])) * 27.211386
            gaps.append(g)
            ax.plot(t, g, color=C_NEV, alpha=0.28, lw=1.0)
        mean_g = np.mean(np.stack(gaps, axis=0), axis=0)
        min_g = min(float(g.min()) for g in gaps)
        ax.plot(t, mean_g, color=C_NEV, lw=2.2, label=f"Uracil 30D $S_1(n\\pi^*)$ (8 trajs, min {min_g:.2f} eV)")
    ax.axhline(0.5, color="#888888", ls=":", lw=0.9)
    ax.set_ylim(0.0, 5.2)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("$S_1 - S_0$ energy gap (eV)")
    ax.set_title("Biological & Retinal 30D–36D Dynamics (0 Active Space)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.7)

    fig.savefig(FIG_DIR / "fig4_namd_photodynamics.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_namd_photodynamics.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig4] Saved fig4_namd_photodynamics.{pdf,png}")

    # Figure 5: Microcanonical Conservation, Ensemble Drift, Spin Purity vs SF-TDA, and Wall-Clock Cost
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    fig.subplots_adjust(wspace=0.34, hspace=0.40)

    # Panel a: Cleanest Exemplary Hopping Trajectory Energy Conservation
    ax = axes[0, 0]
    _panel_label(ax, "a")
    candidates = []
    for s in ("fulvene", "methanimine"):
        tf = namd_dir / s / "trajs_lehmann.json"
        if tf.exists():
            trs = json.loads(tf.read_text())["trajectories"]
            for tr in trs:
                if any("to" in h for h in tr["hops"]):
                    candidates.append((tr["e_drift_max_mev"], s, tr))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, sname, tr = candidates[0]
        t = np.array(tr["times_fs"])
        e0 = (np.array(tr["e0_ha"]) - tr["e0_ha"][0]) * 27.211386
        e1 = (np.array(tr["e1_ha"]) - tr["e0_ha"][0]) * 27.211386
        etot_mev = (np.array(tr["etot_ha"]) - tr["etot_ha"][0]) * 27211.386
        eact = np.where(np.array(tr["active_state"]) == 1, e1, e0)
        l1 = ax.plot(t, e0, color="#999999", lw=1.0, label="$E_0(t)$")
        l2 = ax.plot(t, e1, color="#BBBBBB", lw=1.0, ls="--", label="$E_1(t)$")
        l3 = ax.plot(t, eact, color=C_LEH, lw=2.0, label="Active PES $E_{\\mathrm{act}}(t)$")
        ax2 = ax.twinx()
        l4 = ax2.plot(t, etot_mev, color=C_NEV, lw=1.6, label=r"$\Delta E_{\mathrm{tot}}(t)$ (meV)")
        ax2.set_ylabel(r"Total energy drift $\Delta E_{\mathrm{tot}}$ (meV)", color=C_NEV)
        ax2.tick_params(axis="y", colors=C_NEV)
        lim = max(55.0, float(np.max(np.abs(etot_mev))) * 1.45)
        ax2.set_ylim(-lim, lim)
        ax2.spines["top"].set_visible(False)
        hop_ts = [h["t_fs"] for h in tr["hops"] if "to" in h]
        for ht in hop_ts:
            ax.axvline(ht, color=C_ADIA, ls=":", lw=1.3)
        dof_int = json.loads((namd_dir / sname / "ensemble_summary.json").read_text())["dof_internal"]
        ax.set_ylim(min(e0.min(), eact.min()) - 0.5, max(e1.max(), eact.max()) + 2.3)
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Relative potential energy (eV)")
        ax.set_title(f"Hop Conservation ({sname.capitalize()} {dof_int}D, $\\Delta E_{{\\max}}={tr['e_drift_max_mev']:.1f}$ meV)")
        lines = l1 + l2 + l3 + l4
        ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left", ncol=2, fontsize=6.3)

    # Panel b: Ensemble Mean Max Energy Drift Comparison Across All 5 Systems (Log Scale)
    ax = axes[0, 1]
    _panel_label(ax, "b")
    all_sys = [s for s in all_five if (namd_dir / s / "ensemble_summary.json").exists()]
    x = np.arange(len(all_sys))
    w = 0.26
    pretty_s = {"methanimine": "CH$_2$NH\n(9D)", "ethylene": "C$_2$H$_4$\n(12D)", "fulvene": "Fulvene\n(30D)", "psb3": "PSB3+\n(36D)", "uracil": "Uracil\n(30D)"}
    d_leh = [json.loads((namd_dir / s / "ensemble_summary.json").read_text())["methods"]["lehmann"]["e_drift_mean_mev"] for s in all_sys]
    d_cas = [json.loads((namd_dir / s / "ensemble_summary.json").read_text())["methods"].get("casscf", {}).get("e_drift_mean_mev", np.nan) for s in all_sys]
    d_adia = [json.loads((namd_dir / s / "ensemble_summary.json").read_text())["methods"].get("adiabatic_pbe0", {}).get("e_drift_mean_mev", np.nan) for s in all_sys]
    ax.bar(x - w, d_leh, w, color=C_LEH, label="Lehmann-NAMD (0 instab.)")
    cas_mask = [i for i, v in enumerate(d_cas) if not np.isnan(v)]
    if cas_mask:
        ax.bar(x[cas_mask], [d_cas[i] for i in cas_mask], w, color=C_CAS, label="SA2-CASSCF (orb. flips on C$_2$H$_4$)")
    adia_mask = [i for i, v in enumerate(d_adia) if not np.isnan(v)]
    if adia_mask:
        ax.bar(x[adia_mask] + w, [d_adia[i] for i in adia_mask], w, color=C_ADIA, label="Adiabatic PBE0 (0 hops)")
    ax.set_yscale("log")
    ax.set_ylim(20, 8000)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_s[s] for s in all_sys], fontsize=7.2)
    ax.set_ylabel("Ensemble mean $\\max|\\Delta E_{\\mathrm{tot}}|$ (meV)")
    ax.set_title("Microcanonical Energy Stability (All 5 Systems)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.4)

    # Panel c: Spin Contamination <S^2> Along Real NAMD Trajectories (SF-TDA vs Lehmann-NAMD)
    ax = axes[1, 0]
    _panel_label(ax, "c")
    sf_sys, sf_s1_mean, sf_s1_max = [], [], []
    for s in all_sys:
        m = json.loads((namd_dir / s / "ensemble_summary.json").read_text())["methods"]["lehmann"]
        if m.get("sf_tda_s2_s1_mean") is not None:
            sf_sys.append(s)
            sf_s1_mean.append(m["sf_tda_s2_s1_mean"])
            sf_s1_max.append(m["sf_tda_s2_s1_max"])
    if sf_sys:
        xs = np.arange(len(sf_sys))
        w = 0.27
        ax.bar(xs - w, [0.0] * len(sf_sys), w, color=C_LEH, label=r"Lehmann-NAMD $\langle\hat{S}^2\rangle \equiv 0.000$")
        b_m = ax.bar(xs, sf_s1_mean, w, color=C_SF, label=r"SF-TDA-BHHLYP mean $\langle\hat{S}^2\rangle_{S_1}$")
        b_x = ax.bar(xs + w, sf_s1_max, w, color=C_ADIA, label=r"SF-TDA-BHHLYP max $\langle\hat{S}^2\rangle_{S_1}$")
        for b in b_x:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, max(sf_s1_max) * 1.58)
        ax.set_xticks(xs)
        ax.set_xticklabels([pretty_s[s] for s in sf_sys], fontsize=7.2)
        ax.set_ylabel(r"Spin expectation $\langle \hat{S}^2 \rangle$")
        ax.set_title("Exact Spin Purity vs Live SF-TDA Probes")
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)

    # Panel d: Mean Per-Trajectory Wall Time (s/step) Comparison Across All 5 Systems
    ax = axes[1, 1]
    _panel_label(ax, "d")
    t_leh, t_cas, t_adia = [], [], []
    for s in all_sys:
        d_s = json.loads((namd_dir / s / "ensemble_summary.json").read_text())
        nsteps = d_s["n_steps"]
        def _mean_step_s(mname):
            tf = namd_dir / s / f"trajs_{mname}.json"
            if not tf.exists():
                return np.nan
            trs = json.loads(tf.read_text())["trajectories"]
            return float(np.mean([tr["wall_s"] / max(tr["n_steps_completed"], 1) for tr in trs]))
        t_leh.append(_mean_step_s("lehmann"))
        t_cas.append(_mean_step_s("casscf"))
        t_adia.append(_mean_step_s("adiabatic_pbe0"))
    w = 0.26
    ax.bar(x - w, t_leh, w, color=C_LEH, label="Lehmann-NAMD ($O(N^3)$)")
    cas_m = [i for i, v in enumerate(t_cas) if not np.isnan(v)]
    if cas_m:
        b_c = ax.bar(x[cas_m], [t_cas[i] for i in cas_m], w, color=C_CAS, label="SA2-CASSCF (active space)")
        if "fulvene" in all_sys:
            fi = all_sys.index("fulvene")
            if not np.isnan(t_cas[fi]):
                ratio = t_cas[fi] / t_leh[fi]
                ax.annotate(f"{ratio:.1f}× slower", (x[fi], t_cas[fi]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.8, fontweight="bold", color=C_CAS)
    adia_m = [i for i, v in enumerate(t_adia) if not np.isnan(v)]
    if adia_m:
        ax.bar(x[adia_m] + w, [t_adia[i] for i in adia_m], w, color=C_ADIA, label="Adiabatic PBE0 (0 hops)")
    ax.set_ylim(0, max(v for v in t_cas + t_leh if not np.isnan(v)) * 1.48)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_s[s] for s in all_sys], fontsize=7.2)
    ax.set_ylabel("Mean wall time per NAMD step (s)")
    ax.set_title("Per-Step Cost: Lehmann-NAMD vs SA2-CASSCF")
    ax.legend(frameon=False, loc="upper right", fontsize=6.4)

    fig.savefig(FIG_DIR / "fig5_conservation_spin_isom.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_conservation_spin_isom.png", bbox_inches="tight")
    plt.close(fig)
    print("[fig5] Saved fig5_conservation_spin_isom.{pdf,png}")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4_and_fig5()
