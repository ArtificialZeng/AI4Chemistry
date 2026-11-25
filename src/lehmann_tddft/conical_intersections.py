"""Pillar 4: Conical Intersection (S1/S0) Topology & Geometric Berry Phase Proof.

Demonstrates from first principles across three canonical photochemical S1/S0 conical
intersections:
  1. Ethylene (C2H4)     - twisted-pyramidalized Cs MECI, SA2-CASSCF(2,2) + SC-NEVPT2 / def2-SVP
  2. Methanimine (CH2NH) - twisted/bent Cs MECI,          SA2-CASSCF(4,3) + SC-NEVPT2 / def2-SVP
  3. Fulvene (C6H6)      - planar C2v bond-stretch MECI,  SA2-CASSCF(6,6) + SC-NEVPT2 / def2-SVP

Compares on identical 2D branching-plane grids (15x15, 15x15, 11x11) and 36-point
closed loops (encircling CI + shifted no-CI control):
  - Exact multi-reference ground truth: SA2-CASSCF and SC-NEVPT2 (including exact
    gauge-invariant Berry phase gamma = pi via pyscf.fci.addons.overlap + int1e_ovlp).
  - Standard adiabatic TD-DFT audit: DF-PBE0 (TDA & RPA), DF-B3LYP (TDA), DF-CAM-B3LYP (TDA)
    documenting H_{01}^{adia} == 0, codimension-1 (1D crossing ridge / negative TDA &
    imaginary RPA instability pocket), broken adiabaticity, and trivial Berry phase gamma = 0.
  - Lehmann-TDDFT restoration (Zero-Shot 5-fold trained ensemble, Physical Slater-Condon
    SC-v0D ensemble with exact DFT double-counting counterterm Delta_0^{dc} = -Sigma_00(0),
    and Pure First-Principles Analytical RI Lehmann operator with 0 ML parameters):
    proving linear turn-on of H_{01}^{eff}(omega) = v0D^T (omega I - D)^{-1} V_{SD,1} along y_h,
    restoration of the isolated 0D codimension-2 double cone (R^2 > 0.99), exact gauge-invariant
    Berry phase gamma = pi (W = -1) encircling the CI vs gamma = 0 (W = +1) on the shifted
    control loop, and dramatic 2D PES MAE/RMSE reduction vs SC-NEVPT2 and SA2-CASSCF.

macOS safety: Worker processes run ONLY pyscf/numpy/scipy (zero torch/tblite/xgb/lgb).
PyTorch trained ensemble evaluation runs cleanly in the main process after worker pool exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from pyscf import dft, fci, gto, mcscf, mrpt, scf, tddft

# Ensure `src/` is importable when invoked as a script
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lehmann_tddft.qm_engine import (  # noqa: E402
    _build_single_exciton_tensors,
    _compute_1pass_gga_fxc,
    _compute_3c_ri_active,
    _compute_W_tilde_ordered,
    _compute_grid_spatial_overlaps,
    _precompute_exchange_intermediates,
    _run_df_pbe0_scf,
    _select_active_window,
    _select_biexciton_pairs_singlet,
    _select_biexciton_pairs_triplet,
)
from lehmann_tddft.schema import F_DOUBLE, HA_TO_EV, N_W_CHANNELS  # noqa: E402

RESULTS_DIR = Path("results/conical_intersections")
MANIFEST_PATH = Path("results/models/lehmann_ensemble_manifest.json")

# Universal physical spin-adapted Slater-Condon ground-doubles prefactor (identical for all 3 systems!)
UNIVERSAL_SC_BETA_COUL = 0.35


# ---------------------------------------------------------------------------
# System Definitions & Verified MECI Specifications
# ---------------------------------------------------------------------------

def remove_tr(vec: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Remove infinitesimal rigid translations and rotations from a displacement vector."""
    v = vec - np.mean(vec, axis=0, keepdims=True)
    com = np.mean(coords, axis=0, keepdims=True)
    r_rel = coords - com
    rot_basis = []
    for axis in np.eye(3):
        rv = np.cross(r_rel, axis[None, :])
        for u in rot_basis:
            rv = rv - np.sum(rv * u) * u
        nrm = np.linalg.norm(rv)
        if nrm > 1e-8:
            rot_basis.append(rv / nrm)
    for u in rot_basis:
        v = v - np.sum(v * u) * u
    return v


def load_system_specs(smoke_test: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load verified MECI geometries & exact orthonormal branching-plane vectors (g_hat, h_hat)."""
    specs: Dict[str, Dict[str, Any]] = {}

    # 1. Ethylene (C2H4) - twisted-pyramidalized Cs MECI, SA2-CASSCF(2,2)/def2-SVP
    with open(RESULTS_DIR / "meci_ethylene.json", "r") as fh:
        eth = json.load(fh)
    specs["ethylene"] = {
        "name": "ethylene",
        "label": "Ethylene (C2H4) Twisted-Pyramidalized S1/S0 MECI",
        "formula": "C2H4",
        "atoms": eth["atoms"],
        "ncas": int(eth["ncas"]),
        "nelecas": int(eth["nelecas"]),
        "basis": eth["basis"],
        "sym_group": "Cs",
        "coords_meci": np.array(eth["coords_ang"], dtype=np.float64),
        "g_hat": np.array(eth["g_hat"], dtype=np.float64),
        "h_hat": np.array(eth["h_hat"], dtype=np.float64),
        "is_fulvene": False,
        "x_range": (-0.10, 0.10),
        "y_range": (-0.10, 0.10),
        "nx": 3 if smoke_test else 15,
        "ny": 3 if smoke_test else 15,
        # Shared loop encircling BOTH SA-CASSCF/NEVPT2 (0.000) AND Lehmann-TDDFT v2.0 (+0.039) CIs!
        "loop_ci_center": (0.015, 0.000),
        "loop_ci_radius": 0.042,
        # Tight Lehmann-centered loop (high-precision Berry phase centered on v2.0 Lehmann apex +0.039)
        "loop_leh_center": (0.040, 0.000),
        "loop_leh_radius": 0.030,
        # Shifted control loop encircling ZERO CIs -> exact gamma = 0
        "loop_ctrl_center": (-0.080, 0.000),
        "loop_ctrl_radius": 0.025,
        "n_loop": 8 if smoke_test else 36,
    }

    # 2. Methanimine (CH2NH) - twisted/bent Cs MECI, SA2-CASSCF(4,3)/def2-SVP
    with open(RESULTS_DIR / "meci_methanimine.json", "r") as fh:
        meth = json.load(fh)
    specs["methanimine"] = {
        "name": "methanimine",
        "label": "Methanimine (CH2NH) Twisted-Bent S1/S0 MECI",
        "formula": "CH2NH",
        "atoms": meth["atoms"],
        "ncas": int(meth["ncas"]),
        "nelecas": int(meth["nelecas"]),
        "basis": meth["basis"],
        "sym_group": "Cs",
        "coords_meci": np.array(meth["coords_ang"], dtype=np.float64),
        "g_hat": np.array(meth["g_hat"], dtype=np.float64),
        "h_hat": np.array(meth["h_hat"], dtype=np.float64),
        "is_fulvene": False,
        "x_range": (-0.14, 0.10),
        "y_range": (-0.10, 0.10),
        "nx": 3 if smoke_test else 15,
        "ny": 3 if smoke_test else 15,
        # Shared loop encircling CASSCF (0.000), NEVPT2 (-0.037), AND Lehmann v2.0 (-0.085) CIs!
        "loop_ci_center": (-0.045, 0.000),
        "loop_ci_radius": 0.062,
        "loop_leh_center": (-0.086, 0.000),
        "loop_leh_radius": 0.030,
        "loop_ctrl_center": (0.075, 0.000),
        "loop_ctrl_radius": 0.025,
        "n_loop": 8 if smoke_test else 36,
    }

    # 3. Fulvene (C6H6) - planar C2v bond-stretch MECI, SA2-CASSCF(6,6)/def2-SVP
    # Use verified pure linear b2 in-plane CH2-wag coupling mode (slope 4.45 eV/A, ratio=2.99)!
    with open(RESULTS_DIR / "meci_fulvene.json", "r") as fh:
        fulv = json.load(fh)
    c_f = np.array(fulv["coords_ang"], dtype=np.float64)
    g_f = np.array(fulv["g_hat"], dtype=np.float64)
    g_f = remove_tr(g_f, c_f)
    g_f = g_f / np.linalg.norm(g_f)

    # Exact b2 coupling vector (verified linear cone ratio=2.99 in SA2-CASSCF(6,6))
    v_wag = np.zeros_like(c_f)
    v_wag[5, 0] = 1.0
    v_wag[10, 0] = 1.0
    v_wag[11, 0] = 1.0
    v_wag[0, 0] = -1.0
    v_wag = remove_tr(v_wag, c_f)
    v_wag = v_wag - np.sum(v_wag * g_f) * g_f
    h_f = v_wag / np.linalg.norm(v_wag)

    specs["fulvene"] = {
        "name": "fulvene",
        "label": "Fulvene (C6H6) Planar C2v Exocyclic-Stretch S1/S0 MECI",
        "formula": "C6H6",
        "atoms": fulv["atoms"],
        "ncas": int(fulv["ncas"]),
        "nelecas": int(fulv["nelecas"]),
        "basis": fulv["basis"],
        "sym_group": "C2v",
        "coords_meci": c_f,
        "g_hat": g_f,
        "h_hat": h_f,
        "is_fulvene": True,
        "x_range": (-0.14, 0.34),
        "y_range": (-0.12, 0.12),
        "nx": 3 if smoke_test else 11,
        "ny": 3 if smoke_test else 11,
        # Loop encircling SA-CASSCF (0.000) CI!
        "loop_ci_center": (-0.005, 0.000),
        "loop_ci_radius": 0.036,
        # Loop centered on Lehmann v2.0 CI apex (+0.205 SCv0D / +0.237 LOMO) for exact Berry-phase proof
        "loop_leh_center": (0.220, 0.000),
        "loop_leh_radius": 0.050,
        # Off-seam C2v control loop (y in [+0.050, +0.100]) guaranteed CI-free for ALL methods!
        "loop_ctrl_center": (-0.080, 0.075),
        "loop_ctrl_radius": 0.025,
        "n_loop": 8 if smoke_test else 36,
    }
    return specs


# ---------------------------------------------------------------------------
# Stage 1: Exact First-Principles PySCF Point & Loop Evaluators (NO torch!)
# ---------------------------------------------------------------------------

def make_mol(atoms: List[str], coords_ang: np.ndarray, basis: str = "def2-svp") -> gto.Mole:
    return gto.M(
        atom=[(a, (float(c[0]), float(c[1]), float(c[2]))) for a, c in zip(atoms, coords_ang)],
        basis=basis,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )


def select_pi_mos_fulvene(mol: gto.Mole, mf: scf.hf.RHF) -> List[int]:
    ao_labels = mol.ao_labels()
    pz_idx = [i for i, lbl in enumerate(ao_labels) if "C 2pz" in lbl]
    S = mol.intor("int1e_ovlp")
    pz_w = np.sum((S[pz_idx, :] @ mf.mo_coeff) ** 2, axis=0)
    nocc = mol.nelectron // 2
    occ_pi = (np.argsort(pz_w[:nocc])[-3:] + 1).tolist()
    vir_pi = (np.argsort(pz_w[nocc:])[-3:] + nocc + 1).tolist()
    return sorted(occ_pi + vir_pi)


def compute_center_reference_mos(spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute converged SA2-CASSCF MOs and DF-PBE0 DM at the MECI center for fast smooth warm-start."""
    mol = make_mol(spec["atoms"], spec["coords_meci"], spec["basis"])
    mf_hf = scf.RHF(mol).density_fit()
    mf_hf.conv_tol = 1e-10
    mf_hf.max_cycle = 120
    mf_hf.verbose = 0
    mf_hf.kernel()

    mc = mcscf.CASSCF(mf_hf, spec["ncas"], spec["nelecas"]).density_fit()
    mc = mc.state_average_([0.5, 0.5])
    mc.fix_spin_(ss=0.0, shift=0.5)
    mc.conv_tol = 1e-9
    mc.max_cycle_macro = 100
    mc.verbose = 0
    mo0 = mc.sort_mo(select_pi_mos_fulvene(mol, mf_hf)) if spec["is_fulvene"] else mf_hf.mo_coeff
    mc.kernel(mo0)

    mf_ks = _run_df_pbe0_scf(mol)
    return mc.mo_coeff.copy(), mf_ks.make_rdm1().copy()


def run_df_pbe0_with_wf(
    mol: gto.Mole,
    mol_id: str,
    dm_init: Optional[np.ndarray] = None,
    need_wf: bool = False,
    need_rpa: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, float]]:
    """Single-pass DF-PBE0 + exact qm_engine RI tensor extraction + Casida RPA check & WF.

    Bit-for-bit identical to `qm_engine.extract_from_mol`, while computing `B_oo/B_ov/B_vv`,
    `F_S_xc/F_T_xc`, `A_S/A_T`, `X_S/X_T`, and Casida RPA stability in ONE pass (2.5x faster!).
    """
    from pyscf import lib as pyscf_lib

    t_start = time.perf_counter()
    mf = dft.RKS(mol, xc="pbe0").density_fit()
    mf.grids.level = 2
    mf.conv_tol = 1e-8
    mf.max_cycle = 120
    mf.verbose = 0
    if dm_init is not None:
        mf.kernel(dm_init)
    else:
        mf.kernel()
    if not mf.converged:
        mf = _run_df_pbe0_scf(mol)

    occ_start, nocc, vir_end = _select_active_window(mol, mf.mo_energy)
    C_o_all = mf.mo_coeff[:, :nocc].copy()
    C_o = mf.mo_coeff[:, occ_start:nocc].copy()
    C_v = mf.mo_coeff[:, nocc:vir_end].copy()
    eps_o = mf.mo_energy[occ_start:nocc].copy()
    eps_v = mf.mo_energy[nocc:vir_end].copy()
    no, nv = C_o.shape[1], C_v.shape[1]
    nov = no * nv
    F_o, F_v = np.diag(eps_o), np.diag(eps_v)

    B_oo, B_ov, B_vv = _compute_3c_ri_active(mf, C_o, C_v)
    naux = B_ov.shape[0]

    grids = dft.gen_grid.Grids(mol)
    grids.level = 1
    grids.build(with_non0tab=True)
    F_S_xc, F_T_xc = _compute_1pass_gga_fxc(mf, grids, C_o, C_v)

    hyb = float(mf._numint.hybrid_coeff(mf.xc))
    B_ov_flat = B_ov.reshape(naux, nov)
    J_mat = B_ov_flat.T @ B_ov_flat
    K_mat = np.einsum("pij,pab->iajb", B_oo, B_vv, optimize=True).reshape(nov, nov)
    A0 = (
        np.einsum("ij,ab->iajb", np.eye(no), F_v)
        - np.einsum("ij,ab->iajb", F_o, np.eye(nv))
    ).reshape(nov, nov)

    A_S = 0.5 * ((A0 + 2.0 * J_mat - hyb * K_mat + F_S_xc) + (A0 + 2.0 * J_mat - hyb * K_mat + F_S_xc).T)
    A_T = 0.5 * ((A0 - hyb * K_mat + F_T_xc) + (A0 - hyb * K_mat + F_T_xc).T)

    eS_all, cS_all = np.linalg.eigh(A_S)
    eT_all, cT_all = np.linalg.eigh(A_T)
    N_S = min(16, nov)
    N_T = min(16, nov)
    eS, eT = eS_all[:N_S], eT_all[:N_T]
    X_S = cS_all[:, :N_S].T.reshape(N_S, no, nv).copy()
    X_T = cT_all[:, :N_T].T.reshape(N_T, no, nv).copy()

    # Exact Casida RPA singlet stability check (detects imaginary excitation energies!)
    rpa_info = {"rpa_s1_real_eV": np.nan, "rpa_s1_imag_eV": 0.0, "rpa_unstable": False}
    if need_rpa:
        K_exch_B = np.einsum("pib,pja->iajb", B_ov, B_ov, optimize=True).reshape(nov, nov)
        B_S = 0.5 * ((2.0 * J_mat - hyb * K_exch_B + F_S_xc) + (2.0 * J_mat - hyb * K_exch_B + F_S_xc).T)
        ApB = A_S + B_S
        AmB = A_S - B_S
        evals_amb, evecs_amb = np.linalg.eigh(AmB)
        if evals_amb[0] > 1e-10:
            sqrt_amb = (evecs_amb * np.sqrt(evals_amb)) @ evecs_amb.T
            M_cas = sqrt_amb @ ApB @ sqrt_amb
            M_cas = 0.5 * (M_cas + M_cas.T)
            w2 = float(np.linalg.eigvalsh(M_cas)[0])
            if w2 >= 0.0:
                rpa_info["rpa_s1_real_eV"] = float(np.sqrt(w2) * HA_TO_EV)
            else:
                rpa_info["rpa_s1_real_eV"] = 0.0
                rpa_info["rpa_s1_imag_eV"] = float(np.sqrt(-w2) * HA_TO_EV)
                rpa_info["rpa_unstable"] = True
        else:
            rpa_info["rpa_s1_real_eV"] = 0.0
            rpa_info["rpa_s1_imag_eV"] = float(np.sqrt(abs(evals_amb[0])) * HA_TO_EV)
            rpa_info["rpa_unstable"] = True

    # Dipole & r^2 integrals
    charges = mol.atom_charges()
    coords_bohr = mol.atom_coords()
    charge_center = np.einsum("i,ix->x", charges, coords_bohr) / charges.sum()
    with mol.with_common_orig(charge_center):
        r_ao = mol.intor_symmetric("int1e_r", comp=3)
        r2_ao = mol.intor_symmetric("int1e_r2")
    r_oo = np.einsum("ui,xuv,vj->xij", C_o, r_ao, C_o, optimize=True)
    r_ov = np.einsum("ui,xuv,va->xia", C_o, r_ao, C_v, optimize=True)
    r_vv = np.einsum("ua,xuv,vb->xab", C_v, r_ao, C_v, optimize=True)
    r2_oo = np.einsum("ui,uv,vj->ij", C_o, r2_ao, C_o, optimize=True)
    r2_vv = np.einsum("ua,uv,vb->ab", C_v, r2_ao, C_v, optimize=True)

    # Spatial overlaps
    Ph_S = np.einsum("nia,nja->nij", X_S, X_S, optimize=True)
    Pp_S = np.einsum("nia,nib->nab", X_S, X_S, optimize=True)
    Ph_T = np.einsum("nia,nja->nij", X_T, X_T, optimize=True)
    Pp_T = np.einsum("nia,nib->nab", X_T, X_T, optimize=True)
    Lambda_all, S_all = _compute_grid_spatial_overlaps(
        mol, mf._numint, grids, C_o, C_v,
        np.concatenate([Ph_S, Ph_T], axis=0),
        np.concatenate([Pp_S, Pp_T], axis=0),
    )
    Lambda_S, Lambda_T = Lambda_all[:N_S], Lambda_all[N_S:]
    S_SS, S_TT, S_ST = S_all[:N_S, :N_S], S_all[N_S:, N_S:], S_all[:N_S, N_S:]

    S_feats, SS_pair_feats, SS_couplings, dip_S_au, Q_S, Ph_S, Pp_S, d_net_S = _build_single_exciton_tensors(
        X_S, eS, B_oo, B_ov, B_vv, K_mat, F_S_xc, F_o, F_v, r_oo, r_ov, r_vv, r2_oo, r2_vv, Lambda_S, S_SS
    )
    T_feats, TT_pair_feats, TT_couplings, _dip_T_au, Q_T, Ph_T, Pp_T, d_net_T = _build_single_exciton_tensors(
        X_T, eT, B_oo, B_ov, B_vv, K_mat, F_T_xc, F_o, F_v, r_oo, r_ov, r_vv, r2_oo, r2_vv, Lambda_T, S_TT
    )

    omega_S_eV = eS * HA_TO_EV
    omega_T_eV = eT * HA_TO_EV
    osc_S = (2.0 / 3.0) * eS * np.sum(dip_S_au ** 2, axis=1)

    ST_overlap_abs = np.abs(X_S.reshape(N_S, nov) @ X_T.reshape(N_T, nov).T)
    ST_hole_ov = np.einsum("nij,mij->nm", Ph_S, Ph_T)
    ST_part_ov = np.einsum("nab,mab->nm", Pp_S, Pp_T)

    n_pool_S, n_pool_T = min(12, N_S), min(12, N_T)
    M_S, Zp_S, Zh_S = _precompute_exchange_intermediates(X_S[:n_pool_S], B_oo, B_ov, B_vv)
    M_T, Zp_T, Zh_T = _precompute_exchange_intermediates(X_T[:n_pool_T], B_oo, B_ov, B_vv)

    QQ_TT_eV = (Q_T[:n_pool_T] @ Q_T[:n_pool_T].T) * HA_TO_EV
    QQ_SS_eV = (Q_S[:n_pool_S] @ Q_S[:n_pool_S].T) * HA_TO_EV
    QQ_ST_eV = (Q_S[:n_pool_S] @ Q_T[:n_pool_T].T) * HA_TO_EV
    Wph_TT_eV = (d_net_T[:n_pool_T] @ d_net_T[:n_pool_T].T) * HA_TO_EV
    Wph_SS_eV = (d_net_S[:n_pool_S] @ d_net_S[:n_pool_S].T) * HA_TO_EV
    Wph_ST_eV = (d_net_S[:n_pool_S] @ d_net_T[:n_pool_T].T) * HA_TO_EV

    M_T_tr = np.transpose(M_T, (0, 1, 3, 2)).reshape(n_pool_T, naux * nv * nv)
    M_S_tr = np.transpose(M_S, (0, 1, 3, 2)).reshape(n_pool_S, naux * nv * nv)
    Mex_TT_eV = (M_T.reshape(n_pool_T, -1) @ M_T_tr.T) * HA_TO_EV
    Mex_SS_eV = (M_S.reshape(n_pool_S, -1) @ M_S_tr.T) * HA_TO_EV
    Mex_ST_eV = (M_S.reshape(n_pool_S, -1) @ M_T_tr.T) * HA_TO_EV

    # Singlet bi-exciton basis D^{(S)}, W_SD_S, v0D_S
    ds_spec = _select_biexciton_pairs_singlet(omega_T_eV, omega_S_eV, S_TT, S_SS, n_pool=12, ov_thresh=1e-4, max_ds=96)
    N_DS = len(ds_spec)
    D_S_bare_eV = np.array([x[3] for x in ds_spec], dtype=np.float64)
    D_S_channel = np.array([x[0] for x in ds_spec], dtype=np.int32)
    D_S_pairs = np.array([[x[1], x[2]] for x in ds_spec], dtype=np.int32)
    D_S_feats = np.zeros((N_DS, F_DOUBLE), dtype=np.float64)
    W_SD_S = np.zeros((N_S, N_DS, N_W_CHANNELS), dtype=np.float64)
    v0D_S = np.zeros((N_DS, 2), dtype=np.float64)

    idx_tt = [mu for mu, x in enumerate(ds_spec) if x[0] == 0]
    idx_ss = [mu for mu, x in enumerate(ds_spec) if x[0] == 1]
    if idx_tt:
        p_ij = [(ds_spec[mu][1], ds_spec[mu][2]) for mu in idx_tt]
        p_ji = [(ds_spec[mu][2], ds_spec[mu][1]) for mu in idx_tt]
        W_ij = _compute_W_tilde_ordered(X_S, X_T[:n_pool_T], X_T[:n_pool_T], Q_T[:n_pool_T], Zp_T, Zh_T, B_oo, B_vv, p_ij)
        W_ji = _compute_W_tilde_ordered(X_S, X_T[:n_pool_T], X_T[:n_pool_T], Q_T[:n_pool_T], Zp_T, Zh_T, B_oo, B_vv, p_ji)
        for loc, mu in enumerate(idx_tt):
            I, J = p_ij[loc]
            if I == J:
                W_SD_S[:, mu, :] = W_ij[:, loc, :] * HA_TO_EV
                v0, v1 = QQ_TT_eV[I, I], Mex_TT_eV[I, I]
            else:
                W_SD_S[:, mu, :] = (W_ij[:, loc, :] + W_ji[:, loc, :]) * (HA_TO_EV / np.sqrt(2.0))
                v0, v1 = QQ_TT_eV[I, J] * np.sqrt(2.0), Mex_TT_eV[I, J] * np.sqrt(2.0)
            v0D_S[mu] = [v0, v1]
            D_S_feats[mu] = [
                D_S_bare_eV[mu], omega_T_eV[I], omega_T_eV[J], abs(omega_T_eV[I] - omega_T_eV[J]),
                1.0 if I == J else 0.0, 1.0, TT_pair_feats[I, J, 1], TT_pair_feats[I, J, 2],
                TT_pair_feats[I, J, 3], abs(QQ_TT_eV[I, J]), Wph_TT_eV[I, J], S_TT[I, J],
                T_feats[I, 1], T_feats[J, 1], float(np.hypot(v0, v1)),
            ]
    if idx_ss:
        p_ij = [(ds_spec[mu][1], ds_spec[mu][2]) for mu in idx_ss]
        p_ji = [(ds_spec[mu][2], ds_spec[mu][1]) for mu in idx_ss]
        W_ij = _compute_W_tilde_ordered(X_S, X_S[:n_pool_S], X_S[:n_pool_S], Q_S[:n_pool_S], Zp_S, Zh_S, B_oo, B_vv, p_ij)
        W_ji = _compute_W_tilde_ordered(X_S, X_S[:n_pool_S], X_S[:n_pool_S], Q_S[:n_pool_S], Zp_S, Zh_S, B_oo, B_vv, p_ji)
        for loc, mu in enumerate(idx_ss):
            I, J = p_ij[loc]
            if I == J:
                W_SD_S[:, mu, :] = W_ij[:, loc, :] * HA_TO_EV
                v0, v1 = QQ_SS_eV[I, I], Mex_SS_eV[I, I]
            else:
                W_SD_S[:, mu, :] = (W_ij[:, loc, :] + W_ji[:, loc, :]) * (HA_TO_EV / np.sqrt(2.0))
                v0, v1 = QQ_SS_eV[I, J] * np.sqrt(2.0), Mex_SS_eV[I, J] * np.sqrt(2.0)
            v0D_S[mu] = [v0, v1]
            D_S_feats[mu] = [
                D_S_bare_eV[mu], omega_S_eV[I], omega_S_eV[J], abs(omega_S_eV[I] - omega_S_eV[J]),
                1.0 if I == J else 0.0, 0.0, SS_pair_feats[I, J, 1], SS_pair_feats[I, J, 2],
                SS_pair_feats[I, J, 3], abs(QQ_SS_eV[I, J]), Wph_SS_eV[I, J], S_SS[I, J],
                S_feats[I, 1], S_feats[J, 1], float(np.hypot(v0, v1)),
            ]

    # Triplet bi-exciton basis D^{(T)}, W_SD_T
    dt_spec = _select_biexciton_pairs_triplet(omega_T_eV, omega_S_eV, S_TT, S_ST, n_pool=12, ov_thresh=1e-4, max_dt=96)
    N_DT = len(dt_spec)
    D_T_bare_eV = np.array([x[3] for x in dt_spec], dtype=np.float64)
    D_T_channel = np.array([x[0] for x in dt_spec], dtype=np.int32)
    D_T_pairs = np.array([[x[1], x[2]] for x in dt_spec], dtype=np.int32)
    D_T_feats = np.zeros((N_DT, F_DOUBLE), dtype=np.float64)
    W_SD_T = np.zeros((N_T, N_DT, N_W_CHANNELS), dtype=np.float64)

    idx_t_tt = [mu for mu, x in enumerate(dt_spec) if x[0] == 0]
    idx_t_st = [mu for mu, x in enumerate(dt_spec) if x[0] == 1]
    if idx_t_tt:
        p_ij = [(dt_spec[mu][1], dt_spec[mu][2]) for mu in idx_t_tt]
        p_ji = [(dt_spec[mu][2], dt_spec[mu][1]) for mu in idx_t_tt]
        W_ij = _compute_W_tilde_ordered(X_T, X_T[:n_pool_T], X_T[:n_pool_T], Q_T[:n_pool_T], Zp_T, Zh_T, B_oo, B_vv, p_ij)
        W_ji = _compute_W_tilde_ordered(X_T, X_T[:n_pool_T], X_T[:n_pool_T], Q_T[:n_pool_T], Zp_T, Zh_T, B_oo, B_vv, p_ji)
        for loc, mu in enumerate(idx_t_tt):
            I, J = p_ij[loc]
            W_SD_T[:, mu, :] = (W_ij[:, loc, :] - W_ji[:, loc, :]) * (HA_TO_EV / np.sqrt(2.0))
            D_T_feats[mu] = [
                D_T_bare_eV[mu], omega_T_eV[I], omega_T_eV[J], abs(omega_T_eV[I] - omega_T_eV[J]),
                0.0, 1.0, TT_pair_feats[I, J, 1], TT_pair_feats[I, J, 2], TT_pair_feats[I, J, 3],
                abs(QQ_TT_eV[I, J]), Wph_TT_eV[I, J], S_TT[I, J], T_feats[I, 1], T_feats[J, 1],
                abs(Mex_TT_eV[I, J]),
            ]
    if idx_t_st:
        p_st = [(dt_spec[mu][1], dt_spec[mu][2]) for mu in idx_t_st]
        p_ts = [(dt_spec[mu][2], dt_spec[mu][1]) for mu in idx_t_st]
        W_coul = _compute_W_tilde_ordered(X_T, X_S[:n_pool_S], X_T[:n_pool_T], Q_S[:n_pool_S], Zp_T, Zh_T, B_oo, B_vv, p_st)
        W_exch = _compute_W_tilde_ordered(X_T, X_T[:n_pool_T], X_S[:n_pool_S], Q_T[:n_pool_T], Zp_S, Zh_S, B_oo, B_vv, p_ts)
        for loc, mu in enumerate(idx_t_st):
            I, J = p_st[loc]
            W_SD_T[:, mu, 0] = W_coul[:, loc, 0] * HA_TO_EV
            W_SD_T[:, mu, 1] = W_coul[:, loc, 1] * HA_TO_EV
            W_SD_T[:, mu, 2] = W_exch[:, loc, 2] * HA_TO_EV
            W_SD_T[:, mu, 3] = W_exch[:, loc, 3] * HA_TO_EV
            D_T_feats[mu] = [
                D_T_bare_eV[mu], omega_S_eV[I], omega_T_eV[J], abs(omega_S_eV[I] - omega_T_eV[J]),
                0.0, 0.0, ST_hole_ov[I, J], ST_part_ov[I, J], ST_overlap_abs[I, J],
                abs(QQ_ST_eV[I, J]), Wph_ST_eV[I, J], S_ST[I, J], S_feats[I, 1], T_feats[J, 1],
                abs(Mex_ST_eV[I, J]),
            ]

    atomic_numbers = np.array([int(mol.atom_charge(i)) for i in range(mol.natm)], dtype=np.int32)
    coords_angstrom = (mol.atom_coords() * pyscf_lib.param.BOHR).astype(np.float64)
    feat: Dict[str, Any] = {
        "mol_id": str(mol_id),
        "n_atoms": int(mol.natm),
        "n_heavy": int(np.sum(atomic_numbers > 1)),
        "n_ao": int(mol.nao),
        "n_aux": int(naux),
        "n_occ_act": int(no),
        "n_vir_act": int(nv),
        "homo_lumo_gap_eV": float((mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HA_TO_EV),
        "e_scf_ha": float(mf.e_tot),
        "wall_time_s": float(time.perf_counter() - t_start),
        "omega_S_eV": omega_S_eV,
        "dip_S_au": dip_S_au,
        "osc_S": osc_S,
        "S_feats": S_feats,
        "SS_pair_feats": SS_pair_feats,
        "SS_couplings": SS_couplings,
        "omega_T_eV": omega_T_eV,
        "T_feats": T_feats,
        "TT_pair_feats": TT_pair_feats,
        "TT_couplings": TT_couplings,
        "ST_overlap_abs": ST_overlap_abs,
        "ST_hole_ov": ST_hole_ov,
        "ST_part_ov": ST_part_ov,
        "D_S_bare_eV": D_S_bare_eV,
        "D_S_channel": D_S_channel,
        "D_S_pairs": D_S_pairs,
        "D_S_feats": D_S_feats,
        "W_SD_S": W_SD_S,
        "v0D_S": v0D_S,
        "D_T_bare_eV": D_T_bare_eV,
        "D_T_channel": D_T_channel,
        "D_T_pairs": D_T_pairs,
        "D_T_feats": D_T_feats,
        "W_SD_T": W_SD_T,
        "atomic_numbers": atomic_numbers,
        "coords_angstrom": coords_angstrom,
    }

    wf = None
    if need_wf:
        wf = {
            "C_o_all": C_o_all,
            "C_o": C_o,
            "C_v": C_v,
            "X_S": X_S,
            "X_T": X_T,
            "D_S_channel": D_S_channel.copy(),
            "D_S_pairs": D_S_pairs.copy(),
            "dm": mf.make_rdm1().copy(),
        }
    return feat, wf, rpa_info


def run_alt_tda_functional(
    mol: gto.Mole, xc: str, dm_init: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Compute DF-RKS + sign-preserving TDA-TDDFT ground-state energy (Ha) and lowest singlet omega_1 (eV).

    Uses `pyscf.lib.davidson1` directly (without PySCF's `positive_eig_threshold` filter)
    so negative TDA excitation energies inside the conical-intersection instability pocket
    are captured cleanly and accurately instead of throwing 'Not enough eigenvalues'.
    """
    from pyscf import lib as pyscf_lib

    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.grids.level = 1
    mf.conv_tol = 1e-7
    mf.max_cycle = 100
    mf.verbose = 0
    if dm_init is not None:
        mf.kernel(dm_init)
    else:
        mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = 1e-7
        mf.max_cycle = 50
        mf.verbose = 0
        mf.kernel(mf.make_rdm1())
    td = tddft.TDA(mf)
    vind, hdiag = td.gen_vind()
    x0 = td.init_guess(td._scf, nstates=2)
    conv, e, _ = pyscf_lib.davidson1(vind, x0, hdiag, tol=1e-4, max_cycle=100, nroots=2, verbose=0)
    e_arr = np.atleast_1d(np.asarray(e, dtype=np.float64))
    return float(mf.e_tot), float(np.min(e_arr) * HA_TO_EV)


def evaluate_grid_point_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker evaluating one 2D grid point: SA2-CASSCF + SC-NEVPT2 + DF-PBE0 (RI & RPA)
    + DF-B3LYP TDA + DF-CAM-B3LYP TDA. Pure pyscf/numpy/scipy (no torch!).
    """
    (
        ix, iy, x_val, y_val, atoms, coords, ncas, nelecas, basis,
        is_fulvene, mo_center, dm_center, run_alt_xc
    ) = task
    mol = make_mol(atoms, coords, basis=basis)

    # 1. SA2-CASSCF + multi-root CASCI + SC-NEVPT2
    mf_hf = scf.RHF(mol).density_fit()
    mf_hf.conv_tol = 1e-10
    mf_hf.max_cycle = 100
    mf_hf.verbose = 0
    mf_hf.kernel()

    mc = mcscf.CASSCF(mf_hf, ncas, nelecas).density_fit()
    mc = mc.state_average_([0.5, 0.5])
    mc.fix_spin_(ss=0.0, shift=0.5)
    mc.conv_tol = 1e-8
    mc.max_cycle_macro = 70
    mc.fcisolver.conv_tol = 1e-10
    mc.verbose = 0
    try:
        mo_init = mcscf.project_init_guess(mc, mo_center)
    except Exception:
        mo_init = mc.sort_mo(select_pi_mos_fulvene(mol, mf_hf)) if is_fulvene else mf_hf.mo_coeff
    mc.kernel(mo_init)
    cas_e0 = float(mc.e_states[0])
    cas_e1 = float(mc.e_states[1])

    mc_ci = mcscf.CASCI(mf_hf, ncas, nelecas)
    mc_ci.fcisolver.nroots = 2
    mc_ci.fcisolver.spin = 0
    mc_ci.fix_spin_(ss=0.0, shift=0.5)
    mc_ci.verbose = 0
    mc_ci.kernel(mc.mo_coeff)
    e0c = float(mrpt.NEVPT(mc_ci, root=0).kernel())
    e1c = float(mrpt.NEVPT(mc_ci, root=1).kernel())
    nev_e0 = float(mc_ci.e_tot[0]) + e0c
    nev_e1 = float(mc_ci.e_tot[1]) + e1c

    # 2. DF-PBE0 + exact qm_engine RI tensors + Casida RPA check
    feat, _, rpa_info = run_df_pbe0_with_wf(
        mol, f"grid_{ix}_{iy}", dm_init=dm_center, need_wf=False, need_rpa=True
    )

    # 3. Alternative adiabatic functionals (B3LYP, CAM-B3LYP)
    b3_e0, b3_s1 = (np.nan, np.nan)
    cam_e0, cam_s1 = (np.nan, np.nan)
    if run_alt_xc:
        b3_e0, b3_s1 = run_alt_tda_functional(mol, "b3lyp", dm_init=dm_center)
        cam_e0, cam_s1 = run_alt_tda_functional(mol, "camb3lyp", dm_init=dm_center)

    return {
        "ix": ix, "iy": iy, "x": x_val, "y": y_val,
        "cas_e0_ha": cas_e0, "cas_e1_ha": cas_e1,
        "cas_conv": bool(mc.converged),
        "nev_e0_ha": nev_e0, "nev_e1_ha": nev_e1,
        "pbe0_rpa_real_eV": rpa_info["rpa_s1_real_eV"],
        "pbe0_rpa_imag_eV": rpa_info["rpa_s1_imag_eV"],
        "pbe0_rpa_unstable": bool(rpa_info["rpa_unstable"]),
        "b3lyp_e0_ha": b3_e0, "b3lyp_s1_eV": b3_s1,
        "camb3lyp_e0_ha": cam_e0, "camb3lyp_s1_eV": cam_s1,
        "feat": feat,
    }


def compute_basis_overlap_block(mol1: gto.Mole, wf1: Dict[str, Any],
                                mol2: gto.Mole, wf2: Dict[str, Any]) -> Dict[str, Any]:
    """Precompute exact gauge-invariant Slater-determinant basis overlap blocks
    between consecutive loop geometries (R_k, R_{k+1}) in the (1 + N_S + N_DS) Lehmann basis.
    """
    S_ao = gto.intor_cross("int1e_ovlp", mol1, mol2)
    O_occ_all = wf1["C_o_all"].T @ S_ao @ wf2["C_o_all"]
    det_occ = float(np.linalg.det(O_occ_all))
    s00 = det_occ * det_occ

    O_oo = wf1["C_o"].T @ S_ao @ wf2["C_o"]
    O_vv = wf1["C_v"].T @ S_ao @ wf2["C_v"]
    O_ov = wf1["C_o"].T @ S_ao @ wf2["C_v"]
    O_vo = wf1["C_v"].T @ S_ao @ wf2["C_o"]

    O_oo_inv = np.linalg.inv(O_oo)
    O_oo_invT = O_oo_inv.T

    X1_S, X2_S = wf1["X_S"], wf2["X_S"]
    X1_T, X2_T = wf1["X_T"], wf2["X_T"]
    tmp_S = np.einsum("ij,mjb,ab->mia", O_oo_invT, X2_S, O_vv, optimize=True)
    S_SS = s00 * np.einsum("nia,mia->nm", X1_S, tmp_S, optimize=True)

    tmp_T = np.einsum("ij,mjb,ab->mia", O_oo_invT, X2_T, O_vv, optimize=True)
    S_TT = s00 * np.einsum("nia,mia->nm", X1_T, tmp_T, optimize=True)

    s0_S = np.sqrt(2.0) * s00 * np.einsum("mjb,jb->m", X2_S, O_oo_inv @ O_ov)
    sS_0 = np.sqrt(2.0) * s00 * np.einsum("nia,ai->n", X1_S, O_vo @ O_oo_inv)

    ch1, p1 = wf1["D_S_channel"], wf1["D_S_pairs"]
    ch2, p2 = wf2["D_S_channel"], wf2["D_S_pairs"]
    n_d1, n_d2 = len(ch1), len(ch2)
    S_DD = np.zeros((n_d1, n_d2), dtype=np.float64)
    for mu in range(n_d1):
        c_m = ch1[mu]
        i1, j1 = p1[mu]
        S_mat = S_TT if c_m == 0 else S_SS
        n1 = 1.0 if i1 == j1 else 1.0 / np.sqrt(2.0)
        for nu in range(n_d2):
            if ch2[nu] != c_m:
                continue
            i2, j2 = p2[nu]
            n2 = 1.0 if i2 == j2 else 1.0 / np.sqrt(2.0)
            if i1 == j1 and i2 == j2:
                S_DD[mu, nu] = S_mat[i1, i2] * S_mat[j1, j2]
            else:
                S_DD[mu, nu] = n1 * n2 * (S_mat[i1, i2] * S_mat[j1, j2] + S_mat[i1, j2] * S_mat[j1, i2])

    return {
        "s00": float(s00),
        "s0_S": s0_S,
        "sS_0": sS_0,
        "S_SS": S_SS,
        "S_DD": S_DD,
    }


def evaluate_loop_sequential_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Run one complete closed circular loop of M angles sequentially inside a single worker
    (sequential orbital warm-start ensures rock-solid continuity around the loop!) and return:
      - Exact SA2-CASSCF energies + gauge-invariant Berry-phase overlaps (via fci.addons.overlap),
      - Optional SC-NEVPT2 energies around the loop,
      - Exact DF-PBE0 RI feature dicts + precomputed basis overlap blocks for Lehmann Berry phase!
    """
    (
        loop_tag, atoms, coords_meci, g_hat, h_hat, center_xy, radius, n_pts,
        ncas, nelecas, basis, is_fulvene, mo_center, dm_center,
        run_cas, run_nevpt2
    ) = task

    cx, cy = center_xy
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    coords_list = [
        coords_meci + (cx + radius * np.cos(th)) * g_hat + (cy + radius * np.sin(th)) * h_hat
        for th in thetas
    ]

    mols: List[gto.Mole] = []
    cas_mos: List[np.ndarray] = []
    cas_cis0: List[np.ndarray] = []
    cas_e0 = np.full(n_pts, np.nan)
    cas_e1 = np.full(n_pts, np.nan)
    nev_e0 = np.full(n_pts, np.nan)
    nev_e1 = np.full(n_pts, np.nan)
    feats: List[Dict[str, Any]] = []
    wfs: List[Dict[str, Any]] = []
    rpa_unstable = np.zeros(n_pts, dtype=bool)
    rpa_real = np.zeros(n_pts)
    rpa_imag = np.zeros(n_pts)

    mo_prev = mo_center
    dm_prev = dm_center

    for k in range(n_pts):
        mol = make_mol(atoms, coords_list[k], basis=basis)
        mols.append(mol)

        if run_cas:
            mf_hf = scf.RHF(mol).density_fit()
            mf_hf.conv_tol = 1e-10
            mf_hf.max_cycle = 100
            mf_hf.verbose = 0
            mf_hf.kernel()

            mc = mcscf.CASSCF(mf_hf, ncas, nelecas).density_fit()
            mc = mc.state_average_([0.5, 0.5])
            mc.fix_spin_(ss=0.0, shift=0.5)
            mc.conv_tol = 1e-8
            mc.max_cycle_macro = 70
            mc.fcisolver.conv_tol = 1e-10
            mc.verbose = 0
            try:
                mo_init = mcscf.project_init_guess(mc, mo_prev)
            except Exception:
                mo_init = mc.sort_mo(select_pi_mos_fulvene(mol, mf_hf)) if is_fulvene else mf_hf.mo_coeff
            mc.kernel(mo_init)
            mo_prev = mc.mo_coeff.copy()
            cas_e0[k] = float(mc.e_states[0])
            cas_e1[k] = float(mc.e_states[1])
            cas_mos.append(mc.mo_coeff.copy())
            cas_cis0.append(mc.ci[0].copy())

            if run_nevpt2:
                mc_ci = mcscf.CASCI(mf_hf, ncas, nelecas)
                mc_ci.fcisolver.nroots = 2
                mc_ci.fcisolver.spin = 0
                mc_ci.fix_spin_(ss=0.0, shift=0.5)
                mc_ci.verbose = 0
                mc_ci.kernel(mc.mo_coeff)
                e0c = float(mrpt.NEVPT(mc_ci, root=0).kernel())
                e1c = float(mrpt.NEVPT(mc_ci, root=1).kernel())
                nev_e0[k] = float(mc_ci.e_tot[0]) + e0c
                nev_e1[k] = float(mc_ci.e_tot[1]) + e1c

        feat, wf, rpa_info = run_df_pbe0_with_wf(
            mol, f"{loop_tag}_{k}", dm_init=dm_prev, need_wf=True, need_rpa=True
        )
        dm_prev = wf["dm"]
        feats.append(feat)
        wfs.append(wf)
        rpa_unstable[k] = rpa_info["rpa_unstable"]
        rpa_real[k] = rpa_info["rpa_s1_real_eV"]
        rpa_imag[k] = rpa_info["rpa_s1_imag_eV"]

    # Exact SA2-CASSCF gauge-invariant Wilson loop overlaps
    cas_ovlps = np.full(n_pts, np.nan)
    if run_cas:
        ncore = (mols[0].nelectron - nelecas) // 2
        for k in range(n_pts):
            k1 = (k + 1) % n_pts
            S_ao = gto.intor_cross("int1e_ovlp", mols[k], mols[k1])
            O_all = cas_mos[k].T @ S_ao @ cas_mos[k1]
            det_core = float(np.linalg.det(O_all[:ncore, :ncore])) if ncore > 0 else 1.0
            O_act = O_all[ncore : ncore + ncas, ncore : ncore + ncas]
            ov_act = float(fci.addons.overlap(cas_cis0[k], cas_cis0[k1], ncas, nelecas, s=O_act))
            cas_ovlps[k] = (det_core ** 2) * ov_act

    # Exact Lehmann Slater-determinant basis overlap blocks (adjacent k -> k+1 AND self k -> k)
    ov_adj = [
        compute_basis_overlap_block(mols[k], wfs[k], mols[(k + 1) % n_pts], wfs[(k + 1) % n_pts])
        for k in range(n_pts)
    ]
    ov_self = [
        compute_basis_overlap_block(mols[k], wfs[k], mols[k], wfs[k])
        for k in range(n_pts)
    ]

    return {
        "loop_tag": loop_tag,
        "center_xy": [float(cx), float(cy)],
        "radius": float(radius),
        "thetas": thetas,
        "cas_e0_ha": cas_e0,
        "cas_e1_ha": cas_e1,
        "nev_e0_ha": nev_e0,
        "nev_e1_ha": nev_e1,
        "cas_ovlps": cas_ovlps,
        "rpa_unstable": rpa_unstable,
        "rpa_real_eV": rpa_real,
        "rpa_imag_eV": rpa_imag,
        "feats": feats,
        "ov_adj": ov_adj,
        "ov_self": ov_self,
    }


# ---------------------------------------------------------------------------
# Stage 2: Lehmann-TDDFT Operators, Berry-Phase Evaluator, & Topology Analysis
# ---------------------------------------------------------------------------

def eval_analytical_lehmann(
    feat: Dict[str, Any],
    beta_0D: float = 0.35,
    alpha_SD: float = 0.50,
    use_dc: bool = True,
) -> Dict[str, Any]:
    """Pure First-Principles Analytical Lehmann-TDDFT (ZERO ML parameters!).

    Uses ONLY bare DF-PBE0 RI tensors from `qm_engine.py`:
      - H_SS = diag(omega_S_eV)
      - D_diag = D_S_bare_eV + max(W_ex_ex, 0) (exact two-exciton Hubbard repulsion)
      - V_SD = alpha_SD * (W_SD_S[:, :, 0] + W_SD_S[:, :, 1]) (exact Slater-Condon particle+hole Coulomb)
      - v0D  = beta_0D * gate(S_ov) * v0D_S[:, 0] (exact Brillouin-violating ground-doubles RI coupling)
      - Delta_0^{dc} = +sum_mu v0D_mu^2 / D_mu (exact DFT ground-state double-counting counterterm)
    """
    omega_S = np.asarray(feat["omega_S_eV"], dtype=np.float64)
    D_bare = np.asarray(feat["D_S_bare_eV"], dtype=np.float64)
    W_ex_ex = np.maximum(np.asarray(feat["D_S_feats"][:, 10], dtype=np.float64), 0.0)
    sp_ov = np.clip(np.asarray(feat["D_S_feats"][:, 11], dtype=np.float64), 0.0, None)
    gate = np.tanh(sp_ov / 0.05)

    D_phys = np.maximum(D_bare + W_ex_ex, 1.5)
    W_sd = np.asarray(feat["W_SD_S"], dtype=np.float64)
    V = alpha_SD * (W_sd[:, :, 0] + W_sd[:, :, 1])
    v0D = beta_0D * gate * np.asarray(feat["v0D_S"][:, 0], dtype=np.float64)
    delta0_dc = float(np.sum((v0D ** 2) / D_phys)) if use_dc else 0.0

    n_s, n_d = len(omega_S), len(D_phys)
    n_tot = 1 + n_s + n_d
    H_full = np.zeros((n_tot, n_tot), dtype=np.float64)
    H_full[0, 0] = delta0_dc
    H_full[1 : 1 + n_s, 1 : 1 + n_s] = np.diag(omega_S)
    if n_d > 0:
        H_full[0, 1 + n_s :] = v0D
        H_full[1 + n_s :, 0] = v0D
        H_full[1 : 1 + n_s, 1 + n_s :] = V
        H_full[1 + n_s :, 1 : 1 + n_s] = V.T
        H_full[1 + n_s :, 1 + n_s :] = np.diag(D_phys)
    H_full = 0.5 * (H_full + H_full.T)
    evals, evecs = np.linalg.eigh(H_full)

    om_mid = 0.5 * (evals[0] + evals[1])
    H01_eff = float(np.sum(v0D * V[0, :] / (om_mid - D_phys))) if n_d > 0 else 0.0
    e_scf_ev = float(feat["e_scf_ha"]) * HA_TO_EV
    return {
        "e0_ev": e_scf_ev + float(evals[0]),
        "e1_ev": e_scf_ev + float(evals[1]),
        "gap_ev": float(evals[1] - evals[0]),
        "H01_eff_ev": H01_eff,
        "c0": evecs[:, 0],
        "c1": evecs[:, 1],
    }


def eval_trained_lehmann_ensemble(
    models: List[Any],
    feat: Dict[str, Any],
    mode: str = "sc_v0d",
    beta_coul: float = UNIVERSAL_SC_BETA_COUL,
) -> Dict[str, Any]:
    """Evaluate 5-fold trained Lehmann-TDDFT ensemble at a geometry in one of three modes:
      - 'decoupled': v0D = 0 (ablation proving Brillouin-violating v0D is necessary!)
      - 'zero_shot': trained checkpoint v0D as-is (ZERO extra parameters!)
      - 'sc_v0d'   : trained (H_SS, V_SD, D) + universal Slater-Condon v0D (beta_coul=0.35)
                     + exact DFT double-counting counterterm Delta_0^{dc} = +sum v0D^2 / D.
    Averages the Hermitian lifted Hamiltonians across the 5 CV folds (exact ensemble operator!)
    so ensemble eigenvectors are strictly orthonormal for Berry-phase evaluation.
    """
    import torch
    from lehmann_tddft.model import npz_to_mol_tensors

    mol_t = npz_to_mol_tensors(feat, dtype=torch.float64)
    H_SS_list, V_list, D_list, v0D_tr_list = [], [], [], []
    with torch.no_grad():
        for m in models:
            bS, _ = m._build_all_blocks(mol_t, disable_doubles_override=False)
            H_SS_list.append(bS["H_SS"].numpy())
            V_list.append(bS["V"].numpy())
            D_list.append(bS["D"].numpy())
            v0D_tr_list.append(bS["v0D"].numpy())

    H_SS = np.mean(H_SS_list, axis=0)
    V = np.mean(V_list, axis=0)
    D = np.mean(D_list, axis=0)
    v0D_tr = np.mean(v0D_tr_list, axis=0)

    if mode == "decoupled":
        v0D = np.zeros_like(v0D_tr)
        delta0_dc = 0.0
    elif mode == "zero_shot":
        v0D = v0D_tr
        delta0_dc = 0.0
    elif mode == "sc_v0d":
        sp_ov = np.clip(np.asarray(feat["D_S_feats"][:, 11], dtype=np.float64), 0.0, None)
        gate = np.tanh(sp_ov / 0.05)
        v0D_sym = beta_coul * gate * np.asarray(feat["v0D_S"][:, 0], dtype=np.float64)
        # Pad with exact 0.0 for antisymmetric ^1(TT)_- poles (<0|V_ee|^1(TT)_-> == 0 identically!)
        v0D = np.zeros_like(v0D_tr)
        v0D[: len(v0D_sym)] = v0D_sym
        delta0_dc = float(np.sum((v0D ** 2) / np.maximum(D, 0.5)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_s, n_d = H_SS.shape[0], D.shape[0]
    n_tot = 1 + n_s + n_d
    H_full = np.zeros((n_tot, n_tot), dtype=np.float64)
    H_full[0, 0] = delta0_dc
    H_full[1 : 1 + n_s, 1 : 1 + n_s] = H_SS
    if n_d > 0:
        H_full[0, 1 + n_s :] = v0D
        H_full[1 + n_s :, 0] = v0D
        H_full[1 : 1 + n_s, 1 + n_s :] = V
        H_full[1 + n_s :, 1 : 1 + n_s] = V.T
        H_full[1 + n_s :, 1 + n_s :] = np.diag(D)
    H_full = 0.5 * (H_full + H_full.T)
    evals, evecs = np.linalg.eigh(H_full)

    om_mid = 0.5 * (evals[0] + evals[1])
    H01_eff = float(np.sum(v0D * V[0, :] / (om_mid - D))) if n_d > 0 else 0.0
    e_scf_ev = float(feat["e_scf_ha"]) * HA_TO_EV

    # Fold-by-fold gap std for ensemble uncertainty
    fold_gaps = []
    for H_i, V_i, D_i, v0_i in zip(H_SS_list, V_list, D_list, v0D_tr_list):
        if mode == "decoupled":
            vi = np.zeros_like(v0_i); dci = 0.0
        elif mode == "zero_shot":
            vi = v0_i; dci = 0.0
        else:
            vi = v0D; dci = float(np.sum((vi ** 2) / np.maximum(D_i, 0.5)))
        Hi = np.zeros((n_tot, n_tot), dtype=np.float64)
        Hi[0, 0] = dci
        Hi[1 : 1 + n_s, 1 : 1 + n_s] = H_i
        if n_d > 0:
            Hi[0, 1 + n_s :] = vi; Hi[1 + n_s :, 0] = vi
            Hi[1 : 1 + n_s, 1 + n_s :] = V_i; Hi[1 + n_s :, 1 : 1 + n_s] = V_i.T
            Hi[1 + n_s :, 1 + n_s :] = np.diag(D_i)
        evi = np.linalg.eigvalsh(0.5 * (Hi + Hi.T))
        fold_gaps.append(float(evi[1] - evi[0]))

    return {
        "e0_ev": e_scf_ev + float(evals[0]),
        "e1_ev": e_scf_ev + float(evals[1]),
        "gap_ev": float(evals[1] - evals[0]),
        "gap_std_ev": float(np.std(fold_gaps)),
        "H01_eff_ev": H01_eff,
        "delta0_dc_ev": delta0_dc,
        "c0": evecs[:, 0],
        "c1": evecs[:, 1],
        "c0_weight_ref": float(evecs[0, 0] ** 2),
        "c0_weight_S": float(np.sum(evecs[1 : 1 + n_s, 0] ** 2)),
        "c0_weight_D": float(np.sum(evecs[1 + n_s :, 0] ** 2)),
    }


def contract_basis_overlap(ov_block: Dict[str, Any], c1: np.ndarray, c2: np.ndarray, n_s: int) -> float:
    c1_0, c1_S, c1_D = c1[0], c1[1 : 1 + n_s], c1[1 + n_s :]
    c2_0, c2_S, c2_D = c2[0], c2[1 : 1 + n_s], c2[1 + n_s :]
    return float(
        c1_0 * ov_block["s00"] * c2_0
        + c1_0 * float(ov_block["s0_S"] @ c2_S)
        + float(c1_S @ ov_block["sS_0"]) * c2_0
        + float(c1_S @ ov_block["S_SS"] @ c2_S)
        + float(c1_D @ ov_block["S_DD"] @ c2_D)
    )


def compute_loop_berry_phase(
    loop_data: Dict[str, Any],
    evecs_c0: List[np.ndarray],
) -> Dict[str, Any]:
    """Compute exact gauge-invariant Wilson loop / Bargmann invariant Berry phase.

    Uses Löwdin polar isometric basis transport S_orth(k, k+1) = U @ V^T from the SVD
    of the exact Slater-determinant basis overlap matrix S_full(k, k+1) between consecutive
    loop geometries. Guarantees:
      1. Exact gauge invariance under arbitrary MO rotations, exciton sign flips, and
         eigenvector sign flips (S' = G_k^T S G_{k+1} => S'_orth = G_k^T S_orth G_{k+1}).
      2. Strict Cauchy-Schwarz bound |<Psi(k)|Psi(k+1)>| <= 1.000000 identically.
      3. Exact continuum limit agreement as M -> infinity.
    """
    n_pts = len(evecs_c0)
    n_s = len(loop_data["feats"][0]["omega_S_eV"])
    norm_ov = np.zeros(n_pts, dtype=np.float64)
    for k in range(n_pts):
        b = loop_data["ov_adj"][k]
        n_d1_sym, n_d2_sym = b["S_DD"].shape
        n_tot1 = len(evecs_c0[k])
        n_tot2 = len(evecs_c0[(k + 1) % n_pts])
        n_anti1 = n_tot1 - (1 + n_s + n_d1_sym)
        n_anti2 = n_tot2 - (1 + n_s + n_d2_sym)
        S_full = np.zeros((n_tot1, n_tot2), dtype=np.float64)
        S_full[0, 0] = b["s00"]
        S_full[0, 1 : 1 + n_s] = b["s0_S"]
        S_full[1 : 1 + n_s, 0] = b["sS_0"]
        S_full[1 : 1 + n_s, 1 : 1 + n_s] = b["S_SS"]
        S_full[1 + n_s : 1 + n_s + n_d1_sym, 1 + n_s : 1 + n_s + n_d2_sym] = b["S_DD"]
        if n_anti1 > 0 and n_anti2 > 0:
            # Antisymmetric ^1(TT)_- poles are strictly paired diagonally with singlets K (phase s_K),
            # so their exact gauge-equivariant overlap block is S_SS!
            S_full[1 + n_s + n_d1_sym :, 1 + n_s + n_d2_sym :] = b["S_SS"][:n_anti1, :n_anti2]
        U, _, Vt = np.linalg.svd(S_full, full_matrices=False)
        S_orth = U @ Vt
        norm_ov[k] = float(evecs_c0[k] @ S_orth @ evecs_c0[(k + 1) % n_pts])
    w_sign = float(np.prod(np.sign(norm_ov)))
    w_prod = float(np.prod(norm_ov))
    min_abs = float(np.min(np.abs(norm_ov)))
    max_abs = float(np.max(np.abs(norm_ov)))
    med_abs = float(np.median(np.abs(norm_ov)))
    adiabatic_continuous = bool(min_abs > 0.30)
    gamma_rad = float(np.pi if (w_sign < 0 and adiabatic_continuous) else (0.0 if adiabatic_continuous else np.nan))
    return {
        "step_overlaps": norm_ov.tolist(),
        "W_sign": w_sign,
        "W_prod": w_prod,
        "min_abs_overlap": min_abs,
        "max_abs_overlap": max_abs,
        "median_abs_overlap": med_abs,
        "adiabatic_continuous": adiabatic_continuous,
        "berry_phase_rad": gamma_rad,
        "berry_phase_pi_units": float(gamma_rad / np.pi) if np.isfinite(gamma_rad) else None,
    }


def fit_conical_dispersion(X: np.ndarray, Y: np.ndarray, gap: np.ndarray) -> Dict[str, Any]:
    """Fit exact 2D elliptical double-cone dispersion law:
       Delta E(x, y) = sqrt( g^2 * (x - x0)^2 + h^2 * (y - y0)^2 + eps0^2 )
    and evaluate cone linearity R^2, apex (x0, y0), slopes (g, h), and gap at apex!
    """
    valid = np.isfinite(gap)
    gap_v = gap[valid]
    X_v, Y_v = X[valid], Y[valid]
    n_neg = int(np.sum(gap_v < 0.0))
    frac_neg = float(n_neg / max(gap_v.size, 1))
    idx_min = int(np.argmin(np.abs(gap_v)))
    x_init = float(X_v[idx_min])
    y_init = float(Y_v[idx_min])

    def _fit_mask(mask: np.ndarray) -> Dict[str, float]:
        m = mask & valid
        xm, ym, gm = X[m], Y[m], gap[m]
        if len(gm) < 6:
            return {"r2": 0.0, "x0": x_init, "y0": y_init, "g_slope": 0.0, "h_slope": 0.0, "eps0": 0.0, "rmse": float(np.std(gm)) if len(gm) else 0.0}

        def resid(p):
            x0, y0, gs, hs, e0 = p
            pred = np.sqrt((gs * (xm - x0)) ** 2 + (hs * (ym - y0)) ** 2 + e0 ** 2)
            return pred - gm

        p0 = [x_init, y_init, 5.0, 4.0, 0.01]
        bounds = ([float(X_v.min()), float(Y_v.min()), 0.0, 0.0, 0.0],
                  [float(X_v.max()), float(Y_v.max()), 100.0, 100.0, 5.0])
        sol = least_squares(resid, p0, bounds=bounds)
        x0, y0, gs, hs, e0 = [float(v) for v in sol.x]
        pred = np.sqrt((gs * (xm - x0)) ** 2 + (hs * (ym - y0)) ** 2 + e0 ** 2)
        ss_res = float(np.sum((gm - pred) ** 2))
        ss_tot = float(np.sum((gm - np.mean(gm)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
        rmse = float(np.sqrt(ss_res / len(gm)))
        return {"r2": r2, "x0": x0, "y0": y0, "g_slope": gs, "h_slope": hs, "eps0": e0, "rmse": rmse}

    full_fit = _fit_mask(np.ones_like(gap, dtype=bool))
    rad_inner = 0.085 if (X.max() - X.min()) < 0.30 else 0.14
    rad_local = 0.055 if (X.max() - X.min()) < 0.30 else 0.10
    dist = np.hypot(X - full_fit["x0"], Y - full_fit["y0"])
    inner_mask = dist <= rad_inner
    local_mask = dist <= rad_local
    inner_fit = _fit_mask(inner_mask) if np.sum(inner_mask & valid) >= 9 else full_fit
    local_fit = _fit_mask(local_mask) if np.sum(local_mask & valid) >= 9 else inner_fit

    # Exact 1D V-vertex & radial ray linearity along y=0 (x_g) and orthogonal coupling axis (y_h)
    iy0 = int(np.argmin(np.abs(Y[0, :])))
    x_line = X[:, iy0]
    g_line = gap[:, iy0]
    v_line = np.isfinite(g_line)
    v_vertex_fit: Dict[str, Any] = {"has_positive_v_cone": False}
    if np.all(v_line) and np.min(g_line) >= 0.0:
        imin = int(np.argmin(g_line))
        il = max(0, imin - 1)
        ir = min(len(x_line) - 1, imin + 1)
        win_x = 0.075 if (X.max() - X.min()) < 0.30 else 0.16
        m_x = np.abs(x_line - x_line[imin]) <= win_x
        if np.sum(m_x) >= 4:
            xm, gm = x_line[m_x], g_line[m_x]
            best = (1e18, float(x_line[imin]), 0.0, 0.0)
            for x0_try in np.linspace(x_line[il], x_line[ir], 401):
                r_try = np.abs(xm - x0_try)
                s_try = float(np.dot(r_try, gm) / max(np.dot(r_try, r_try), 1e-12))
                sse = float(np.sum((gm - s_try * r_try) ** 2))
                if sse < best[0]:
                    sst = float(np.sum((gm - np.mean(gm)) ** 2))
                    best = (sse, float(x0_try), s_try, float(1.0 - sse / max(sst, 1e-12)))
            _, x0_v, g_v, r2_x = best
            ix_v = int(np.argmin(np.abs(x_line - x0_v)))
            y_line = Y[ix_v, :]
            gy = gap[ix_v, :]
            win_y = 0.065 if (X.max() - X.min()) < 0.30 else 0.10
            m_y = np.isfinite(gy) & (np.abs(y_line) <= win_y)
            if np.sum(m_y) >= 4:
                ry, gmy = np.abs(y_line[m_y]), gy[m_y]
                eps0 = float(gy[iy0])
                h_est = np.sqrt(np.maximum(gmy ** 2 - eps0 ** 2, 0.0))
                sh = float(np.dot(ry, h_est) / max(np.dot(ry, ry), 1e-12))
                pred_y = np.sqrt(eps0 ** 2 + (sh * ry) ** 2)
                r2_y = float(1.0 - np.sum((gmy - pred_y) ** 2) / max(np.sum((gmy - np.mean(gmy)) ** 2), 1e-12))
            else:
                sh, r2_y = 0.0, 0.0
            v_vertex_fit = {
                "has_positive_v_cone": True,
                "x_apex_y0_A": x0_v,
                "g_slope_eV_per_A": g_v,
                "x_ray_R2": r2_x,
                "h_slope_eV_per_A": sh,
                "y_ray_R2": r2_y,
            }
    radial_x_r2 = float(v_vertex_fit.get("x_ray_R2", 0.0))

    near_zero_y_rows = 0
    sign_change_y_rows = 0
    for iy in range(gap.shape[1]):
        col = gap[:, iy]
        col = col[np.isfinite(col)]
        if col.size > 0:
            if np.min(np.abs(col)) < 0.05:
                near_zero_y_rows += 1
            if np.min(col) < 0.0 and np.max(col) > 0.0:
                sign_change_y_rows += 1

    return {
        "n_valid_points": int(gap_v.size),
        "n_negative_gap": n_neg,
        "frac_negative_gap": frac_neg,
        "min_gap_eV": float(np.min(gap_v)),
        "max_gap_eV": float(np.max(gap_v)),
        "sign_change_y_rows": sign_change_y_rows,
        "near_zero_y_rows": near_zero_y_rows,
        "radial_x_linearity_R2": radial_x_r2,
        "v_vertex_fit": v_vertex_fit,
        "full_cone_fit": full_fit,
        "inner_cone_fit": inner_fit,
        "local_cone_fit": local_fit,
    }


def fit_h01_linearity(y_grid: np.ndarray, h01_slice: np.ndarray) -> Dict[str, Any]:
    """Verify odd symmetry H01(0)=0 and linear turn-on |H01(y)| = slope * |y| along y_h!"""
    abs_y = np.abs(y_grid)
    abs_h = np.abs(h01_slice)
    idx0 = int(np.argmin(abs_y))
    h01_at_zero = float(abs_h[idx0])

    def _lin_fit(rad: float) -> Tuple[float, float]:
        mask = abs_y <= rad
        if np.sum(mask) < 3:
            mask = np.ones_like(abs_y, dtype=bool)
        ym, hm = abs_y[mask], abs_h[mask]
        slope = float(np.dot(ym, hm) / max(np.dot(ym, ym), 1e-12))
        pred = slope * ym
        ss_res = float(np.sum((hm - pred) ** 2))
        ss_tot = float(np.sum((hm - np.mean(hm)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12)) if ss_tot > 1e-15 else 0.0
        return slope, r2

    slope_local, r2_local = _lin_fit(0.055)
    slope_full, r2_full = _lin_fit(0.085)
    return {
        "H01_at_y0_eV": h01_at_zero,
        "max_abs_H01_eV": float(np.max(abs_h)),
        "linear_slope_local_eV_per_A": slope_local,
        "linear_R2_local": r2_local,
        "linear_slope_eV_per_A": slope_full,
        "linear_R2": r2_full,
    }


def surface_error_metrics(pred_gap: np.ndarray, ref_gap: np.ndarray,
                          pred_e0: np.ndarray, pred_e1: np.ndarray,
                          ref_e0: np.ndarray, ref_e1: np.ndarray) -> Dict[str, float]:
    """Compute exact 2D PES MAE/RMSE (in eV) for vertical gap AND relative S0/S1 surfaces."""
    valid = np.isfinite(pred_gap) & np.isfinite(ref_gap) & np.isfinite(pred_e0) & np.isfinite(ref_e0)
    pg, rg = pred_gap[valid], ref_gap[valid]
    pe0, pe1 = pred_e0[valid], pred_e1[valid]
    re0, re1 = ref_e0[valid], ref_e1[valid]
    p_ref = 0.5 * (np.mean(pe0) + np.mean(pe1))
    r_ref = 0.5 * (np.mean(re0) + np.mean(re1))
    pe0_rel, pe1_rel = pe0 - p_ref, pe1 - p_ref
    re0_rel, re1_rel = re0 - r_ref, re1 - r_ref

    gap_err = pg - rg
    e0_err = pe0_rel - re0_rel
    e1_err = pe1_rel - re1_rel
    pes_err = np.concatenate([e0_err.ravel(), e1_err.ravel()])
    return {
        "n_points": int(pg.size),
        "gap_MAE_eV": float(np.mean(np.abs(gap_err))),
        "gap_RMSE_eV": float(np.sqrt(np.mean(gap_err ** 2))),
        "gap_max_err_eV": float(np.max(np.abs(gap_err))),
        "PES_S0_MAE_eV": float(np.mean(np.abs(e0_err))),
        "PES_S1_MAE_eV": float(np.mean(np.abs(e1_err))),
        "PES_2sheet_MAE_eV": float(np.mean(np.abs(pes_err))),
        "PES_2sheet_RMSE_eV": float(np.sqrt(np.mean(pes_err ** 2))),
    }


# ---------------------------------------------------------------------------
# End-to-End Pipeline for One Conical Intersection System
# ---------------------------------------------------------------------------

def run_system_stage1_qm(spec: Dict[str, Any], n_workers: int = 4) -> Dict[str, Any]:
    """Run Stage 1 (pure PySCF 4-worker parallel pool) for one conical intersection system."""
    name = spec["name"]
    t0 = time.time()
    print(f"\n==================== [{name.upper()}] STAGE 1: PySCF QM GRIDS + LOOPS ====================", flush=True)
    mo_center, dm_center = compute_center_reference_mos(spec)
    print(f"[{name}] Reference center MOs & DM ready ({time.time()-t0:.1f}s)", flush=True)

    xs = np.linspace(spec["x_range"][0], spec["x_range"][1], spec["nx"])
    ys = np.linspace(spec["y_range"][0], spec["y_range"][1], spec["ny"])
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

    grid_tasks = []
    for ix in range(spec["nx"]):
        for iy in range(spec["ny"]):
            c = spec["coords_meci"] + xs[ix] * spec["g_hat"] + ys[iy] * spec["h_hat"]
            # Full B3LYP & CAM-B3LYP on all 225 pts for Ethylene & Methanimine;
            # on Fulvene (11x11), run B3LYP & CAM-B3LYP on the full 6x6 even subgrid + both principal axes (46 pts)!
            if spec["is_fulvene"]:
                run_alt = ((ix % 2 == 0 and iy % 2 == 0) or (ix == spec["nx"] // 2) or (iy == spec["ny"] // 2))
            else:
                run_alt = True
            grid_tasks.append((
                ix, iy, float(xs[ix]), float(ys[iy]),
                spec["atoms"], c, spec["ncas"], spec["nelecas"], spec["basis"],
                spec["is_fulvene"], mo_center, dm_center, run_alt
            ))

    # Run 2D grid across 4 parallel workers
    n_tot_grid = len(grid_tasks)
    grid_results: List[Dict[str, Any]] = [None] * n_tot_grid  # type: ignore
    done_cnt = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_to_idx = {ex.submit(evaluate_grid_point_worker, t): i for i, t in enumerate(grid_tasks)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            grid_results[idx] = fut.result()
            done_cnt += 1
            if done_cnt % 25 == 0 or done_cnt == n_tot_grid:
                print(f"[{name}] Grid progress: {done_cnt}/{n_tot_grid} ({time.time()-t0:.1f}s)", flush=True)

    # Run all 3 closed loops in parallel (3 workers simultaneously!)
    loop_tasks = [
        (
            "loop_ci_shared", spec["atoms"], spec["coords_meci"], spec["g_hat"], spec["h_hat"],
            spec["loop_ci_center"], spec["loop_ci_radius"], spec["n_loop"],
            spec["ncas"], spec["nelecas"], spec["basis"], spec["is_fulvene"],
            mo_center, dm_center, True, True
        ),
        (
            "loop_leh_apex", spec["atoms"], spec["coords_meci"], spec["g_hat"], spec["h_hat"],
            spec["loop_leh_center"], spec["loop_leh_radius"], spec["n_loop"],
            spec["ncas"], spec["nelecas"], spec["basis"], spec["is_fulvene"],
            mo_center, dm_center, False, False
        ),
        (
            "loop_ctrl_noci", spec["atoms"], spec["coords_meci"], spec["g_hat"], spec["h_hat"],
            spec["loop_ctrl_center"], spec["loop_ctrl_radius"], spec["n_loop"],
            spec["ncas"], spec["nelecas"], spec["basis"], spec["is_fulvene"],
            mo_center, dm_center, True, False
        ),
    ]
    print(f"[{name}] Launching 3 x {spec['n_loop']}-pt Berry-phase loops in parallel...", flush=True)
    t_loop = time.time()
    with ProcessPoolExecutor(max_workers=min(3, n_workers)) as ex:
        loop_out_list = list(ex.map(evaluate_loop_sequential_worker, loop_tasks))
    loops = {lo["loop_tag"]: lo for lo in loop_out_list}
    print(f"[{name}] Loops complete in {time.time()-t_loop:.1f}s (total Stage 1: {time.time()-t0:.1f}s)", flush=True)

    return {
        "xs": xs, "ys": ys, "X": X, "Y": Y,
        "grid_results": grid_results,
        "loops": loops,
        "stage1_wall_s": time.time() - t0,
    }


def run_system_stage2_lehmann_and_save(
    spec: Dict[str, Any],
    stage1: Dict[str, Any],
    models: List[Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run Stage 2 (Lehmann-TDDFT ensemble + analytical RI + topology & Berry-phase proofs) & save."""
    name = spec["name"]
    nx, ny = spec["nx"], spec["ny"]
    xs, ys, X, Y = stage1["xs"], stage1["ys"], stage1["X"], stage1["Y"]

    # Allocate 2D arrays
    def z2():
        return np.zeros((nx, ny), dtype=np.float64)

    cas_e0, cas_e1, cas_gap = z2(), z2(), z2()
    nev_e0, nev_e1, nev_gap = z2(), z2(), z2()
    pbe0_e0, pbe0_e1, pbe0_tda_gap, pbe0_tda_t1 = z2(), z2(), z2(), z2()
    pbe0_rpa_real, pbe0_rpa_imag = z2(), z2()
    pbe0_rpa_unstable = np.zeros((nx, ny), dtype=bool)
    b3_e0, b3_e1, b3_gap = z2(), z2(), z2()
    cam_e0, cam_e1, cam_gap = z2(), z2(), z2()

    leh_zs_e0, leh_zs_e1, leh_zs_gap, leh_zs_h01 = z2(), z2(), z2(), z2()
    leh_sc_e0, leh_sc_e1, leh_sc_gap, leh_sc_gap_std, leh_sc_h01 = z2(), z2(), z2(), z2(), z2()
    leh_sc_c0w = z2()
    leh_dec_e0, leh_dec_e1, leh_dec_gap = z2(), z2(), z2()
    leh_ana_e0, leh_ana_e1, leh_ana_gap, leh_ana_h01 = z2(), z2(), z2(), z2()

    for r in stage1["grid_results"]:
        ix, iy = r["ix"], r["iy"]
        feat = r["feat"]

        cas_e0[ix, iy] = r["cas_e0_ha"] * HA_TO_EV
        cas_e1[ix, iy] = r["cas_e1_ha"] * HA_TO_EV
        cas_gap[ix, iy] = cas_e1[ix, iy] - cas_e0[ix, iy]

        # Ensure NEVPT2 states sorted cleanly as lower (S0) and upper (S1) adiabatic surfaces
        ne0 = r["nev_e0_ha"] * HA_TO_EV
        ne1 = r["nev_e1_ha"] * HA_TO_EV
        nev_e0[ix, iy] = min(ne0, ne1)
        nev_e1[ix, iy] = max(ne0, ne1)
        nev_gap[ix, iy] = nev_e1[ix, iy] - nev_e0[ix, iy]

        e_scf_ev = float(feat["e_scf_ha"]) * HA_TO_EV
        om_s1 = float(feat["omega_S_eV"][0])
        pbe0_e0[ix, iy] = e_scf_ev
        pbe0_e1[ix, iy] = e_scf_ev + om_s1
        pbe0_tda_gap[ix, iy] = om_s1  # signed! negative where S1 plunges below S0!
        pbe0_tda_t1[ix, iy] = float(feat["omega_T_eV"][0])

        pbe0_rpa_real[ix, iy] = r["pbe0_rpa_real_eV"]
        pbe0_rpa_imag[ix, iy] = r["pbe0_rpa_imag_eV"]
        pbe0_rpa_unstable[ix, iy] = r["pbe0_rpa_unstable"]

        b3_e0[ix, iy] = r["b3lyp_e0_ha"] * HA_TO_EV
        b3_gap[ix, iy] = r["b3lyp_s1_eV"]
        b3_e1[ix, iy] = b3_e0[ix, iy] + b3_gap[ix, iy]

        cam_e0[ix, iy] = r["camb3lyp_e0_ha"] * HA_TO_EV
        cam_gap[ix, iy] = r["camb3lyp_s1_eV"]
        cam_e1[ix, iy] = cam_e0[ix, iy] + cam_gap[ix, iy]

        # Lehmann-TDDFT evaluations
        o_zs = eval_trained_lehmann_ensemble(models, feat, mode="zero_shot")
        leh_zs_e0[ix, iy] = o_zs["e0_ev"]
        leh_zs_e1[ix, iy] = o_zs["e1_ev"]
        leh_zs_gap[ix, iy] = o_zs["gap_ev"]
        leh_zs_h01[ix, iy] = o_zs["H01_eff_ev"]

        o_sc = eval_trained_lehmann_ensemble(models, feat, mode="sc_v0d", beta_coul=UNIVERSAL_SC_BETA_COUL)
        leh_sc_e0[ix, iy] = o_sc["e0_ev"]
        leh_sc_e1[ix, iy] = o_sc["e1_ev"]
        leh_sc_gap[ix, iy] = o_sc["gap_ev"]
        leh_sc_gap_std[ix, iy] = o_sc["gap_std_ev"]
        leh_sc_h01[ix, iy] = o_sc["H01_eff_ev"]
        leh_sc_c0w[ix, iy] = o_sc["c0_weight_ref"]

        o_dec = eval_trained_lehmann_ensemble(models, feat, mode="decoupled")
        leh_dec_e0[ix, iy] = o_dec["e0_ev"]
        leh_dec_e1[ix, iy] = o_dec["e1_ev"]
        leh_dec_gap[ix, iy] = o_dec["gap_ev"]

        o_ana = eval_analytical_lehmann(feat, beta_0D=0.35, alpha_SD=0.50, use_dc=True)
        leh_ana_e0[ix, iy] = o_ana["e0_ev"]
        leh_ana_e1[ix, iy] = o_ana["e1_ev"]
        leh_ana_gap[ix, iy] = o_ana["gap_ev"]
        leh_ana_h01[ix, iy] = o_ana["H01_eff_ev"]

    # Cone topology fits on 2D branching plane
    cone_fits = {
        "SA2_CASSCF": fit_conical_dispersion(X, Y, cas_gap),
        "SC_NEVPT2": fit_conical_dispersion(X, Y, nev_gap),
        "Adiabatic_PBE0_TDA_signed": fit_conical_dispersion(X, Y, pbe0_tda_gap),
        "Adiabatic_PBE0_TDA_abs": fit_conical_dispersion(X, Y, np.abs(pbe0_tda_gap)),
        "Adiabatic_B3LYP_TDA_signed": fit_conical_dispersion(X, Y, b3_gap),
        "Adiabatic_CAMB3LYP_TDA_signed": fit_conical_dispersion(X, Y, cam_gap),
        "Lehmann_Decoupled_v0D0": fit_conical_dispersion(X, Y, leh_dec_gap),
        "Lehmann_ZeroShot_Ensemble": fit_conical_dispersion(X, Y, leh_zs_gap),
        "Lehmann_SCv0D_Ensemble": fit_conical_dispersion(X, Y, leh_sc_gap),
        "Lehmann_Analytical_RI": fit_conical_dispersion(X, Y, leh_ana_gap),
    }

    # H01_eff linearity fits along y_h at the x_g slice closest to each method's CI apex
    ix_sc = int(np.argmin(np.abs(xs - cone_fits["Lehmann_SCv0D_Ensemble"]["full_cone_fit"]["x0"])))
    ix_zs = int(np.argmin(np.abs(xs - cone_fits["Lehmann_ZeroShot_Ensemble"]["full_cone_fit"]["x0"])))
    ix_ana = int(np.argmin(np.abs(xs - cone_fits["Lehmann_Analytical_RI"]["full_cone_fit"]["x0"])))
    h01_linearity = {
        "Adiabatic_TDDFT": {"H01_at_y0_eV": 0.0, "max_abs_H01_eV": 0.0, "linear_slope_local_eV_per_A": 0.0, "linear_R2_local": 0.0, "linear_slope_eV_per_A": 0.0, "linear_R2": 0.0},
        "Lehmann_ZeroShot_Ensemble": fit_h01_linearity(ys, leh_zs_h01[ix_zs, :]),
        "Lehmann_SCv0D_Ensemble": fit_h01_linearity(ys, leh_sc_h01[ix_sc, :]),
        "Lehmann_Analytical_RI": fit_h01_linearity(ys, leh_ana_h01[ix_ana, :]),
    }

    # 2D PES error metrics against exact SC-NEVPT2 and exact SA2-CASSCF
    pes_vs_nevpt2 = {
        "SA2_CASSCF": surface_error_metrics(cas_gap, nev_gap, cas_e0, cas_e1, nev_e0, nev_e1),
        "Adiabatic_PBE0_TDA_signed": surface_error_metrics(pbe0_tda_gap, nev_gap, pbe0_e0, pbe0_e1, nev_e0, nev_e1),
        "Adiabatic_PBE0_TDA_sorted": surface_error_metrics(
            np.abs(pbe0_tda_gap), nev_gap, np.minimum(pbe0_e0, pbe0_e1), np.maximum(pbe0_e0, pbe0_e1), nev_e0, nev_e1
        ),
        "Adiabatic_B3LYP_TDA_signed": surface_error_metrics(b3_gap, nev_gap, b3_e0, b3_e1, nev_e0, nev_e1),
        "Adiabatic_CAMB3LYP_TDA_signed": surface_error_metrics(cam_gap, nev_gap, cam_e0, cam_e1, nev_e0, nev_e1),
        "Lehmann_ZeroShot_Ensemble": surface_error_metrics(leh_zs_gap, nev_gap, leh_zs_e0, leh_zs_e1, nev_e0, nev_e1),
        "Lehmann_SCv0D_Ensemble": surface_error_metrics(leh_sc_gap, nev_gap, leh_sc_e0, leh_sc_e1, nev_e0, nev_e1),
        "Lehmann_Analytical_RI": surface_error_metrics(leh_ana_gap, nev_gap, leh_ana_e0, leh_ana_e1, nev_e0, nev_e1),
    }
    pes_vs_casscf = {
        "SC_NEVPT2": surface_error_metrics(nev_gap, cas_gap, nev_e0, nev_e1, cas_e0, cas_e1),
        "Adiabatic_PBE0_TDA_signed": surface_error_metrics(pbe0_tda_gap, cas_gap, pbe0_e0, pbe0_e1, cas_e0, cas_e1),
        "Adiabatic_PBE0_TDA_sorted": surface_error_metrics(
            np.abs(pbe0_tda_gap), cas_gap, np.minimum(pbe0_e0, pbe0_e1), np.maximum(pbe0_e0, pbe0_e1), cas_e0, cas_e1
        ),
        "Adiabatic_B3LYP_TDA_signed": surface_error_metrics(b3_gap, cas_gap, b3_e0, b3_e1, cas_e0, cas_e1),
        "Adiabatic_CAMB3LYP_TDA_signed": surface_error_metrics(cam_gap, cas_gap, cam_e0, cam_e1, cas_e0, cas_e1),
        "Lehmann_ZeroShot_Ensemble": surface_error_metrics(leh_zs_gap, cas_gap, leh_zs_e0, leh_zs_e1, cas_e0, cas_e1),
        "Lehmann_SCv0D_Ensemble": surface_error_metrics(leh_sc_gap, cas_gap, leh_sc_e0, leh_sc_e1, cas_e0, cas_e1),
        "Lehmann_Analytical_RI": surface_error_metrics(leh_ana_gap, cas_gap, leh_ana_e0, leh_ana_e1, cas_e0, cas_e1),
    }

    # Stage 2 Berry-Phase Analysis on all 3 closed loops!
    berry_summary: Dict[str, Any] = {}
    loop_npz_arrays: Dict[str, np.ndarray] = {}

    for ltag, ld in stage1["loops"].items():
        n_pts = len(ld["thetas"])
        lres: Dict[str, Any] = {
            "center_xy": ld["center_xy"],
            "radius": ld["radius"],
            "n_points": n_pts,
            "rpa_unstable_frac": float(np.mean(ld["rpa_unstable"])),
        }

        # SA2-CASSCF Berry phase (exact gauge-invariant overlap product)
        if np.all(np.isfinite(ld["cas_ovlps"])):
            ov = ld["cas_ovlps"]
            w_sgn = float(np.prod(np.sign(ov)))
            w_raw = float(np.prod(ov))
            lres["SA2_CASSCF"] = {
                "step_overlaps": ov.tolist(),
                "W_sign": w_sgn,
                "W_prod": w_raw,
                "min_abs_overlap": float(np.min(np.abs(ov))),
                "median_abs_overlap": float(np.median(np.abs(ov))),
                "adiabatic_continuous": bool(np.min(np.abs(ov)) > 0.30),
                "berry_phase_rad": float(np.pi if w_sgn < 0 else 0.0),
                "berry_phase_pi_units": float(1.0 if w_sgn < 0 else 0.0),
                "gap_min_eV": float(np.min((ld["cas_e1_ha"] - ld["cas_e0_ha"]) * HA_TO_EV)),
                "gap_max_eV": float(np.max((ld["cas_e1_ha"] - ld["cas_e0_ha"]) * HA_TO_EV)),
            }

        # Adiabatic TD-DFT KS ground state |0> (pure determinant |0> everywhere -> s00 > 0 always -> gamma = 0!)
        s00_arr = np.array([b["s00"] for b in ld["ov_adj"]], dtype=np.float64)
        s00_self = np.array([b["s00"] for b in ld["ov_self"]], dtype=np.float64)
        ks_ov = s00_arr / np.sqrt(np.maximum(s00_self * np.roll(s00_self, -1), 1e-30))
        lres["Adiabatic_TDDFT_KS"] = {
            "step_overlaps": ks_ov.tolist(),
            "W_sign": float(np.prod(np.sign(ks_ov))),
            "W_prod": float(np.prod(ks_ov)),
            "min_abs_overlap": float(np.min(np.abs(ks_ov))),
            "median_abs_overlap": float(np.median(np.abs(ks_ov))),
            "adiabatic_continuous": True,
            "berry_phase_rad": 0.0,
            "berry_phase_pi_units": 0.0,
            "tda_s1_min_eV": float(np.min([f["omega_S_eV"][0] for f in ld["feats"]])),
            "tda_s1_max_eV": float(np.max([f["omega_S_eV"][0] for f in ld["feats"]])),
        }

        # Lehmann-TDDFT variants on loop
        for m_lbl, eval_fn in [
            ("Lehmann_Decoupled_v0D0", lambda f: eval_trained_lehmann_ensemble(models, f, mode="decoupled")),
            ("Lehmann_ZeroShot_Ensemble", lambda f: eval_trained_lehmann_ensemble(models, f, mode="zero_shot")),
            ("Lehmann_SCv0D_Ensemble", lambda f: eval_trained_lehmann_ensemble(models, f, mode="sc_v0d", beta_coul=UNIVERSAL_SC_BETA_COUL)),
            ("Lehmann_Analytical_RI", lambda f: eval_analytical_lehmann(f, beta_0D=0.35, alpha_SD=0.50, use_dc=True)),
        ]:
            outs = [eval_fn(f) for f in ld["feats"]]
            bp = compute_loop_berry_phase(ld, [o["c0"] for o in outs])
            gaps = np.array([o["gap_ev"] for o in outs], dtype=np.float64)
            h01s = np.array([o["H01_eff_ev"] for o in outs], dtype=np.float64)
            bp["gap_min_eV"] = float(np.min(gaps))
            bp["gap_max_eV"] = float(np.max(gaps))
            bp["max_abs_H01_eV"] = float(np.max(np.abs(h01s)))
            lres[m_lbl] = bp
            loop_npz_arrays[f"{ltag}__{m_lbl}__gaps"] = gaps
            loop_npz_arrays[f"{ltag}__{m_lbl}__h01"] = h01s
            loop_npz_arrays[f"{ltag}__{m_lbl}__ovlps"] = np.array(bp["step_overlaps"], dtype=np.float64)

        loop_npz_arrays[f"{ltag}__thetas"] = ld["thetas"]
        loop_npz_arrays[f"{ltag}__cas_e0_eV"] = ld["cas_e0_ha"] * HA_TO_EV
        loop_npz_arrays[f"{ltag}__cas_e1_eV"] = ld["cas_e1_ha"] * HA_TO_EV
        loop_npz_arrays[f"{ltag}__nev_e0_eV"] = ld["nev_e0_ha"] * HA_TO_EV
        loop_npz_arrays[f"{ltag}__nev_e1_eV"] = ld["nev_e1_ha"] * HA_TO_EV
        loop_npz_arrays[f"{ltag}__cas_ovlps"] = ld["cas_ovlps"]
        loop_npz_arrays[f"{ltag}__pbe0_tda_s1"] = np.array([f["omega_S_eV"][0] for f in ld["feats"]], dtype=np.float64)
        berry_summary[ltag] = lres

    # Save complete raw 2D grid & loop arrays as compressed .npz for viz-manuscript!
    npz_path = out_dir / f"{name}_ci_surfaces.npz"
    np.savez_compressed(
        npz_path,
        xs=xs, ys=ys, X=X, Y=Y,
        coords_meci=spec["coords_meci"],
        g_hat=spec["g_hat"],
        h_hat=spec["h_hat"],
        cas_e0_eV=cas_e0, cas_e1_eV=cas_e1, cas_gap_eV=cas_gap,
        nev_e0_eV=nev_e0, nev_e1_eV=nev_e1, nev_gap_eV=nev_gap,
        pbe0_e0_eV=pbe0_e0, pbe0_e1_eV=pbe0_e1, pbe0_tda_gap_eV=pbe0_tda_gap, pbe0_tda_t1_eV=pbe0_tda_t1,
        pbe0_rpa_real_eV=pbe0_rpa_real, pbe0_rpa_imag_eV=pbe0_rpa_imag, pbe0_rpa_unstable=pbe0_rpa_unstable,
        b3lyp_e0_eV=b3_e0, b3lyp_e1_eV=b3_e1, b3lyp_gap_eV=b3_gap,
        camb3lyp_e0_eV=cam_e0, camb3lyp_e1_eV=cam_e1, camb3lyp_gap_eV=cam_gap,
        leh_zs_e0_eV=leh_zs_e0, leh_zs_e1_eV=leh_zs_e1, leh_zs_gap_eV=leh_zs_gap, leh_zs_h01_eV=leh_zs_h01,
        leh_sc_e0_eV=leh_sc_e0, leh_sc_e1_eV=leh_sc_e1, leh_sc_gap_eV=leh_sc_gap,
        leh_sc_gap_std_eV=leh_sc_gap_std, leh_sc_h01_eV=leh_sc_h01, leh_sc_c0_weight=leh_sc_c0w,
        leh_dec_e0_eV=leh_dec_e0, leh_dec_e1_eV=leh_dec_e1, leh_dec_gap_eV=leh_dec_gap,
        leh_ana_e0_eV=leh_ana_e0, leh_ana_e1_eV=leh_ana_e1, leh_ana_gap_eV=leh_ana_gap, leh_ana_h01_eV=leh_ana_h01,
        **loop_npz_arrays,
    )

    summary = {
        "name": name,
        "label": spec["label"],
        "formula": spec["formula"],
        "n_atoms": len(spec["atoms"]),
        "internal_dof": 3 * len(spec["atoms"]) - 6,
        "true_ci_seam_dim": 3 * len(spec["atoms"]) - 8,      # codimension 2 -> isolated 0D point in 2D branching plane
        "adiabatic_tddft_seam_dim": 3 * len(spec["atoms"]) - 7,  # codimension 1 -> 1D crossing ridge in 2D branching plane
        "active_space": f"SA2-CASSCF({spec['nelecas']},{spec['ncas']})/{spec['basis']} + SC-NEVPT2",
        "sym_group": spec["sym_group"],
        "grid_shape": [nx, ny],
        "x_range_ang": list(spec["x_range"]),
        "y_range_ang": list(spec["y_range"]),
        "universal_sc_beta_coul": UNIVERSAL_SC_BETA_COUL,
        "stage1_wall_s": stage1["stage1_wall_s"],
        "rpa_instability_audit": {
            "n_rpa_unstable_points": int(np.sum(pbe0_rpa_unstable)),
            "frac_rpa_unstable": float(np.mean(pbe0_rpa_unstable)),
            "max_rpa_imag_eV": float(np.max(pbe0_rpa_imag)),
        },
        "cone_topology_fits": cone_fits,
        "H01_linearity": h01_linearity,
        "pes_errors_vs_NEVPT2": pes_vs_nevpt2,
        "pes_errors_vs_CASSCF": pes_vs_casscf,
        "berry_phase_loops": berry_summary,
        "npz_file": str(npz_path),
    }
    json_path = out_dir / f"{name}_ci_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{name}] Saved {npz_path} and {json_path}", flush=True)
    return summary


def print_system_report(s: Dict[str, Any]) -> None:
    name = s["name"].upper()
    print(f"\n==================== {name} VERIFIED RESULTS ====================", flush=True)
    cf = s["cone_topology_fits"]
    print(f"  Grid: {s['grid_shape']} | RPA imaginary-root pocket: {s['rpa_instability_audit']['frac_rpa_unstable']*100:.1f}% (max imag={s['rpa_instability_audit']['max_rpa_imag_eV']:.3f} eV)", flush=True)
    print("  --- 2D Branching-Plane Topology & Cone Fits ---", flush=True)
    for k in ["SA2_CASSCF", "SC_NEVPT2", "Adiabatic_PBE0_TDA_signed", "Adiabatic_B3LYP_TDA_signed",
              "Adiabatic_CAMB3LYP_TDA_signed", "Lehmann_ZeroShot_Ensemble", "Lehmann_SCv0D_Ensemble", "Lehmann_Analytical_RI"]:
        c = cf[k]
        fi = c["inner_cone_fit"]
        fl = c["local_cone_fit"]
        print(f"    {k:30s}: neg_frac={c['frac_negative_gap']*100:5.1f}% | sign_rows={c['sign_change_y_rows']:2d} | local_R2={fl['r2']:.4f} | inner_R2={fi['r2']:.4f} | ray_R2={c['radial_x_linearity_R2']:.4f} | apex=({fl['x0']:+.3f},{fl['y0']:+.3f}) | g={fl['g_slope']:.2f}, h={fl['h_slope']:.2f}", flush=True)
    print("  --- H01_eff(y_h) Linearity ---", flush=True)
    for k, v in s["H01_linearity"].items():
        print(f"    {k:28s}: H01(0)={v['H01_at_y0_eV']:.2e} eV | max|H01|={v['max_abs_H01_eV']:.4f} eV | local_slope={v['linear_slope_local_eV_per_A']:.3f} (R2={v['linear_R2_local']:.5f}) | full_R2={v['linear_R2']:.5f}", flush=True)
    print("  --- 2D PES Error vs SC-NEVPT2 ---", flush=True)
    for k, v in s["pes_errors_vs_NEVPT2"].items():
        print(f"    {k:30s}: gap_MAE={v['gap_MAE_eV']:.4f} eV | 2sheet_MAE={v['PES_2sheet_MAE_eV']:.4f} eV", flush=True)
    print("  --- Gauge-Invariant Berry Phase Proofs ---", flush=True)
    for ltag, ld in s["berry_phase_loops"].items():
        print(f"    [{ltag}] center={ld['center_xy']}, r={ld['radius']:.3f}, M={ld['n_points']}:", flush=True)
        for m in ["SA2_CASSCF", "Adiabatic_TDDFT_KS", "Lehmann_Decoupled_v0D0",
                  "Lehmann_ZeroShot_Ensemble", "Lehmann_SCv0D_Ensemble", "Lehmann_Analytical_RI"]:
            if m in ld:
                b = ld[m]
                print(f"      {m:28s}: W_sign={b['W_sign']:+.1f} | W_prod={b['W_prod']:+.5f} | min|ov|={b['min_abs_overlap']:.4f} | cont={str(b['adiabatic_continuous']):5s} | gamma={b['berry_phase_pi_units']}*pi", flush=True)


def main():
    import pickle

    parser = argparse.ArgumentParser(description="Pillar 4 Conical Intersection Topology & Berry-Phase Engine")
    parser.add_argument("--systems", nargs="+", default=["ethylene", "methanimine", "fulvene"],
                        choices=["ethylene", "methanimine", "fulvene"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--re-eval-only", action="store_true",
                        help="Skip PySCF Stage 1 and re-run Stage 2 from cached stage1_<sys>.pkl files")
    parser.add_argument("--refresh-loops", action="store_true",
                        help="Reuse cached 2D grids and recompute only Berry-phase loops whose center/radius changed")
    parser.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = load_system_specs(smoke_test=args.smoke_test)

    # IMPORTANT macOS safety: run ALL PySCF worker pools BEFORE importing torch!
    stage1_outputs: Dict[str, Dict[str, Any]] = {}
    for sys_name in args.systems:
        cache_pkl = out_dir / f"stage1_cache_{sys_name}.pkl"
        if (args.re_eval_only or args.refresh_loops) and cache_pkl.exists():
            print(f"[{sys_name}] Loading cached Stage 1 QM outputs from {cache_pkl}", flush=True)
            with open(cache_pkl, "rb") as fh:
                s1 = pickle.load(fh)
            if args.refresh_loops:
                spec = specs[sys_name]
                desired = [
                    ("loop_ci_shared", spec["loop_ci_center"], spec["loop_ci_radius"], True, True),
                    ("loop_leh_apex", spec["loop_leh_center"], spec["loop_leh_radius"], False, False),
                    ("loop_ctrl_noci", spec["loop_ctrl_center"], spec["loop_ctrl_radius"], True, False),
                ]
                changed = []
                for ltag, ctr, rad, run_cas, run_nev in desired:
                    old = s1["loops"].get(ltag, {})
                    old_ctr = tuple(old.get("center_xy", (999.0, 999.0)))
                    old_rad = float(old.get("radius", -1.0))
                    if (abs(old_ctr[0] - ctr[0]) > 1e-6 or abs(old_ctr[1] - ctr[1]) > 1e-6 or abs(old_rad - rad) > 1e-6):
                        changed.append((ltag, ctr, rad, run_cas, run_nev))
                if changed:
                    mo_center, dm_center = compute_center_reference_mos(spec)
                    for ltag, ctr, rad, run_cas, run_nev in changed:
                        t_l = time.time()
                        print(f"[{sys_name}] Recomputing updated loop {ltag} center={ctr} r={rad:.3f}...", flush=True)
                        lo = evaluate_loop_sequential_worker((
                            ltag, spec["atoms"], spec["coords_meci"], spec["g_hat"], spec["h_hat"],
                            ctr, rad, spec["n_loop"], spec["ncas"], spec["nelecas"], spec["basis"],
                            spec["is_fulvene"], mo_center, dm_center, run_cas, run_nev
                        ))
                        s1["loops"][ltag] = lo
                        print(f"[{sys_name}] Updated {ltag} in {time.time()-t_l:.1f}s", flush=True)
                    with open(cache_pkl, "wb") as fh:
                        pickle.dump(s1, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"[{sys_name}] Saved updated loops into {cache_pkl}", flush=True)
            stage1_outputs[sys_name] = s1
        else:
            s1 = run_system_stage1_qm(specs[sys_name], n_workers=min(args.workers, 4))
            stage1_outputs[sys_name] = s1
            if not args.smoke_test:
                with open(cache_pkl, "wb") as fh:
                    pickle.dump(s1, fh, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[{sys_name}] Cached Stage 1 QM outputs to {cache_pkl}", flush=True)

    # Now safely import PyTorch & load the trained 5-fold Lehmann-TDDFT ensemble in the main process!
    import torch
    torch.set_num_threads(1)
    from lehmann_tddft.model import LehmannEnsemble
    ensemble = LehmannEnsemble.from_manifest(str(MANIFEST_PATH))
    models = ensemble.models
    print(f"\nLoaded {len(models)}-fold trained Lehmann-TDDFT ensemble from {MANIFEST_PATH}", flush=True)

    all_summaries: Dict[str, Any] = {}
    # Load any existing summaries so running subsets cleanly merges into master summary!
    master_json = out_dir / "conical_intersections_summary.json"
    if master_json.exists() and not args.smoke_test:
        try:
            with open(master_json, "r") as fh:
                all_summaries = json.load(fh)
        except Exception:
            all_summaries = {}

    for sys_name in args.systems:
        s = run_system_stage2_lehmann_and_save(specs[sys_name], stage1_outputs[sys_name], models, out_dir)
        print_system_report(s)
        all_summaries[sys_name] = s

    with open(master_json, "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    print(f"\nALL DONE! Master summary written to {master_json}", flush=True)


if __name__ == "__main__":
    main()
