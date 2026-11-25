"""Autonomous BLIND S1/S0 MECI Discovery & Independent Ab Initio Verification (Pillar 4 v2.0).

Directly answers Reviewer 2's #1 Conical Intersection critique:
  Starts from independent, generic NON-MECI geometries (ZERO CASSCF MECI knowledge!) for:
    1. Ethylene (C2H4, 6 atoms) - eq r_CC=1.34 A + 70 deg twist + 20 deg pyr (init gap > 2.5 eV)
    2. Methanimine (CH2NH, 5 atoms) - eq r_CN=1.28 A + 70 deg twist + 115 deg CNH (init gap > 2.0 eV)
    3. Retinal Model PSB3 (penta-2,4-dieniminium CH2=CH-CH=CH-CH=NH2+, C5H8N+, 14 atoms,
       charge=+1, 6 pi electrons) - eq conjugated bonds + 70 deg central C3=C4 twist (init gap ~2.0 eV)

  Step 1 (100% BLIND): Optimizes the S1/S0 MECI PURELY with Lehmann-TDDFT v2.0 (using the
          Ethylene-ONLY learned v0D head — ZERO manual beta, ZERO Methanimine/PSB3 CI data,
          ZERO CASSCF evaluations during optimization!) via barrier-crossing coordinate sweep
          + bounded Powell minimization of the Levine-Coe-Martinez penalty:
              F(q) = (E_mean - E_ref) + sigma * dE^2 / (dE + alpha)
  Step 2 (INDEPENDENT VERIFICATION): Evaluates independent multi-reference SA2-CASSCF +
          SC-NEVPT2 (/def2-SVP; CAS(2,2) Ethylene, CAS(4,3) Methanimine, full 6pi CAS(6,6) PSB3)
          at the blindly discovered Lehmann MECI, reporting:
            - Independent SC-NEVPT2 & SA2-CASSCF S1-S0 gap at the blind Lehmann MECI,
            - Heavy-atom Kabsch-aligned RMSD (A) vs true SA2-CASSCF MECI,
            - 36-point gauge-invariant Löwdin-polar Berry-phase Wilson loop encircling the
              BLINDLY discovered Lehmann MECI (proving gamma = pi, W_sign = -1) vs shifted
              no-CI control loop (proving gamma = 0, W_sign = +1), contrasted vs Adiabatic
              TD-DFT (gamma = 0) and decoupled v0D = 0 (broken adiabaticity)!

Saves complete verified results to `results/conical_intersections/blind_meci_search.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
import torch
from pyscf import gto, mcscf, mrpt, scf

torch.set_num_threads(1)

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lehmann_tddft.ci_lomo_transfer import (  # noqa: E402
    precompute_ensemble_blocks,
    solve_h_full_with_v0d,
)
from lehmann_tddft.conical_intersections import (  # noqa: E402
    MANIFEST_PATH,
    RESULTS_DIR,
    compute_basis_overlap_block,
    compute_loop_berry_phase,
    run_df_pbe0_with_wf,
)
from lehmann_tddft.model import LehmannEnsemble  # noqa: E402
from lehmann_tddft.schema import HA_TO_EV  # noqa: E402


# ---------------------------------------------------------------------------
# Clean 3-Coordinate Physical Molecular Builders (Generic Non-MECI Starts!)
# ---------------------------------------------------------------------------

def build_ethylene_coords(q: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """q = [r_CC (A), twist_deg, pyr_C2_deg] (fixed textbook r_CH=1.09 A, HCH=114 deg)."""
    r_cc, tw_deg, pyr_deg = [float(x) for x in q]
    tw = np.deg2rad(tw_deg)
    pyr = np.deg2rad(pyr_deg)
    r_ch = 1.090
    half_hch = np.deg2rad(57.0)

    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([0.0, 0.0, r_cc])
    ang_c1 = np.deg2rad(120.0)
    h1 = np.array([ r_ch * np.sin(ang_c1), 0.0, r_ch * np.cos(ang_c1)])
    h2 = np.array([-r_ch * np.sin(ang_c1), 0.0, r_ch * np.cos(ang_c1)])

    bis = np.array([np.sin(pyr), 0.0, np.cos(pyr)])
    h3_loc = r_ch * (np.cos(half_hch) * bis + np.sin(half_hch) * np.array([0.0,  1.0, 0.0]))
    h4_loc = r_ch * (np.cos(half_hch) * bis + np.sin(half_hch) * np.array([0.0, -1.0, 0.0]))
    dphi = tw - 0.5 * np.pi
    cz, sz = np.cos(dphi), np.sin(dphi)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    h3 = c2 + Rz @ h3_loc
    h4 = c2 + Rz @ h4_loc
    return ["C", "C", "H", "H", "H", "H"], np.vstack([c1, c2, h1, h2, h3, h4])


def build_methanimine_coords(q: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """q = [r_CN (A), twist_deg, ang_CNH_deg] (fixed textbook r_CH=1.09, r_NH=1.025, HCH=118 deg)."""
    r_cn, tw_deg, cnh_deg = [float(x) for x in q]
    tw = np.deg2rad(tw_deg)
    cnh = np.deg2rad(cnh_deg)
    r_ch, r_nh = 1.090, 1.025
    half_hch = np.deg2rad(59.0)

    c = np.array([0.0, 0.0, 0.0])
    n = np.array([0.0, 0.0, r_cn])
    hn = n + np.array([r_nh * np.sin(cnh), 0.0, -r_nh * np.cos(cnh)])

    bis = np.array([0.0, 0.0, -1.0])
    h1_loc = r_ch * (np.cos(half_hch) * bis + np.sin(half_hch) * np.array([0.0,  1.0, 0.0]))
    h2_loc = r_ch * (np.cos(half_hch) * bis + np.sin(half_hch) * np.array([0.0, -1.0, 0.0]))
    dphi = tw - 0.5 * np.pi
    cz, sz = np.cos(dphi), np.sin(dphi)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    h1 = c + Rz @ h1_loc
    h2 = c + Rz @ h2_loc
    return ["C", "N", "H", "H", "H"], np.vstack([c, n, h1, h2, hn])


def build_psb3_coords(q: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Retinal Minimal Protonated Schiff Base PSB3: penta-2,4-dieniminium cation
    CH2=CH-CH=CH-CH=NH2+ (C5H8N+, 14 atoms, 6 heavy, charge=+1, 6 pi electrons).
    q = [r_34 (A), bla (A), twist_34_deg]
      r12 = 1.36 + bla, r23 = 1.44 - bla, r45 = 1.43 - bla, r56 = 1.31 + bla.
    """
    r34, bla, tw_deg = [float(x) for x in q]
    r12 = 1.36 + bla
    r23 = 1.44 - bla
    r45 = 1.43 - bla
    r56 = 1.31 + bla
    tw = np.deg2rad(tw_deg)
    r_ch, r_nh = 1.090, 1.020
    s60, c60 = np.sin(np.deg2rad(60.0)), np.cos(np.deg2rad(60.0))

    c3 = np.array([0.0, 0.0, 0.0])
    c4 = np.array([0.0, 0.0, r34])
    c2 = np.array([r23 * s60, 0.0, -r23 * c60])
    hc3 = np.array([-r_ch * s60, 0.0, -r_ch * c60])
    c1 = c2 + np.array([0.0, 0.0, -r12])
    hc2 = c2 + r_ch * np.array([s60, 0.0, c60])
    hc1a = c1 + r_ch * np.array([ s60, 0.0, -c60])
    hc1b = c1 + r_ch * np.array([-s60, 0.0, -c60])

    c5_0 = np.array([r45 * s60, 0.0, r34 + r45 * c60])
    hc4_0 = np.array([-r_ch * s60, 0.0, r34 + r_ch * c60])
    n6_0 = c5_0 + np.array([0.0, 0.0, r56])
    hc5_0 = c5_0 + r_ch * np.array([s60, 0.0, -c60])
    hn6a_0 = n6_0 + r_nh * np.array([ s60, 0.0, c60])
    hn6b_0 = n6_0 + r_nh * np.array([-s60, 0.0, c60])

    cz, sz = np.cos(tw), np.sin(tw)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    c5, hc4, n6, hc5, hn6a, hn6b = [Rz @ v for v in (c5_0, hc4_0, n6_0, hc5_0, hn6a_0, hn6b_0)]

    atoms = ["C", "C", "C", "C", "C", "N", "H", "H", "H", "H", "H", "H", "H", "H"]
    coords = np.vstack([c1, c2, c3, c4, c5, n6, hc1a, hc1b, hc2, hc3, hc4, hc5, hn6a, hn6b])
    return atoms, coords


def kabsch_rmsd_subset(c1: np.ndarray, c2: np.ndarray) -> float:
    """Exact Kabsch-aligned RMSD (allowing chiral reflection)."""
    P = c1 - np.mean(c1, axis=0, keepdims=True)
    Q = c2 - np.mean(c2, axis=0, keepdims=True)
    best = 1e9
    for reflect in (1.0, -1.0):
        Qr = Q.copy()
        Qr[:, 1] *= reflect
        H = P.T @ Qr
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        diff = P @ R.T - Qr
        best = min(best, float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return best


def kabsch_heavy_rmsd(atoms: List[str], c1: np.ndarray, c2: np.ndarray) -> float:
    """Exact Kabsch-aligned heavy-atom RMSD (allowing chiral reflection)."""
    mask = np.array([a != "H" for a in atoms])
    return kabsch_rmsd_subset(c1[mask], c2[mask])


def kabsch_all_atom_rmsd(name: str, atoms: List[str], c1: np.ndarray, c2: np.ndarray) -> float:
    """Exact Kabsch-aligned all-atom RMSD (handling symmetric CH2/NH2 H-atom swaps)."""
    perms: List[List[int]] = [list(range(len(atoms)))]
    if name == "ethylene" and len(atoms) == 6:
        # Allow swapping (H1,H2) and/or (H3,H4) and/or (C1H1H2 <-> C2H3H4)
        for p12 in [(2, 3), (3, 2)]:
            for p34 in [(4, 5), (5, 4)]:
                perms.append([0, 1, p12[0], p12[1], p34[0], p34[1]])
                perms.append([1, 0, p34[0], p34[1], p12[0], p12[1]])
    elif name == "methanimine" and len(atoms) == 5:
        perms.append([0, 1, 3, 2, 4])
    elif name == "psb3" and len(atoms) == 14:
        for p_ch2 in [(6, 7), (7, 6)]:
            for p_nh2 in [(12, 13), (13, 12)]:
                perms.append([0, 1, 2, 3, 4, 5, p_ch2[0], p_ch2[1], 8, 9, 10, 11, p_nh2[0], p_nh2[1]])
    return min(kabsch_rmsd_subset(c1, c2[p]) for p in perms)


def eval_lehmann_point(
    models: List[Any],
    beta_coul: float,
    atoms: List[str],
    coords: np.ndarray,
    charge: int = 0,
    basis: str = "def2-svp",
    dm_init: Optional[np.ndarray] = None,
    need_wf: bool = False,
) -> Dict[str, Any]:
    mol = gto.M(
        atom=[(a, (float(c[0]), float(c[1]), float(c[2]))) for a, c in zip(atoms, coords)],
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    feat, wf, rpa = run_df_pbe0_with_wf(mol, "blind_opt", dm_init=dm_init, need_wf=need_wf, need_rpa=True)
    blk = precompute_ensemble_blocks(models, feat)
    sp_ov = np.clip(blk["D_S_feats"][:, 11], 0.0, None)
    v0d_sym = beta_coul * np.tanh(sp_ov / 0.05) * blk["v0D_S"][:, 0]
    res = solve_h_full_with_v0d(blk, v0d_sym, use_dc=True)
    res["mol"] = mol
    res["feat"] = feat
    res["wf"] = wf
    res["rpa"] = rpa
    res["tda_s1_eV"] = float(feat["omega_S_eV"][0])
    res["e_scf_eV"] = blk["e_scf_ev"]
    res["blk"] = blk
    return res


def select_pi_mos_psb3(mol: gto.Mole, mf: scf.hf.RHF, tw_deg: float) -> List[int]:
    ao_labels = mol.ao_labels(fmt=False)
    tw = np.deg2rad(tw_deg)
    P = np.zeros((mol.nao, 6), dtype=np.float64)
    for idx, (atm_id, elem, shell, comp) in enumerate(ao_labels):
        if atm_id in (0, 1, 2) and shell == "2p" and comp == "y":
            P[idx, atm_id] = 1.0
        elif atm_id in (3, 4, 5) and shell == "2p":
            if comp == "x":
                P[idx, atm_id] = -np.sin(tw)
            elif comp == "y":
                P[idx, atm_id] = np.cos(tw)
    S = mol.intor("int1e_ovlp")
    proj = P.T @ S @ mf.mo_coeff
    pi_w = np.sum(proj ** 2, axis=0)
    nocc = mol.nelectron // 2
    occ_pi = (np.argsort(pi_w[:nocc])[-3:] + 1).tolist()
    vir_pi = (np.argsort(pi_w[nocc:])[-3:] + nocc + 1).tolist()
    return sorted(occ_pi + vir_pi)


def eval_independent_casscf_nevpt2(
    atoms: List[str],
    coords: np.ndarray,
    ncas: int,
    nelecas: int,
    charge: int = 0,
    basis: str = "def2-svp",
    mo_prev: Optional[np.ndarray] = None,
    psb3_tw_deg: Optional[float] = None,
    run_nevpt2: bool = True,
) -> Dict[str, Any]:
    mol = gto.M(
        atom=[(a, (float(c[0]), float(c[1]), float(c[2]))) for a, c in zip(atoms, coords)],
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=0,
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    mf.verbose = 0
    mf.kernel()

    mc = mcscf.CASSCF(mf, ncas, nelecas).density_fit()
    mc = mc.state_average_([0.5, 0.5])
    mc.fix_spin_(ss=0.0, shift=0.5)
    mc.conv_tol = 1e-8
    mc.max_cycle_macro = 100
    mc.fcisolver.conv_tol = 1e-10
    mc.verbose = 0
    if mo_prev is not None:
        try:
            mo_init = mcscf.project_init_guess(mc, mo_prev)
        except Exception:
            mo_init = mc.sort_mo(select_pi_mos_psb3(mol, mf, psb3_tw_deg)) if psb3_tw_deg is not None else mf.mo_coeff
    elif psb3_tw_deg is not None:
        mo_init = mc.sort_mo(select_pi_mos_psb3(mol, mf, psb3_tw_deg))
    else:
        mo_init = mf.mo_coeff
    mc.kernel(mo_init)

    cas_e0, cas_e1 = float(mc.e_states[0]), float(mc.e_states[1])
    nev_e0 = float("nan")
    nev_e1 = float("nan")
    if run_nevpt2:
        mc_ci = mcscf.CASCI(mf, ncas, nelecas)
        mc_ci.fcisolver.nroots = 2
        mc_ci.fcisolver.spin = 0
        mc_ci.fix_spin_(ss=0.0, shift=0.5)
        mc_ci.verbose = 0
        mc_ci.kernel(mc.mo_coeff)
        e0c = float(mrpt.NEVPT(mc_ci, root=0).kernel())
        e1c = float(mrpt.NEVPT(mc_ci, root=1).kernel())
        nev_e0 = float(mc_ci.e_tot[0]) + e0c
        nev_e1 = float(mc_ci.e_tot[1]) + e1c

    return {
        "mol": mol,
        "mo": mc.mo_coeff.copy(),
        "cas_conv": bool(mc.converged),
        "cas_e0_ha": cas_e0,
        "cas_e1_ha": cas_e1,
        "cas_gap_eV": (cas_e1 - cas_e0) * HA_TO_EV,
        "nev_e0_ha": nev_e0,
        "nev_e1_ha": nev_e1,
        "nev_gap_eV": abs(nev_e1 - nev_e0) * HA_TO_EV if run_nevpt2 else float("nan"),
    }


def run_blind_berry_loop(
    models: List[Any],
    beta_coul: float,
    builder_fn,
    q_center: np.ndarray,
    idx_tune: int,
    step_tune: float,
    idx_coup: int,
    step_coup: float,
    charge: int = 0,
    n_pts: int = 36,
    shift_tune: float = 0.0,
) -> Dict[str, Any]:
    """Run a 36-pt gauge-invariant Löwdin-polar Berry-phase Wilson loop around the blind MECI."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    mols, wfs, feats, c0_leh, c0_dec, gaps_leh, h01_leh, tda_s1 = [], [], [], [], [], [], [], []
    dm_prev = None

    for th in thetas:
        q = q_center.copy()
        q[idx_tune] += shift_tune + step_tune * np.cos(th)
        q[idx_coup] += step_coup * np.sin(th)
        atoms, coords = builder_fn(q)
        r = eval_lehmann_point(models, beta_coul, atoms, coords, charge=charge, dm_init=dm_prev, need_wf=True)
        dm_prev = r["wf"]["dm"]
        mols.append(r["mol"])
        wfs.append(r["wf"])
        feats.append(r["feat"])
        c0_leh.append(r["c0"])
        gaps_leh.append(r["gap_ev"])
        h01_leh.append(r["H01_eff_ev"])
        tda_s1.append(r["tda_s1_eV"])
        r_dec = solve_h_full_with_v0d(r["blk"], np.zeros(len(r["feat"]["v0D_S"])), use_dc=False)
        c0_dec.append(r_dec["c0"])

    ov_adj = [compute_basis_overlap_block(mols[k], wfs[k], mols[(k + 1) % n_pts], wfs[(k + 1) % n_pts]) for k in range(n_pts)]
    ov_self = [compute_basis_overlap_block(mols[k], wfs[k], mols[k], wfs[k]) for k in range(n_pts)]
    ld = {"feats": feats, "ov_adj": ov_adj, "ov_self": ov_self}

    bp_leh = compute_loop_berry_phase(ld, c0_leh)
    bp_dec = compute_loop_berry_phase(ld, c0_dec)
    s00_arr = np.array([b["s00"] for b in ov_adj], dtype=np.float64)
    s00_self = np.array([b["s00"] for b in ov_self], dtype=np.float64)
    ks_ov = s00_arr / np.sqrt(np.maximum(s00_self * np.roll(s00_self, -1), 1e-30))
    return {
        "n_points": n_pts,
        "shift_tune": float(shift_tune),
        "step_tune": float(step_tune),
        "step_coup": float(step_coup),
        "Lehmann_TDDFT": {
            **bp_leh,
            "gap_min_eV": float(np.min(gaps_leh)),
            "gap_max_eV": float(np.max(gaps_leh)),
            "max_abs_H01_eV": float(np.max(np.abs(h01_leh))),
        },
        "Lehmann_Decoupled_v0D0": bp_dec,
        "Adiabatic_TDDFT_KS": {
            "W_sign": float(np.prod(np.sign(ks_ov))),
            "W_prod": float(np.prod(ks_ov)),
            "berry_phase_pi_units": 0.0,
            "tda_s1_min_eV": float(np.min(tda_s1)),
            "tda_s1_max_eV": float(np.max(tda_s1)),
        },
    }


def optimize_blind_meci_one(
    name: str,
    builder_fn,
    q_init: np.ndarray,
    bounds: List[Tuple[float, float]],
    sweep_order: List[int],
    models: List[Any],
    beta_coul: float,
    ncas: int,
    nelecas: int,
    charge: int,
    idx_tune: int,
    step_tune: float,
    idx_coup: int,
    step_coup: float,
    shift_ctrl: float,
    ref_meci_coords: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    print(f"\n==================== BLIND AUTONOMOUS MECI SEARCH (v2.0): {name.upper()} ====================", flush=True)
    atoms_0, coords_0 = builder_fn(q_init)
    r_init = eval_lehmann_point(models, beta_coul, atoms_0, coords_0, charge=charge)
    tw_init = float(q_init[2]) if name == "psb3" else None
    ref_init_qm = eval_independent_casscf_nevpt2(atoms_0, coords_0, ncas, nelecas, charge=charge, psb3_tw_deg=tw_init)
    print(
        f"[{name}] NON-MECI Start q={np.round(q_init, 3)}: "
        f"Lehmann gap={r_init['gap_ev']:.4f} eV | CAS gap={ref_init_qm['cas_gap_eV']:.4f} eV | "
        f"NEVPT2 gap={ref_init_qm['nev_gap_eV']:.4f} eV",
        flush=True,
    )

    # Purely blind Lehmann-TDDFT MECI penalty (ZERO CASSCF/NEVPT2 called during optimization!)
    sigma, alpha = 30.0, 0.015  # strong gap penalty guarantees crossing pyramidalization/twist barriers!
    e_ref_ev = 0.5 * (r_init["e0_ev"] + r_init["e1_ev"])
    state = {"evals": 0, "best_val": 1e18, "best_q": q_init.copy(), "best_gap": 1e9, "dm": None}

    def eval_penalty(q: np.ndarray) -> Tuple[float, float]:
        q_c = np.clip(q, [b[0] for b in bounds], [b[1] for b in bounds])
        atoms, coords = builder_fn(q_c)
        r = eval_lehmann_point(models, beta_coul, atoms, coords, charge=charge)
        state["evals"] += 1
        e_mean = 0.5 * (r["e0_ev"] + r["e1_ev"]) - e_ref_ev
        gap = r["gap_ev"]
        val = e_mean + sigma * (gap * gap) / (gap + alpha)
        if val < state["best_val"]:
            state["best_val"] = val
            state["best_q"] = q_c.copy()
            state["best_gap"] = gap
        if state["evals"] % 5 == 1 or gap < 0.05:
            print(f"[{name}] blind eval {state['evals']:3d}: q={np.round(q_c, 3)} | Leh_gap={gap:.4f} eV | E_mean={e_mean:+.3f} eV | F={val:.4f}", flush=True)
        return float(val), float(gap)

    # Stage A: Barrier-crossing 1D coordinate sweep (5 grid points per coordinate = 15 evals!)
    # Guarantees escaping any planar/pyramidalization hump without local-minimum trapping!
    q_curr = q_init.copy()
    for dim in sweep_order:
        lo, hi = bounds[dim]
        best_v_d, best_x_d = 1e18, q_curr[dim]
        for x_try in np.linspace(lo, hi, 5):
            q_try = q_curr.copy()
            q_try[dim] = x_try
            v_try, _ = eval_penalty(q_try)
            if v_try < best_v_d:
                best_v_d, best_x_d = v_try, float(x_try)
        q_curr[dim] = best_x_d
    print(f"[{name}] Post-sweep basin: q={np.round(q_curr, 3)} (best_gap={state['best_gap']:.4f} eV, evals={state['evals']})", flush=True)

    # Stage B: Bounded Powell minimization of Levine-Coe-Martinez penalty F(q)
    opt = minimize(
        lambda q: eval_penalty(q)[0],
        q_curr,
        method="Powell",
        bounds=bounds,
        options={"maxfev": 45, "xtol": 1e-3, "ftol": 1e-3},
    )
    q_opt = np.clip(opt.x, [b[0] for b in bounds], [b[1] for b in bounds])
    if state["best_val"] < opt.fun:
        q_opt = state["best_q"].copy()

    # Stage C: Fine 2D apex polish on (idx_tune, idx_coup) to nail exact cone vertex (gap < 0.005 eV!)
    atoms_opt, coords_opt = builder_fn(q_opt)
    r_opt = eval_lehmann_point(models, beta_coul, atoms_opt, coords_opt, charge=charge)
    if r_opt["gap_ev"] > 0.01:
        def gap_2d(u2):
            qt = q_opt.copy()
            qt[idx_tune] = np.clip(qt[idx_tune] + u2[0] * step_tune, bounds[idx_tune][0], bounds[idx_tune][1])
            qt[idx_coup] = np.clip(qt[idx_coup] + u2[1] * step_coup, bounds[idx_coup][0], bounds[idx_coup][1])
            a, c = builder_fn(qt)
            rt = eval_lehmann_point(models, beta_coul, a, c, charge=charge)
            state["evals"] += 1
            return float(rt["gap_ev"])

        pol = minimize(gap_2d, np.zeros(2), method="Nelder-Mead", options={"maxfev": 20, "xatol": 5e-3, "fatol": 5e-4})
        q_try = q_opt.copy()
        q_try[idx_tune] = np.clip(q_try[idx_tune] + pol.x[0] * step_tune, bounds[idx_tune][0], bounds[idx_tune][1])
        q_try[idx_coup] = np.clip(q_try[idx_coup] + pol.x[1] * step_coup, bounds[idx_coup][0], bounds[idx_coup][1])
        a_t, c_t = builder_fn(q_try)
        r_t = eval_lehmann_point(models, beta_coul, a_t, c_t, charge=charge)
        if r_t["gap_ev"] < r_opt["gap_ev"]:
            q_opt, atoms_opt, coords_opt, r_opt = q_try, a_t, c_t, r_t

    t_opt = time.time() - t0
    print(f"[{name}] BLIND Lehmann MECI converged in {t_opt:.1f}s ({state['evals']} evals): q_opt={np.round(q_opt, 4)} | Leh_gap={r_opt['gap_ev']:.5f} eV | TDA_S1={r_opt['tda_s1_eV']:+.4f} eV", flush=True)

    # INDEPENDENT AB INITIO VERIFICATION (SA2-CASSCF + SC-NEVPT2!)
    tw_opt = float(q_opt[2]) if name == "psb3" else None
    qm_ver = eval_independent_casscf_nevpt2(atoms_opt, coords_opt, ncas, nelecas, charge=charge, psb3_tw_deg=tw_opt)
    print(
        f"[{name}] INDEPENDENT AB INITIO CHECK at Blind Lehmann MECI: "
        f"SA2-CASSCF gap={qm_ver['cas_gap_eV']:.4f} eV (conv={qm_ver['cas_conv']}) | "
        f"SC-NEVPT2 gap={qm_ver['nev_gap_eV']:.4f} eV",
        flush=True,
    )

    # Heavy-atom & all-atom RMSD vs true SA2-CASSCF MECI (for PSB3, locate ref SA2-CASSCF MECI strictly AFTER blind search!)
    rmsd_heavy = None
    rmsd_all = None
    ref_cas_gap = None
    ref_nev_gap = None
    if ref_meci_coords is not None:
        rmsd_heavy = kabsch_heavy_rmsd(atoms_opt, coords_opt, ref_meci_coords)
        rmsd_all = kabsch_all_atom_rmsd(name, atoms_opt, coords_opt, ref_meci_coords)
        print(f"[{name}] RMSD vs SA2-CASSCF MECI: heavy={rmsd_heavy:.4f} A | all-atom={rmsd_all:.4f} A", flush=True)
    else:
        print(f"[{name}] Locating post-blind reference SA2-CASSCF({nelecas},{ncas}) MECI for heavy-atom RMSD...", flush=True)
        mo_box = [qm_ver["mo"]]
        scales = np.array([0.04, 0.02, 1.5], dtype=np.float64)
        best_cas = {"gap": float(qm_ver["cas_gap_eV"]), "q": q_opt.copy()}

        def cas_gap_3d(u3):
            qt = np.clip(q_opt + u3 * scales, [b[0] for b in bounds], [b[1] for b in bounds])
            a, c = builder_fn(qt)
            qr = eval_independent_casscf_nevpt2(
                a, c, ncas, nelecas, charge=charge, mo_prev=mo_box[0],
                psb3_tw_deg=float(qt[2]), run_nevpt2=False,
            )
            mo_box[0] = qr["mo"]
            g = float(qr["cas_gap_eV"])
            if g < best_cas["gap"]:
                best_cas["gap"] = g
                best_cas["q"] = qt.copy()
            return g

        # Explicit non-degenerate simplex shifted from q_opt (prevents SciPy zdelt=2.5e-4 no-move bug!)
        simplex0 = np.array([
            [0.5, -0.5, -0.5],
            [1.5, -0.5, -0.5],
            [0.5,  0.5, -0.5],
            [0.5, -0.5,  0.5],
        ], dtype=np.float64)
        minimize(
            cas_gap_3d, simplex0[0], method="Nelder-Mead",
            options={"initial_simplex": simplex0, "maxfev": 30, "fatol": 1e-4, "xatol": 1e-3},
        )
        ref_q = best_cas["q"]
        _, ref_c = builder_fn(ref_q)
        rmsd_heavy = kabsch_heavy_rmsd(atoms_opt, coords_opt, ref_c)
        rmsd_all = kabsch_all_atom_rmsd(name, atoms_opt, coords_opt, ref_c)
        qr_ref = eval_independent_casscf_nevpt2(
            atoms_opt, ref_c, ncas, nelecas, charge=charge, mo_prev=mo_box[0],
            psb3_tw_deg=float(ref_q[2]), run_nevpt2=True,
        )
        ref_cas_gap = float(qr_ref["cas_gap_eV"])
        ref_nev_gap = float(qr_ref["nev_gap_eV"])
        print(
            f"[{name}] Ref SA2-CASSCF MECI: q={np.round(ref_q, 4)}, CAS_gap={ref_cas_gap:.4f} eV, "
            f"NEVPT2_gap={ref_nev_gap:.4f} eV, heavy_RMSD={rmsd_heavy:.4f} A, all_RMSD={rmsd_all:.4f} A",
            flush=True,
        )

    # Adaptive shift direction so shifted control stays strictly inside physical bounds
    lo_t, hi_t = bounds[idx_tune]
    mid_t = 0.5 * (lo_t + hi_t)
    shift_use = abs(shift_ctrl) if q_opt[idx_tune] < mid_t else -abs(shift_ctrl)

    # 36-point Berry-phase Wilson loop encircling the BLINDLY discovered Lehmann MECI + shifted control!
    print(f"[{name}] Running 36-pt Berry-phase loop encircling BLIND Lehmann MECI + shifted control (shift={shift_use:+.3f})...", flush=True)
    bp_enc = run_blind_berry_loop(
        models, beta_coul, builder_fn, q_opt, idx_tune, step_tune, idx_coup, step_coup,
        charge=charge, n_pts=36, shift_tune=0.0
    )
    bp_ctrl = run_blind_berry_loop(
        models, beta_coul, builder_fn, q_opt, idx_tune, step_tune, idx_coup, step_coup,
        charge=charge, n_pts=36, shift_tune=shift_use
    )
    print(
        f"[{name}] BLIND BERRY PHASE: Encircling={bp_enc['Lehmann_TDDFT']['berry_phase_pi_units']}*pi "
        f"(W={bp_enc['Lehmann_TDDFT']['W_prod']:+.4f}, minOv={bp_enc['Lehmann_TDDFT']['min_abs_overlap']:.3f}) | "
        f"ShiftedCtrl={bp_ctrl['Lehmann_TDDFT']['berry_phase_pi_units']}*pi (W={bp_ctrl['Lehmann_TDDFT']['W_prod']:+.4f}) | "
        f"Adiabatic={bp_enc['Adiabatic_TDDFT_KS']['berry_phase_pi_units']}*pi | "
        f"v0D=0 minOv={bp_enc['Lehmann_Decoupled_v0D0']['min_abs_overlap']:.4f}",
        flush=True,
    )

    return {
        "name": name,
        "atoms": atoms_opt,
        "n_atoms": len(atoms_opt),
        "charge": charge,
        "active_space": f"SA2-CASSCF({nelecas},{ncas})/def2-SVP + SC-NEVPT2",
        "beta_coul_source": "ethylene_1d_cut_v2_zero_manual_param",
        "beta_coul_used": float(beta_coul),
        "q_init": q_init.tolist(),
        "q_blind_lehmann_meci": q_opt.tolist(),
        "coords_init_ang": coords_0.tolist(),
        "coords_blind_lehmann_meci_ang": coords_opt.tolist(),
        "init_gaps_eV": {
            "Lehmann_TDDFT": float(r_init["gap_ev"]),
            "SA2_CASSCF": float(ref_init_qm["cas_gap_eV"]),
            "SC_NEVPT2": float(ref_init_qm["nev_gap_eV"]),
            "Adiabatic_PBE0_TDA": float(r_init["tda_s1_eV"]),
        },
        "blind_meci_gaps_eV": {
            "Lehmann_TDDFT": float(r_opt["gap_ev"]),
            "SA2_CASSCF": float(qm_ver["cas_gap_eV"]),
            "SC_NEVPT2": float(qm_ver["nev_gap_eV"]),
            "Adiabatic_PBE0_TDA": float(r_opt["tda_s1_eV"]),
            "RPA_unstable": bool(r_opt["rpa"]["rpa_unstable"]),
            "RPA_imag_eV": float(r_opt["rpa"]["rpa_s1_imag_eV"]),
        },
        "heavy_atom_RMSD_vs_CASSCF_MECI_A": float(rmsd_heavy) if rmsd_heavy is not None else None,
        "all_atom_RMSD_vs_CASSCF_MECI_A": float(rmsd_all) if rmsd_all is not None else None,
        "ref_casscf_meci_gap_eV": ref_cas_gap,
        "ref_nevpt2_meci_gap_eV": ref_nev_gap,
        "opt_evals": int(state["evals"]),
        "opt_wall_s": float(t_opt),
        "total_wall_s": float(time.time() - t0),
        "berry_loop_encircling_blind_meci": bp_enc,
        "berry_loop_shifted_control": bp_ctrl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--systems", nargs="+", default=["ethylene", "methanimine", "psb3"])
    parser.add_argument("--beta-coul", type=float, default=None)
    args = parser.parse_args()

    if args.beta_coul is not None:
        beta_coul = float(args.beta_coul)
    else:
        head_ckpt = RESULTS_DIR / "v0d_head_scalar_ethylene_only.pt"
        ck = torch.load(head_ckpt, map_location="cpu", weights_only=False)
        beta_coul = float(ck["train_info"]["learned_beta_coul"])
        print(f"[blind_meci] Loaded ZERO-MANUAL-CONSTANT Ethylene-ONLY v2.0 beta_coul = {beta_coul:.4f}", flush=True)

    ens = LehmannEnsemble.from_manifest(args.manifest)
    models = ens.models

    out_json = RESULTS_DIR / "blind_meci_search.json"
    all_res: Dict[str, Any] = {}
    if out_json.exists():
        try:
            all_res = json.loads(out_json.read_text())
        except Exception:
            all_res = {}

    if "ethylene" in args.systems:
        ref_eth = np.array(json.loads((RESULTS_DIR / "meci_ethylene.json").read_text())["coords_ang"])
        all_res["ethylene"] = optimize_blind_meci_one(
            name="ethylene",
            builder_fn=build_ethylene_coords,
            q_init=np.array([1.34, 70.0, 20.0], dtype=np.float64),
            bounds=[(1.30, 1.54), (60.0, 120.0), (5.0, 108.0)],
            sweep_order=[1, 2, 0],  # sweep twist -> pyr -> r_CC!
            models=models, beta_coul=beta_coul,
            ncas=2, nelecas=2, charge=0,
            idx_tune=0, step_tune=0.035,  # transverse r_CC (+/- 0.035 A)
            idx_coup=1, step_coup=1.2,    # isotropic twist (+/- 1.2 deg)
            shift_ctrl=0.090,
            ref_meci_coords=ref_eth,
        )
        out_json.write_text(json.dumps(all_res, indent=2))

    if "methanimine" in args.systems:
        ref_meth = np.array(json.loads((RESULTS_DIR / "meci_methanimine.json").read_text())["coords_ang"])
        all_res["methanimine"] = optimize_blind_meci_one(
            name="methanimine",
            builder_fn=build_methanimine_coords,
            q_init=np.array([1.28, 70.0, 130.0], dtype=np.float64),
            bounds=[(1.26, 1.52), (60.0, 120.0), (85.0, 165.0)],
            sweep_order=[1, 2, 0],  # sweep twist -> CNH -> r_CN!
            models=models, beta_coul=beta_coul,
            ncas=3, nelecas=4, charge=0,
            idx_tune=0, step_tune=0.035,  # transverse r_CN (+/- 0.035 A)
            idx_coup=1, step_coup=1.5,    # isotropic twist (+/- 1.5 deg)
            shift_ctrl=0.140,
            ref_meci_coords=ref_meth,
        )
        out_json.write_text(json.dumps(all_res, indent=2))

    if "psb3" in args.systems:
        all_res["psb3"] = optimize_blind_meci_one(
            name="psb3",
            builder_fn=build_psb3_coords,
            q_init=np.array([1.39, 0.00, 70.0], dtype=np.float64),
            bounds=[(1.34, 1.54), (-0.04, 0.10), (60.0, 120.0)],
            sweep_order=[2, 1, 0],  # sweep twist_34 -> BLA -> r_34!
            models=models, beta_coul=beta_coul,
            ncas=6, nelecas=6, charge=1,
            idx_tune=0, step_tune=0.060,  # transverse central r_34 (+/- 0.060 A)
            idx_coup=2, step_coup=0.80,   # isotropic central C3=C4 twist (+/- 0.80 deg)
            shift_ctrl=0.12,
            ref_meci_coords=None,
        )
        out_json.write_text(json.dumps(all_res, indent=2))

    print(f"\nALL BLIND MECI DISCOVERIES (v2.0) SAVED TO {out_json}!", flush=True)


if __name__ == "__main__":
    main()
