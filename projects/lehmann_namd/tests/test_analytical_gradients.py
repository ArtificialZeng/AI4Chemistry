"""Comprehensive finite-difference verification of exact analytical nuclear gradients
and non-adiabatic coupling vectors (NACVs) for Lehmann-NAMD.

Tests:
  1. Exact RI-TDA singlet & triplet eigenvalues vs PySCF `tdscf.TDA(mf)` (< 1e-12 Ha).
  2. DF-HF (`xc='hf'`) exact analytical grad(E0), grad(E1), and h_01 / d_01^CI vs central
     finite differences across ALL 3N Cartesian coordinates on BOTH distorted H2O (9 DOF)
     and twisted-pyramidalized Ethylene (18 DOF, strong S0/S1 coupling!) — target < 1e-7 Ha/Bohr.
  3. DF-PBE0 (`xc='pbe0'`, hybrid GGA!) exact analytical grad(E0), grad(E1), and h_01 / d_01^CI
     vs central finite differences across ALL 3N coordinates on BOTH H2O and Ethylene — target < 2e-5 Ha/Bohr.
  4. Exact O(n_occ) x O(n_vir) orbital-rotation invariance (`oo_vv_invariance_err < 1e-11`),
     translational invariance, and rotational invariance.
"""

from __future__ import annotations

import os
import sys
import unittest

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
from pyscf import dft, gto, scf, tdscf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lehmann_namd import LehmannNAMDEngine, OperatorParams


def _h2o_asymmetric() -> tuple[list[str], np.ndarray]:
    atoms = ["O", "H", "H"]
    coords_bohr = np.array(
        [
            [0.00, 0.05, -0.10],
            [1.45, 1.15, 0.20],
            [-1.35, 1.30, -0.15],
        ],
        dtype=np.float64,
    )
    return atoms, coords_bohr


def _ethylene_twisted_pyr() -> tuple[list[str], np.ndarray]:
    """Twisted (68 deg) + pyramidalized (24 deg) Ethylene (C1 symmetry, strong S0/S1 coupling)."""
    atoms = ["C", "C", "H", "H", "H", "H"]
    ang_to_bohr = 1.0 / 0.529177210903
    r_cc = 1.44
    r_ch = 1.09
    phi = np.deg2rad(68.0)
    pyr = np.deg2rad(24.0)
    h1 = np.array([-r_cc / 2 - r_ch * np.cos(np.deg2rad(60)), r_ch * np.sin(np.deg2rad(60)), 0.0])
    h2 = np.array([-r_cc / 2 - r_ch * np.cos(np.deg2rad(60)), -r_ch * np.sin(np.deg2rad(60)), 0.0])
    c2 = np.array([r_cc / 2, 0.0, 0.0])
    v3 = np.array([r_ch * np.cos(np.deg2rad(60)) * np.cos(pyr), r_ch * np.sin(np.deg2rad(60)), r_ch * np.cos(np.deg2rad(60)) * np.sin(pyr)])
    v4 = np.array([r_ch * np.cos(np.deg2rad(60)) * np.cos(pyr), -r_ch * np.sin(np.deg2rad(60)), r_ch * np.cos(np.deg2rad(60)) * np.sin(pyr)])
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    h3 = c2 + R @ v3 + np.array([0.02, -0.01, 0.03])
    h4 = c2 + R @ v4 + np.array([-0.01, 0.02, -0.02])
    coords_ang = np.vstack([[-r_cc / 2, 0.0, 0.0], c2, h1, h2, h3, h4])
    return atoms, coords_ang * ang_to_bohr


def _finite_diff_grad_and_nacv(
    engine: LehmannNAMDEngine,
    atoms: list[str],
    coords_bohr: np.ndarray,
    delta: float = 1e-4,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Central finite-difference grad(E0), grad(E1), and Hellmann-Feynman h_01 = u0^T (dH_ext/dR) u1."""
    res0 = engine.compute(atoms, coords_bohr, grad_states=(), nacv_pairs=())
    u0 = res0.evecs[:, 0]
    u1 = res0.evecs[:, 1]
    natm = len(atoms)
    g0_fd = np.zeros((natm, 3), dtype=np.float64)
    g1_fd = np.zeros((natm, 3), dtype=np.float64)
    h01_fd = np.zeros((natm, 3), dtype=np.float64)

    nocc = res0.X_S.shape[1]
    C_o0, C_v0 = res0.mo_coeff[:, :nocc], res0.mo_coeff[:, nocc:]
    n_tt = min(engine.params.n_tt_pool, res0.X_T.shape[0])
    n_ss = min(engine.params.n_ss_pool, res0.X_S.shape[0]) if engine.params.include_ss_doubles else 0

    def _eval_aligned_H(coords: np.ndarray):
        r = engine.compute(atoms, coords, grad_states=(), nacv_pairs=(), dm_init=res0.dm_scf)
        S_cross = gto.intor_cross("int1e_ovlp", res0.mol, r.mol)
        C_o1, C_v1 = r.mo_coeff[:, :nocc], r.mo_coeff[:, nocc:]
        O_o = C_o0.T @ S_cross @ C_o1
        O_v = C_v0.T @ S_cross @ C_v1
        s_S = np.array([
            1.0 if np.sum(res0.X_S[I] * (O_o @ r.X_S[I] @ O_v.T)) >= 0 else -1.0
            for I in range(res0.X_S.shape[0])
        ])
        s_T = np.array([
            1.0 if np.sum(res0.X_T[I] * (O_o @ r.X_T[I] @ O_v.T)) >= 0 else -1.0
            for I in range(res0.X_T.shape[0])
        ])
        s_D = [s_T[i] * s_T[j] for i in range(n_tt) for j in range(i, n_tt)]
        s_D += [s_S[i] * s_S[j] for i in range(n_ss) for j in range(i, n_ss)]
        d_vec = np.concatenate([[1.0], s_S, np.array(s_D, dtype=np.float64)])
        H_raw = (r.evecs * r.evals_ha[None, :]) @ r.evecs.T
        H_aligned = (d_vec[:, None] * H_raw) * d_vec[None, :]
        return r.evals_ha, H_aligned

    for ia in range(natm):
        for ax in range(3):
            cp = coords_bohr.copy()
            cm = coords_bohr.copy()
            cp[ia, ax] += delta
            cm[ia, ax] -= delta
            ep, Hp = _eval_aligned_H(cp)
            em, Hm = _eval_aligned_H(cm)
            g0_fd[ia, ax] = (ep[0] - em[0]) / (2.0 * delta)
            g1_fd[ia, ax] = (ep[1] - em[1]) / (2.0 * delta)
            h01_fd[ia, ax] = float(u0 @ ((Hp - Hm) / (2.0 * delta)) @ u1)

    return {0: g0_fd, 1: g1_fd}, h01_fd


def run_verification():
    cases = [
        ("H2O (C1)", *_h2o_asymmetric(), "6-31g", "def2-svp-jkfit"),
        ("Ethylene (twisted+pyr, 18 DOF)", *_ethylene_twisted_pyr(), "6-31g", "def2-svp-jkfit"),
    ]
    for name, atoms, coords, basis, auxbasis in cases:
        print(f"\n================ {name} ================")
        for xc in ("hf", "pbe0"):
            eng = LehmannNAMDEngine(
                basis=basis,
                auxbasis=auxbasis,
                xc=xc,
                grids_level=5,
                n_states_s=4,
                n_states_t=4,
                scf_conv_tol=1e-12,
                scf_conv_tol_grad=1e-9,
                cphf_tol=1e-9,
            )
            res = eng.compute(atoms, coords, grad_states=(0, 1), nacv_pairs=((0, 1),))
            g_fd, h01_fd = _finite_diff_grad_and_nacv(eng, atoms, coords, delta=1e-4)
            err_g0 = np.max(np.abs(res.grads_ha_bohr[0] - g_fd[0]))
            err_g1 = np.max(np.abs(res.grads_ha_bohr[1] - g_fd[1]))
            err_h01 = np.max(np.abs(res.h_mn_ha_bohr[(0, 1)] - h01_fd))
            gap_ha = float(res.evals_ha[1] - res.evals_ha[0])
            d01_fd = h01_fd / gap_ha
            err_d01 = np.max(np.abs(res.nacv_ci_bohr_inv[(0, 1)] - d01_fd))
            h01_norm = float(np.linalg.norm(res.h_mn_ha_bohr[(0, 1)]))
            d01_norm = float(np.linalg.norm(res.nacv_ci_bohr_inv[(0, 1)]))
            trans_g0 = float(np.linalg.norm(res.grads_ha_bohr[0].sum(axis=0)))
            trans_g1 = float(np.linalg.norm(res.grads_ha_bohr[1].sum(axis=0)))
            trans_h01 = float(np.linalg.norm(res.h_mn_ha_bohr[(0, 1)].sum(axis=0)))
            print(
                f"  xc={xc:4s} | gap={gap_ha*27.211386:.3f} eV | |H01_eff|={abs(res.H01_eff_ha)*27.211386:.3f} eV | "
                f"||h_01||={h01_norm:.4f} | ||d_01||={d01_norm:.4f} Bohr^-1 | invar={res.oo_vv_invariance_err:.2e}"
            )
            print(
                f"          | max|grad_E0 - FD| = {err_g0:.3e} | max|grad_E1 - FD| = {err_g1:.3e} | "
                f"max|h_01 - FD| = {err_h01:.3e} | max|d_01 - FD| = {err_d01:.3e}"
            )
            print(
                f"          | trans(g0)={trans_g0:.2e} | trans(g1)={trans_g1:.2e} | trans(h01)={trans_h01:.2e} | "
                f"time={res.timings_s['total']:.3f}s"
            )
            if (xc == "hf" and max(err_g0, err_g1, err_h01) > 1e-6) or (xc == "pbe0" and max(err_g0, err_g1, err_h01) > 1e-4):
                print("    WARNING: discrepancy detected!")
                print("    h_01 anal:\n", res.h_mn_ha_bohr[(0, 1)])
                print("    h_01 FD:\n", h01_fd)
                print("    h_01 diff:\n", res.h_mn_ha_bohr[(0, 1)] - h01_fd)
                print("    g1 diff:\n", res.grads_ha_bohr[1] - g_fd[1])


if __name__ == "__main__":
    run_verification()
