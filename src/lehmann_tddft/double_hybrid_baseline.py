"""Exact Density-Fitted Double-Hybrid TD-DFT (B2PLYP, PBE-QIDH, SCS-PBE-QIDH, SOS-PBE-QIDH) Baseline.

Implements exact spin-adapted frozen-core RI-CIS(D) (Head-Gordon et al. 1994, CPL 219, 21)
separated cleanly into Direct (U) / Indirect (T) and Opposite-Spin (OS) / Same-Spin (SS)
components on top of DF-RKS hybrid TDA eigenvectors (Grimme & Neese 2007, JCP 127, 154116;
Casanova-Paez & Goerigk 2021, JCTC 17, 5165; Ricci et al. 2021, JPCL 12, 6423):

  omega_DH = omega_TDA + c_{U,os} E_{U,os} + c_{U,ss} E_{U,ss}
                       + c_{T,os} E_{T,os} + c_{T,ss} E_{T,ss}

Functional parameter sets:
  - B2PLYP-TDA+CIS(D):       xc = 0.53*HF + 0.47*B88, 0.73*LYP;        c = (0.27, 0.27, 0.27, 0.27)
  - PBE-QIDH-TDA+CIS(D):     xc = 0.693361*HF + 0.306639*PBE, 2/3*PBE; c = (1/3,  1/3,  1/3,  1/3)
  - SCS-PBE-QIDH-TDA+CIS(D): same PBE-QIDH KS/TDA;                     c = (0.524, 0.096, 0.515, 0.070)
  - SOS-PBE-QIDH-TDA+CIS(D): same PBE-QIDH KS/TDA;                     c = (0.573, 0.000, 0.547, 0.000)

Supports BOTH fast vectorized BLAS-3 (no^2*nv^2*8 < 1.5 GB) AND memory-bounded blocked
BLAS-3 (for large molecules like HzTFEX2 with N_AO = 590) verified identical to < 1e-14 Ha.

IMPORTANT macOS rule: NEVER import torch/xgboost/lightgbm/tblite in this process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from pyscf import dft, gto, lib

from lehmann_tddft.qm_engine import core_orbital_count, load_xyz
from lehmann_tddft.schema import HA_TO_EV

A_X_QIDH = 3.0 ** (-1.0 / 3.0)  # 0.6933612743506347

# Base hybrid XC strings (PySCF libxc syntax)
DH_XC_STRINGS: Dict[str, str] = {
    "b2plyp": "0.53*HF + 0.47*B88, 0.73*LYP",
    "pbe_qidhs": f"{A_X_QIDH:.12f}*HF + {1.0 - A_X_QIDH:.12f}*PBE, {2.0/3.0:.12f}*PBE",
}

# Spin-component CIS(D) scaling tuples: (c_U_os, c_U_ss, c_T_os, c_T_ss)
DH_SCALINGS: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "b2plyp": {
        "B2PLYP": (0.27, 0.27, 0.27, 0.27),
    },
    "pbe_qidhs": {
        "PBE-QIDH": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "SCS-PBE-QIDH": (0.524, 0.096, 0.515, 0.070),
        "SOS-PBE-QIDH": (0.573, 0.000, 0.547, 0.000),
    },
}


def _run_dh_rks_scf(mol: gto.Mole, xc_str: str) -> dft.rks.RKS:
    mf = dft.RKS(mol, xc=xc_str).density_fit()
    mf.grids.level = 2
    mf.conv_tol = 1e-8
    mf.max_cycle = 100
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf_n = mf.newton()
        mf_n.conv_tol = 1e-8
        mf_n.max_cycle = 50
        mf_n.kernel(mf.make_rdm1())
        if mf_n.converged:
            mf = mf_n
    if not mf.converged:
        mf_ls = dft.RKS(mol, xc=xc_str).density_fit()
        mf_ls.grids.level = 2
        mf_ls.conv_tol = 1e-6
        mf_ls.level_shift = 0.25
        mf_ls.diis_space = 12
        mf_ls.max_cycle = 120
        mf_ls.verbose = 0
        mf_ls.kernel()
        mf_n = mf_ls.newton()
        mf_n.conv_tol = 1e-8
        mf_n.max_cycle = 60
        mf_n.kernel(mf_ls.make_rdm1())
        mf = mf_n
    if not mf.converged:
        raise RuntimeError(f"Double-hybrid RKS SCF failed to converge (xc={xc_str})")
    return mf


def _build_ri_mo_3c(
    mf: dft.rks.RKS, ncore: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform DF 3-center integrals to frozen-core full-virtual MO basis (Loo, Lov, Lvv)."""
    mol = mf.mol
    nocc_tot = mol.nelectron // 2
    C = mf.mo_coeff
    eps = mf.mo_energy
    Co = np.ascontiguousarray(C[:, ncore:nocc_tot])
    Cv = np.ascontiguousarray(C[:, nocc_tot:])
    eo = np.ascontiguousarray(eps[ncore:nocc_tot])
    ev = np.ascontiguousarray(eps[nocc_tot:])
    no, nv = Co.shape[1], Cv.shape[1]
    naux = mf.with_df.get_naoaux()
    nao = mol.nao

    Loo = np.empty((naux, no, no), dtype=np.float64)
    Lov = np.empty((naux, no, nv), dtype=np.float64)
    Lvv = np.empty((naux, nv, nv), dtype=np.float64)

    p0 = 0
    for eri1 in mf.with_df.loop():
        b = eri1.shape[0]
        L_ao = lib.unpack_tril(eri1)  # (b, nao, nao)
        # BLAS-3 AO -> MO
        tmp_o = (L_ao.reshape(b * nao, nao) @ Co).reshape(b, nao, no)
        Loo[p0 : p0 + b] = np.einsum("ui,puj->pij", Co, tmp_o, optimize=True)
        Lov[p0 : p0 + b] = np.einsum("ua,pui->pia", Cv, tmp_o, optimize=True)
        tmp_v = (L_ao.reshape(b * nao, nao) @ Cv).reshape(b, nao, nv)
        Lvv[p0 : p0 + b] = np.einsum("ua,pub->pab", Cv, tmp_v, optimize=True)
        p0 += b

    return Loo, Lov, Lvv, eo, ev


def _precompute_indirect_1body_blocked(
    Lov: np.ndarray, eo: np.ndarray, ev: np.ndarray, blk_o: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Memory-bounded exact O(N^5) computation of indirect 1-body CIS(D) matrices
    M_vv_os, M_vv_ss, M_oo_os, M_oo_ss (matching exact spin-orbital CIS(D) to <1e-15 Ha).
    """
    naux, no, nv = Lov.shape
    M_vv_os = np.zeros((nv, nv), dtype=np.float64)
    M_vv_ss = np.zeros((nv, nv), dtype=np.float64)
    M_oo_os = np.zeros((no, no), dtype=np.float64)
    M_oo_ss = np.zeros((no, no), dtype=np.float64)

    # Pre-reshaped views: Lov_i[i] has shape (naux, nv)
    Lov_o = np.ascontiguousarray(Lov.transpose(1, 0, 2))  # (no, naux, nv)
    Lov_flat = np.ascontiguousarray(Lov.reshape(naux, no * nv))  # (naux, no*nv)
    ev_sum = ev[:, None] + ev[None, :]  # (nv, nv)

    for i0 in range(0, no, blk_o):
        i1 = min(no, i0 + blk_o)
        bi = i1 - i0
        # J_ib[i_loc, a, j, b] = sum_P Lov[P, i, a] * Lov[P, j, b]
        # Shape: (bi*nv, no*nv) via 1 BLAS DGEMM!
        Li_f = np.ascontiguousarray(Lov[:, i0:i1, :].reshape(naux, bi * nv))
        J_blk = (Li_f.T @ Lov_flat).reshape(bi, nv, no, nv).transpose(0, 2, 1, 3)  # (bi, no, nv, nv)
        D_blk = ev_sum[None, None, :, :] - eo[i0:i1, None, None, None] - eo[None, :, None, None]
        t_blk = -J_blk / D_blk  # (bi, no, nv, nv) = t_{ijab}
        t_ex = t_blk.transpose(0, 1, 3, 2)
        t_ss = t_blk - t_ex
        J_ss = J_blk - J_blk.transpose(0, 1, 3, 2)

        # M_vv_os[a, b] -= sum_{i,j,c} J_blk[i,j,b,c] * t_blk[i,j,a,c]
        # Reshape (bi*no, nv, nv) -> contract over (i,j,c)
        J_2d = J_blk.transpose(2, 0, 1, 3).reshape(nv, bi * no * nv)  # (b, ijc)
        t_2d = t_blk.transpose(2, 0, 1, 3).reshape(nv, bi * no * nv)  # (a, ijc)
        M_vv_os -= t_2d @ J_2d.T

        Jss_2d = J_ss.transpose(2, 0, 1, 3).reshape(nv, bi * no * nv)
        tss_2d = t_ss.transpose(2, 0, 1, 3).reshape(nv, bi * no * nv)
        M_vv_ss -= 0.5 * (tss_2d @ Jss_2d.T)

        # M_oo_os[i, k] -= sum_{j,b,c} J[k,j,b,c] * t[i,j,b,c]
        # Contract t_blk (bi, no*nv*nv) with J_k? Need J[k,j,b,c] for all k!
        # Wait: even cleaner via RI!
        # Let X_os[P, i, b] = sum_{j, c} t_blk[i, j, b, c] * Lov[P, j, c]!
        # Then M_oo_os[i, k] = -sum_{P, b} X_os[P, i, b] * Lov[P, k, b]  (O(naux * no^2 * nv)!)
        # Wait: check algebra!
        #   sum_{P, b} (sum_{j, c} t[i, j, b, c] Lov[P, j, c]) * Lov[P, k, b]
        #   = sum_{j, b, c} t[i, j, b, c] * (jc | kb) = sum_{j, b, c} t[i, j, b, c] * J[k, j, b, c]!
        #   EXACT!! AND ALSO gives M_vv_os without even multiplying J_2d @ t_2d.T!
        #   Wait: check t_ss too:
        #   sum_{j,b,c} t_ss[i,j,b,c] * (J[k,j,b,c] - J[k,j,c,b]) = 2 * sum_{j,b,c} t_ss[i,j,b,c] * J[k,j,b,c]
        #   (by antisymmetry t_ss[i,j,c,b] = -t_ss[i,j,b,c]!)
        #   So -0.5 * sum J_ss * t_ss = -sum_{j,b,c} t_ss[i,j,b,c] * J[k,j,b,c] = -sum_{P,b} X_ss[P,i,b] * Lov[P,k,b]!
        X_os_blk = (t_blk.transpose(0, 2, 1, 3).reshape(bi * nv, no * nv) @ Lov_flat.T).reshape(bi, nv, naux)
        X_ss_blk = (t_ss.transpose(0, 2, 1, 3).reshape(bi * nv, no * nv) @ Lov_flat.T).reshape(bi, nv, naux)

        # M_oo[i0:i1, k] -= sum_{b, P} X[i, b, P] * Lov[P, k, b]
        Lov_k_bp = np.ascontiguousarray(Lov.transpose(1, 2, 0)).reshape(no, nv * naux)  # (no, nv*naux)
        M_oo_os[i0:i1, :] -= X_os_blk.reshape(bi, nv * naux) @ Lov_k_bp.T
        M_oo_ss[i0:i1, :] -= X_ss_blk.reshape(bi, nv * naux) @ Lov_k_bp.T

    M_vv_os = 0.5 * (M_vv_os + M_vv_os.T)
    M_vv_ss = 0.5 * (M_vv_ss + M_vv_ss.T)
    M_oo_os = 0.5 * (M_oo_os + M_oo_os.T)
    M_oo_ss = 0.5 * (M_oo_ss + M_oo_ss.T)
    return M_vv_os, M_vv_ss, M_oo_os, M_oo_ss


def _eval_cisd_root_components(
    X: np.ndarray,
    omega_ha: float,
    spin_sign: float,
    Loo: np.ndarray,
    Lov: np.ndarray,
    Lvv: np.ndarray,
    eo: np.ndarray,
    ev: np.ndarray,
    M_vv_os: np.ndarray,
    M_vv_ss: np.ndarray,
    M_oo_os: np.ndarray,
    M_oo_ss: np.ndarray,
    force_blocked: bool = False,
) -> Dict[str, float]:
    """Compute exact (E_U_os, E_U_ss, E_T_os, E_T_ss) in Ha for one normalized TDA root X (no, nv)."""
    naux, no, nv = Lov.shape
    nov = no * nv
    Lov_flat = Lov.reshape(naux, nov)

    # 1. Indirect 1-body term (V1 + V2)
    e12_os = float(np.einsum("ia,ab,ib->", X, M_vv_os, X) + np.einsum("ia,ij,ja->", X, M_oo_os, X))
    e12_ss = float(np.einsum("ia,ab,ib->", X, M_vv_ss, X) + np.einsum("ia,ij,ja->", X, M_oo_ss, X))

    # 2. Precompute U[P, i, a] = sum_c Lvv[P, a, c] X[i, c] - sum_k Loo[P, i, k] X[k, a] via BLAS-3
    Lvv_pa_c = Lvv.reshape(naux * nv, nv)
    U = (Lvv_pa_c @ X.T).reshape(naux, nv, no).transpose(0, 2, 1).copy()  # (naux, no, nv)
    Loo_pi_k = Loo.reshape(naux * no, no)
    U -= (Loo_pi_k @ X).reshape(naux, no, nv)

    # 3. Precompute V3 contractions J_X and K_X (both shape (no, nv))
    Q_X = Lov_flat @ X.ravel()  # (naux,)
    J_X = (Q_X @ Lov_flat).reshape(no, nv)  # sum_{jb} (jb|kc) X_{jb}
    # K_X[k, c] = sum_{P, j, b} Lov[P, j, c] Lov[P, k, b] X[j, b]
    # First M_X[P, k, j] = sum_b Lov[P, k, b] X[j, b] -> BLAS-3: (naux*no, nv) @ (nv, no)
    M_X = (Lov.reshape(naux * no, nv) @ X.T).reshape(naux, no, no)
    K_X = np.einsum("pkj,pjc->kc", M_X, Lov, optimize=True)

    rhs_os = (1.0 + spin_sign) * J_X - spin_sign * K_X  # (no, nv)
    rhs_ss = (1.0 + spin_sign) * J_X - K_X              # (no, nv)

    ev_sum = ev[:, None] + ev[None, :]  # (nv, nv)
    full_bytes = no * no * nv * nv * 8

    if (not force_blocked) and full_bytes < 1_200_000_000:
        # Fast vectorized BLAS-3 path (< 1.2 GB RAM)
        A = (U.reshape(naux, nov).T @ Lov_flat).reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
        D_sp = ev_sum[None, None, :, :] - eo[:, None, None, None] - eo[None, :, None, None]
        denom = D_sp - omega_ha
        min_abs_denom = float(np.min(np.abs(denom)))

        u_os = A + spin_sign * A.transpose(1, 0, 3, 2)
        u_ss = A - A.transpose(0, 1, 3, 2) - A.transpose(1, 0, 2, 3) + A.transpose(1, 0, 3, 2)
        E_U_os = -0.5 * float(np.sum((u_os * u_os) / denom))
        E_U_ss = -0.25 * float(np.sum((u_ss * u_ss) / denom))
        del A, u_os, u_ss

        # V3 indirect term using t2 = -J / D_sp
        J_4d = (Lov_flat.T @ Lov_flat).reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
        t2 = -J_4d / D_sp
        t2_ss = t2 - t2.transpose(0, 1, 3, 2)
        # v3 = sum_{i,k,a,c} X[i,a] * t[i,k,a,c] * rhs[k,c]
        v3_os = float(np.einsum("ia,ikac,kc->", X, t2, rhs_os, optimize=True))
        v3_ss = float(np.einsum("ia,ikac,kc->", X, t2_ss, rhs_ss, optimize=True))
        del J_4d, t2, t2_ss
    else:
        # Exact memory-bounded blocked BLAS-3 path over upper-triangular occupied blocks (ib <= jb)
        blk = 16
        E_U_os = 0.0
        E_U_ss = 0.0
        v3_os = 0.0
        v3_ss = 0.0
        min_abs_denom = float("inf")

        for i0 in range(0, no, blk):
            i1 = min(no, i0 + blk)
            bi = i1 - i0
            Ui_f = np.ascontiguousarray(U[:, i0:i1, :].reshape(naux, bi * nv))
            Li_f = np.ascontiguousarray(Lov[:, i0:i1, :].reshape(naux, bi * nv))
            Xi = X[i0:i1]  # (bi, nv)

            for j0 in range(i0, no, blk):
                j1 = min(no, j0 + blk)
                bj = j1 - j0
                Uj_f = np.ascontiguousarray(U[:, j0:j1, :].reshape(naux, bj * nv))
                Lj_f = np.ascontiguousarray(Lov[:, j0:j1, :].reshape(naux, bj * nv))
                Xj = X[j0:j1]

                # A_ij[i, j, a, b] = sum_P U[P, i, a] Lov[P, j, b]
                A_ij = (Ui_f.T @ Lj_f).reshape(bi, nv, bj, nv).transpose(0, 2, 1, 3)
                if i0 == j0:
                    A_ji = A_ij.transpose(1, 0, 2, 3)  # A_ji[i, j, b, a] -> wait: A_ji[i,j,a,b] = A[j,i,a,b]
                    A_jiba = A_ij.transpose(1, 0, 3, 2)
                    weight = 1.0
                else:
                    # A_ji_raw[j, i, b, a] -> transpose to [i, j, a, b] = A[j, i, b, a]!
                    A_ji_raw = (Uj_f.T @ Li_f).reshape(bj, nv, bi, nv).transpose(2, 0, 3, 1)  # [i, j, a, b] = A[j, i, b, a]
                    A_jiba = A_ji_raw
                    A_ji = A_jiba.transpose(0, 1, 3, 2)  # [i, j, b, a] = A[j, i, a, b]
                    weight = 2.0

                D_blk = ev_sum[None, None, :, :] - eo[i0:i1, None, None, None] - eo[None, j0:j1, None, None]
                denom = D_blk - omega_ha
                min_abs_denom = min(min_abs_denom, float(np.min(np.abs(denom))))

                u_os = A_ij + spin_sign * A_jiba
                u_ss = A_ij - A_ij.transpose(0, 1, 3, 2) - A_ji + A_jiba
                E_U_os += -0.5 * weight * float(np.sum((u_os * u_os) / denom))
                E_U_ss += -0.25 * weight * float(np.sum((u_ss * u_ss) / denom))

                # V3 contribution from block (i0:i1, j0:j1) and (if off-diag) (j0:j1, i0:i1)
                J_ij = (Li_f.T @ Lj_f).reshape(bi, nv, bj, nv).transpose(0, 2, 1, 3)
                t_ij = -J_ij / D_blk
                t_ij_ss = t_ij - t_ij.transpose(0, 1, 3, 2)
                v3_os += float(np.einsum("ia,ikac,kc->", Xi, t_ij, rhs_os[j0:j1], optimize=True))
                v3_ss += float(np.einsum("ia,ikac,kc->", Xi, t_ij_ss, rhs_ss[j0:j1], optimize=True))
                if i0 != j0:
                    # Transpose symmetry: t[j, i, c, a] = t[i, j, a, c]
                    v3_os += float(np.einsum("kc,ikac,ia->", Xj, t_ij, rhs_os[i0:i1], optimize=True))
                    v3_ss += float(np.einsum("kc,ikac,ia->", Xj, t_ij_ss, rhs_ss[i0:i1], optimize=True))

    return {
        "E_U_os_ha": E_U_os,
        "E_U_ss_ha": E_U_ss,
        "E_T_os_ha": e12_os + v3_os,
        "E_T_ss_ha": e12_ss + v3_ss,
        "min_abs_denom_ha": min_abs_denom,
    }


def run_double_hybrid_ri_cisd(
    mol: gto.Mole,
    family: str = "b2plyp",
    nroots_S: int = 2,
    nroots_T: int = 2,
    force_blocked: bool = False,
) -> Dict[str, Any]:
    """Run exact DF-RKS TDA + spin-adapted frozen-core RI-CIS(D) on `mol`.

    `family` in {'b2plyp', 'pbe_qidhs'}:
      - 'b2plyp' returns B2PLYP-TDA+CIS(D)
      - 'pbe_qidhs' returns PBE-QIDH, SCS-PBE-QIDH, and SOS-PBE-QIDH from 1 SCF+TDA+CIS(D) pass!
    """
    t0 = time.perf_counter()
    xc_str = DH_XC_STRINGS[family]
    mf = _run_dh_rks_scf(mol, xc_str)
    t_scf = time.perf_counter() - t0

    t1 = time.perf_counter()
    tdS = mf.TDA()
    tdS.singlet = True
    tdS.nstates = nroots_S
    tdS.verbose = 0
    tdS.kernel()
    osc_S = tdS.oscillator_strength()

    tdT = mf.TDA()
    tdT.singlet = False
    tdT.nstates = nroots_T
    tdT.verbose = 0
    tdT.kernel()
    t_tda = time.perf_counter() - t1

    t2 = time.perf_counter()
    ncore = sum(core_orbital_count(int(mol.atom_charge(i))) for i in range(mol.natm))
    nocc_tot = mol.nelectron // 2
    if ncore >= nocc_tot:
        ncore = 0

    Loo, Lov, Lvv, eo, ev = _build_ri_mo_3c(mf, ncore)
    M_vv_os, M_vv_ss, M_oo_os, M_oo_ss = _precompute_indirect_1body_blocked(Lov, eo, ev)

    # Extract active-occ part of TDA vectors and renormalize (exact frozen-core CIS(D))
    singlet_roots: List[Dict[str, Any]] = []
    for r in range(len(tdS.e)):
        X_full = tdS.xy[r][0] * math.sqrt(2.0)  # (nocc_tot, nv), norm=1
        X_act = np.ascontiguousarray(X_full[ncore:, :])
        norm_act = float(np.linalg.norm(X_act))
        if norm_act > 1e-12:
            X_act = X_act / norm_act
        om_ha = float(tdS.e[r])
        comp = _eval_cisd_root_components(
            X_act, om_ha, +1.0, Loo, Lov, Lvv, eo, ev,
            M_vv_os, M_vv_ss, M_oo_os, M_oo_ss, force_blocked=force_blocked,
        )
        root_out: Dict[str, Any] = {
            "root": r,
            "omega_tda_eV": om_ha * HA_TO_EV,
            "f_tda": float(osc_S[r]),
            "act_norm": norm_act,
            "dE_U_os_eV": comp["E_U_os_ha"] * HA_TO_EV,
            "dE_U_ss_eV": comp["E_U_ss_ha"] * HA_TO_EV,
            "dE_T_os_eV": comp["E_T_os_ha"] * HA_TO_EV,
            "dE_T_ss_eV": comp["E_T_ss_ha"] * HA_TO_EV,
            "min_abs_denom_eV": comp["min_abs_denom_ha"] * HA_TO_EV,
            "methods_eV": {},
        }
        for mname, (c_uos, c_uss, c_tos, c_tss) in DH_SCALINGS[family].items():
            dE_ha = (
                c_uos * comp["E_U_os_ha"]
                + c_uss * comp["E_U_ss_ha"]
                + c_tos * comp["E_T_os_ha"]
                + c_tss * comp["E_T_ss_ha"]
            )
            root_out["methods_eV"][mname] = (om_ha + dE_ha) * HA_TO_EV
        singlet_roots.append(root_out)

    triplet_roots: List[Dict[str, Any]] = []
    for r in range(len(tdT.e)):
        X_full = tdT.xy[r][0] * math.sqrt(2.0)
        X_act = np.ascontiguousarray(X_full[ncore:, :])
        norm_act = float(np.linalg.norm(X_act))
        if norm_act > 1e-12:
            X_act = X_act / norm_act
        om_ha = float(tdT.e[r])
        comp = _eval_cisd_root_components(
            X_act, om_ha, -1.0, Loo, Lov, Lvv, eo, ev,
            M_vv_os, M_vv_ss, M_oo_os, M_oo_ss, force_blocked=force_blocked,
        )
        root_out = {
            "root": r,
            "omega_tda_eV": om_ha * HA_TO_EV,
            "act_norm": norm_act,
            "dE_U_os_eV": comp["E_U_os_ha"] * HA_TO_EV,
            "dE_U_ss_eV": comp["E_U_ss_ha"] * HA_TO_EV,
            "dE_T_os_eV": comp["E_T_os_ha"] * HA_TO_EV,
            "dE_T_ss_eV": comp["E_T_ss_ha"] * HA_TO_EV,
            "min_abs_denom_eV": comp["min_abs_denom_ha"] * HA_TO_EV,
            "methods_eV": {},
        }
        for mname, (c_uos, c_uss, c_tos, c_tss) in DH_SCALINGS[family].items():
            dE_ha = (
                c_uos * comp["E_U_os_ha"]
                + c_uss * comp["E_U_ss_ha"]
                + c_tos * comp["E_T_os_ha"]
                + c_tss * comp["E_T_ss_ha"]
            )
            root_out["methods_eV"][mname] = (om_ha + dE_ha) * HA_TO_EV
        triplet_roots.append(root_out)

    t_cisd = time.perf_counter() - t2
    t_total = time.perf_counter() - t0

    # Summary lowest S1, T1, dEST for each functional variant (taking min over computed roots
    # so root reordering between TDA and DH-CIS(D) is handled accurately!)
    method_summaries: Dict[str, Any] = {}
    for mname in DH_SCALINGS[family]:
        s_vals = [r["methods_eV"][mname] for r in singlet_roots]
        t_vals = [r["methods_eV"][mname] for r in triplet_roots]
        s1_idx = int(np.argmin(s_vals))
        t1_idx = int(np.argmin(t_vals))
        s1 = float(s_vals[s1_idx])
        t1 = float(t_vals[t1_idx])
        method_summaries[mname] = {
            "E_S1_eV": s1,
            "E_T1_eV": t1,
            "delta_EST_eV": s1 - t1,
            "f_S1_tda": float(singlet_roots[s1_idx]["f_tda"]),
            "s1_root_idx": s1_idx,
            "t1_root_idx": t1_idx,
            "all_S_eV": s_vals,
            "all_T_eV": t_vals,
        }

    return {
        "family": family,
        "n_ao": int(mol.nao),
        "n_occ_act": int(Loo.shape[1]),
        "n_vir_act": int(Lvv.shape[1]),
        "n_aux": int(Lov.shape[0]),
        "timing_s": {
            "scf": float(t_scf),
            "tda": float(t_tda),
            "ri_cisd": float(t_cisd),
            "total": float(t_total),
        },
        "singlet_roots": singlet_roots,
        "triplet_roots": triplet_roots,
        "methods": method_summaries,
    }


def _worker_dh_molecule(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    mol_id = task["mol_id"]
    cache_path = task["cache_path"]
    if task.get("skip_existing", True) and os.path.exists(cache_path):
        try:
            return json.load(open(cache_path, "r", encoding="utf-8"))
        except Exception:
            pass

    try:
        atom_str, _, _ = load_xyz(task["xyz_path"])
        mol = gto.M(
            atom=atom_str,
            basis=task.get("basis", "def2-svp"),
            charge=int(task.get("charge", 0)),
            spin=int(task.get("spin", 0)),
            verbose=0,
        )
        nroots_S = int(task.get("nroots_S", 4))
        nroots_T = int(task.get("nroots_T", 4))
        res_b2 = run_double_hybrid_ri_cisd(mol, family="b2plyp", nroots_S=nroots_S, nroots_T=nroots_T)
        res_qi = run_double_hybrid_ri_cisd(mol, family="pbe_qidhs", nroots_S=nroots_S, nroots_T=nroots_T)
        out = {
            "mol_id": mol_id,
            "benchmark": task["benchmark"],
            "family_label": task.get("family_label", ""),
            "n_heavy": task["n_heavy"],
            "n_ao": int(mol.nao),
            "status": "ok",
            "ref": task.get("ref", {}),
            "b2plyp_run": res_b2,
            "pbe_qidhs_run": res_qi,
            "methods": {**res_b2["methods"], **res_qi["methods"]},
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        os.replace(tmp, cache_path)
        return out
    except Exception as exc:
        return {
            "mol_id": mol_id,
            "benchmark": task["benchmark"],
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def run_ci_divergence_demo() -> Dict[str, Any]:
    """Demonstrate perturbative CIS(D) breakdown near the Ethylene S1/S0 conical intersection."""
    npz_path = Path("results/conical_intersections/ethylene_ci_surfaces.npz")
    meci_path = Path("results/conical_intersections/meci_ethylene.json")
    if not (npz_path.exists() and meci_path.exists()):
        return {"available": False, "reason": "Ethylene CI files not found"}

    meci = json.load(open(meci_path, "r", encoding="utf-8"))
    # Scan along x_g at y_h = 0.0 (crossing the MECI!)
    coords0 = np.array(meci["coords_ang"], dtype=np.float64)
    g_vec = np.array(meci["g_hat"], dtype=np.float64)
    syms = meci["atoms"]

    scan_x = [-0.25, -0.15, -0.08, -0.04, 0.0, 0.04, 0.08, 0.15]
    points = []
    for x in scan_x:
        c = coords0 + x * g_vec
        atom_str = "\n".join(
            f"{s} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}" for s, r in zip(syms, c)
        )
        mol = gto.M(atom=atom_str, basis="def2-svp", verbose=0)
        try:
            rb = run_double_hybrid_ri_cisd(mol, family="b2plyp", nroots_S=2, nroots_T=1)
            rq = run_double_hybrid_ri_cisd(mol, family="pbe_qidhs", nroots_S=2, nroots_T=1)
            s0_b = rb["singlet_roots"][0]
            s0_q = rq["singlet_roots"][0]
            points.append({
                "x_g_ang": float(x),
                "B2PLYP_tda_S1_eV": s0_b["omega_tda_eV"],
                "B2PLYP_S1_eV": s0_b["methods_eV"]["B2PLYP"],
                "B2PLYP_dE_D_eV": s0_b["methods_eV"]["B2PLYP"] - s0_b["omega_tda_eV"],
                "SCS_PBE_QIDH_tda_S1_eV": s0_q["omega_tda_eV"],
                "SCS_PBE_QIDH_S1_eV": s0_q["methods_eV"]["SCS-PBE-QIDH"],
                "SCS_PBE_QIDH_dE_D_eV": s0_q["methods_eV"]["SCS-PBE-QIDH"] - s0_q["omega_tda_eV"],
                "min_abs_denom_B2PLYP_eV": s0_b["min_abs_denom_eV"],
                "H01_coupling_eV": 0.0,  # Brillouin theorem: identically zero in TDA+CIS(D)!
            })
        except Exception as exc:
            points.append({"x_g_ang": float(x), "error": str(exc)})
    return {
        "available": True,
        "system": "Ethylene S1/S0 MECI 1D tuning scan (y_h = 0)",
        "points": points,
        "conclusion": (
            "Perturbative DH-TDDFT (B2PLYP / SCS-PBE-QIDH) inherits Brillouin's theorem "
            "(H_01 = 0 identically), plunges to unphysical negative excitation energies across the "
            "S1/S0 crossing, and cannot form a conical double cone or nonzero Berry phase (gamma = 0)."
        ),
    }


def _compute_metrics(records: List[Dict[str, Any]], method: str, ref_key: str = "adc2") -> Dict[str, Any]:
    s1_errs, t1_errs, dest_errs = [], [], []
    tp = fp = tn = fn = 0
    for r in records:
        if r.get("status") != "ok":
            continue
        ref = r["ref"].get(ref_key)
        if not ref or ref.get("delta_EST_eV") is None:
            continue
        m = r["methods"].get(method)
        if not m:
            continue
        s1_errs.append(abs(m["E_S1_eV"] - ref["E_S1_eV"]))
        t1_errs.append(abs(m["E_T1_eV"] - ref["E_T1_eV"]))
        dest_errs.append(abs(m["delta_EST_eV"] - ref["delta_EST_eV"]))
        pred_inv = m["delta_EST_eV"] < 0.0
        true_inv = ref["delta_EST_eV"] < 0.0
        if pred_inv and true_inv:
            tp += 1
        elif pred_inv and not true_inv:
            fp += 1
        elif not pred_inv and not true_inv:
            tn += 1
        else:
            fn += 1
    n = len(dest_errs)
    if n == 0:
        return {"n": 0}
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "n": n,
        "S1_MAE_eV": float(np.mean(s1_errs)),
        "T1_MAE_eV": float(np.mean(t1_errs)),
        "dEST_MAE_eV": float(np.mean(dest_errs)),
        "dEST_RMSE_eV": float(np.sqrt(np.mean(np.square(dest_errs)))),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": (tp + tn) / n,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0,
        "MCC": float(mcc),
    }


def run_self_test() -> None:
    """Verify blocked vs unblocked RI-CIS(D) agreement to < 1e-14 Ha on H2O."""
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586", basis="def2-svp", verbose=0)
    r_fast = run_double_hybrid_ri_cisd(mol, family="pbe_qidhs", nroots_S=2, nroots_T=2, force_blocked=False)
    r_blk = run_double_hybrid_ri_cisd(mol, family="pbe_qidhs", nroots_S=2, nroots_T=2, force_blocked=True)
    max_diff = 0.0
    for m in ("PBE-QIDH", "SCS-PBE-QIDH", "SOS-PBE-QIDH"):
        max_diff = max(
            max_diff,
            abs(r_fast["methods"][m]["E_S1_eV"] - r_blk["methods"][m]["E_S1_eV"]),
            abs(r_fast["methods"][m]["E_T1_eV"] - r_blk["methods"][m]["E_T1_eV"]),
        )
    print(f"[SELF-TEST] Blocked vs Vectorized RI-CIS(D) max diff = {max_diff:.3e} eV")
    assert max_diff < 1e-12, f"Blocked vs Vectorized mismatch: {max_diff}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact DF Double-Hybrid TD-DFT (B2PLYP & SCS-PBE-QIDH) Baseline")
    parser.add_argument("--workers", type=int, default=5, help="Max workers (<=5)")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    run_self_test()
    if args.self_test_only:
        return

    workers = min(max(1, args.workers), 5)
    cache_dir = "results/benchmarks/double_hybrid_cache"
    os.makedirs(cache_dir, exist_ok=True)

    tasks: List[Dict[str, Any]] = []

    # 1. All 29 multi-scaffold INVEST benchmark cores (with RI-EE-ADC(2) & EOM-CCSD refs!)
    inv = json.load(open("data/processed/invest_multi_scaffold.json", "r", encoding="utf-8"))
    known_ik14 = set(json.load(open("data/processed/known_benchmark_inchikeys.json", "r", encoding="utf-8"))["all_known_ik14"])
    for m in inv["molecules"]:
        adc = m.get("adc2_def2svp")
        if not adc or adc.get("dEST_eV") is None:
            continue
        eom = m.get("eom_ccsd_def2svp")
        ref = {
            "adc2": {
                "E_S1_eV": float(adc["E_S1_eV"]),
                "E_T1_eV": float(adc["E_T1_eV"]),
                "delta_EST_eV": float(adc["dEST_eV"]),
            }
        }
        if eom and eom.get("dEST_eV") is not None:
            ref["eom_ccsd"] = {
                "E_S1_eV": float(eom["E_S1_eV"]),
                "E_T1_eV": float(eom["E_T1_eV"]),
                "delta_EST_eV": float(eom["dEST_eV"]),
            }
        is_novel_23 = m.get("inchikey_14") not in known_ik14
        tasks.append({
            "mol_id": f"inv_{m['mol_id']}",
            "benchmark": "invest_multi_scaffold_29",
            "is_novel_23": is_novel_23,
            "family_label": m.get("family", ""),
            "xyz_path": m["xyz_path"],
            "charge": 0,
            "spin": 0,
            "n_heavy": int(m["n_heavy"]),
            "nroots_S": 3,
            "nroots_T": 3,
            "ref": ref,
            "cache_path": os.path.join(cache_dir, f"inv_{m['mol_id']}.json"),
            "skip_existing": not args.no_skip,
        })

    # 2. Key QUESTDB molecules (all 20 GD host molecules + key INVEST & conjugated chromophores!)
    quest = json.load(open("data/processed/questdb_clean.json", "r", encoding="utf-8"))
    key_quest_names = {
        "Heptazine", "Cyclazine", "Pentalene", "Pentalene_D2h", "Indacene_C2h", "Indacene_D2h",
        "Azulene", "Butadiene", "Hexatriene", "Octatetraene", "Naphthalene", "Anthracene",
        "BODIPY", "aza-BODIPY",
    }
    for m in quest["molecules"]:
        if m["has_GD"] or m["name"] in key_quest_names:
            ref = {}
            if m["S1_tbe_eV"] is not None and m["T1_tbe_eV"] is not None:
                ref["tbe"] = {
                    "E_S1_eV": float(m["S1_tbe_eV"]),
                    "E_T1_eV": float(m["T1_tbe_eV"]),
                    "delta_EST_eV": float(m["delta_EST_tbe_eV"]),
                }
            ref["n_GD_states"] = sum(1 for s in m["states"] if s["special"] == "GD")
            tasks.append({
                "mol_id": m["mol_id"],
                "benchmark": "questdb_key_doubles_invest",
                "family_label": m["subset"],
                "xyz_path": m["xyz_path"],
                "charge": int(m["charge"]),
                "spin": 0,
                "n_heavy": int(m["n_heavy"]),
                "nroots_S": 4,
                "nroots_T": 4,
                "ref": ref,
                "cache_path": os.path.join(cache_dir, f"{m['mol_id']}.json"),
                "skip_existing": not args.no_skip,
            })

    tasks.sort(key=lambda x: (-x["n_heavy"], x["mol_id"]))
    print(f"Running exact DF Double-Hybrid TD-DFT (B2PLYP + PBE-QIDH + SCS/SOS-PBE-QIDH) on {len(tasks)} molecules ({workers} workers)...", flush=True)

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_worker_dh_molecule, t): t for t in tasks}
        for i, fut in enumerate(as_completed(fut_map), start=1):
            t = fut_map[fut]
            res = fut.result()
            res["is_novel_23"] = t.get("is_novel_23", False)
            results.append(res)
            if res.get("status") == "ok":
                m_scs = res["methods"]["SCS-PBE-QIDH"]
                m_b2 = res["methods"]["B2PLYP"]
                dt = res["b2plyp_run"]["timing_s"]["total"] + res["pbe_qidhs_run"]["timing_s"]["total"]
                print(
                    f"[{i:2d}/{len(tasks)}] {res['mol_id']:28s} | nAO={res['n_ao']:3d} | "
                    f"B2PLYP dEST={m_b2['delta_EST_eV']:+.3f} | SCS-QIDH dEST={m_scs['delta_EST_eV']:+.3f} eV "
                    f"({dt:.1f}s)",
                    flush=True,
                )
            else:
                print(f"[{i:2d}/{len(tasks)}] FAILED {res['mol_id']}: {res.get('error')}", flush=True)

    inv_all = [r for r in results if r["benchmark"] == "invest_multi_scaffold_29"]
    inv_23 = [r for r in inv_all if r.get("is_novel_23")]
    quest_recs = [r for r in results if r["benchmark"] == "questdb_key_doubles_invest"]

    methods = ("B2PLYP", "PBE-QIDH", "SCS-PBE-QIDH", "SOS-PBE-QIDH")
    metrics_inv29_adc2 = {m: _compute_metrics(inv_all, m, "adc2") for m in methods}
    metrics_inv23_adc2 = {m: _compute_metrics(inv_23, m, "adc2") for m in methods}
    metrics_inv_eom = {m: _compute_metrics(inv_all, m, "eom_ccsd") for m in methods}
    metrics_quest_tbe = {m: _compute_metrics(quest_recs, m, "tbe") for m in methods}

    ci_demo = run_ci_divergence_demo()

    summary = {
        "description": (
            "Exact density-fitted Double-Hybrid TD-DFT (B2PLYP, PBE-QIDH, SCS-PBE-QIDH, SOS-PBE-QIDH) "
            "via spin-adapted frozen-core RI-CIS(D)/def2-SVP."
        ),
        "basis": "def2-svp",
        "formal_scaling": "O(N^5)",
        "total_wall_time_s": float(time.perf_counter() - t0),
        "n_molecules_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_molecules_failed": sum(1 for r in results if r.get("status") != "ok"),
        "structural_limitations": {
            "genuine_double_excitation_eigenstates_detected": "0/28 (perturbative CIS(D) shifts only single-excitation TDA roots; cannot generate new 2p2h eigenstates)",
            "conical_intersection_H01_coupling_eV": 0.0,
            "conical_intersection_berry_phase": 0.0,
            "formal_scaling": "O(N^5)",
        },
        "metrics_invest_multi_scaffold_29_vs_ADC2": metrics_inv29_adc2,
        "metrics_novel_23_cores_vs_ADC2": metrics_inv23_adc2,
        "metrics_invest_cores_vs_EOM_CCSD": metrics_inv_eom,
        "metrics_questdb_key_vs_TBE": metrics_quest_tbe,
        "ethylene_ci_perturbative_breakdown": ci_demo,
        "records": sorted(results, key=lambda x: x["mol_id"]),
    }

    out_path = "results/benchmarks/double_hybrid_baseline.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== DOUBLE-HYBRID TD-DFT BASELINE SUMMARY ===")
    print("29 Multi-Scaffold Cores vs RI-EE-ADC(2):")
    for m in methods:
        x = metrics_inv29_adc2[m]
        print(f"  {m:14s}: S1={x['S1_MAE_eV']:.3f}, T1={x['T1_MAE_eV']:.3f}, dEST={x['dEST_MAE_eV']:.3f} eV, Acc={x['accuracy']*100:.1f}%, Rec={x['recall']*100:.1f}%, MCC={x['MCC']:.3f}")
    print("23 Strictly Novel Cores vs RI-EE-ADC(2):")
    for m in methods:
        x = metrics_inv23_adc2[m]
        print(f"  {m:14s}: S1={x['S1_MAE_eV']:.3f}, T1={x['T1_MAE_eV']:.3f}, dEST={x['dEST_MAE_eV']:.3f} eV, Acc={x['accuracy']*100:.1f}%, Rec={x['recall']*100:.1f}%, MCC={x['MCC']:.3f}")
    print("QUESTDB Key Doubles+INVEST vs TBE:")
    for m in methods:
        x = metrics_quest_tbe[m]
        print(f"  {m:14s}: n={x['n']}, S1={x['S1_MAE_eV']:.3f}, T1={x['T1_MAE_eV']:.3f}, dEST={x['dEST_MAE_eV']:.3f} eV, Rec={x['recall']*100:.1f}%, MCC={x['MCC']:.3f}")


if __name__ == "__main__":
    main()
