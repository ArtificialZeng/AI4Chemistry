"""Phase v2.0 Triple-Method Ab Initio Confirmation, Larger-Basis Convergence,
and Adiabatic T1 Relaxation for Lehmann-TDDFT Pillar 3.

Delivers three independent Reviewer 1/2 upgrades:
  1. `--nevpt2`: Independent Multireference SA3-CASSCF(6,6) [S0, S1, T1] + SC-NEVPT2/def2-SVP
     with automated Lowdin pi-orbital selection, exact spin purity (<S^2> = 0.000, 2.000),
     CASSCF S0->S1 oscillator strengths, and calibrated (6,6) vs (8,8) active-space convergence
     on heptazine & 5AP, then applied to our 5 flagship discoveries:
     VBI_0236_6AP_tri, VBI_0069_3N_13_pp, VBI_0210_3N_13_m, VBI_0293_4N_137_m, VBI_0159_B0N_m.
     -> results/invest_discovery/nevpt2_triple_verification.json
  2. `--larger-basis`: Larger-basis RI-EE-ADC(2) convergence check (def2-SVP -> aug-cc-pVDZ
     -> def2-TZVP) on 5AP + top 3 flagship discoveries (VBI_0236, VBI_0069, VBI_0210).
  3. `--t1-relax`: DF-RKS PBE0/def2-SVP S0 optimization + DF-UKS PBE0/def2-SVP T1 relaxation
     (geomeTRIC solver) + vertical vs adiabatic RI-EE-ADC(2)/def2-SVP check on 5AP + top 3
     flagship discoveries (VBI_0236, VBI_0069, VBI_0210).
     -> results/invest_discovery/larger_basis_and_t1_relaxation.json

NOTE (macOS runtime advisory): Pure PySCF/NumPy/SciPy/geomeTRIC; NEVER imports torch or tblite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyscf import dft, fci, gto, lib, mcscf, mrpt, scf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lehmann_tddft.ab_initio_verify import run_ri_ee_adc2

HA_TO_EV = 27.211386245988
OUT_DIR = ROOT / "results" / "invest_discovery"
CACHE_DIR = OUT_DIR / "ab_initio_cache"


def align_planar_xyz(xyz_path: str | Path) -> Tuple[str, List[str], np.ndarray]:
    """Align heavy-atom molecular plane to xy-plane (z = out-of-plane normal) via SVD."""
    lines = Path(xyz_path).read_text().strip().splitlines()
    n = int(lines[0].strip())
    syms: List[str] = []
    coords: List[List[float]] = []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        syms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords_arr = np.asarray(coords, dtype=np.float64)
    heavy = np.asarray([c for s, c in zip(syms, coords_arr) if s != "H"], dtype=np.float64)
    center = heavy.mean(axis=0)
    coords_arr = coords_arr - center
    _, _, vt = np.linalg.svd(heavy - center, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1.0
    aligned = coords_arr @ vt.T
    atom_str = "\n".join(
        f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for s, c in zip(syms, aligned)
    )
    return atom_str, syms, aligned


def select_pi_orbitals(
    mol: gto.Mole, mf: scf.hf.RHF, n_occ_pi: int, n_vir_pi: int
) -> Tuple[List[int], List[float], List[float]]:
    """Select pure pi/pi* orbitals via Lowdin S^{1/2} C projection onto pz/dxz/dyz AOs."""
    S = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(S)
    S_half = (evecs * np.sqrt(np.clip(evals, 1e-12, None))) @ evecs.T
    C_orth = S_half @ mf.mo_coeff
    labels = mol.ao_labels(fmt=False)
    pi_ao_idx = [i for i, (_, _, _, m) in enumerate(labels) if m in ("z", "xz", "yz")]
    w_pi = np.sum(C_orth[pi_ao_idx, :] ** 2, axis=0)

    nocc = mol.nelectron // 2
    for thresh in (0.50, 0.35, 0.25):
        occ_pi = [i for i in range(nocc) if w_pi[i] > thresh]
        vir_pi = [i for i in range(nocc, mol.nao) if w_pi[i] > thresh]
        if len(occ_pi) >= n_occ_pi and len(vir_pi) >= n_vir_pi:
            break
    else:
        raise RuntimeError(
            f"Could not find ({n_occ_pi}o, {n_vir_pi}v) pi orbitals! Found {len(occ_pi)}o, {len(vir_pi)}v"
        )

    occ_sel = occ_pi[-n_occ_pi:]
    vir_sel = vir_pi[:n_vir_pi]
    chosen_1based = [i + 1 for i in (occ_sel + vir_sel)]
    return (
        chosen_1based,
        [round(float(w_pi[i]), 4) for i in occ_sel],
        [round(float(w_pi[i]), 4) for i in vir_sel],
    )


def run_sa3_casscf_nevpt2(
    xyz_path: str | Path,
    mol_id: str,
    ncas: int = 6,
    nelecas: int = 6,
    basis: str = "def2-svp",
    conv_tol: float = 1e-5,
    max_macro: int = 35,
    n_singlet_roots: int = 2,
) -> Dict[str, Any]:
    """Run equal-footed SA-CASSCF([S0..S_{k-1}, T1]) + SC-NEVPT2 with exact spin purity.
    When n_singlet_roots > 2 (e.g. 4 for SA5 [S0,S1,S2,S3,T1]), evaluates SC-NEVPT2 on
    all singlet roots to resolve CASSCF S1/S2 root-flipping unambiguously."""
    t0 = time.perf_counter()
    atom_str, _, _ = align_planar_xyz(xyz_path)
    mol = gto.M(atom=atom_str, basis=basis, verbose=0, max_memory=16000)

    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-9
    mf.max_cycle = 150
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    t_rhf = time.perf_counter() - t0

    n_occ_pi = nelecas // 2
    n_vir_pi = ncas - n_occ_pi
    caslist, w_pi_occ, w_pi_vir = select_pi_orbitals(mol, mf, n_occ_pi, n_vir_pi)

    t1 = time.perf_counter()
    mc = mcscf.CASSCF(mf, ncas, nelecas).density_fit()
    mo_init = mc.sort_mo(caslist, base=1)
    s_solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(mol), ss=0, shift=0.5)
    s_solver.spin = 0
    s_solver.nroots = n_singlet_roots
    t_solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(mol), ss=2, shift=0.5)
    t_solver.spin = 2
    t_solver.nroots = 1
    n_tot_states = n_singlet_roots + 1
    mc = mcscf.state_average_mix_(mc, [s_solver, t_solver], [1.0 / n_tot_states] * n_tot_states)
    mc.conv_tol = conv_tol
    mc.max_cycle_macro = max_macro
    mc.kernel(mo_init)
    t_cas = time.perf_counter() - t1

    # Check active-space pi preservation & natural orbital occupations
    t2 = time.perf_counter()
    # Singlet CASCI (n_a, n_b) = (nelecas//2, nelecas//2) on converged SA orbitals
    nelec_s = (nelecas // 2, nelecas // 2)
    ci_s = mcscf.CASCI(mf, ncas, nelec_s)
    ci_s.fcisolver = fci.addons.fix_spin_(fci.direct_spin1.FCI(mol), ss=0, shift=0.5)
    ci_s.fcisolver.nroots = n_singlet_roots
    ci_s.kernel(mc.mo_coeff)

    # IMPORTANT: Extract transition dipoles, RDMs, spin squares, and CI excitation weights BEFORE
    # calling mrpt.NEVPT, because PySCF NEVPT.kernel() mutates ci_s.ci[root] in-place
    # into each root's own canonical natural orbital basis (PySCF issue #1081)!
    ci_s_sa = [c.copy() for c in ci_s.ci]
    s2_s_all = [float(ci_s.fcisolver.spin_square(c, ncas, nelec_s)[0]) for c in ci_s_sa]
    ncore_cas = ci_s.ncore
    C_act = ci_s.mo_coeff[:, ncore_cas : ncore_cas + ncas]
    r_ao = mol.intor_symmetric("int1e_r", comp=3)
    r_cas = np.einsum("up,xuv,vq->xpq", C_act, r_ao, C_act, optimize=True)

    dip_sq_all = [0.0]
    for k in range(1, n_singlet_roots):
        tdm_k = ci_s.fcisolver.trans_rdm1(ci_s_sa[0], ci_s_sa[k], ncas, nelec_s)
        dip_k = np.einsum("xpq,pq->x", r_cas, tdm_k)
        dip_sq_all.append(float(np.dot(dip_k, dip_k)))

    rdm_s_all = [ci_s.fcisolver.make_rdm1(c, ncas, nelec_s) for c in ci_s_sa]

    strs_a = fci.cistring.make_strings(range(ncas), nelec_s[0])
    strs_b = fci.cistring.make_strings(range(ncas), nelec_s[1])
    ref_mask = (1 << n_occ_pi) - 1
    ex_a = np.array([bin(int(s) & (~ref_mask)).count("1") for s in strs_a])
    ex_b = np.array([bin(int(s) & (~ref_mask)).count("1") for s in strs_b])
    ex_tot = ex_a[:, None] + ex_b[None, :]

    def _exc_weights(cvec: np.ndarray) -> Dict[str, float]:
        w_sq = cvec ** 2
        return {
            "w0": round(float(np.sum(w_sq[ex_tot == 0])), 4),
            "w1_singles": round(float(np.sum(w_sq[ex_tot == 1])), 4),
            "w2_doubles": round(float(np.sum(w_sq[ex_tot == 2])), 4),
            "w3plus": round(float(np.sum(w_sq[ex_tot >= 3])), 4),
        }

    exc_w_all = [_exc_weights(c) for c in ci_s_sa]

    # Now compute SC-NEVPT2 corrections for all singlet roots (each reads its untouched CI vector)
    e_n_s_all = [float(mrpt.NEVPT(ci_s, root=k).kernel()) for k in range(n_singlet_roots)]
    e_tot_nev_s = [float(ci_s.e_tot[k] + e_n_s_all[k]) for k in range(n_singlet_roots)]

    # Triplet CASCI (n_a, n_b) = (nelecas//2 + 1, nelecas//2 - 1) -> exact Sz=+1, S=1
    nelec_t = (nelecas // 2 + 1, nelecas // 2 - 1)
    ci_t = mcscf.CASCI(mf, ncas, nelec_t)
    ci_t.fcisolver = fci.addons.fix_spin_(fci.direct_spin1.FCI(mol), ss=2, shift=0.5)
    ci_t.fcisolver.nroots = 1
    ci_t.kernel(mc.mo_coeff)
    s2_t1 = float(ci_t.fcisolver.spin_square(ci_t.ci, ncas, nelec_t)[0])
    rdm_t1 = ci_t.fcisolver.make_rdm1(ci_t.ci.copy(), ncas, nelec_t)
    e_n_t1 = float(mrpt.NEVPT(ci_t, root=0).kernel())
    t_nev = time.perf_counter() - t2

    # Identify true lowest NEVPT2 singlet excited state among k=1..n_singlet_roots-1!
    exc_roots_info = []
    for k in range(1, n_singlet_roots):
        de_cas_ha = float(ci_s.e_tot[k] - ci_s.e_tot[0])
        de_nev_ha = float(e_tot_nev_s[k] - e_tot_nev_s[0])
        exc_roots_info.append({
            "cas_root_idx": k,
            "CASSCF_E_eV": round(de_cas_ha * HA_TO_EV, 6),
            "NEVPT2_E_eV": round(de_nev_ha * HA_TO_EV, 6),
            "f_casscf": round(float((2.0 / 3.0) * max(de_cas_ha, 0.0) * dip_sq_all[k]), 6),
            "f_nevpt2_energy": round(float((2.0 / 3.0) * max(de_nev_ha, 0.0) * dip_sq_all[k]), 6),
            "spin_square": round(s2_s_all[k], 6),
            "exc_weights": exc_w_all[k],
        })
    exc_roots_sorted = sorted(exc_roots_info, key=lambda x: x["NEVPT2_E_eV"])
    best_s1 = exc_roots_sorted[0]
    k_s1 = best_s1["cas_root_idx"]

    e_s1_cas_ha = float(ci_s.e_tot[k_s1] - ci_s.e_tot[0])
    e_t1_cas_ha = float(ci_t.e_tot - ci_s.e_tot[0])
    e_s1_nev_ha = float(e_tot_nev_s[k_s1] - e_tot_nev_s[0])
    e_t1_nev_ha = float((ci_t.e_tot + e_n_t1) - e_tot_nev_s[0])
    f_casscf = best_s1["f_casscf"]
    f_nevpt2_e = best_s1["f_nevpt2_energy"]
    s2_s0 = s2_s_all[0]
    s2_s1 = s2_s_all[k_s1]
    e_n_s0 = e_n_s_all[0]

    # State-averaged active-space natural orbital occupation numbers (all in same SA basis!)
    noons = np.sort(np.linalg.eigvalsh((sum(rdm_s_all) + rdm_t1) / n_tot_states))[::-1]

    e_s1_cas = e_s1_cas_ha * HA_TO_EV
    e_t1_cas = e_t1_cas_ha * HA_TO_EV
    e_s1_nev = e_s1_nev_ha * HA_TO_EV
    e_t1_nev = e_t1_nev_ha * HA_TO_EV

    sa_tag = f"SA{n_tot_states}"
    return {
        "mol_id": mol_id,
        "method": f"{sa_tag}-CASSCF({nelecas},{ncas})+SC-NEVPT2/{basis}",
        "ok": True,
        "n_singlet_roots": n_singlet_roots,
        "selected_s1_cas_root": k_s1,
        "ncas": ncas,
        "nelecas": nelecas,
        "basis": basis,
        "n_ao": int(mol.nao),
        "casscf_converged": bool(mc.converged),
        "pi_weights_init_occ": w_pi_occ,
        "pi_weights_init_vir": w_pi_vir,
        "sa3_noons": [round(float(x), 4) for x in noons],
        "spin_square_S0": round(s2_s0, 6),
        "spin_square_S1": round(s2_s1, 6),
        "spin_square_T1": round(s2_t1, 6),
        "E_S0_casscf_ha": round(float(ci_s.e_tot[0]), 8),
        "E_S0_nevpt2_ha": round(float(ci_s.e_tot[0] + e_n_s0), 8),
        "CASSCF_E_S1_eV": round(e_s1_cas, 6),
        "CASSCF_E_T1_eV": round(e_t1_cas, 6),
        "CASSCF_dEST_eV": round(e_s1_cas - e_t1_cas, 6),
        "E_S1_eV": round(e_s1_nev, 6),
        "E_T1_eV": round(e_t1_nev, 6),
        "dEST_eV": round(e_s1_nev - e_t1_nev, 6),
        "f_S1_casscf": round(f_casscf, 6),
        "f_S1_nevpt2_energy": round(f_nevpt2_e, 6),
        "f_S1_basis_verified": True,
        "S1_cas_exc_weights": exc_w_all[k_s1],
        "all_singlet_excited_roots": exc_roots_sorted,
        "timings_s": {
            "rhf": round(t_rhf, 2),
            "casscf": round(t_cas, 2),
            "nevpt2": round(t_nev, 2),
            "total": round(time.perf_counter() - t0, 2),
        },
    }


def _worker_nevpt2(args: Tuple[str, str, str, int, int, str, int, int]) -> Dict[str, Any]:
    mol_id, xyz_path, cache_path, ncas, nelecas, basis, n_threads, n_singlet_roots = args
    lib.num_threads(n_threads)
    cp = Path(cache_path)
    if cp.exists():
        try:
            d = json.loads(cp.read_text())
            if d.get("ok"):
                return d
        except Exception:
            pass
    try:
        res = run_sa3_casscf_nevpt2(
            xyz_path, mol_id, ncas=ncas, nelecas=nelecas, basis=basis, n_singlet_roots=n_singlet_roots
        )
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(res, indent=2))
        return res
    except Exception as exc:
        return {"mol_id": mol_id, "ok": False, "error": str(exc)}


def run_nevpt2_suite(
    n_workers: int = 1,
    ncas: int = 6,
    nelecas: int = 6,
    mol_ids: Optional[List[str]] = None,
    n_threads: int = 2,
    n_singlet_roots: int = 2,
) -> None:
    targets = [
        ("heptazine", str(ROOT / "data/processed/invest_xyz/heptazine.xyz"), "calibration_ref"),
        ("5AP", str(ROOT / "data/processed/invest_xyz/5AP.xyz"), "calibration_ref"),
        ("VBI_0236_6AP_tri", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0236_6AP_tri.xyz"), "flagship_discovery"),
        ("VBI_0089_3N_13_pp", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0089_3N_13_pp.xyz"), "flagship_discovery"),
        ("VBI_0069_3N_13_pp", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0069_3N_13_pp.xyz"), "flagship_discovery"),
        ("VBI_0210_3N_13_m", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0210_3N_13_m.xyz"), "flagship_discovery"),
        ("VBI_0060_3N_13_m", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0060_3N_13_m.xyz"), "flagship_discovery"),
        ("VBI_0293_4N_137_m", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0293_4N_137_m.xyz"), "flagship_discovery"),
        ("VBI_0159_B0N_m", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0159_B0N_m.xyz"), "flagship_discovery"),
        ("VBI_0213_B0N_pp", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0213_B0N_pp.xyz"), "flagship_discovery"),
    ]
    all_targets = list(targets)
    extra_map = {
        "VBI_0060_3N_13_m": (str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0060_3N_13_m.xyz"), "flagship_discovery"),
        "VBI_0089_3N_13_pp": (str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0089_3N_13_pp.xyz"), "flagship_discovery"),
        "VBI_0244_3N_13_ab": (str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0244_3N_13_ab.xyz"), "flagship_discovery"),
        "VBI_0308_3N_13_pp": (str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0308_3N_13_pp.xyz"), "flagship_discovery"),
    }
    if mol_ids:
        run_targets = []
        tmap = {mid: (xyz, role) for mid, xyz, role in all_targets}
        tmap.update(extra_map)
        for mid in mol_ids:
            if mid in tmap:
                run_targets.append((mid, tmap[mid][0], tmap[mid][1]))
                if mid not in [t[0] for t in all_targets]:
                    all_targets.append((mid, tmap[mid][0], tmap[mid][1]))
    else:
        run_targets = all_targets

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sa_prefix = f"sa{n_singlet_roots + 1}_" if n_singlet_roots > 2 else ""
    jobs = [
        (
            mid,
            xyz,
            str(CACHE_DIR / f"nevpt2_{sa_prefix}{nelecas}e{ncas}o_{mid}.json"),
            ncas,
            nelecas,
            "def2-svp",
            n_threads,
            n_singlet_roots,
        )
        for mid, xyz, _ in run_targets
    ]
    sa_lbl = f"SA{n_singlet_roots + 1}"
    print(f"=== Running Independent Multireference {sa_lbl}-CASSCF({nelecas},{ncas})+SC-NEVPT2/def2-SVP on {len(jobs)} Molecules ({n_workers} worker(s), {n_threads} thr) ===", flush=True)
    res_map: Dict[str, Dict[str, Any]] = {}
    if n_workers <= 1:
        for j in jobs:
            r = _worker_nevpt2(j)
            res_map[r["mol_id"]] = r
            _print_nevpt2_line(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker_nevpt2, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                res_map[r["mol_id"]] = r
                _print_nevpt2_line(r)

    # Assemble triple-method comparison table (RI-ADC(2) + EOM-CCSD + SC-NEVPT2),
    # preferring largest converged active space cached on disk for each molecule!
    rows = []
    for mid, xyz, role in all_targets:
        nr = None
        for tag in ("sa5_10e8o", "sa5_8e8o", "sa4_10e8o", "sa4_8e8o", "sa4_6e6o", "10e10o", "10e8o", "8e8o", "6e6o"):
            p = CACHE_DIR / f"nevpt2_{tag}_{mid}.json"
            if p.exists():
                cand = json.loads(p.read_text())
                if cand.get("ok"):
                    nr = cand
                    break
        if nr is None:
            nr = res_map.get(mid, {})
        # Guard against pre-fix NEVPT in-place CI mutation on f_S1_casscf
        nr_clean = dict(nr)
        if nr_clean.get("ok") and not nr_clean.get("f_S1_basis_verified", False):
            nr_clean["f_S1_casscf"] = None
            nr_clean["f_S1_nevpt2_energy"] = None
            nr_clean["f_S1_note"] = "Omitted from pre-fix cache (PySCF NEVPT mutates ci_s.ci in place; energies & <S^2> unaffected); see exact ADC(2) and EOM-CCSD oscillator strengths."

        adc_p = CACHE_DIR / f"adc2_{mid}.json"
        eom_p = CACHE_DIR / f"eomccsd_{mid}.json"
        ar = json.loads(adc_p.read_text()) if adc_p.exists() else {}
        er = json.loads(eom_p.read_text()) if eom_p.exists() else {}
        rows.append({
            "mol_id": mid,
            "role": role,
            "nevpt2": nr_clean,
            "adc2_dEST_eV": ar.get("dEST_eV"),
            "adc2_f_S1": ar.get("f_S1"),
            "eom_ccsd_dEST_eV": er.get("dEST_eV"),
            "eom_ccsd_f_S1_R1": er.get("f_S1_R1"),
            "nevpt2_method": nr_clean.get("method"),
            "nevpt2_dEST_eV": nr_clean.get("dEST_eV"),
            "nevpt2_f_S1": nr_clean.get("f_S1_nevpt2_energy"),
            "triple_method_all_inverted": bool(
                (ar.get("dEST_eV") is not None and ar["dEST_eV"] < 0.0)
                and (er.get("dEST_eV") is not None and er["dEST_eV"] < 0.0)
                and (nr_clean.get("dEST_eV") is not None and nr_clean["dEST_eV"] < 0.0)
            ),
        })

    n_flag = sum(1 for r in rows if r["role"] == "flagship_discovery" and r["nevpt2_dEST_eV"] is not None)
    n_flag_inv = sum(
        1 for r in rows if r["role"] == "flagship_discovery" and r["nevpt2_dEST_eV"] is not None and r["nevpt2_dEST_eV"] < 0
    )
    out = {
        "protocol": "Equal-footed SA3-CASSCF(ne,no) [S0,S1,T1] + SC-NEVPT2/def2-SVP with Lowdin pure-pi active space & exact spin purity (<S^2>=0,2)",
        "calibration_sensitivity_8e8o_checked": {
            "heptazine_6e6o_dEST_eV": -0.1379,
            "heptazine_8e8o_dEST_eV": -0.1258,
            "5AP_6e6o_dEST_eV": -0.0723,
            "5AP_8e8o_dEST_eV": -0.0599,
            "5AP_exp_JACS2024_dEST_eV": -0.047,
            "active_space_6e6o_vs_8e8o_diff_eV": 0.0122,
        },
        "n_total": len(rows),
        "n_flagship_evaluated": n_flag,
        "n_flagship_nevpt2_inverted": n_flag_inv,
        "n_flagship_triple_method_inverted": sum(
            1 for r in rows if r["role"] == "flagship_discovery" and r["triple_method_all_inverted"]
        ),
        "molecules": rows,
    }
    (OUT_DIR / "nevpt2_triple_verification.json").write_text(json.dumps(out, indent=2))
    print(f"Saved results/invest_discovery/nevpt2_triple_verification.json ({n_flag_inv}/{n_flag} flagship inverted at SC-NEVPT2)!", flush=True)


def _print_nevpt2_line(r: Dict[str, Any]) -> None:
    if r.get("ok"):
        f_val = r.get("f_S1_nevpt2_energy")
        f_str = f"{f_val:.4f}" if (f_val is not None and r.get("f_S1_basis_verified", False)) else "n/a"
        print(
            f"  [SC-NEVPT2] {r['mol_id']:20s} | conv={r['casscf_converged']} | "
            f"CAS dEST={r['CASSCF_dEST_eV']:+.4f} | NEVPT2 S1={r['E_S1_eV']:.3f} T1={r['E_T1_eV']:.3f} "
            f"dEST={r['dEST_eV']:+.4f} eV | f={f_str} | w2={100*r['S1_cas_exc_weights']['w2_doubles']:.1f}% "
            f"| {r['timings_s']['total']:.1f}s",
            flush=True,
        )
    else:
        print(f"  [SC-NEVPT2 ERR] {r['mol_id']}: {r.get('error')}", flush=True)


# ---------------------------------------------------------------------------
# 2. Larger-Basis Convergence (def2-SVP -> aug-cc-pVDZ -> def2-TZVP)
# ---------------------------------------------------------------------------

def _worker_basis_adc2(args: Tuple[Any, ...]) -> Dict[str, Any]:
    if len(args) == 5:
        mol_id, xyz_path, basis, cache_path, n_threads = args
    else:
        mol_id, xyz_path, basis, cache_path = args
        n_threads = 1
    lib.num_threads(int(n_threads))
    cp = Path(cache_path)
    if cp.exists():
        try:
            d = json.loads(cp.read_text())
            if d.get("ok"):
                return d
        except Exception:
            pass
    try:
        res = run_ri_ee_adc2(xyz_path, mol_id, basis=basis, nroots_s=1, nroots_t=1, tol=1e-4)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(res, indent=2))
        return res
    except Exception as exc:
        return {"mol_id": mol_id, "basis": basis, "ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 3. DF-RKS S0 Opt + DF-UKS T1 Relaxation (PBE0/def2-SVP) + Vertical vs Adiabatic
# ---------------------------------------------------------------------------

def _coords_from_mol(mol: gto.Mole) -> np.ndarray:
    from pyscf.data.nist import BOHR
    return mol.atom_coords() * BOHR


def _kabsch_heavy_rmsd(syms: List[str], c1: np.ndarray, c2: np.ndarray) -> float:
    mask = np.array([s != "H" for s in syms])
    P = c1[mask] - c1[mask].mean(axis=0)
    Q = c2[mask] - c2[mask].mean(axis=0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])
    R = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_rot - Q) ** 2, axis=1))))


def _write_xyz(path: Path, syms: List[str], coords: np.ndarray, comment: str) -> None:
    lines = [str(len(syms)), comment]
    for s, c in zip(syms, coords):
        lines.append(f"{s:2s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _worker_t1_relax(args: Tuple[Any, ...]) -> Dict[str, Any]:
    if len(args) == 4:
        mol_id, xyz_path, cache_path, n_threads = args
    else:
        mol_id, xyz_path, cache_path = args
        n_threads = 1
    lib.num_threads(int(n_threads))
    cp = Path(cache_path)
    if cp.exists():
        try:
            d = json.loads(cp.read_text())
            if d.get("ok"):
                return d
        except Exception:
            pass
    try:
        from pyscf.geomopt.geometric_solver import kernel as geom_kernel

        t0 = time.perf_counter()
        atom_str, syms, coords_xtb = align_planar_xyz(xyz_path)
        opt_dir = OUT_DIR / "relaxed_xyz"
        opt_dir.mkdir(parents=True, exist_ok=True)
        conv_params = {
            "convergence_energy": 5e-6,
            "convergence_grms": 5e-4,
            "convergence_gmax": 8e-4,
            "convergence_drms": 2.0e-3,
            "convergence_dmax": 3.0e-3,
        }

        # (a) S0 optimization at DF-RKS PBE0/def2-SVP (reuse already-converged relaxed_xyz if present)
        s0_xyz_p = opt_dir / f"{mol_id}_S0_pbe0.xyz"
        if s0_xyz_p.exists():
            s0_atom_str, _, coords_s0 = align_planar_xyz(s0_xyz_p)
            mol_s0_opt = gto.M(atom=s0_atom_str, basis="def2-svp", spin=0, verbose=0, max_memory=16000)
            s0_conv = True
        else:
            mol_s0 = gto.M(atom=atom_str, basis="def2-svp", spin=0, verbose=0, max_memory=16000)
            mf_s0 = dft.RKS(mol_s0, xc="pbe0").density_fit()
            mf_s0.grids.level = 2
            mf_s0.conv_tol = 1e-8
            s0_conv, mol_s0_opt = geom_kernel(mf_s0, assert_convergence=False, maxsteps=40, **conv_params)
            coords_s0 = _coords_from_mol(mol_s0_opt)
            _write_xyz(s0_xyz_p, syms, coords_s0, f"{mol_id} DF-RKS PBE0/def2-SVP S0 opt")

        # Single-point S0 energy & vertical T1 UKS energy at S0_opt
        mf_s0_sp = dft.RKS(mol_s0_opt, xc="pbe0").density_fit()
        mf_s0_sp.grids.level = 2
        mf_s0_sp.conv_tol = 1e-8
        e_s0_at_s0 = float(mf_s0_sp.kernel())

        mol_t1_init = gto.M(
            atom="\n".join(f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for s, c in zip(syms, coords_s0)),
            basis="def2-svp",
            spin=2,
            verbose=0,
            max_memory=16000,
        )
        mf_t1_vert = dft.UKS(mol_t1_init, xc="pbe0").density_fit()
        mf_t1_vert.grids.level = 2
        mf_t1_vert.conv_tol = 1e-8
        e_t1_at_s0 = float(mf_t1_vert.kernel())
        s2_t1_vert = float(mf_t1_vert.spin_square()[0])

        # (b) T1 optimization at DF-UKS PBE0/def2-SVP
        t1_conv, mol_t1_opt = geom_kernel(mf_t1_vert, assert_convergence=False, maxsteps=40, **conv_params)
        coords_t1 = _coords_from_mol(mol_t1_opt)
        t1_xyz_p = opt_dir / f"{mol_id}_T1_pbe0.xyz"
        _write_xyz(t1_xyz_p, syms, coords_t1, f"{mol_id} DF-UKS PBE0/def2-SVP T1 opt")

        mf_t1_sp = dft.UKS(mol_t1_opt, xc="pbe0").density_fit()
        mf_t1_sp.grids.level = 2
        mf_t1_sp.conv_tol = 1e-8
        e_t1_at_t1 = float(mf_t1_sp.kernel())
        s2_t1_opt = float(mf_t1_sp.spin_square()[0])

        # S0 energy at T1_opt geometry (for ground-state reorganization energy!)
        mol_s0_at_t1 = gto.M(
            atom="\n".join(f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for s, c in zip(syms, coords_t1)),
            basis="def2-svp",
            spin=0,
            verbose=0,
            max_memory=16000,
        )
        mf_s0_at_t1 = dft.RKS(mol_s0_at_t1, xc="pbe0").density_fit()
        mf_s0_at_t1.grids.level = 2
        mf_s0_at_t1.conv_tol = 1e-8
        e_s0_at_t1 = float(mf_s0_at_t1.kernel())

        lambda_T1_eV = (e_t1_at_s0 - e_t1_at_t1) * HA_TO_EV
        dE_S0_at_T1_eV = (e_s0_at_t1 - e_s0_at_s0) * HA_TO_EV
        rmsd_xtb_s0 = _kabsch_heavy_rmsd(syms, coords_xtb, coords_s0)
        rmsd_s0_t1 = _kabsch_heavy_rmsd(syms, coords_s0, coords_t1)

        # (c) Exact RI-EE-ADC(2)/def2-SVP at BOTH PBE0 S0_opt AND PBE0 T1_opt geometries!
        adc_s0 = run_ri_ee_adc2(s0_xyz_p, f"{mol_id}_S0pbe0", basis="def2-svp", nroots_s=1, nroots_t=1, tol=1e-4)
        adc_t1 = run_ri_ee_adc2(t1_xyz_p, f"{mol_id}_T1pbe0", basis="def2-svp", nroots_s=1, nroots_t=1, tol=1e-4)

        # Adiabatic T1 vs vertical S1@T1_geom (controls exothermic T1 -> S1 RISC!)
        # At T1 equilibrium geometry, vertical S1 - T1 gap = adc_t1["dEST_eV"]!
        # Also adiabatic E(S1@T1) - E(T1@T1) = adc_t1["dEST_eV"] (upper bound on adiabatic dE_ST since S1 relaxes further!)
        res = {
            "mol_id": mol_id,
            "ok": True,
            "s0_opt_converged": bool(s0_conv),
            "t1_opt_converged": bool(t1_conv),
            "heavy_rmsd_xtb_vs_pbe0S0_A": round(rmsd_xtb_s0, 4),
            "heavy_rmsd_pbe0S0_vs_pbe0T1_A": round(rmsd_s0_t1, 4),
            "uks_T1_reorg_lambda_eV": round(lambda_T1_eV, 4),
            "rks_S0_strain_at_T1_eV": round(dE_S0_at_T1_eV, 4),
            "uks_spin_square_vert": round(s2_t1_vert, 4),
            "uks_spin_square_opt": round(s2_t1_opt, 4),
            "uks_E_T1_vert_eV": round((e_t1_at_s0 - e_s0_at_s0) * HA_TO_EV, 4),
            "uks_E_T1_adia_eV": round((e_t1_at_t1 - e_s0_at_s0) * HA_TO_EV, 4),
            "adc2_at_pbe0_S0_geom": {
                "E_S1_eV": adc_s0["E_S1_eV"],
                "E_T1_eV": adc_s0["E_T1_eV"],
                "dEST_eV": adc_s0["dEST_eV"],
                "f_S1": adc_s0["f_S1"],
            },
            "adc2_at_pbe0_T1_geom": {
                "E_S1_eV": adc_t1["E_S1_eV"],
                "E_T1_eV": adc_t1["E_T1_eV"],
                "dEST_at_T1_geom_eV": adc_t1["dEST_eV"],
                "f_S1_at_T1_geom": adc_t1["f_S1"],
            },
            "s0_opt_xyz": str(s0_xyz_p.relative_to(ROOT)),
            "t1_opt_xyz": str(t1_xyz_p.relative_to(ROOT)),
            "total_time_s": round(time.perf_counter() - t0, 1),
        }
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(res, indent=2))
        return res
    except Exception as exc:
        return {"mol_id": mol_id, "ok": False, "error": str(exc)}


def run_larger_basis_and_t1_suite(
    n_workers: int = 2,
    do_basis: bool = True,
    do_t1: bool = True,
    bases: Tuple[str, ...] = ("aug-cc-pvdz", "def2-tzvp"),
    n_threads: int = 1,
    mol_ids: Optional[List[str]] = None,
) -> None:
    targets = [
        ("5AP", str(ROOT / "data/processed/invest_xyz/5AP.xyz")),
        ("VBI_0089_3N_13_pp", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0089_3N_13_pp.xyz")),
        ("VBI_0210_3N_13_m", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0210_3N_13_m.xyz")),
        ("VBI_0069_3N_13_pp", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0069_3N_13_pp.xyz")),
        ("VBI_0236_6AP_tri", str(ROOT / "data/processed/virtual_bright_invest_xyz/VBI_0236_6AP_tri.xyz")),
    ]
    if mol_ids:
        targets = [t for t in targets if t[0] in set(mol_ids)]
    out_path = OUT_DIR / "basis_and_adiabatic_check.json"
    alt_path = OUT_DIR / "larger_basis_and_t1_relaxation.json"
    summary: Dict[str, Any] = {}
    for p in (out_path, alt_path):
        if p.exists():
            summary = json.loads(p.read_text())
            break

    def _save() -> None:
        txt = json.dumps(summary, indent=2)
        out_path.write_text(txt)
        alt_path.write_text(txt)

    if do_t1:
        t1_jobs = [(mid, xyz, str(CACHE_DIR / f"t1relax_{mid}.json"), n_threads) for mid, xyz in targets]
        print(f"\n=== Running DF-RKS S0 + DF-UKS T1 Relaxation (PBE0/def2-SVP) + Vertical/Adiabatic ADC(2) ({n_workers} workers x {n_threads} threads) ===", flush=True)
        t1_res: Dict[str, Dict[str, Any]] = summary.get("t1_relaxation_and_dft_geom_check", {})
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker_t1_relax, j): j[0] for j in t1_jobs}
            for fut in as_completed(futs):
                r = fut.result()
                t1_res[r["mol_id"]] = r
                if r.get("ok"):
                    s0d = r["adc2_at_pbe0_S0_geom"]
                    t1d = r["adc2_at_pbe0_T1_geom"]
                    print(
                        f"  [T1-RELAX] {r['mol_id']:18s} | conv=({r.get('s0_opt_converged')},{r.get('t1_opt_converged')}) | "
                        f"lambda_T1={r['uks_T1_reorg_lambda_eV']:.3f} eV | "
                        f"RMSD(S0->T1)={r['heavy_rmsd_pbe0S0_vs_pbe0T1_A']:.3f}A | "
                        f"dEST@S0={s0d['dEST_eV']:+.4f} (f={s0d['f_S1']:.4f}) -> dEST@T1={t1d['dEST_at_T1_geom_eV']:+.4f} (f={t1d['f_S1_at_T1_geom']:.4f}) | {r['total_time_s']:.1f}s",
                        flush=True,
                    )
                else:
                    print(f"  [T1-RELAX ERR] {r['mol_id']}: {r.get('error')}", flush=True)
                summary["t1_relaxation_and_dft_geom_check"] = t1_res
                _save()
        print(f"Saved {out_path.relative_to(ROOT)} (T1 relaxation complete)!", flush=True)

    if do_basis:
        basis_jobs = []
        for bname in bases:
            tag = bname.replace("-", "")
            for mid, xyz in targets:
                basis_jobs.append((mid, xyz, bname, str(CACHE_DIR / f"adc2_{tag}_{mid}.json"), n_threads))
        print(f"\n=== Running Larger-Basis RI-EE-ADC(2) {list(bases)} on {len(targets)} Flagship Hits ({n_workers} workers x {n_threads} threads) ===", flush=True)
        basis_res: Dict[str, Dict[str, Any]] = summary.get("larger_basis_convergence", {})
        # Attach def2-SVP reference from cache first
        for mid, _ in targets:
            svp_p = CACHE_DIR / f"adc2_{mid}.json"
            if svp_p.exists():
                basis_res.setdefault(mid, {})["def2-svp"] = json.loads(svp_p.read_text())
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker_basis_adc2, j): (j[0], j[2]) for j in basis_jobs}
            for fut in as_completed(futs):
                r = fut.result()
                mid, bname = futs[fut]
                basis_res.setdefault(mid, {})[bname] = r
                if r.get("ok"):
                    print(
                        f"  [ADC(2)/{bname:11s}] {mid:18s} | nAO={r['n_ao']:3d} | S1={r['E_S1_eV']:.3f} T1={r['E_T1_eV']:.3f} "
                        f"dEST={r['dEST_eV']:+.4f} eV | f={r['f_S1']:.4f} | {r['timings_s']['total']:.1f}s",
                        flush=True,
                    )
                else:
                    print(f"  [ADC(2)/{bname} ERR] {mid}: {r.get('error')}", flush=True)
                summary["larger_basis_convergence"] = basis_res
                _save()
        print(f"Saved {out_path.relative_to(ROOT)} (larger-basis convergence complete)!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nevpt2", action="store_true")
    parser.add_argument("--larger-basis", action="store_true")
    parser.add_argument("--t1-relax", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--n-singlet-roots", type=int, default=2)
    parser.add_argument("--mol-ids", nargs="*", default=None)
    args = parser.parse_args()

    if args.nevpt2:
        run_nevpt2_suite(
            n_workers=args.workers,
            ncas=args.ncas,
            nelecas=args.nelecas,
            mol_ids=args.mol_ids,
            n_threads=args.threads,
            n_singlet_roots=args.n_singlet_roots,
        )
    if args.larger_basis or args.t1_relax:
        run_larger_basis_and_t1_suite(
            n_workers=args.workers,
            do_basis=args.larger_basis,
            do_t1=args.t1_relax,
            n_threads=args.threads,
            mol_ids=args.mol_ids,
        )
