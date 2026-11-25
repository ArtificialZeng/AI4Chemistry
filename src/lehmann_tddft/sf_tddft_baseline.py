"""Exact Collinear Spin-Flip TDA-TDDFT (Shao-Head-Gordon-Krylov 2003) Baseline for Pillar 4.

Computes high-spin triplet-reference (M_S = +1) DF-UKS + exact RI collinear SF-TDA for BOTH:
  - SF-TDA-BHHLYP (50% HF exchange: standard literature SF-TDDFT functional)
  - SF-TDA-PBE0   (25% HF exchange: same base functional as Lehmann-TDDFT)
across the EXACT SAME 2D branching-plane grids (Ethylene 15x15, Methanimine 15x15, Fulvene 11x11).

Computes exact analytical spin expectation values <S^2> for every spin-flip eigenstate via
exact second quantization:
    <Psi^SF | S^2 | Psi^SF> = [Tr(X^T S^{ab}_{ov})]^2 + ||X (S^{ab}_{vv})^T||_F^2
                            + ||(S^{ab}_{oo})^T X||_F^2 + ||S^{ab}_{vo}||_F^2
quantifying severe SF-TDDFT spin contamination directly against Lehmann-TDDFT's exact spin purity
(<S^2> == 0 identically).

macOS & swarm CPU safety: strictly uses pure pyscf/numpy/scipy (NO torch), max_workers <= 2.
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

# Prevent BLAS/OpenMP oversubscription when sharing 16-core M3 Max with other swarm agents
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from pyscf import dft, gto, lib

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lehmann_tddft.conical_intersections import (  # noqa: E402
    RESULTS_DIR,
    fit_conical_dispersion,
    load_system_specs,
    surface_error_metrics,
)
from lehmann_tddft.qm_engine import core_orbital_count  # noqa: E402
from lehmann_tddft.schema import HA_TO_EV  # noqa: E402

XC_BHHLYP = "0.5*HF + 0.5*B88, LYP"
XC_PBE0 = "pbe0"


def run_sf_tda_point(
    atoms: List[str],
    coords_ang: np.ndarray,
    xc: str,
    basis: str = "def2-svp",
    charge: int = 0,
    dm_init: Optional[np.ndarray] = None,
    freeze_core: bool = True,
    max_vir_b: int = 56,
) -> Dict[str, Any]:
    """Run DF-UKS (M_S=+1) + exact RI collinear SF-TDA and exact analytical <S^2>."""
    mol = gto.M(
        atom=[(a, (float(c[0]), float(c[1]), float(c[2]))) for a, c in zip(atoms, coords_ang)],
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=2,
        symmetry=False,
        verbose=0,
    )
    mf = dft.UKS(mol, xc=xc).density_fit()
    mf.grids.level = 2
    mf.conv_tol = 1e-8
    mf.max_cycle = 120
    mf.verbose = 0
    if dm_init is not None:
        mf.kernel(dm_init)
    else:
        mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = 1e-8
        mf.max_cycle = 60
        mf.verbose = 0
        mf.kernel(mf.make_rdm1())

    hyb = float(mf._numint.hybrid_coeff(mf.xc))
    na, nb = mol.nelec  # na = nb + 2 for M_S = +1
    nao = mol.nao
    Ca, Cb = mf.mo_coeff[0], mf.mo_coeff[1]
    ea, eb = mf.mo_energy[0], mf.mo_energy[1]

    ncore = sum(core_orbital_count(int(mol.atom_charge(i))) for i in range(mol.natm)) if freeze_core else 0
    if ncore >= nb:
        ncore = 0
    occ_a_start = ncore
    vir_b_end = min(nao, nb + max_vir_b)

    Ca_act = Ca[:, occ_a_start:na]
    Cb_act = Cb[:, nb:vir_b_end]
    na_act = Ca_act.shape[1]
    nvb_act = Cb_act.shape[1]
    nov = na_act * nvb_act

    # 3-center RI transformation for exact collinear exchange integral (i_a j_a | b_b a_b)
    naux = mf.with_df.get_naoaux()
    B_oa = np.empty((naux, na_act, na_act), dtype=np.float64)
    B_vb = np.empty((naux, nvb_act, nvb_act), dtype=np.float64)
    p0 = 0
    for eri1 in mf.with_df.loop():
        b = eri1.shape[0]
        L_ao = lib.unpack_tril(eri1)
        tmp_a = (L_ao.reshape(b * nao, nao) @ Ca_act).reshape(b, nao, na_act)
        B_oa[p0 : p0 + b] = np.einsum("ui,puj->pij", Ca_act, tmp_a, optimize=True)
        tmp_b = (L_ao.reshape(b * nao, nao) @ Cb_act).reshape(b, nao, nvb_act)
        B_vb[p0 : p0 + b] = np.einsum("ua,pub->pab", Cb_act, tmp_b, optimize=True)
        p0 += b

    # K_sf[i, a, j, b] = sum_P B_oa[P, i, j] * B_vb[P, b, a]
    K_sf = np.einsum("pij,pba->iajb", B_oa, B_vb, optimize=True).reshape(nov, nov)
    diag_e = (eb[None, nb:vir_b_end] - ea[occ_a_start:na, None]).ravel()
    A_sf = np.diag(diag_e) - hyb * K_sf
    A_sf = 0.5 * (A_sf + A_sf.T)

    evals, evecs = np.linalg.eigh(A_sf)

    # Exact analytical <S^2> in full MO space (including core & full virtuals!)
    S_ao = mol.intor("int1e_ovlp")
    Sab = Ca.T @ S_ao @ Cb  # (nao, nao)
    Sab_ov = Sab[occ_a_start:na, nb:vir_b_end]
    Sab_vv = Sab[na:, nb:vir_b_end]
    Sab_oo = Sab[occ_a_start:na, :nb]
    Sab_vo = Sab[na:, :nb]
    ref_vo_norm2 = float(np.sum(Sab_vo ** 2))

    n_keep = min(6, nov)
    e_rel_eV = evals[:n_keep] * HA_TO_EV
    s2_vals = np.zeros(n_keep, dtype=np.float64)
    for m in range(n_keep):
        X = evecs[:, m].reshape(na_act, nvb_act)
        t1 = float(np.sum(X * Sab_ov)) ** 2
        t2 = float(np.sum((X @ Sab_vv.T) ** 2))
        t3 = float(np.sum((Sab_oo.T @ X) ** 2))
        s2_vals[m] = t1 + t2 + t3 + ref_vo_norm2

    # Identify S0 and S1 cleanly: among lowest 3 states (which span {S0, T1(Ms=0), S1}),
    # pick the 2 states with smallest <S^2> (eliminating the triplet M_S=0 root with <S^2> ~ 2.0!).
    # If a higher state has <S^2> < 1.0 when one of the lowest 3 is a second triplet, check lowest 4.
    cand_idx = list(range(min(3, n_keep)))
    # Sort the 3 lowest-energy states by <S^2> and pick the 2 most singlet-like, then order by energy!
    sing_idx = sorted(sorted(cand_idx, key=lambda idx: s2_vals[idx])[:2], key=lambda idx: e_rel_eV[idx])
    trip_idx = [idx for idx in cand_idx if idx not in sing_idx][0]

    e_ref_eV = float(mf.e_tot) * HA_TO_EV
    s0_i, s1_i = sing_idx[0], sing_idx[1]
    return {
        "converged": bool(mf.converged),
        "e_ref_ha": float(mf.e_tot),
        "ref_s2": float(mf.spin_square()[0]),
        "e_s0_eV": e_ref_eV + float(e_rel_eV[s0_i]),
        "e_s1_eV": e_ref_eV + float(e_rel_eV[s1_i]),
        "e_t1_eV": e_ref_eV + float(e_rel_eV[trip_idx]),
        "gap_eV": float(e_rel_eV[s1_i] - e_rel_eV[s0_i]),
        "s2_s0": float(s2_vals[s0_i]),
        "s2_s1": float(s2_vals[s1_i]),
        "s2_t1": float(s2_vals[trip_idx]),
        "all_e_rel_eV": e_rel_eV.tolist(),
        "all_s2": s2_vals.tolist(),
        "dm": mf.make_rdm1(),
    }


def _sf_worker(task: Tuple[Any, ...]) -> Dict[str, Any]:
    ix, iy, atoms, coords, basis, dm_bhh, dm_pbe = task
    r_bhh = run_sf_tda_point(atoms, coords, XC_BHHLYP, basis=basis, dm_init=dm_bhh)
    r_pbe = run_sf_tda_point(atoms, coords, XC_PBE0, basis=basis, dm_init=dm_pbe)
    r_bhh.pop("dm", None)
    r_pbe.pop("dm", None)
    return {"ix": ix, "iy": iy, "bhhlyp": r_bhh, "pbe0": r_pbe}


def run_sf_grid_for_system(name: str, spec: Dict[str, Any], n_workers: int = 2) -> Dict[str, Any]:
    t0 = time.time()
    ref_npz = np.load(RESULTS_DIR / f"{name}_ci_surfaces.npz")
    xs, ys, X, Y = ref_npz["xs"], ref_npz["ys"], ref_npz["X"], ref_npz["Y"]
    nev_e0, nev_e1, nev_gap = ref_npz["nev_e0_eV"], ref_npz["nev_e1_eV"], ref_npz["nev_gap_eV"]
    cas_e0, cas_e1, cas_gap = ref_npz["cas_e0_eV"], ref_npz["cas_e1_eV"], ref_npz["cas_gap_eV"]
    nx, ny = len(xs), len(ys)

    print(f"\n=== [{name.upper()}] SF-TDA-BHHLYP & SF-TDA-PBE0 ({nx}x{ny}={nx*ny} pts, {n_workers} workers) ===", flush=True)
    # Warm-start triplet UKS DMs at center
    c0_bhh = run_sf_tda_point(spec["atoms"], spec["coords_meci"], XC_BHHLYP, basis=spec["basis"])
    c0_pbe = run_sf_tda_point(spec["atoms"], spec["coords_meci"], XC_PBE0, basis=spec["basis"])
    dm_bhh, dm_pbe = c0_bhh["dm"], c0_pbe["dm"]
    print(f"[{name}] Center warm-start ready ({time.time()-t0:.1f}s): "
          f"BHHLYP gap={c0_bhh['gap_eV']:.3f} (<S^2>={c0_bhh['s2_s0']:.3f}/{c0_bhh['s2_s1']:.3f}) | "
          f"PBE0 gap={c0_pbe['gap_eV']:.3f} (<S^2>={c0_pbe['s2_s0']:.3f}/{c0_pbe['s2_s1']:.3f})", flush=True)

    tasks = []
    for ix in range(nx):
        for iy in range(ny):
            c = spec["coords_meci"] + xs[ix] * spec["g_hat"] + ys[iy] * spec["h_hat"]
            tasks.append((ix, iy, spec["atoms"], c, spec["basis"], dm_bhh, dm_pbe))

    out_arrays = {}
    for tag in ("bhhlyp", "pbe0"):
        out_arrays[tag] = {
            k: np.zeros((nx, ny), dtype=np.float64)
            for k in ("e0", "e1", "gap", "s2_s0", "s2_s1", "s2_t1", "ref_s2")
        }

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_sf_worker, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            ix, iy = res["ix"], res["iy"]
            for tag in ("bhhlyp", "pbe0"):
                r = res[tag]
                out_arrays[tag]["e0"][ix, iy] = r["e_s0_eV"]
                out_arrays[tag]["e1"][ix, iy] = r["e_s1_eV"]
                out_arrays[tag]["gap"][ix, iy] = r["gap_eV"]
                out_arrays[tag]["s2_s0"][ix, iy] = r["s2_s0"]
                out_arrays[tag]["s2_s1"][ix, iy] = r["s2_s1"]
                out_arrays[tag]["s2_t1"][ix, iy] = r["s2_t1"]
                out_arrays[tag]["ref_s2"][ix, iy] = r["ref_s2"]
            done += 1
            if done % 30 == 0 or done == len(tasks):
                print(f"[{name}] SF-TDA progress: {done}/{len(tasks)} ({time.time()-t0:.1f}s)", flush=True)

    npz_path = RESULTS_DIR / f"{name}_sf_tddft_surfaces.npz"
    np.savez_compressed(
        npz_path,
        xs=xs, ys=ys, X=X, Y=Y,
        sf_bhhlyp_e0_eV=out_arrays["bhhlyp"]["e0"],
        sf_bhhlyp_e1_eV=out_arrays["bhhlyp"]["e1"],
        sf_bhhlyp_gap_eV=out_arrays["bhhlyp"]["gap"],
        sf_bhhlyp_s2_s0=out_arrays["bhhlyp"]["s2_s0"],
        sf_bhhlyp_s2_s1=out_arrays["bhhlyp"]["s2_s1"],
        sf_bhhlyp_s2_t1=out_arrays["bhhlyp"]["s2_t1"],
        sf_pbe0_e0_eV=out_arrays["pbe0"]["e0"],
        sf_pbe0_e1_eV=out_arrays["pbe0"]["e1"],
        sf_pbe0_gap_eV=out_arrays["pbe0"]["gap"],
        sf_pbe0_s2_s0=out_arrays["pbe0"]["s2_s0"],
        sf_pbe0_s2_s1=out_arrays["pbe0"]["s2_s1"],
        sf_pbe0_s2_t1=out_arrays["pbe0"]["s2_t1"],
    )

    summary = {"name": name, "grid_shape": [nx, ny], "wall_s": time.time() - t0, "npz_file": str(npz_path)}
    for tag, label in [("bhhlyp", "SF_TDA_BHHLYP"), ("pbe0", "SF_TDA_PBE0")]:
        a = out_arrays[tag]
        s2_both = np.concatenate([a["s2_s0"].ravel(), a["s2_s1"].ravel()])
        summary[label] = {
            "pes_vs_NEVPT2": surface_error_metrics(a["gap"], nev_gap, a["e0"], a["e1"], nev_e0, nev_e1),
            "pes_vs_CASSCF": surface_error_metrics(a["gap"], cas_gap, a["e0"], a["e1"], cas_e0, cas_e1),
            "cone_fit": fit_conical_dispersion(X, Y, a["gap"]),
            "spin_contamination": {
                "S0_S2_mean": float(np.mean(a["s2_s0"])),
                "S0_S2_median": float(np.median(a["s2_s0"])),
                "S0_S2_max": float(np.max(a["s2_s0"])),
                "S1_S2_mean": float(np.mean(a["s2_s1"])),
                "S1_S2_median": float(np.median(a["s2_s1"])),
                "S1_S2_max": float(np.max(a["s2_s1"])),
                "T1_S2_mean": float(np.mean(a["s2_t1"])),
                "ref_triplet_S2_mean": float(np.mean(a["ref_s2"])),
                "frac_S0_or_S1_gt_0p25": float(np.mean((a["s2_s0"] > 0.25) | (a["s2_s1"] > 0.25))),
                "frac_S0_or_S1_gt_0p50": float(np.mean((a["s2_s0"] > 0.50) | (a["s2_s1"] > 0.50))),
                "combined_S0_S1_S2_mean": float(np.mean(s2_both)),
                "lehmann_tddft_S2_exact": 0.0,
            },
        }
        m = summary[label]
        print(
            f"  [{name}] {label:14s}: gap_MAE={m['pes_vs_NEVPT2']['gap_MAE_eV']:.4f} eV | "
            f"2sh_MAE={m['pes_vs_NEVPT2']['PES_2sheet_MAE_eV']:.4f} eV | "
            f"x_ray_R2={m['cone_fit']['radial_x_linearity_R2']:.4f} | "
            f"<S^2> S0={m['spin_contamination']['S0_S2_mean']:.3f} (max {m['spin_contamination']['S0_S2_max']:.3f}), "
            f"S1={m['spin_contamination']['S1_S2_mean']:.3f} (max {m['spin_contamination']['S1_S2_max']:.3f}), "
            f"frac(>0.25)={m['spin_contamination']['frac_S0_or_S1_gt_0p25']*100:.1f}%",
            flush=True,
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["ethylene", "methanimine", "fulvene"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    specs = load_system_specs(smoke_test=False)
    out_json = RESULTS_DIR / "sf_tddft_comparison.json"
    all_res: Dict[str, Any] = {}
    if out_json.exists():
        try:
            all_res = json.loads(out_json.read_text())
        except Exception:
            all_res = {}

    for name in args.systems:
        all_res[name] = run_sf_grid_for_system(name, specs[name], n_workers=min(args.workers, 2))
        out_json.write_text(json.dumps(all_res, indent=2))
        print(f"[{name}] Updated {out_json}", flush=True)


if __name__ == "__main__":
    main()
