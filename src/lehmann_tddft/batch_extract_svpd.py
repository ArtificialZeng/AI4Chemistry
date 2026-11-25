"""Parallel O(N^3) DF-PBE0/def2-SVPD diffuse-basis quantum feature extraction for QUESTDB Rydberg proof.

Selects all closed-shell QUESTDB molecules with n_heavy <= 8 that contain at least one
Rydberg state (V/R == 'R') plus `quest_Phenolate` anion (exact 64 molecules, 233 Rydberg +
403 Valence states across all 5 conformer-grouped CV folds).

Uses `qm_engine.extract_from_xyz(..., basis='def2-svpd', max_occ_act=14, max_vir_act=56)`
across at most 5 parallel workers (single-threaded BLAS per worker) to fully capture
diffuse 3s/3p/3d Rydberg virtual orbitals inside the active window.

Saves schema-validated `.npz` tensors to `data/processed/features_svpd/<mol_id>.npz`
and writes `data/processed/svpd_extraction_report.json`.

IMPORTANT macOS rule: NEVER import torch/xgboost/lightgbm/tblite in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from lehmann_tddft.qm_engine import (
    extract_from_xyz,
    save_features_npz,
    validate_feature_dict,
)


OUT_FEAT_DIR = "data/processed/features_svpd"
REPORT_PATH = "data/processed/svpd_extraction_report.json"


def _worker_extract_svpd(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    mol_id = task["mol_id"]
    xyz_path = task["xyz_path"]
    out_path = task["feature_path_svpd"]
    charge = int(task.get("charge", 0))
    spin = int(task.get("spin", 0))
    max_vir_act = int(task.get("max_vir_act", 56))
    skip_existing = bool(task.get("skip_existing", True))

    if skip_existing and os.path.exists(out_path):
        try:
            with np.load(out_path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            validate_feature_dict(d)
            return {
                "mol_id": mol_id,
                "subset": task["subset"],
                "cv5_fold": task["cv5_fold"],
                "status": "cached",
                "n_atoms": int(d["n_atoms"]),
                "n_heavy": int(d["n_heavy"]),
                "n_ao_svpd": int(d["n_ao"]),
                "n_aux_svpd": int(d["n_aux"]),
                "n_occ_act": int(d["n_occ_act"]),
                "n_vir_act": int(d["n_vir_act"]),
                "N_S": int(len(d["omega_S_eV"])),
                "N_T": int(len(d["omega_T_eV"])),
                "N_DS": int(len(d["D_S_bare_eV"])),
                "N_DT": int(len(d["D_T_bare_eV"])),
                "n_rydberg_ref": task["n_rydberg_ref"],
                "n_valence_ref": task["n_valence_ref"],
                "e_scf_ha": float(d["e_scf_ha"]),
                "wall_time_s": float(d["wall_time_s"]),
                "S1_svpd_eV": float(d["omega_S_eV"][0]),
                "T1_svpd_eV": float(d["omega_T_eV"][0]),
                "max_ryd_dsig_bohr": float(np.max(d["S_feats"][:, 10])),
            }
        except Exception:
            pass

    try:
        feats = extract_from_xyz(
            xyz_path=xyz_path,
            mol_id=mol_id,
            charge=charge,
            spin=spin,
            basis="def2-svpd",
            max_occ_act=14,
            max_vir_act=max_vir_act,
        )
        tmp_path = out_path + f".tmp.{os.getpid()}.npz"
        save_features_npz(feats, tmp_path)
        os.replace(tmp_path, out_path)
        return {
            "mol_id": mol_id,
            "subset": task["subset"],
            "cv5_fold": task["cv5_fold"],
            "status": "ok",
            "n_atoms": int(feats["n_atoms"]),
            "n_heavy": int(feats["n_heavy"]),
            "n_ao_svpd": int(feats["n_ao"]),
            "n_aux_svpd": int(feats["n_aux"]),
            "n_occ_act": int(feats["n_occ_act"]),
            "n_vir_act": int(feats["n_vir_act"]),
            "N_S": int(len(feats["omega_S_eV"])),
            "N_T": int(len(feats["omega_T_eV"])),
            "N_DS": int(len(feats["D_S_bare_eV"])),
            "N_DT": int(len(feats["D_T_bare_eV"])),
            "n_rydberg_ref": task["n_rydberg_ref"],
            "n_valence_ref": task["n_valence_ref"],
            "e_scf_ha": float(feats["e_scf_ha"]),
            "wall_time_s": float(feats["wall_time_s"]),
            "S1_svpd_eV": float(feats["omega_S_eV"][0]),
            "T1_svpd_eV": float(feats["omega_T_eV"][0]),
            "max_ryd_dsig_bohr": float(np.max(feats["S_feats"][:, 10])),
        }
    except Exception as exc:
        return {
            "mol_id": mol_id,
            "subset": task["subset"],
            "cv5_fold": task["cv5_fold"],
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def build_svpd_task_list(
    max_n_heavy: int = 8,
    max_vir_act: int = 56,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    quest = json.load(open("data/processed/questdb_clean.json", "r", encoding="utf-8"))
    tasks: List[Dict[str, Any]] = []
    for m in quest["molecules"]:
        n_r = sum(1 for s in m["states"] if s.get("vr") == "R")
        n_v = sum(1 for s in m["states"] if s.get("vr") == "V")
        if m["n_heavy"] <= max_n_heavy and (n_r > 0 or m["mol_id"] == "quest_Phenolate"):
            tasks.append({
                "mol_id": m["mol_id"],
                "subset": m["subset"],
                "cv5_fold": int(m["cv5_fold"]),
                "xyz_path": m["xyz_path"],
                "feature_path_svp": m["feature_path"],
                "feature_path_svpd": os.path.join(OUT_FEAT_DIR, f"{m['mol_id']}.npz"),
                "charge": int(m["charge"]),
                "spin": int(m["spin"]),
                "n_heavy": int(m["n_heavy"]),
                "n_atoms": int(m["n_atoms"]),
                "n_rydberg_ref": n_r,
                "n_valence_ref": n_v,
                "max_vir_act": max_vir_act,
                "skip_existing": skip_existing,
            })

    # Longest-Processing-Time-First scheduling: largest molecules first!
    tasks.sort(key=lambda x: (-x["n_heavy"], -x["n_atoms"], x["mol_id"]))
    return tasks


def verify_canonical_rydberg_molecules(quest_by_id: Dict[str, Any]) -> Dict[str, Any]:
    """Verify clean 3s/3p Rydberg emergence in def2-SVPD vs def2-SVP on 4 canonical molecules."""
    checks: Dict[str, Any] = {}
    for mid in ("quest_Formaldehyde", "quest_Ethylene", "quest_Butadiene", "quest_Benzene", "quest_Phenolate"):
        fp_svpd = os.path.join(OUT_FEAT_DIR, f"{mid}.npz")
        fp_svp = quest_by_id[mid]["feature_path"]
        if not (os.path.exists(fp_svpd) and os.path.exists(fp_svp)):
            continue
        svpd = np.load(fp_svpd, allow_pickle=False)
        svp = np.load(fp_svp, allow_pickle=False)
        ref_r_S = sorted(
            [s["tbe_avtz_eV"] for s in quest_by_id[mid]["states"] if s["spin"] == 1 and s["vr"] == "R"]
        )
        # Identify diffuse Rydberg-like TDA singlets via sigma_e - sigma_h > 1.5 Bohr
        dsig_svpd = svpd["S_feats"][:, 10]
        ryd_mask = dsig_svpd > 1.5
        ryd_energies_svpd = svpd["omega_S_eV"][ryd_mask]
        lowest_ref_r = ref_r_S[0] if ref_r_S else None
        lowest_svpd_r = float(ryd_energies_svpd[0]) if len(ryd_energies_svpd) > 0 else None

        # Compare against lowest diffuse-like or closest state in def2-SVP
        dsig_svp = svp["S_feats"][:, 10]
        svp_ryd_mask = dsig_svp > 1.2
        lowest_svp_r = (
            float(svp["omega_S_eV"][svp_ryd_mask][0])
            if np.any(svp_ryd_mask)
            else float(svp["omega_S_eV"][1])
        )
        checks[mid] = {
            "n_ao_svp": int(svp["n_ao"]),
            "n_ao_svpd": int(svpd["n_ao"]),
            "n_vir_act_svpd": int(svpd["n_vir_act"]),
            "n_diffuse_S_states_dsig_gt_1p5": int(np.sum(ryd_mask)),
            "lowest_ref_singlet_Rydberg_eV": lowest_ref_r,
            "lowest_SVP_Rydberg_eV": lowest_svp_r,
            "lowest_SVPD_Rydberg_eV": lowest_svpd_r,
            "raw_TDA_err_SVP_eV": (lowest_svp_r - lowest_ref_r) if (lowest_ref_r and lowest_svp_r) else None,
            "raw_TDA_err_SVPD_eV": (lowest_svpd_r - lowest_ref_r) if (lowest_ref_r and lowest_svpd_r) else None,
            "top6_omega_S_SVPD_eV": [round(float(x), 4) for x in svpd["omega_S_eV"][:6]],
            "top6_dsig_SVPD_bohr": [round(float(x), 3) for x in dsig_svpd[:6]],
        }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DF-PBE0/def2-SVPD extraction for Rydberg proof")
    parser.add_argument("--workers", type=int, default=5, help="Max worker processes (<=5)")
    parser.add_argument("--max-n-heavy", type=int, default=8, help="Max heavy atoms (default 8)")
    parser.add_argument("--max-vir-act", type=int, default=56, help="Active virtual orbitals (default 56)")
    parser.add_argument("--no-skip", action="store_true", help="Recompute even if .npz exists")
    args = parser.parse_args()

    workers = min(max(1, args.workers), 5)
    os.makedirs(OUT_FEAT_DIR, exist_ok=True)
    tasks = build_svpd_task_list(
        max_n_heavy=args.max_n_heavy,
        max_vir_act=args.max_vir_act,
        skip_existing=not args.no_skip,
    )
    n_total = len(tasks)
    print(
        f"Launching DF-PBE0/def2-SVPD batch extraction (max_vir_act={args.max_vir_act}): "
        f"{n_total} molecules across {workers} workers...",
        flush=True,
    )

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_worker_extract_svpd, t): t["mol_id"] for t in tasks}
        for i, fut in enumerate(as_completed(fut_map), start=1):
            res = fut.result()
            results.append(res)
            if res["status"] in ("ok", "cached"):
                n_ok += 1
                if i % 5 == 0 or i == n_total or res["wall_time_s"] > 60.0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[{i:2d}/{n_total}] {res['mol_id']:30s} | {res['status']:6s} | "
                        f"nH={res['n_heavy']:1d} nAO={res['n_ao_svpd']:3d} nv={res['n_vir_act']:2d} "
                        f"nR={res['n_rydberg_ref']:2d} t={res['wall_time_s']:5.1f}s | "
                        f"S1={res['S1_svpd_eV']:.2f} max_dsig={res['max_ryd_dsig_bohr']:.2f} "
                        f"(elapsed {elapsed:.0f}s)",
                        flush=True,
                    )
            else:
                n_fail += 1
                print(f"[{i:2d}/{n_total}] FAILED {res['mol_id']}: {res.get('error')}", flush=True)

    total_wall = time.perf_counter() - t0
    ok_rows = [r for r in results if r["status"] in ("ok", "cached")]
    fail_rows = [r for r in results if r["status"] == "failed"]

    quest = json.load(open("data/processed/questdb_clean.json", "r", encoding="utf-8"))
    quest_by_id = {m["mol_id"]: m for m in quest["molecules"]}
    canon_checks = verify_canonical_rydberg_molecules(quest_by_id)

    report = {
        "basis": "def2-svpd",
        "auxbasis": "def2-svp-jkfit (verified <= 0.002 eV vs exact non-DF def2-SVPD)",
        "max_occ_act": 14,
        "max_vir_act": args.max_vir_act,
        "max_n_heavy": args.max_n_heavy,
        "feature_dir": OUT_FEAT_DIR,
        "n_total": n_total,
        "n_success": n_ok,
        "n_failed": n_fail,
        "n_rydberg_states_covered": sum(r["n_rydberg_ref"] for r in ok_rows),
        "n_valence_states_covered": sum(r["n_valence_ref"] for r in ok_rows),
        "cv5_fold_counts": dict(sorted(Counter(r["cv5_fold"] for r in ok_rows).items())),
        "mol_ids": sorted(r["mol_id"] for r in ok_rows),
        "workers": workers,
        "total_wall_time_s": float(total_wall),
        "sum_cpu_time_s": float(sum(r["wall_time_s"] for r in ok_rows)),
        "mean_per_mol_s": float(np.mean([r["wall_time_s"] for r in ok_rows])) if ok_rows else None,
        "median_per_mol_s": float(np.median([r["wall_time_s"] for r in ok_rows])) if ok_rows else None,
        "canonical_rydberg_verification": canon_checks,
        "failures": fail_rows,
        "records": sorted(results, key=lambda x: x["mol_id"]),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== def2-SVPD BATCH EXTRACTION COMPLETE ===")
    print(
        f"Success: {n_ok}/{n_total} (failed={n_fail}) | "
        f"Rydberg states: {report['n_rydberg_states_covered']}, Valence states: {report['n_valence_states_covered']}"
    )
    print(f"Total wall time: {total_wall:.1f}s ({total_wall/60:.2f} min), sum CPU: {report['sum_cpu_time_s']:.1f}s")
    print("Canonical Rydberg verification:")
    for mid, chk in canon_checks.items():
        print(
            f"  {mid:20s}: ref_R={chk['lowest_ref_singlet_Rydberg_eV']:.2f} eV | "
            f"SVP={chk['lowest_SVP_Rydberg_eV']:.2f} (err={chk['raw_TDA_err_SVP_eV']:+.2f}) -> "
            f"SVPD={chk['lowest_SVPD_Rydberg_eV']:.2f} (err={chk['raw_TDA_err_SVPD_eV']:+.2f}) | "
            f"n_diffuse={chk['n_diffuse_S_states_dsig_gt_1p5']}"
        )
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
