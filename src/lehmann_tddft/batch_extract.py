"""Parallel O(N^3) quantum feature extraction for QUESTDB + HEPTA-INVEST44.

Uses ProcessPoolExecutor with at most 8 worker processes (single-threaded BLAS per worker)
and longest-processing-time-first (largest n_heavy first) scheduling for optimal load balance.
Saves validated `.npz` caches to `data/processed/features/<mol_id>.npz` and writes a
detailed scaling & verification report to `data/processed/extraction_report.json`.

IMPORTANT macOS rule: NEVER import torch/xgboost/lightgbm/tblite in this process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

# Ensure single-threaded BLAS per worker BEFORE importing numpy/pyscf in child processes
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


def _worker_extract(task: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and atomically save one molecule's .npz quantum tensors."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    mol_id = task["mol_id"]
    xyz_path = task["xyz_path"]
    out_path = task["feature_path"]
    charge = int(task.get("charge", 0))
    spin = int(task.get("spin", 0))
    skip_existing = bool(task.get("skip_existing", True))

    if skip_existing and os.path.exists(out_path):
        try:
            with np.load(out_path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            validate_feature_dict(d)
            return {
                "mol_id": mol_id,
                "dataset": task["dataset"],
                "subset": task["subset"],
                "status": "cached",
                "n_atoms": int(d["n_atoms"]),
                "n_heavy": int(d["n_heavy"]),
                "n_ao": int(d["n_ao"]),
                "n_aux": int(d["n_aux"]),
                "N_S": int(len(d["omega_S_eV"])),
                "N_T": int(len(d["omega_T_eV"])),
                "N_DS": int(len(d["D_S_bare_eV"])),
                "N_DT": int(len(d["D_T_bare_eV"])),
                "e_scf_ha": float(d["e_scf_ha"]),
                "wall_time_s": float(d["wall_time_s"]),
                "S1_tda_eV": float(d["omega_S_eV"][0]),
                "T1_tda_eV": float(d["omega_T_eV"][0]),
            }
        except Exception:
            pass  # Corrupt/incomplete cache; recompute cleanly

    try:
        feats = extract_from_xyz(
            xyz_path=xyz_path,
            mol_id=mol_id,
            charge=charge,
            spin=spin,
            basis="def2-svp",
        )
        tmp_path = out_path + f".tmp.{os.getpid()}.npz"
        save_features_npz(feats, tmp_path)
        os.replace(tmp_path, out_path)
        return {
            "mol_id": mol_id,
            "dataset": task["dataset"],
            "subset": task["subset"],
            "status": "ok",
            "n_atoms": int(feats["n_atoms"]),
            "n_heavy": int(feats["n_heavy"]),
            "n_ao": int(feats["n_ao"]),
            "n_aux": int(feats["n_aux"]),
            "N_S": int(len(feats["omega_S_eV"])),
            "N_T": int(len(feats["omega_T_eV"])),
            "N_DS": int(len(feats["D_S_bare_eV"])),
            "N_DT": int(len(feats["D_T_bare_eV"])),
            "e_scf_ha": float(feats["e_scf_ha"]),
            "wall_time_s": float(feats["wall_time_s"]),
            "S1_tda_eV": float(feats["omega_S_eV"][0]),
            "T1_tda_eV": float(feats["omega_T_eV"][0]),
        }
    except Exception as exc:
        return {
            "mol_id": mol_id,
            "dataset": task["dataset"],
            "subset": task["subset"],
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def build_task_list(skip_existing: bool = True) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    quest = json.load(open("data/processed/questdb_clean.json", "r", encoding="utf-8"))
    for m in quest["molecules"]:
        tasks.append({
            "mol_id": m["mol_id"],
            "dataset": "QUESTDB",
            "subset": m["subset"],
            "xyz_path": m["xyz_path"],
            "feature_path": m["feature_path"],
            "charge": m["charge"],
            "spin": m["spin"],
            "n_heavy": m["n_heavy"],
            "n_atoms": m["n_atoms"],
            "skip_existing": skip_existing,
        })

    hepta = json.load(open("data/processed/hepta_invest_clean.json", "r", encoding="utf-8"))
    for m in hepta["molecules"]:
        tasks.append({
            "mol_id": m["mol_id"],
            "dataset": "HEPTA-INVEST44",
            "subset": m["substitution_type"],
            "xyz_path": m["xyz_path"],
            "feature_path": m["feature_path"],
            "charge": m["charge"],
            "spin": m["spin"],
            "n_heavy": m["n_heavy"],
            "n_atoms": m["n_atoms"],
            "skip_existing": skip_existing,
        })

    # Longest-Processing-Time-First scheduling: largest molecules first!
    tasks.sort(key=lambda x: (-x["n_heavy"], -x["n_atoms"], x["mol_id"]))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch O(N^3) QM feature extraction")
    parser.add_argument("--workers", type=int, default=8, help="Max worker processes (<=8)")
    parser.add_argument("--no-skip", action="store_true", help="Recompute even if .npz exists")
    args = parser.parse_args()

    workers = min(max(1, args.workers), 8)
    os.makedirs("data/processed/features", exist_ok=True)
    tasks = build_task_list(skip_existing=not args.no_skip)
    n_total = len(tasks)
    print(f"Launching batch QM extraction: {n_total} molecules across {workers} workers...", flush=True)

    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_worker_extract, t): t["mol_id"] for t in tasks}
        for i, fut in enumerate(as_completed(fut_map), start=1):
            res = fut.result()
            results.append(res)
            if res["status"] in ("ok", "cached"):
                n_ok += 1
                if i % 10 == 0 or i == n_total or res["wall_time_s"] > 20.0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[{i:3d}/{n_total}] {res['mol_id']:36s} | {res['status']:6s} | "
                        f"nH={res['n_heavy']:2d} nAO={res['n_ao']:3d} N_DS={res['N_DS']:2d} "
                        f"t={res['wall_time_s']:5.1f}s | S1={res['S1_tda_eV']:.2f} T1={res['T1_tda_eV']:.2f} eV "
                        f"(elapsed {elapsed:.0f}s)",
                        flush=True,
                    )
            else:
                n_fail += 1
                print(
                    f"[{i:3d}/{n_total}] FAILED {res['mol_id']}: {res.get('error')}",
                    flush=True,
                )

    total_wall = time.perf_counter() - t0
    ok_rows = [r for r in results if r["status"] in ("ok", "cached")]
    fail_rows = [r for r in results if r["status"] == "failed"]

    # Compute scaling statistics by n_heavy bin
    scaling_bins: Dict[str, Any] = {}
    for label, lo, hi in [("1-5", 1, 5), ("6-9", 6, 9), ("10-13", 10, 13), ("14-24", 14, 24)]:
        sub = [r for r in ok_rows if lo <= r["n_heavy"] <= hi]
        if sub:
            times = [r["wall_time_s"] for r in sub]
            aos = [r["n_ao"] for r in sub]
            scaling_bins[label] = {
                "count": len(sub),
                "mean_n_ao": float(np.mean(aos)),
                "mean_wall_time_s": float(np.mean(times)),
                "median_wall_time_s": float(np.median(times)),
                "max_wall_time_s": float(np.max(times)),
            }

    report = {
        "n_total": n_total,
        "n_success": n_ok,
        "n_failed": n_fail,
        "n_questdb_success": sum(1 for r in ok_rows if r["dataset"] == "QUESTDB"),
        "n_hepta_success": sum(1 for r in ok_rows if r["dataset"] == "HEPTA-INVEST44"),
        "workers": workers,
        "total_wall_time_s": float(total_wall),
        "sum_cpu_time_s": float(sum(r["wall_time_s"] for r in ok_rows)),
        "mean_per_mol_s": float(np.mean([r["wall_time_s"] for r in ok_rows])) if ok_rows else None,
        "median_per_mol_s": float(np.median([r["wall_time_s"] for r in ok_rows])) if ok_rows else None,
        "scaling_by_n_heavy": scaling_bins,
        "failures": fail_rows,
        "records": sorted(results, key=lambda x: x["mol_id"]),
    }

    report_path = "data/processed/extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== BATCH EXTRACTION COMPLETE ===")
    print(f"Success: {n_ok}/{n_total} (QUESTDB: {report['n_questdb_success']}/143, HEPTA-INVEST44: {report['n_hepta_success']}/44)")
    print(f"Failed:  {n_fail}")
    print(f"Total wall time: {total_wall:.1f}s ({total_wall/60:.2f} min), sum per-mol time: {report['sum_cpu_time_s']:.1f}s")
    print("Scaling by n_heavy:", json.dumps(scaling_bins, indent=2))
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
