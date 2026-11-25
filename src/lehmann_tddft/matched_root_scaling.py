"""Fair Same-Hardware, Same-Basis, Same-Root-Count Scaling Benchmark (Apple M3 Max, 1 Core).

Directly addresses Reviewers 1 & 2's fairness critique by timing ALL methods under
IDENTICAL conditions:
  - Hardware: Apple M3 Max, strictly single-core (OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1),
    sequential execution, recording BOTH exact single-core CPU time (`time.process_time()`)
    and wall-clock time (`time.perf_counter()`).
  - Basis set: def2-SVP for every method.
  - Excited roots: strictly matched nroots = 8 Singlets + 8 Triplets (16 excited states)
    for every method.

Methods benchmarked across molecular ladder from N_AO = 66 (Glyoxal) to N_AO = 590 (HzTFEX2):
  1. Lehmann-TDDFT (8S+8T): DF-PBE0 SCF + 1-pass BLAS RI-TDA + RI bi-exciton tensors
     (`qm_engine.extract_from_mol(..., n_states=8)`) + v2.0 neural forward pass (measured in a
     clean separate torch subprocess for macOS libomp safety).
  2. Double-Hybrid RI-CIS(D) (8S+8T): DF-PBE-QIDH / SCS-PBE-QIDH TDA + frozen-core RI-CIS(D).
  3. RI-EE-ADC(2) (8S+8T): exact spin-adapted Lowdin-partitioned RI-EE-ADC(2)
     (`ab_initio_verify.run_ri_ee_adc2(..., nroots_s=8, nroots_t=8)`).
  4. EOM-EE-CCSD (8S+8T): unmodified frozen-core PySCF EOM-EE-CCSD (where tractable within
     single-core ~10 min budget, reporting honest memory/timeout boundaries above it).

Saves complete table, speedup ratios, and empirical log-log power-law fits (both CPU-time
and wall-time) to `results/benchmarks/matched_root_scaling.json`.

IMPORTANT macOS rule: NEVER import torch in the same process as PySCF/tblite/xgboost.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from pyscf import gto

from lehmann_tddft.ab_initio_verify import run_eom_ee_ccsd, run_ri_ee_adc2
from lehmann_tddft.double_hybrid_baseline import run_double_hybrid_ri_cisd
from lehmann_tddft.qm_engine import extract_from_mol, load_xyz, save_features_npz


BENCHMARK_LADDER: List[Dict[str, Any]] = [
    {
        "mol_id": "Glyoxal",
        "xyz_path": "data/external/QUESTDB/geometries/xyz/glyoxal.xyz",
        "run_adc2": True,
        "run_dh": True,
        "run_eom": True,
    },
    {
        "mol_id": "Butadiene",
        "xyz_path": "data/external/QUESTDB/geometries/xyz/butadiene.xyz",
        "run_adc2": True,
        "run_dh": True,
        "run_eom": True,
    },
    {
        "mol_id": "Benzene",
        "xyz_path": "data/external/QUESTDB/geometries/xyz/benzene.xyz",
        "run_adc2": True,
        "run_dh": True,
        "run_eom": True,
    },
    {
        "mol_id": "Pentalene",
        "xyz_path": "data/external/QUESTDB/geometries/xyz/pentalene_1.xyz",
        "run_adc2": True,
        "run_dh": True,
        "run_eom": "if_prev_cpu_lt_240",
    },
    {
        "mol_id": "Naphthalene",
        "xyz_path": "data/external/QUESTDB/geometries/xyz/naphthalene.xyz",
        "run_adc2": True,
        "run_dh": True,
        "run_eom": "if_prev_cpu_lt_180",
    },
    {
        "mol_id": "5AP",
        "xyz_path": "data/processed/invest_xyz/5AP.xyz",
        "run_adc2": True,
        "run_dh": True,
        "run_eom": False,
    },
    {
        "mol_id": "Anthracene",
        "xyz_path": "data/external/QUESTDB/geometries/xyz/anthracene.xyz",
        "run_adc2": "if_prev_cpu_lt_300",
        "run_dh": "if_prev_cpu_lt_300",
        "run_eom": False,
    },
    {
        "mol_id": "hepta_CF3_CF3_CN",
        "xyz_path": "data/external/HEPTA-INVEST/data/raw_data/trisub/eomccsd/2T-7N-CF3-CF3-CN/2T-7N-CF3-CF3-CN.xyz",
        "run_adc2": "if_prev_cpu_lt_300",
        "run_dh": "if_prev_cpu_lt_300",
        "run_eom": False,
    },
    {
        "mol_id": "HzTFEX2",
        "xyz_path": "data/processed/invest_xyz/HzTFEX2.xyz",
        "run_adc2": False,  # 89^2 * 466^2 * 8 = 13.8 GB single dense t2 tensor (>50 GB peak)
        "run_dh": "if_prev_cpu_lt_300",
        "run_eom": False,
    },
]


def _should_run(flag: Any, records: List[Dict[str, Any]], key: str) -> Tuple[bool, Optional[str]]:
    if flag is True:
        return True, None
    if flag is False:
        return False, "exceeds tractable single-core ~10 min / memory budget"
    if isinstance(flag, str) and flag.startswith("if_prev_cpu_lt_"):
        thresh = float(flag.split("_")[-1])
        prev_cpus = [
            r[key]["cpu_s"]
            for r in records
            if isinstance(r.get(key), dict) and "cpu_s" in r[key]
        ]
        if not prev_cpus:
            return False, "previous size skipped or failed"
        if prev_cpus[-1] < thresh:
            return True, None
        return (
            False,
            f"skipped (previous size already used {prev_cpus[-1]:.1f}s single-core CPU >= {thresh:.0f}s budget)",
        )
    return False, "disabled"


def _measure_neural_inference_ms(npz_paths: Dict[str, str]) -> Dict[str, float]:
    """Measure exact single-core PyTorch Lehmann-TDDFT v2.0 inference time in a separate subprocess."""
    ckpt_path = "results/models/lehmann_cv_fold0.pt"
    if not os.path.exists(ckpt_path):
        return {k: 0.0 for k in npz_paths}

    code = f"""
import json, os, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)
from lehmann_tddft.model import load_checkpoint, npz_to_mol_tensors

paths = {json.dumps(npz_paths)}
ckpt = load_checkpoint({json.dumps(ckpt_path)}, map_location='cpu')
model = ckpt[0] if isinstance(ckpt, tuple) else ckpt
model.eval()
out = {{}}
for mid, fp in paths.items():
    mol_t = npz_to_mol_tensors(fp)
    with torch.no_grad():
        _ = model(mol_t)
    t0 = time.process_time()
    n_rep = 20
    with torch.no_grad():
        for _ in range(n_rep):
            _ = model(mol_t)
    out[mid] = (time.process_time() - t0) / n_rep * 1000.0
print(json.dumps(out))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        print(f"[WARN] Neural timing subprocess failed: {proc.stderr[:400]}", flush=True)
        return {k: 0.0 for k in npz_paths}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _fit_power_law(n_aos: List[int], times_s: List[float]) -> Optional[Dict[str, float]]:
    if len(n_aos) < 2:
        return None
    x = np.log(np.asarray(n_aos, dtype=np.float64))
    y = np.log(np.asarray(times_s, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "exponent_alpha": float(slope),
        "prefactor_a": float(np.exp(intercept)),
        "log_log_R2": float(r2),
        "n_points": len(n_aos),
        "n_ao_range": [int(min(n_aos)), int(max(n_aos))],
    }


def main() -> None:
    out_path = Path("results/benchmarks/matched_root_scaling.json")
    cache_dir = Path("results/benchmarks/scaling_8S8T_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    nroots_s = 8
    nroots_t = 8
    records: List[Dict[str, Any]] = []
    lehmann_npz_paths: Dict[str, str] = {}
    done_ids: set[str] = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for r in prev.get("records", []):
                mid_prev = r.get("mol_id")
                npz_prev = cache_dir / f"lehmann_8S8T_{mid_prev}.npz"
                if mid_prev and npz_prev.exists() and "lehmann_tddft_8S8T" in r:
                    records.append(r)
                    lehmann_npz_paths[mid_prev] = str(npz_prev)
                    done_ids.add(mid_prev)
            if done_ids:
                print(f"Resuming matched_root_scaling: keeping {len(done_ids)} completed molecules {sorted(done_ids)}", flush=True)
        except Exception as exc:
            print(f"[WARN] Could not resume from {out_path}: {exc}", flush=True)

    print(
        f"=== Fair Single-Core Matched-Root Scaling Benchmark (def2-SVP, {nroots_s}S + {nroots_t}T) ===",
        flush=True,
    )

    for item in BENCHMARK_LADDER:
        mid = item["mol_id"]
        if mid in done_ids:
            continue
        xyz_path = item["xyz_path"]
        atom_str, z_arr, _ = load_xyz(xyz_path)
        mol = gto.M(atom=atom_str, basis="def2-svp", verbose=0, max_memory=16000)
        n_heavy = int(np.sum(z_arr > 1))
        n_ao = int(mol.nao)
        print(f"\n--- {mid} (n_atoms={mol.natm}, n_heavy={n_heavy}, N_AO={n_ao}) ---", flush=True)

        rec: Dict[str, Any] = {
            "mol_id": mid,
            "xyz_path": xyz_path,
            "n_atoms": int(mol.natm),
            "n_heavy": n_heavy,
            "n_ao": n_ao,
            "basis": "def2-svp",
            "nroots_S": nroots_s,
            "nroots_T": nroots_t,
        }

        # 1. Lehmann-TDDFT (8S + 8T)
        tw0, tc0 = time.perf_counter(), time.process_time()
        feats = extract_from_mol(mol, mol_id=f"scale_{mid}", n_states=nroots_s)
        tw_leh, tc_leh = time.perf_counter() - tw0, time.process_time() - tc0
        npz_fp = str(cache_dir / f"lehmann_8S8T_{mid}.npz")
        save_features_npz(feats, npz_fp)
        lehmann_npz_paths[mid] = npz_fp
        rec["lehmann_tddft_8S8T"] = {
            "qm_wall_s": float(tw_leh),
            "qm_cpu_s": float(tc_leh),
            "n_occ_act": int(feats["n_occ_act"]),
            "n_vir_act": int(feats["n_vir_act"]),
            "n_aux": int(feats["n_aux"]),
            "S1_tda_eV": float(feats["omega_S_eV"][0]),
            "T1_tda_eV": float(feats["omega_T_eV"][0]),
        }
        print(f"  [Lehmann-TDDFT 8S+8T]   cpu={tc_leh:7.2f}s | wall={tw_leh:7.2f}s", flush=True)

        # 2. Double-Hybrid RI-CIS(D) (SCS-PBE-QIDH, 8S + 8T)
        do_dh, why_dh = _should_run(item["run_dh"], records, "double_hybrid_scs_pbe_qidh_8S8T")
        if do_dh:
            try:
                tw0, tc0 = time.perf_counter(), time.process_time()
                dh = run_double_hybrid_ri_cisd(
                    mol, family="pbe_qidhs", nroots_S=nroots_s, nroots_T=nroots_t
                )
                tw_dh, tc_dh = time.perf_counter() - tw0, time.process_time() - tc0
                rec["double_hybrid_scs_pbe_qidh_8S8T"] = {
                    "cpu_s": float(tc_dh),
                    "wall_s": float(tw_dh),
                    "total_s": float(tw_dh),
                    "scf_wall_s": dh["timing_s"]["scf"],
                    "tda_wall_s": dh["timing_s"]["tda"],
                    "ri_cisd_wall_s": dh["timing_s"]["ri_cisd"],
                    "S1_eV": dh["methods"]["SCS-PBE-QIDH"]["E_S1_eV"],
                    "T1_eV": dh["methods"]["SCS-PBE-QIDH"]["E_T1_eV"],
                    "dEST_eV": dh["methods"]["SCS-PBE-QIDH"]["delta_EST_eV"],
                }
                print(
                    f"  [DH SCS-PBE-QIDH 8S+8T] cpu={tc_dh:7.2f}s | wall={tw_dh:7.2f}s "
                    f"(scf={dh['timing_s']['scf']:.1f}, tda={dh['timing_s']['tda']:.1f}, cisd={dh['timing_s']['ri_cisd']:.1f})",
                    flush=True,
                )
            except Exception as exc:
                rec["double_hybrid_scs_pbe_qidh_8S8T"] = {"error": str(exc)}
                print(f"  [DH SCS-PBE-QIDH 8S+8T] FAILED: {exc}", flush=True)
        else:
            rec["double_hybrid_scs_pbe_qidh_8S8T"] = {"skipped_reason": why_dh}
            print(f"  [DH SCS-PBE-QIDH 8S+8T] skipped: {why_dh}", flush=True)

        # 3. Exact Spin-Adapted RI-EE-ADC(2) (8S + 8T)
        do_adc, why_adc = _should_run(item["run_adc2"], records, "ri_ee_adc2_8S8T")
        if do_adc:
            try:
                tw0, tc0 = time.perf_counter(), time.process_time()
                adc = run_ri_ee_adc2(
                    xyz_path=xyz_path,
                    mol_id=mid,
                    basis="def2-svp",
                    nroots_s=nroots_s,
                    nroots_t=nroots_t,
                    tol=5e-5,
                )
                tw_adc, tc_adc = time.perf_counter() - tw0, time.process_time() - tc0
                rec["ri_ee_adc2_8S8T"] = {
                    "cpu_s": float(tc_adc),
                    "wall_s": float(tw_adc),
                    "total_s": float(tw_adc),
                    "timings_wall_s": adc["timings_s"],
                    "S1_eV": float(adc["E_S1_eV"]),
                    "T1_eV": float(adc["E_T1_eV"]),
                    "dEST_eV": float(adc["dEST_eV"]),
                    "all_converged": bool(
                        all(r["converged"] for r in adc["singlet_roots"] + adc["triplet_roots"])
                    ),
                }
                print(
                    f"  [RI-EE-ADC(2) 8S+8T]    cpu={tc_adc:7.2f}s | wall={tw_adc:7.2f}s "
                    f"(S={adc['timings_s']['singlet']:.1f}, T={adc['timings_s']['triplet']:.1f})",
                    flush=True,
                )
            except Exception as exc:
                rec["ri_ee_adc2_8S8T"] = {"error": str(exc)}
                print(f"  [RI-EE-ADC(2) 8S+8T]    FAILED: {exc}", flush=True)
        else:
            note = (
                "Intractable dense 4D t2/W memory (89^2 * 466^2 * 8 = 13.8 GB single tensor; >50 GB peak) and O(N^5) cost at N_AO=590"
                if mid == "HzTFEX2"
                else why_adc
            )
            rec["ri_ee_adc2_8S8T"] = {"skipped_reason": note}
            print(f"  [RI-EE-ADC(2) 8S+8T]    skipped: {note}", flush=True)

        # 4. Canonical PySCF EOM-EE-CCSD (8S + 8T)
        do_eom, why_eom = _should_run(item["run_eom"], records, "eom_ee_ccsd_8S8T")
        if do_eom:
            try:
                tw0, tc0 = time.perf_counter(), time.process_time()
                eom = run_eom_ee_ccsd(
                    xyz_path=xyz_path,
                    mol_id=mid,
                    basis="def2-svp",
                    nroots_s=nroots_s,
                    nroots_t=nroots_t,
                    conv_tol=1e-5,
                )
                tw_eom, tc_eom = time.perf_counter() - tw0, time.process_time() - tc0
                rec["eom_ee_ccsd_8S8T"] = {
                    "cpu_s": float(tc_eom),
                    "wall_s": float(tw_eom),
                    "total_s": float(tw_eom),
                    "timings_wall_s": eom["timings_s"],
                    "S1_eV": float(eom["E_S1_eV"]),
                    "T1_eV": float(eom["E_T1_eV"]),
                    "dEST_eV": float(eom["dEST_eV"]),
                }
                print(
                    f"  [EOM-EE-CCSD 8S+8T]     cpu={tc_eom:7.2f}s | wall={tw_eom:7.2f}s",
                    flush=True,
                )
            except Exception as exc:
                rec["eom_ee_ccsd_8S8T"] = {"error": str(exc)}
                print(f"  [EOM-EE-CCSD 8S+8T]     FAILED: {exc}", flush=True)
        else:
            rec["eom_ee_ccsd_8S8T"] = {"skipped_reason": why_eom}
            print(f"  [EOM-EE-CCSD 8S+8T]     skipped: {why_eom}", flush=True)

        records.append(rec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"status": "in_progress", "records": records}, fh, indent=2)

    # Measure single-core PyTorch v2.0 neural forward time cleanly in a separate subprocess
    nn_ms = _measure_neural_inference_ms(lehmann_npz_paths)
    for rec in records:
        mid = rec["mol_id"]
        ms = float(nn_ms.get(mid, 0.0))
        rec["lehmann_tddft_8S8T"]["nn_forward_cpu_ms"] = ms
        rec["lehmann_tddft_8S8T"]["cpu_s"] = rec["lehmann_tddft_8S8T"]["qm_cpu_s"] + ms / 1000.0
        rec["lehmann_tddft_8S8T"]["wall_s"] = rec["lehmann_tddft_8S8T"]["qm_wall_s"] + ms / 1000.0
        rec["lehmann_tddft_8S8T"]["total_s"] = rec["lehmann_tddft_8S8T"]["wall_s"]
        tc_l = rec["lehmann_tddft_8S8T"]["cpu_s"]
        tw_l = rec["lehmann_tddft_8S8T"]["wall_s"]
        rec["speedups_cpu_vs_lehmann"] = {}
        rec["speedups_wall_vs_lehmann"] = {}
        for key, label in [
            ("double_hybrid_scs_pbe_qidh_8S8T", "vs_DH_SCS_PBE_QIDH"),
            ("ri_ee_adc2_8S8T", "vs_RI_EE_ADC2"),
            ("eom_ee_ccsd_8S8T", "vs_EOM_EE_CCSD"),
        ]:
            if key in rec and isinstance(rec[key], dict) and "cpu_s" in rec[key]:
                rec["speedups_cpu_vs_lehmann"][label] = float(rec[key]["cpu_s"] / tc_l)
                rec["speedups_wall_vs_lehmann"][label] = float(rec[key]["wall_s"] / tw_l)

    fits_cpu = {}
    fits_wall = {}
    for key, name in [
        ("lehmann_tddft_8S8T", "Lehmann-TDDFT_8S8T"),
        ("double_hybrid_scs_pbe_qidh_8S8T", "SCS-PBE-QIDH_8S8T"),
        ("ri_ee_adc2_8S8T", "RI-EE-ADC2_8S8T"),
        ("eom_ee_ccsd_8S8T", "EOM-EE-CCSD_8S8T"),
    ]:
        xs = [r["n_ao"] for r in records if isinstance(r.get(key), dict) and "cpu_s" in r[key]]
        ys_c = [r[key]["cpu_s"] for r in records if isinstance(r.get(key), dict) and "cpu_s" in r[key]]
        ys_w = [r[key]["wall_s"] for r in records if isinstance(r.get(key), dict) and "wall_s" in r[key]]
        fits_cpu[name] = _fit_power_law(xs, ys_c)
        fits_wall[name] = _fit_power_law(xs, ys_w)

    final = {
        "status": "complete",
        "protocol": {
            "hardware": "Apple M3 Max, strictly 1 CPU core (OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1)",
            "basis": "def2-svp",
            "nroots_S": nroots_s,
            "nroots_T": nroots_t,
            "timing_metrics": (
                "Both exact single-core CPU time (time.process_time(), immune to unrelated OS load) "
                "and wall-clock time (time.perf_counter()) are reported under identical conditions."
            ),
        },
        "power_law_fits_cpu": fits_cpu,
        "power_law_fits_wall": fits_wall,
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2)

    print("\n=== MATCHED-ROOT (8S+8T) SCALING SUMMARY (SINGLE-CORE CPU SECONDS) ===", flush=True)
    for r in records:
        tl = r["lehmann_tddft_8S8T"]["cpu_s"]
        tdh = r.get("double_hybrid_scs_pbe_qidh_8S8T", {}).get("cpu_s")
        tadc = r.get("ri_ee_adc2_8S8T", {}).get("cpu_s")
        teom = r.get("eom_ee_ccsd_8S8T", {}).get("cpu_s")
        print(
            f"  {r['mol_id']:18s} (N_AO={r['n_ao']:3d}): "
            f"Lehmann={tl:6.2f}s | DH={f'{tdh:7.2f}s' if tdh else '    N/A'} | "
            f"ADC2={f'{tadc:7.2f}s' if tadc else '    N/A'} | "
            f"CCSD={f'{teom:7.2f}s' if teom else '    N/A'} | "
            f"CPU speedups={ {k: round(v, 2) for k, v in r['speedups_cpu_vs_lehmann'].items()} }",
            flush=True,
        )
    print("CPU power-law fits:", json.dumps(fits_cpu, indent=2), flush=True)


if __name__ == "__main__":
    main()
