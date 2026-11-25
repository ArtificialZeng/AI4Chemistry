"""Finalize Pillar 3 (De Novo Bright Non-Heptazine INVEST Discovery) outputs,
synced strictly to the final leak-free 5-fold Lehmann-TDDFT ensemble
(`results/models/lehmann_ensemble_manifest.json`) while preserving the exact
pre-registered blind Top-20 selection and linking every ab initio number directly
to raw `results/invest_discovery/ab_initio_cache/{adc2,eomccsd}_*.json` files.

Produces:
  - results/invest_discovery/multi_scaffold_benchmark_eval.json
  - results/invest_discovery/virtual_screening_all.{json,csv}
  - results/invest_discovery/pareto_frontier.json
  - results/invest_discovery/quantum_phase_diagram.json
  - results/invest_discovery/blind_verification_summary.json
  - results/invest_discovery/top20_discoveries_table.csv
  - results/invest_discovery/timing_scaling_table.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC = ROOT / "data" / "processed"
BENCH_FEAT_DIR = DATA_PROC / "invest_features"
VIRT_FEAT_DIR = DATA_PROC / "virtual_bright_invest_features"
OUT_DIR = ROOT / "results" / "invest_discovery"
CACHE_DIR = OUT_DIR / "ab_initio_cache"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def compute_pareto_frontier(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inv = [r for r in records if r["delta_EST_eV"] < 0.0 and r["f_S1"] > 1e-4]
    pareto = []
    for a in inv:
        dominated = False
        for b in inv:
            if b["mol_id"] == a["mol_id"]:
                continue
            if (b["delta_EST_eV"] <= a["delta_EST_eV"] and b["f_S1"] >= a["f_S1"]) and (
                b["delta_EST_eV"] < a["delta_EST_eV"] or b["f_S1"] > a["f_S1"]
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    pareto.sort(key=lambda x: x["delta_EST_eV"])
    return pareto


def main() -> None:
    import torch
    from lehmann_tddft.model import LehmannEnsemble

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    known = json.loads((DATA_PROC / "known_benchmark_inchikeys.json").read_text())
    known_ik14 = set(known["all_known_ik14"])

    manifest_path = ROOT / "results/models/lehmann_ensemble_manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    model_version = manifest_data.get("version", "v2.0")
    ens = LehmannEnsemble.from_manifest(str(manifest_path), dtype=torch.float64)
    print(f"Loaded leak-free LehmannEnsemble ({model_version}, {len(ens.models)} checkpoints) from {manifest_path.relative_to(ROOT)}", flush=True)

    # -----------------------------------------------------------------------
    # 1. Re-evaluate all 34 Multi-Scaffold Benchmark Molecules
    # -----------------------------------------------------------------------
    bench = json.loads((DATA_PROC / "invest_multi_scaffold.json").read_text())
    bench_ik14 = {m["inchikey_14"] for m in bench["molecules"]}
    bench_preds = []
    timing_rows = []

    for m in bench["molecules"]:
        mid = m["mol_id"]
        npz_p = BENCH_FEAT_DIR / f"{mid}.npz"
        if not npz_p.exists():
            continue
        npz_raw = np.load(npz_p, allow_pickle=False)
        qm_wall_s = float(npz_raw["wall_time_s"])
        n_ao = int(npz_raw["n_ao"])
        t0 = time.perf_counter()
        pred = ens.predict_molecule(str(npz_p))
        inf_s = time.perf_counter() - t0

        rec: Dict[str, Any] = {
            **{k: m[k] for k in ["mol_id", "name", "family", "symmetry_class", "smiles", "inchikey", "inchikey_14", "formula", "n_heavy", "sa_score", "qed", "xyz_path"]},
            "n_ao": n_ao,
            "lit_ref": m.get("lit_ref", {}),
            "qm_wall_time_s": round(qm_wall_s, 3),
            "nn_infer_time_s": round(inf_s, 4),
            "lehmann_total_time_s": round(qm_wall_s + inf_s, 3),
            **{k: v for k, v in pred.items() if k != "mol_id"},
        }
        # Load directly from authoritative ab_initio_cache files!
        adc_f = CACHE_DIR / f"adc2_{mid}.json"
        if adc_f.exists():
            ad = json.loads(adc_f.read_text())
            if ad.get("ok"):
                rec["adc2_def2svp"] = ad
        eom_f = CACHE_DIR / f"eomccsd_{mid}.json"
        if eom_f.exists():
            ed = json.loads(eom_f.read_text())
            if ed.get("ok"):
                rec["eom_ccsd_def2svp"] = ed

        bench_preds.append(rec)
        timing_rows.append({
            "dataset": "multi_scaffold_benchmark",
            "mol_id": mid,
            "n_heavy": m["n_heavy"],
            "n_ao": n_ao,
            "lehmann_qm_s": round(qm_wall_s, 2),
            "lehmann_nn_s": round(inf_s, 4),
            "lehmann_total_s": round(qm_wall_s + inf_s, 2),
            "adc2_total_s": rec["adc2_def2svp"]["timings_s"]["total"] if "adc2_def2svp" in rec else None,
            "eom_ccsd_total_s": rec["eom_ccsd_def2svp"]["timings_s"]["total"] if "eom_ccsd_def2svp" in rec else None,
        })

    # Benchmark summary stats vs ADC(2) and canonical EOM-EE-CCSD
    adc_errs, tda_adc_errs, adc_sign_ok, tda_adc_sign_ok = [], [], 0, 0
    eom_errs, tda_eom_errs, eom_sign_ok, tda_eom_sign_ok = [], [], 0, 0
    eom_inv_errs = []
    for r in bench_preds:
        if "adc2_def2svp" in r:
            gt = r["adc2_def2svp"]["dEST_eV"]
            adc_errs.append(abs(r["delta_EST_eV"] - gt))
            tda_adc_errs.append(abs(r["delta_EST_bare_tda_eV"] - gt))
            adc_sign_ok += int((r["delta_EST_eV"] < 0) == (gt < 0))
            tda_adc_sign_ok += int((r["delta_EST_bare_tda_eV"] < 0) == (gt < 0))
        if "eom_ccsd_def2svp" in r:
            gt = r["eom_ccsd_def2svp"]["dEST_eV"]
            eom_errs.append(abs(r["delta_EST_eV"] - gt))
            tda_eom_errs.append(abs(r["delta_EST_bare_tda_eV"] - gt))
            eom_sign_ok += int((r["delta_EST_eV"] < 0) == (gt < 0))
            tda_eom_sign_ok += int((r["delta_EST_bare_tda_eV"] < 0) == (gt < 0))
            if gt < 0:
                eom_inv_errs.append(abs(r["delta_EST_eV"] - gt))

    bench_summary = {
        "ensemble_manifest": "results/models/lehmann_ensemble_manifest.json",
        "model_version": model_version,
        "leak_free_cv5_synced": True,
        "n_bench_screened": len(bench_preds),
        "vs_adc2_n": len(adc_errs),
        "vs_adc2_lehmann_dEST_MAE_eV": round(float(np.mean(adc_errs)), 4),
        "vs_adc2_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_adc_errs)), 4),
        "vs_adc2_lehmann_sign_acc": round(adc_sign_ok / len(adc_errs), 4),
        "vs_adc2_tda_pbe0_sign_acc": round(tda_adc_sign_ok / len(adc_errs), 4),
        "vs_eom_ccsd_n": len(eom_errs),
        "vs_eom_ccsd_lehmann_dEST_MAE_eV": round(float(np.mean(eom_errs)), 4),
        "vs_eom_ccsd_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_eom_errs)), 4),
        "vs_eom_ccsd_lehmann_sign_acc": round(eom_sign_ok / len(eom_errs), 4),
        "vs_eom_ccsd_tda_pbe0_sign_acc": round(tda_eom_sign_ok / len(eom_errs), 4),
        "vs_eom_ccsd_inverted_cores_n": len(eom_inv_errs),
        "vs_eom_ccsd_inverted_cores_MAE_eV": round(float(np.mean(eom_inv_errs)), 4),
        "molecules": bench_preds,
    }
    (OUT_DIR / "multi_scaffold_benchmark_eval.json").write_text(json.dumps(bench_summary, indent=2))
    print(
        f"Benchmark (n=34): vs ADC(2) (n={len(adc_errs)}) MAE={bench_summary['vs_adc2_lehmann_dEST_MAE_eV']} eV "
        f"(sign {adc_sign_ok}/{len(adc_errs)}); vs EOM-CCSD (n={len(eom_errs)}) MAE={bench_summary['vs_eom_ccsd_lehmann_dEST_MAE_eV']} eV "
        f"(inverted cores n={len(eom_inv_errs)} MAE={bench_summary['vs_eom_ccsd_inverted_cores_MAE_eV']} eV)",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # 2. Re-evaluate ALL 320 Extracted Virtual Screening Candidates
    # -----------------------------------------------------------------------
    virt = json.loads((DATA_PROC / "virtual_bright_invest_library.json").read_text())
    virt_preds = []
    t_scr0 = time.perf_counter()
    for m in virt["molecules"]:
        mid = m["mol_id"]
        npz_p = VIRT_FEAT_DIR / f"{mid}.npz"
        if not npz_p.exists():
            continue
        npz_raw = np.load(npz_p, allow_pickle=False)
        qm_wall_s = float(npz_raw["wall_time_s"])
        n_ao = int(npz_raw["n_ao"])
        t0 = time.perf_counter()
        pred = ens.predict_molecule(str(npz_p))
        inf_s = time.perf_counter() - t0
        rec = {
            **{k: m[k] for k in ["mol_id", "scaffold_family", "core_tag", "design_strategy", "smiles", "inchikey", "inchikey_14", "formula", "n_heavy", "n_atoms", "sa_score", "qed", "planarity_rmsd_A", "gfn2_dipole_debye", "xyz_path"]},
            "n_ao": n_ao,
            "qm_wall_time_s": round(qm_wall_s, 3),
            "nn_infer_time_s": round(inf_s, 4),
            "lehmann_total_time_s": round(qm_wall_s + inf_s, 3),
            **{k: v for k, v in pred.items() if k != "mol_id"},
        }
        virt_preds.append(rec)
    t_scr = time.perf_counter() - t_scr0

    # Strict novelty audit
    overlap_known = [r["mol_id"] for r in virt_preds if r["inchikey_14"] in known_ik14]
    assert len(overlap_known) == 0, f"Overlap with QUESTDB/HEPTA-INVEST: {overlap_known}"

    n_scr = len(virt_preds)
    n_tda_inv = sum(1 for r in virt_preds if r["delta_EST_bare_tda_eV"] < 0.0)
    n_leh_inv = sum(1 for r in virt_preds if r["delta_EST_eV"] < 0.0)
    n_leh_strong = sum(1 for r in virt_preds if r["delta_EST_eV"] <= -0.05)
    n_leh_bright01 = sum(1 for r in virt_preds if r["delta_EST_eV"] < 0.0 and r["f_S1"] >= 0.01)
    n_leh_bright02 = sum(1 for r in virt_preds if r["delta_EST_eV"] < 0.0 and r["f_S1"] >= 0.02)

    pareto = compute_pareto_frontier([r for r in virt_preds if r["inchikey_14"] not in bench_ik14])

    (OUT_DIR / "virtual_screening_all.json").write_text(
        json.dumps(
            {
                "ensemble_manifest": "results/models/lehmann_ensemble_manifest.json",
                "model_version": model_version,
                "leak_free_cv5_synced": True,
                "n_screened": n_scr,
                "verified_zero_overlap_questdb_heptainvest": True,
                "n_tda_pbe0_invest": n_tda_inv,
                "n_lehmann_invest": n_leh_inv,
                "n_lehmann_strong_invest_005": n_leh_strong,
                "n_lehmann_bright01_invest": n_leh_bright01,
                "n_lehmann_dipole_allowed01_invest": n_leh_bright01,
                "n_lehmann_bright02_invest": n_leh_bright02,
                "n_lehmann_dipole_allowed02_invest": n_leh_bright02,
                "ensemble_inference_total_s": round(t_scr, 3),
                "mean_qm_feature_time_s": round(float(np.mean([r["qm_wall_time_s"] for r in virt_preds])), 3),
                "molecules": virt_preds,
            },
            indent=2,
        )
    )

    csv_cols = [
        "mol_id", "scaffold_family", "core_tag", "design_strategy", "formula", "n_heavy", "n_ao",
        "sa_score", "qed", "smiles", "inchikey", "E_S1_eV", "E_T1_eV", "delta_EST_eV",
        "delta_EST_eV_std", "delta_EST_bare_tda_eV", "f_S1", "f_S1_std", "pct_T2_S1",
        "two_K_HL_eV", "delta_EST_static_eV", "delta_Sigma_ST_eV", "Sigma_S1_TT_eV",
        "Sigma_S1_TT_anti_eV", "Sigma_S1_SS_eV", "p_invest_ensemble", "lehmann_total_time_s",
    ]
    with open(OUT_DIR / "virtual_screening_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in virt_preds:
            w.writerow(r)

    (OUT_DIR / "pareto_frontier.json").write_text(json.dumps({"model_version": model_version, "n_pareto": len(pareto), "molecules": pareto}, indent=2))

    # 2D Quantum Phase Diagram combining benchmark + 320 novel virtual candidates
    phase_rows = []
    for src_tag, lst in [("multi_scaffold_benchmark", bench_preds), ("virtual_library", virt_preds)]:
        for r in lst:
            phase_rows.append({
                "source": src_tag,
                "mol_id": r["mol_id"],
                "family": r.get("family", r.get("scaffold_family")),
                "two_K_HL_eV": r["two_K_HL_eV"],
                "delta_EST_static_eV": r["delta_EST_static_eV"],
                "delta_Sigma_ST_eV": r["delta_Sigma_ST_eV"],
                "delta_EST_eV": r["delta_EST_eV"],
                "delta_EST_bare_tda_eV": r["delta_EST_bare_tda_eV"],
                "f_S1": r["f_S1"],
                "pct_T2_S1": r["pct_T2_S1"],
                "Sigma_S1_TT_eV": r["Sigma_S1_TT_eV"],
                "Sigma_S1_TT_anti_eV": r.get("Sigma_S1_TT_anti_eV", 0.0),
                "Sigma_S1_SS_eV": r["Sigma_S1_SS_eV"],
                "is_invest_pred": bool(r["delta_EST_eV"] < 0.0),
                "dynamically_inverted": bool(r["delta_EST_static_eV"] > 0.0 and r["delta_EST_eV"] < 0.0),
                "adc2_dEST_eV": r.get("adc2_def2svp", {}).get("dEST_eV"),
                "eom_dEST_eV": r.get("eom_ccsd_def2svp", {}).get("dEST_eV"),
            })
    (OUT_DIR / "quantum_phase_diagram.json").write_text(json.dumps({"model_version": model_version, "n_points": len(phase_rows), "points": phase_rows}, indent=2))

    # -----------------------------------------------------------------------
    # 3. Assemble Final Audited Blind Verification Summary (Top 20 ADC(2) + Expanded EOM-CCSD + NEVPT2)
    # -----------------------------------------------------------------------
    orig_sel = json.loads((OUT_DIR / "top20_selected_for_verification.json").read_text())
    virt_map = {r["mol_id"]: r for r in virt_preds}

    # Optional triple-method NEVPT2 lookup
    nev_map: Dict[str, Dict[str, Any]] = {}
    nev_flagship_n = 0
    nev_summary_path = OUT_DIR / "nevpt2_triple_verification.json"
    nev_summary_data: Dict[str, Any] = {}
    if nev_summary_path.exists():
        nev_summary_data = json.loads(nev_summary_path.read_text())
        for nr in nev_summary_data.get("molecules", []):
            if nr.get("nevpt2", {}).get("ok"):
                nev_map[nr["mol_id"]] = nr["nevpt2"]
                if nr.get("role") == "flagship_discovery":
                    nev_flagship_n += 1

    verified_rows = []
    adc_dEST_errs, tda_adc_errs = [], []
    adc_S1_errs, adc_T1_errs, adc_f_pred, adc_f_true = [], [], [], []
    eom_dEST_errs, tda_eom_errs, eom_S1_errs, eom_T1_errs, eom_f_pred, eom_f_true = [], [], [], [], [], []
    adc_times, eom_times, leh_times_adc, leh_times_eom = [], [], [], []
    adc_eom_offsets = []
    sel_shifts = []

    hits_invest_adc2 = 0
    hits_bright01_adc2 = 0
    hits_bright015_adc2 = 0
    hits_bright02_adc2 = 0
    hits_invest_eom = 0
    hits_bright01_eom = 0
    hits_dyn_inverted_pred = 0
    n_cur_pred_neg = 0
    n_cur_pred_neg_adc_inv = 0
    n_cur_pred_neg_eom_tot = 0
    n_cur_pred_neg_eom_inv = 0
    n_cur_eom_sign_ok = 0
    n_cur_adc_sign_ok = 0

    for rank, old_m in enumerate(orig_sel["molecules"], 1):
        mid = old_m["mol_id"]
        cur = virt_map[mid]
        assert cur["inchikey_14"] not in known_ik14 and cur["inchikey_14"] not in bench_ik14, f"Novelty fail on {mid}"

        row = dict(cur)
        row["rank"] = rank
        row["selection_delta_EST_eV"] = old_m.get("selection_delta_EST_eV", old_m["delta_EST_eV"])
        row["selection_f_S1"] = old_m.get("selection_f_S1", old_m["f_S1"])
        row["v1_delta_EST_eV"] = old_m.get("v1_delta_EST_eV", old_m["delta_EST_eV"])
        sel_shifts.append(abs(cur["delta_EST_eV"] - row["selection_delta_EST_eV"]))
        row["dynamically_inverted_pred"] = bool(cur["delta_EST_static_eV"] > 0.0 and cur["delta_EST_eV"] < 0.0)
        if row["dynamically_inverted_pred"]:
            hits_dyn_inverted_pred += 1
        if cur["delta_EST_eV"] < 0.0:
            n_cur_pred_neg += 1

        adc_path = CACHE_DIR / f"adc2_{mid}.json"
        eom_path = CACHE_DIR / f"eomccsd_{mid}.json"
        ar = json.loads(adc_path.read_text()) if adc_path.exists() else {}
        er = json.loads(eom_path.read_text()) if eom_path.exists() else {}
        nr = nev_map.get(mid, {})

        if ar.get("ok"):
            row["adc2_E_S1_eV"] = ar["E_S1_eV"]
            row["adc2_E_T1_eV"] = ar["E_T1_eV"]
            row["adc2_dEST_eV"] = ar["dEST_eV"]
            row["adc2_f_S1"] = ar["f_S1"]
            row["adc2_pct_T1_S1"] = ar["pct_T1_S1"]
            row["adc2_pct_T2_S1"] = ar["pct_T2_S1"]
            row["adc2_time_s"] = ar["timings_s"]["total"]
            adc_dEST_errs.append(abs(cur["delta_EST_eV"] - ar["dEST_eV"]))
            tda_adc_errs.append(abs(cur["delta_EST_bare_tda_eV"] - ar["dEST_eV"]))
            adc_S1_errs.append(abs(cur["E_S1_eV"] - ar["E_S1_eV"]))
            adc_T1_errs.append(abs(cur["E_T1_eV"] - ar["E_T1_eV"]))
            adc_f_pred.append(cur["f_S1"])
            adc_f_true.append(ar["f_S1"])
            adc_times.append(ar["timings_s"]["total"])
            leh_times_adc.append(cur["lehmann_total_time_s"])
            if (cur["delta_EST_eV"] < 0.0) == (ar["dEST_eV"] < 0.0):
                n_cur_adc_sign_ok += 1
            if ar["dEST_eV"] < 0.0:
                hits_invest_adc2 += 1
                if cur["delta_EST_eV"] < 0.0:
                    n_cur_pred_neg_adc_inv += 1
                if ar["f_S1"] >= 0.01:
                    hits_bright01_adc2 += 1
                if ar["f_S1"] >= 0.015:
                    hits_bright015_adc2 += 1
                if ar["f_S1"] >= 0.02:
                    hits_bright02_adc2 += 1

        if er.get("ok"):
            row["eom_E_S1_eV"] = er["E_S1_eV"]
            row["eom_E_T1_eV"] = er["E_T1_eV"]
            row["eom_dEST_eV"] = er["dEST_eV"]
            row["eom_f_S1_R1"] = er["f_S1_R1"]
            row["eom_pct_R1_S1"] = er["pct_R1_S1"]
            row["eom_time_s"] = er["timings_s"]["total"]
            eom_dEST_errs.append(abs(cur["delta_EST_eV"] - er["dEST_eV"]))
            tda_eom_errs.append(abs(cur["delta_EST_bare_tda_eV"] - er["dEST_eV"]))
            eom_S1_errs.append(abs(cur["E_S1_eV"] - er["E_S1_eV"]))
            eom_T1_errs.append(abs(cur["E_T1_eV"] - er["E_T1_eV"]))
            eom_f_pred.append(cur["f_S1"])
            eom_f_true.append(er["f_S1_R1"])
            eom_times.append(er["timings_s"]["total"])
            leh_times_eom.append(cur["lehmann_total_time_s"])
            if (cur["delta_EST_eV"] < 0.0) == (er["dEST_eV"] < 0.0):
                n_cur_eom_sign_ok += 1
            if cur["delta_EST_eV"] < 0.0:
                n_cur_pred_neg_eom_tot += 1
                if er["dEST_eV"] < 0.0:
                    n_cur_pred_neg_eom_inv += 1
            if ar.get("ok"):
                adc_eom_offsets.append(er["dEST_eV"] - ar["dEST_eV"])
            if er["dEST_eV"] < 0.0:
                hits_invest_eom += 1
                if er["f_S1_R1"] >= 0.01:
                    hits_bright01_eom += 1

        if nr.get("ok"):
            row["nevpt2_method"] = nr["method"]
            row["nevpt2_E_S1_eV"] = nr["E_S1_eV"]
            row["nevpt2_E_T1_eV"] = nr["E_T1_eV"]
            row["nevpt2_dEST_eV"] = nr["dEST_eV"]
            if "f_S1_nevpt2_energy" in nr and nr["f_S1_nevpt2_energy"] is not None:
                row["nevpt2_f_S1"] = nr["f_S1_nevpt2_energy"]

        timing_rows.append({
            "dataset": "blind_top20_discoveries",
            "mol_id": mid,
            "n_heavy": cur["n_heavy"],
            "n_ao": cur["n_ao"],
            "lehmann_qm_s": cur["qm_wall_time_s"],
            "lehmann_nn_s": cur["nn_infer_time_s"],
            "lehmann_total_s": cur["lehmann_total_time_s"],
            "adc2_total_s": ar["timings_s"]["total"] if ar.get("ok") else None,
            "eom_ccsd_total_s": er["timings_s"]["total"] if er.get("ok") else None,
        })
        verified_rows.append(row)

    n_adc = len(adc_dEST_errs)
    n_eom = len(eom_dEST_errs)
    w_adc_p = float(wilcoxon(adc_dEST_errs, tda_adc_errs).pvalue) if n_adc >= 5 else None
    w_eom_p = float(wilcoxon(eom_dEST_errs, tda_eom_errs).pvalue) if n_eom >= 5 else None
    rho_f_adc = float(spearmanr(adc_f_pred, adc_f_true).statistic) if n_adc >= 3 else None
    rho_f_eom = float(spearmanr(eom_f_pred, eom_f_true).statistic) if n_eom >= 3 else None

    mean_leh_adc = float(np.mean(leh_times_adc)) if leh_times_adc else 1.0
    mean_adc = float(np.mean(adc_times)) if adc_times else 0.0
    mean_leh_eom = float(np.mean(leh_times_eom)) if leh_times_eom else 1.0
    mean_eom = float(np.mean(eom_times)) if eom_times else 0.0
    speedups_eom = [et / lt for et, lt in zip(eom_times, leh_times_eom)] if eom_times else []
    mean_sel_shift = float(np.mean(sel_shifts)) if sel_shifts else 0.0

    summary = {
        "ensemble_manifest": "results/models/lehmann_ensemble_manifest.json",
        "model_version": model_version,
        "leak_free_cv5_synced": True,
        "selection_protocol": (
            f"Pre-registered blind Top-20 selected from 320 novel non-heptazine candidates "
            f"(0 IK14 overlap with QUESTDB, HEPTA-INVEST, or 34 Phase-A benchmark cores); "
            f"predictions refreshed with {model_version} 5-fold CV ensemble "
            f"({n_cur_pred_neg}/20 predicted negative; mean |shift| vs selection = {mean_sel_shift:.4f} eV)."
        ),
        "n_virtual_library_total_gfn2xtb": len(virt["molecules"]),
        "n_virtual_library_screened": n_scr,
        "n_top_selected": len(verified_rows),
        "n_top20_pred_neg_current": n_cur_pred_neg,
        "verified_zero_overlap_all_known": True,
        "n_dynamically_inverted_top20_pred": hits_dyn_inverted_pred,
        "n_adc2_verified": n_adc,
        "adc2_blind_invest_hits": hits_invest_adc2,
        "adc2_blind_invest_hit_rate": round(hits_invest_adc2 / n_adc, 4) if n_adc else None,
        "adc2_blind_invest_wilson95_ci": list(wilson_ci(hits_invest_adc2, n_adc)),
        "adc2_current_sign_acc": round(n_cur_adc_sign_ok / n_adc, 4) if n_adc else None,
        "adc2_current_pred_neg_precision": round(n_cur_pred_neg_adc_inv / n_cur_pred_neg, 4) if n_cur_pred_neg else None,
        "adc2_blind_bright01_invest_hits": hits_bright01_adc2,
        "adc2_blind_dipole_allowed01_invest_hits": hits_bright01_adc2,
        "adc2_blind_bright015_invest_hits": hits_bright015_adc2,
        "adc2_blind_bright02_invest_hits": hits_bright02_adc2,
        "adc2_lehmann_dEST_MAE_eV": round(float(np.mean(adc_dEST_errs)), 4) if n_adc else None,
        "adc2_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_adc_errs)), 4) if n_adc else None,
        "adc2_wilcoxon_p_vs_tda": w_adc_p,
        "adc2_lehmann_S1_MAE_eV": round(float(np.mean(adc_S1_errs)), 4) if n_adc else None,
        "adc2_lehmann_T1_MAE_eV": round(float(np.mean(adc_T1_errs)), 4) if n_adc else None,
        "adc2_f_S1_MAE": round(float(np.mean(np.abs(np.array(adc_f_pred) - np.array(adc_f_true)))), 5) if n_adc else None,
        "adc2_f_S1_spearman": round(rho_f_adc, 4) if rho_f_adc is not None else None,
        "n_eom_ccsd_verified": n_eom,
        "eom_ccsd_blind_invest_hits": hits_invest_eom,
        "eom_ccsd_blind_invest_hit_rate": round(hits_invest_eom / n_eom, 4) if n_eom else None,
        "eom_ccsd_blind_invest_wilson95_ci": list(wilson_ci(hits_invest_eom, n_eom)),
        "eom_ccsd_current_sign_acc": round(n_cur_eom_sign_ok / n_eom, 4) if n_eom else None,
        "eom_ccsd_current_pred_neg_n": n_cur_pred_neg_eom_tot,
        "eom_ccsd_current_pred_neg_hits": n_cur_pred_neg_eom_inv,
        "eom_ccsd_current_pred_neg_precision": round(n_cur_pred_neg_eom_inv / n_cur_pred_neg_eom_tot, 4) if n_cur_pred_neg_eom_tot else None,
        "eom_ccsd_blind_bright01_invest_hits": hits_bright01_eom,
        "eom_ccsd_blind_dipole_allowed01_invest_hits": hits_bright01_eom,
        "eom_ccsd_lehmann_dEST_MAE_eV": round(float(np.mean(eom_dEST_errs)), 4) if n_eom else None,
        "eom_ccsd_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_eom_errs)), 4) if n_eom else None,
        "eom_ccsd_wilcoxon_p_vs_tda": w_eom_p,
        "eom_ccsd_lehmann_S1_MAE_eV": round(float(np.mean(eom_S1_errs)), 4) if n_eom else None,
        "eom_ccsd_lehmann_T1_MAE_eV": round(float(np.mean(eom_T1_errs)), 4) if n_eom else None,
        "eom_ccsd_f_S1_MAE": round(float(np.mean(np.abs(np.array(eom_f_pred) - np.array(eom_f_true)))), 5) if n_eom else None,
        "eom_ccsd_f_S1_spearman": round(rho_f_eom, 4) if rho_f_eom is not None else None,
        "mean_eom_minus_adc2_dEST_offset_eV": round(float(np.mean(adc_eom_offsets)), 4) if adc_eom_offsets else None,
        "mean_lehmann_total_time_on_top20_s": round(mean_leh_adc, 2),
        "mean_adc2_time_on_top20_s": round(mean_adc, 2),
        "speedup_vs_custom_ri_adc2_mean": round(mean_adc / mean_leh_adc, 2),
        "mean_lehmann_total_time_on_eom_s": round(mean_leh_eom, 2),
        "mean_eom_ccsd_time_on_eom_s": round(mean_eom, 2),
        "speedup_vs_eom_ccsd_mean": round(mean_eom / mean_leh_eom, 1) if eom_times else None,
        "speedup_vs_eom_ccsd_median": round(float(np.median(speedups_eom)), 1) if speedups_eom else None,
        "speedup_vs_eom_ccsd_range": [round(min(speedups_eom), 1), round(max(speedups_eom), 1)] if speedups_eom else None,
        "n_nevpt2_flagship_verified": nev_flagship_n,
        "nevpt2_triple_verification_file": "results/invest_discovery/nevpt2_triple_verification.json" if nev_summary_path.exists() else None,
        "basis_and_adiabatic_check_file": "results/invest_discovery/basis_and_adiabatic_check.json" if (OUT_DIR / "basis_and_adiabatic_check.json").exists() else None,
        "larger_basis_and_t1_file": "results/invest_discovery/larger_basis_and_t1_relaxation.json" if (OUT_DIR / "larger_basis_and_t1_relaxation.json").exists() else None,
        "verified_discoveries": verified_rows,
    }
    (OUT_DIR / "blind_verification_summary.json").write_text(json.dumps(summary, indent=2))

    # Refresh top20_selected_for_verification.json with synced leak-free fields while preserving immutable selection fields
    (OUT_DIR / "top20_selected_for_verification.json").write_text(
        json.dumps(
            {
                "n_selected": len(verified_rows),
                "model_version": model_version,
                "verified_zero_overlap_questdb_heptainvest": True,
                "verified_zero_overlap_bench34": True,
                "leak_free_cv5_synced": True,
                "molecules": verified_rows,
            },
            indent=2,
        )
    )

    # Save publication-ready Top 20 CSV table
    tcols = [
        "rank", "mol_id", "scaffold_family", "design_strategy", "formula", "n_heavy", "n_ao",
        "sa_score", "qed", "smiles", "inchikey",
        "delta_EST_bare_tda_eV", "E_S1_eV", "E_T1_eV", "delta_EST_eV", "delta_EST_eV_std",
        "f_S1", "f_S1_std", "pct_T2_S1", "two_K_HL_eV", "delta_EST_static_eV", "delta_Sigma_ST_eV",
        "Sigma_S1_TT_eV", "Sigma_S1_TT_anti_eV", "Sigma_S1_SS_eV",
        "dynamically_inverted_pred",
        "adc2_E_S1_eV", "adc2_E_T1_eV", "adc2_dEST_eV", "adc2_f_S1", "adc2_pct_T2_S1", "adc2_time_s",
        "eom_E_S1_eV", "eom_E_T1_eV", "eom_dEST_eV", "eom_f_S1_R1", "eom_pct_R1_S1", "eom_time_s",
        "nevpt2_dEST_eV",
        "lehmann_total_time_s",
    ]
    with open(OUT_DIR / "top20_discoveries_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tcols, extrasaction="ignore")
        w.writeheader()
        for r in verified_rows:
            w.writerow(r)

    # Save timing & scaling CSV table for Fig 5a & audit
    with open(OUT_DIR / "timing_scaling_table.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "mol_id", "n_heavy", "n_ao", "lehmann_qm_s", "lehmann_nn_s", "lehmann_total_s", "adc2_total_s", "eom_ccsd_total_s"],
        )
        w.writeheader()
        for r in timing_rows:
            w.writerow(r)

    print("\n================ FINAL LEAK-FREE BLIND DISCOVERY SUMMARY ================", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "verified_discoveries"}, indent=2), flush=True)
    print("\n=== TOP 20 BLIND DISCOVERIES (Lehmann vs Blind ADC(2) & Blind EOM-EE-CCSD) ===", flush=True)
    for r in verified_rows:
        eom_str = f"EOM={r['eom_dEST_eV']:+.4f} (f={r['eom_f_S1_R1']:.4f})" if "eom_dEST_eV" in r else "EOM=   -   "
        print(
            f"  #{r['rank']:02d} {r['mol_id']:22s} | {r['scaffold_family']:22s} | "
            f"TDA={r['delta_EST_bare_tda_eV']:+.3f} | Leh={r['delta_EST_eV']:+.3f} (f={r['f_S1']:.4f}) | "
            f"ADC2={r['adc2_dEST_eV']:+.4f} (f={r['adc2_f_S1']:.4f}) | {eom_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
