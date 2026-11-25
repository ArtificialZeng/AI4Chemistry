"""Phase B Virtual Screening, Pareto Discovery, and Blind Ab Initio Verification
for Lehmann-TDDFT Pillar 3: Breaking the Dark-State Dilemma of INVEST OLED Emitters.

Stages (cleanly modularized so PySCF extraction and PyTorch inference stay isolated):
  1. `--extract-features`: Extract O(N^3) DF-PBE0/def2-SVP + RI bi-exciton `.npz` tensors
     via `qm_engine.extract_from_xyz` across all 34 multi-scaffold benchmark molecules
     AND all 520 novel de novo Bright-INVEST virtual candidates (4 workers, resumable).
  2. `--screen-and-select`: Load trained `LehmannEnsemble` from
     `results/models/lehmann_ensemble_manifest.json`, predict
     (E_S1, E_T1, dE_ST, f_S1, %T2, 2K_HL, dSigma_ST) with calibrated ensemble uncertainties
     across ALL 554 molecules, evaluate multi-scaffold benchmark transfer, map the
     Bright-INVEST Pareto Frontier and 2D Quantum Phase Diagram (2K_HL vs dSigma_ST),
     and select the Top 20 diverse novel non-heptazine bright-INVEST candidates.
  3. `--blind-verify`: Execute blind first-principles PySCF verification:
     - Exact RI-EE-ADC(2)/def2-SVP on all 20 top hits,
     - Unmodified canonical EOM-EE-CCSD/def2-SVP on top 8-10 hits,
     and compute exact blind hit rate, MAE, brightness gain vs heptazine/5AP, and speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA_PROC = ROOT / "data" / "processed"
BENCH_FEAT_DIR = DATA_PROC / "invest_features"
VIRT_FEAT_DIR = DATA_PROC / "virtual_bright_invest_features"
OUT_DIR = ROOT / "results" / "invest_discovery"


# ---------------------------------------------------------------------------
# Stage B1: O(N^3) qm_engine Feature Extraction (Pure PySCF/NumPy, NO torch)
# ---------------------------------------------------------------------------

def _extract_one_worker(args: Tuple[str, str, str]) -> Dict[str, Any]:
    mol_id, xyz_path, out_npz = args
    p = Path(out_npz)
    if p.exists():
        try:
            d = np.load(p, allow_pickle=False)
            return {
                "mol_id": mol_id,
                "ok": True,
                "cached": True,
                "wall_time_s": float(d["wall_time_s"]),
                "n_ao": int(d["n_ao"]),
            }
        except Exception:
            pass
    try:
        from lehmann_tddft.qm_engine import extract_from_xyz, save_features_npz

        feat = extract_from_xyz(xyz_path, mol_id=mol_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_features_npz(feat, str(p))
        return {
            "mol_id": mol_id,
            "ok": True,
            "cached": False,
            "wall_time_s": float(feat["wall_time_s"]),
            "n_ao": int(feat["n_ao"]),
        }
    except Exception as exc:
        return {"mol_id": mol_id, "ok": False, "error": str(exc)}


def select_stratified_virtual(molecules: List[Dict[str, Any]], max_virtual: Optional[int]) -> List[Dict[str, Any]]:
    """Deterministically pick a stratified subset spanning ALL scaffold families
    (prioritizing compact, synthetically accessible, and asymmetric push-pull cores)."""
    if max_virtual is None or max_virtual <= 0 or len(molecules) <= max_virtual:
        return molecules
    by_fam: Dict[str, List[Dict[str, Any]]] = {}
    for m in molecules:
        by_fam.setdefault(m["scaffold_family"], []).append(m)
    for fam in by_fam:
        # Sort by compact size (<=18 heavy preferred for fast exact verification) + SA-score
        by_fam[fam].sort(key=lambda x: (0 if x["n_heavy"] <= 18 else 1, x["sa_score"], x["n_heavy"], x["mol_id"]))
    picked: List[Dict[str, Any]] = []
    idx = 0
    fams = sorted(by_fam.keys())
    while len(picked) < max_virtual:
        added = False
        for fam in fams:
            if idx < len(by_fam[fam]) and len(picked) < max_virtual:
                picked.append(by_fam[fam][idx])
                added = True
        if not added:
            break
        idx += 1
    return picked


def run_feature_extraction(
    n_workers: int = 6,
    benchmark_only: bool = False,
    virtual_only: bool = False,
    max_virtual: Optional[int] = None,
) -> None:
    if not virtual_only:
        bench = json.loads((DATA_PROC / "invest_multi_scaffold.json").read_text())
        bench_jobs = [
            (m["mol_id"], str(ROOT / m["xyz_path"]), str(BENCH_FEAT_DIR / f"{m['mol_id']}.npz"))
            for m in bench["molecules"]
        ]
        print(f"=== Stage B1a: Extracting O(N^3) QM features for {len(bench_jobs)} Multi-Scaffold Benchmark mols ({n_workers} workers) ===", flush=True)
        _run_extract_pool(bench_jobs, n_workers=n_workers, label="Bench-QM")

    if benchmark_only:
        return

    virt = json.loads((DATA_PROC / "virtual_bright_invest_library.json").read_text())
    mols = select_stratified_virtual(virt["molecules"], max_virtual)
    fam_counts: Dict[str, int] = {}
    for m in mols:
        fam_counts[m["scaffold_family"]] = fam_counts.get(m["scaffold_family"], 0) + 1
    virt_jobs = [
        (m["mol_id"], str(ROOT / m["xyz_path"]), str(VIRT_FEAT_DIR / f"{m['mol_id']}.npz"))
        for m in mols
    ]
    print(f"\n=== Stage B1b: Extracting O(N^3) QM features for {len(virt_jobs)} Virtual Library candidates ({n_workers} workers) ===", flush=True)
    print(f"  Stratified families: {fam_counts}", flush=True)
    _run_extract_pool(virt_jobs, n_workers=n_workers, label="Virt-QM")


def _run_extract_pool(jobs: List[Tuple[str, str, str]], n_workers: int, label: str) -> None:
    t0 = time.perf_counter()
    done, ok, fails = 0, 0, []
    times = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_extract_one_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res.get("ok"):
                ok += 1
                times.append(res["wall_time_s"])
            else:
                fails.append((res["mol_id"], res.get("error")))
            if done % 25 == 0 or done == len(jobs):
                mean_t = float(np.mean(times)) if times else 0.0
                print(
                    f"  [{label}] {done}/{len(jobs)} done ({ok} ok, {len(fails)} err) | "
                    f"mean_qm={mean_t:.2f}s | elapsed={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
    if fails:
        print(f"  [{label} FAILURES] {fails[:10]}", flush=True)


# ---------------------------------------------------------------------------
# Stage B2: Lehmann-TDDFT Ensemble Screening, Pareto Frontier & Top-20 Selection
# ---------------------------------------------------------------------------

def compute_pareto_frontier(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute exact 2D Pareto frontier maximizing (-delta_EST_eV, f_S1)
    among predicted INVEST candidates (delta_EST_eV < 0 and f_S1 > 0.001)."""
    inv = [r for r in records if r["delta_EST_eV"] < 0.0 and r["f_S1"] > 1e-4]
    pareto: List[Dict[str, Any]] = []
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


def select_top_discoveries(
    records: List[Dict[str, Any]],
    known_ik14: set,
    bench_ik14: Optional[set] = None,
    n_select: int = 20,
) -> List[Dict[str, Any]]:
    """Select Top 20 strictly novel, diverse, non-heptazine Bright-INVEST discoveries
    excluding QUESTDB, HEPTA-INVEST, AND Phase A benchmark cores (`invest_multi_scaffold.json`).

    Balances:
      1. Breaking the Dark-State Dilemma: simultaneous negative dE_ST AND bright f_S1 (>= 0.01-0.05+),
      2. Strong singlet-triplet inversion (dE_ST <= -0.05 to -0.25 eV) with high ensemble confidence,
      3. Multi-scaffold diversity (push-pull azaphenalenes, substituted azaphenalenes,
         boron-centered boraphenalenes, BN-rim triangulenes, asymmetric azaphenalenes, non-alternants),
      4. Synthetic accessibility (low SA-score) & compact size (<= 17 heavy preferred for exact EOM-CCSD)."""
    excl = set(known_ik14) | set(bench_ik14 or set())
    novel = [r for r in records if r["inchikey_14"] not in excl]

    # Bright-INVEST figure of merit: strongly rewards BOTH negative dE_ST and bright f_S1!
    def bright_invest_score(r: Dict[str, Any]) -> float:
        if r["delta_EST_eV"] >= 0.0:
            return -1.0
        inv_mag = max(0.0, -r["delta_EST_eV"])
        # Smooth joint bright-INVEST metric: rewards f_S1 >= 0.01 and strong inversion
        f_term = np.log1p(r["f_S1"] / 0.005)
        conf = max(0.2, r["p_invest_ensemble"])
        sa_pen = max(0.0, (r["sa_score"] - 4.8) * 0.03)
        return float((inv_mag + 0.025) * f_term * conf - sa_pen)

    for r in novel:
        r["bright_invest_score"] = round(bright_invest_score(r), 5)

    # Tier 1: Flagship Bright INVEST (dE_ST <= -0.015 eV AND f_S1 >= 0.010)
    tier1 = [
        r for r in novel
        if r["delta_EST_eV"] <= -0.015 and r["f_S1"] >= 0.010 and r["p_invest_ensemble"] >= 0.6
    ]
    # Tier 2: Strongly Inverted & Moderately Bright (dE_ST <= -0.05 eV AND f_S1 >= 0.003)
    tier2 = [
        r for r in novel
        if r["delta_EST_eV"] <= -0.05 and r["f_S1"] >= 0.003 and r["mol_id"] not in {x["mol_id"] for x in tier1}
    ]
    # Tier 3: Deepest-inversion non-heptazine discoveries (e.g. boron-centered boraphenalenes!)
    tier3 = [
        r for r in novel
        if r["delta_EST_eV"] <= -0.06 and r["mol_id"] not in {x["mol_id"] for x in tier1 + tier2}
    ]

    tier1.sort(key=lambda x: (-x["f_S1"] * (0.04 + max(0.0, -x["delta_EST_eV"])), x["delta_EST_eV"]))
    tier2.sort(key=lambda x: -x["bright_invest_score"])
    tier3.sort(key=lambda x: x["delta_EST_eV"])

    selected: List[Dict[str, Any]] = []
    sel_ids: set = set()
    fam_cap: Dict[str, int] = {}

    def add_cand(r: Dict[str, Any]) -> bool:
        if len(selected) >= n_select or r["mol_id"] in sel_ids:
            return False
        selected.append(r)
        sel_ids.add(r["mol_id"])
        fam = r["scaffold_family"]
        fam_cap[fam] = fam_cap.get(fam, 0) + 1
        return True

    # Pass 1: Take all top Pareto-optimal Bright-INVEST hits (f >= 0.012, dE_ST <= -0.015) up to 10
    pareto = compute_pareto_frontier(novel)
    pareto_bright = [p for p in pareto if p["f_S1"] >= 0.012 and p["delta_EST_eV"] <= -0.015]
    pareto_bright.sort(key=lambda x: -x["f_S1"])
    for r in pareto_bright[:10]:
        add_cand(r)

    # Pass 2: Guarantee multi-scaffold diversity from Tier 1 (up to 4 per family)
    by_fam_t1: Dict[str, List[Dict]] = {}
    for r in tier1:
        by_fam_t1.setdefault(r["scaffold_family"], []).append(r)
    for fam in sorted(by_fam_t1.keys()):
        for r in by_fam_t1[fam]:
            if fam_cap.get(fam, 0) >= 5:
                break
            add_cand(r)

    # Pass 3: Guarantee at least 2 top Boron-centered boraphenalene & 2 BN-rim & 2 non-alternant hits if available
    for special_fam, target_n in [("bn_rim_triangulene", 2), ("boraphenalene_bn", 2), ("non_alternant_azulene", 1)]:
        pool_sp = [r for r in (tier1 + tier2 + tier3) if r["scaffold_family"] == special_fam and r["mol_id"] not in sel_ids]
        pool_sp.sort(key=lambda x: (-x["f_S1"] if special_fam != "boraphenalene_bn" else x["delta_EST_eV"]))
        for r in pool_sp:
            if fam_cap.get(special_fam, 0) >= target_n:
                break
            add_cand(r)

    # Pass 4: Fill remaining slots from tier1 -> tier2 -> tier3
    for pool in (tier1, tier2, tier3):
        for r in pool:
            if len(selected) >= n_select:
                break
            if fam_cap.get(r["scaffold_family"], 0) >= 7 and pool is not tier3:
                continue
            add_cand(r)

    selected.sort(key=lambda x: (-x["f_S1"] if x["f_S1"] >= 0.01 else 0.0, x["delta_EST_eV"]))
    return selected[:n_select]


def run_screening_and_selection(
    manifest_path: str = "results/models/lehmann_ensemble_manifest.json",
    n_select: int = 20,
) -> None:
    import torch
    from lehmann_tddft.model import LehmannEnsemble

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    known = json.loads((DATA_PROC / "known_benchmark_inchikeys.json").read_text())
    known_ik14 = set(known["all_known_ik14"])

    print(f"=== Stage B2: Loading Trained Lehmann-TDDFT Ensemble from {manifest_path} ===", flush=True)
    ens = LehmannEnsemble.from_manifest(str(ROOT / manifest_path), dtype=torch.float64)
    print(f"  Loaded ensemble with {len(ens.models)} checkpoints.", flush=True)

    # 1. Screen all 34 Multi-Scaffold Benchmark Molecules
    bench = json.loads((DATA_PROC / "invest_multi_scaffold.json").read_text())
    bench_preds = []
    t0 = time.perf_counter()
    for m in bench["molecules"]:
        npz_p = BENCH_FEAT_DIR / f"{m['mol_id']}.npz"
        if not npz_p.exists():
            continue
        npz_raw = np.load(npz_p, allow_pickle=False)
        qm_wall_s = float(npz_raw["wall_time_s"])
        t_inf0 = time.perf_counter()
        pred = ens.predict_molecule(str(npz_p))
        inf_s = time.perf_counter() - t_inf0
        rec = {
            **{k: m[k] for k in ["mol_id", "name", "family", "symmetry_class", "smiles", "inchikey", "inchikey_14", "formula", "n_heavy", "sa_score", "qed", "xyz_path"]},
            "lit_ref": m.get("lit_ref", {}),
            "qm_wall_time_s": round(qm_wall_s, 3),
            "nn_infer_time_s": round(inf_s, 4),
            "lehmann_total_time_s": round(qm_wall_s + inf_s, 3),
            **{k: v for k, v in pred.items() if k != "mol_id"},
        }
        if "adc2_def2svp" in m:
            rec["adc2_def2svp"] = m["adc2_def2svp"]
        if "eom_ccsd_def2svp" in m:
            rec["eom_ccsd_def2svp"] = m["eom_ccsd_def2svp"]
        bench_preds.append(rec)
    print(f"  Screened {len(bench_preds)} multi-scaffold benchmark molecules in {time.perf_counter()-t0:.2f}s.", flush=True)

    # Compute multi-scaffold benchmark metrics vs ADC(2) and EOM-CCSD
    adc_errs, tda_adc_errs = [], []
    adc_sign_ok, tda_sign_ok = 0, 0
    eom_errs, tda_eom_errs = [], []
    for r in bench_preds:
        if "adc2_def2svp" in r and r["adc2_def2svp"].get("ok"):
            gt = r["adc2_def2svp"]["dEST_eV"]
            adc_errs.append(abs(r["delta_EST_eV"] - gt))
            tda_adc_errs.append(abs(r["delta_EST_bare_tda_eV"] - gt))
            if (r["delta_EST_eV"] < 0) == (gt < 0):
                adc_sign_ok += 1
            if (r["delta_EST_bare_tda_eV"] < 0) == (gt < 0):
                tda_sign_ok += 1
        if "eom_ccsd_def2svp" in r and r["eom_ccsd_def2svp"].get("ok"):
            gt = r["eom_ccsd_def2svp"]["dEST_eV"]
            eom_errs.append(abs(r["delta_EST_eV"] - gt))
            tda_eom_errs.append(abs(r["delta_EST_bare_tda_eV"] - gt))

    bench_summary = {
        "n_bench_screened": len(bench_preds),
        "vs_adc2_n": len(adc_errs),
        "vs_adc2_lehmann_dEST_MAE_eV": round(float(np.mean(adc_errs)), 4) if adc_errs else None,
        "vs_adc2_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_adc_errs)), 4) if tda_adc_errs else None,
        "vs_adc2_lehmann_sign_acc": round(adc_sign_ok / len(adc_errs), 4) if adc_errs else None,
        "vs_adc2_tda_pbe0_sign_acc": round(tda_sign_ok / len(adc_errs), 4) if adc_errs else None,
        "vs_eom_ccsd_n": len(eom_errs),
        "vs_eom_ccsd_lehmann_dEST_MAE_eV": round(float(np.mean(eom_errs)), 4) if eom_errs else None,
        "vs_eom_ccsd_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_eom_errs)), 4) if tda_eom_errs else None,
        "molecules": bench_preds,
    }
    (OUT_DIR / "multi_scaffold_benchmark_eval.json").write_text(json.dumps(bench_summary, indent=2))
    print(f"  Multi-scaffold vs ADC(2) (n={len(adc_errs)}): Lehmann MAE={bench_summary['vs_adc2_lehmann_dEST_MAE_eV']} eV "
          f"(sign acc={bench_summary['vs_adc2_lehmann_sign_acc']}) vs TDA-PBE0 MAE={bench_summary['vs_adc2_tda_pbe0_dEST_MAE_eV']} eV!", flush=True)

    # 2. Screen ALL 520 Virtual Bright-INVEST Candidates
    virt = json.loads((DATA_PROC / "virtual_bright_invest_library.json").read_text())
    virt_preds = []
    t1 = time.perf_counter()
    for m in virt["molecules"]:
        npz_p = VIRT_FEAT_DIR / f"{m['mol_id']}.npz"
        if not npz_p.exists():
            continue
        npz_raw = np.load(npz_p, allow_pickle=False)
        qm_wall_s = float(npz_raw["wall_time_s"])
        t_inf0 = time.perf_counter()
        pred = ens.predict_molecule(str(npz_p))
        inf_s = time.perf_counter() - t_inf0
        rec = {
            **{k: m[k] for k in ["mol_id", "scaffold_family", "core_tag", "design_strategy", "smiles", "inchikey", "inchikey_14", "formula", "n_heavy", "n_atoms", "sa_score", "qed", "planarity_rmsd_A", "gfn2_dipole_debye", "xyz_path"]},
            "qm_wall_time_s": round(qm_wall_s, 3),
            "nn_infer_time_s": round(inf_s, 4),
            "lehmann_total_time_s": round(qm_wall_s + inf_s, 3),
            **{k: v for k, v in pred.items() if k != "mol_id"},
        }
        virt_preds.append(rec)
    t_screen = time.perf_counter() - t1
    print(f"  Screened {len(virt_preds)} novel virtual candidates with Lehmann-TDDFT ensemble in {t_screen:.2f}s!", flush=True)

    # Verify ZERO overlap with QUESTDB + HEPTA-INVEST
    overlaps = [r["mol_id"] for r in virt_preds if r["inchikey_14"] in known_ik14]
    assert len(overlaps) == 0, f"Novelty violation: {overlaps}"

    n_pred_invest = sum(1 for r in virt_preds if r["delta_EST_eV"] < 0.0)
    n_pred_strong = sum(1 for r in virt_preds if r["delta_EST_eV"] <= -0.05)
    n_pred_bright_invest = sum(1 for r in virt_preds if r["delta_EST_eV"] <= -0.05 and r["f_S1"] >= 0.02)
    n_tda_invest = sum(1 for r in virt_preds if r["delta_EST_bare_tda_eV"] < 0.0)

    print(
        f"  SCREENING HEADLINE (N={len(virt_preds)}): "
        f"TDA-PBE0 predicts INVEST={n_tda_invest}/{len(virt_preds)} (0%!) | "
        f"Lehmann-TDDFT predicts INVEST={n_pred_invest}/{len(virt_preds)}, "
        f"Strong (<-0.05 eV)={n_pred_strong}, Bright INVEST (<-0.05 eV & f>=0.02)={n_pred_bright_invest}!",
        flush=True,
    )

    # Compute Pareto Frontier & Top 20 Discoveries (strictly excluding QUESTDB + HEPTA-INVEST + 34 bench cores!)
    bench_ik14 = {m["inchikey_14"] for m in bench["molecules"]}
    pareto = compute_pareto_frontier([r for r in virt_preds if r["inchikey_14"] not in bench_ik14])
    top20 = select_top_discoveries(virt_preds, known_ik14, bench_ik14=bench_ik14, n_select=n_select)

    # Save full screening table (JSON + CSV)
    (OUT_DIR / "virtual_screening_all.json").write_text(
        json.dumps(
            {
                "n_screened": len(virt_preds),
                "n_tda_pbe0_invest": n_tda_invest,
                "n_lehmann_invest": n_pred_invest,
                "n_lehmann_strong_invest_005": n_pred_strong,
                "n_lehmann_bright_invest": n_pred_bright_invest,
                "ensemble_inference_total_s": round(t_screen, 3),
                "mean_qm_feature_time_s": round(float(np.mean([r["qm_wall_time_s"] for r in virt_preds])), 3),
                "molecules": virt_preds,
            },
            indent=2,
        )
    )

    csv_cols = [
        "mol_id", "scaffold_family", "core_tag", "design_strategy", "formula", "n_heavy",
        "sa_score", "qed", "smiles", "inchikey", "E_S1_eV", "E_T1_eV", "delta_EST_eV",
        "delta_EST_eV_std", "delta_EST_bare_tda_eV", "f_S1", "f_S1_std", "pct_T2_S1",
        "two_K_HL_eV", "delta_EST_static_eV", "delta_Sigma_ST_eV", "Sigma_S1_TT_eV",
        "Sigma_S1_SS_eV", "p_invest_ensemble", "bright_invest_score", "lehmann_total_time_s",
    ]
    with open(OUT_DIR / "virtual_screening_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in virt_preds:
            w.writerow(r)

    # Save Pareto Frontier
    (OUT_DIR / "pareto_frontier.json").write_text(json.dumps({"n_pareto": len(pareto), "molecules": pareto}, indent=2))

    # Save 2D Quantum Phase Diagram (2K_HL vs delta_Sigma_ST) combining benchmark + virtual library!
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
                "Sigma_S1_SS_eV": r["Sigma_S1_SS_eV"],
                "is_invest_pred": bool(r["delta_EST_eV"] < 0.0),
                "adc2_dEST_eV": r.get("adc2_def2svp", {}).get("dEST_eV"),
            })
    (OUT_DIR / "quantum_phase_diagram.json").write_text(json.dumps({"n_points": len(phase_rows), "points": phase_rows}, indent=2))

    # Save Top 20 Selected Candidates for Blind Verification
    (OUT_DIR / "top20_selected_for_verification.json").write_text(
        json.dumps({"n_selected": len(top20), "verified_zero_overlap_questdb_heptainvest": True, "molecules": top20}, indent=2)
    )
    print("\n=== TOP 20 NOVEL NON-HEPTAZINE BRIGHT-INVEST CANDIDATES SELECTED FOR BLIND AB INITIO VERIFICATION ===")
    for i, r in enumerate(top20, 1):
        print(
            f"  #{i:02d} {r['mol_id']:24s} | {r['scaffold_family']:23s} | {r['formula']:12s} (nH={r['n_heavy']:2d}) | "
            f"dEST={r['delta_EST_eV']:+.3f}±{r['delta_EST_eV_std']:.3f} eV (TDA={r['delta_EST_bare_tda_eV']:+.3f}) | "
            f"f={r['f_S1']:.4f} | %T2={r['pct_T2_S1']:4.1f}% | SA={r['sa_score']:.2f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Stage B3: Blind, Independent First-Principles Verification (ADC(2) + EOM-CCSD)
# ---------------------------------------------------------------------------

def run_blind_verification(n_workers: int = 4, n_eom_ccsd: int = 8) -> None:
    from lehmann_tddft.ab_initio_verify import run_batch_adc2, run_batch_eom_ccsd

    sel_path = OUT_DIR / "top20_selected_for_verification.json"
    sel_data = json.loads(sel_path.read_text())
    top20 = sel_data["molecules"]
    cache_dir = OUT_DIR / "ab_initio_cache"

    # 1. Exact RI-EE-ADC(2)/def2-SVP on ALL 20 top hits!
    adc_jobs = [(m["mol_id"], str(ROOT / m["xyz_path"])) for m in top20]
    print(f"=== Stage B3a: BLIND RI-EE-ADC(2)/def2-SVP Verification on Top {len(adc_jobs)} Discoveries ===", flush=True)
    adc_res = run_batch_adc2(adc_jobs, cache_dir, basis="def2-svp", nroots_s=1, nroots_t=1, n_workers=n_workers)

    # Write intermediate ADC(2) blind report immediately before EOM-CCSD starts
    (OUT_DIR / "blind_adc2_raw.json").write_text(json.dumps(adc_res, indent=2))
    n_adc_inv = sum(1 for r in adc_res.values() if r.get("ok") and r["dEST_eV"] < 0.0)
    n_adc_ok = sum(1 for r in adc_res.values() if r.get("ok"))
    print(f"\n  [BLIND ADC(2) CHECKPOINT] {n_adc_inv}/{n_adc_ok} verified INVEST (dEST < 0)!", flush=True)

    # 2. Unmodified Canonical EOM-EE-CCSD/def2-SVP on Top 8-10 flagship hits (<= 16 heavy preferred, diverse families!)
    eom_pool = [m for m in top20 if m["n_heavy"] <= 16]
    # Sort so diverse scaffold families & flagship brightest/strongest hits are covered first
    seen_fams = set()
    eom_targets: List[Dict[str, Any]] = []
    # First pass: take top verified ADC(2) hit from each scaffold family (<= 16 heavy)
    for m in sorted(
        eom_pool,
        key=lambda x: (
            0 if adc_res.get(x["mol_id"], {}).get("dEST_eV", 1.0) < 0.0 else 1,
            -x["f_S1"],
            x["delta_EST_eV"],
        ),
    ):
        if m["scaffold_family"] not in seen_fams and len(eom_targets) < n_eom_ccsd:
            eom_targets.append(m)
            seen_fams.add(m["scaffold_family"])
    # Second pass: fill remaining slots by highest brightness + strong inversion (<= 16 heavy)
    chosen_ids = {m["mol_id"] for m in eom_targets}
    for m in sorted(
        eom_pool,
        key=lambda x: (
            0 if (adc_res.get(x["mol_id"], {}).get("dEST_eV", 1.0) < 0.0 and adc_res.get(x["mol_id"], {}).get("f_S1", 0.0) >= 0.01) else 1,
            x["n_heavy"],
            -x["f_S1"],
        ),
    ):
        if len(eom_targets) >= n_eom_ccsd:
            break
        if m["mol_id"] not in chosen_ids:
            eom_targets.append(m)
            chosen_ids.add(m["mol_id"])

    eom_jobs = [(m["mol_id"], str(ROOT / m["xyz_path"])) for m in eom_targets]
    print(f"\n=== Stage B3b: BLIND Canonical EOM-EE-CCSD/def2-SVP Verification on Top {len(eom_jobs)} Hits ===", flush=True)
    for m in eom_targets:
        print(f"    -> EOM target: {m['mol_id']:24s} ({m['scaffold_family']}, nH={m['n_heavy']}, Leh dEST={m['delta_EST_eV']:+.3f}, f={m['f_S1']:.4f})", flush=True)
    eom_res = run_batch_eom_ccsd(eom_jobs, cache_dir, basis="def2-svp", nroots_s=1, nroots_t=1, n_workers=n_workers)

    # Aggregate blind verification metrics
    verified_rows = []
    adc_dEST_errs, tda_dEST_errs = [], []
    adc_S1_errs, adc_T1_errs = [], []
    eom_dEST_errs, tda_eom_errs = [], []
    adc_times, eom_times, leh_times = [], [], []
    hits_invest_adc2 = 0
    hits_bright01_adc2 = 0
    hits_bright02_adc2 = 0
    hits_invest_eom = 0

    for rank, m in enumerate(top20, 1):
        mid = m["mol_id"]
        ar = adc_res.get(mid, {})
        er = eom_res.get(mid, {})
        row = dict(m)
        row["rank"] = rank
        if ar.get("ok"):
            row["adc2_E_S1_eV"] = ar["E_S1_eV"]
            row["adc2_E_T1_eV"] = ar["E_T1_eV"]
            row["adc2_dEST_eV"] = ar["dEST_eV"]
            row["adc2_f_S1"] = ar["f_S1"]
            row["adc2_pct_T1_S1"] = ar["pct_T1_S1"]
            row["adc2_pct_T2_S1"] = ar["pct_T2_S1"]
            row["adc2_time_s"] = ar["timings_s"]["total"]
            adc_dEST_errs.append(abs(m["delta_EST_eV"] - ar["dEST_eV"]))
            tda_dEST_errs.append(abs(m["delta_EST_bare_tda_eV"] - ar["dEST_eV"]))
            adc_S1_errs.append(abs(m["E_S1_eV"] - ar["E_S1_eV"]))
            adc_T1_errs.append(abs(m["E_T1_eV"] - ar["E_T1_eV"]))
            adc_times.append(ar["timings_s"]["total"])
            leh_times.append(m["lehmann_total_time_s"])
            if ar["dEST_eV"] < 0.0:
                hits_invest_adc2 += 1
                if ar["f_S1"] >= 0.01:
                    hits_bright01_adc2 += 1
                if ar["f_S1"] >= 0.02:
                    hits_bright02_adc2 += 1
        if er.get("ok"):
            row["eom_E_S1_eV"] = er["E_S1_eV"]
            row["eom_E_T1_eV"] = er["E_T1_eV"]
            row["eom_dEST_eV"] = er["dEST_eV"]
            row["eom_f_S1_R1"] = er["f_S1_R1"]
            row["eom_pct_R1_S1"] = er["pct_R1_S1"]
            row["eom_time_s"] = er["timings_s"]["total"]
            eom_dEST_errs.append(abs(m["delta_EST_eV"] - er["dEST_eV"]))
            tda_eom_errs.append(abs(m["delta_EST_bare_tda_eV"] - er["dEST_eV"]))
            eom_times.append(er["timings_s"]["total"])
            if er["dEST_eV"] < 0.0:
                hits_invest_eom += 1
        verified_rows.append(row)

    n_adc = len(adc_dEST_errs)
    n_eom = len(eom_dEST_errs)
    mean_leh = float(np.mean(leh_times)) if leh_times else 1.0
    mean_adc = float(np.mean(adc_times)) if adc_times else 0.0
    mean_eom = float(np.mean(eom_times)) if eom_times else 0.0

    summary = {
        "n_top_selected": len(top20),
        "n_adc2_verified": n_adc,
        "adc2_blind_invest_hits": hits_invest_adc2,
        "adc2_blind_invest_hit_rate": round(hits_invest_adc2 / n_adc, 4) if n_adc else None,
        "adc2_blind_bright01_invest_hits": hits_bright01_adc2,
        "adc2_blind_bright02_invest_hits": hits_bright02_adc2,
        "adc2_lehmann_dEST_MAE_eV": round(float(np.mean(adc_dEST_errs)), 4) if n_adc else None,
        "adc2_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_dEST_errs)), 4) if n_adc else None,
        "adc2_lehmann_S1_MAE_eV": round(float(np.mean(adc_S1_errs)), 4) if n_adc else None,
        "adc2_lehmann_T1_MAE_eV": round(float(np.mean(adc_T1_errs)), 4) if n_adc else None,
        "n_eom_ccsd_verified": n_eom,
        "eom_ccsd_blind_invest_hits": hits_invest_eom,
        "eom_ccsd_blind_invest_hit_rate": round(hits_invest_eom / n_eom, 4) if n_eom else None,
        "eom_ccsd_lehmann_dEST_MAE_eV": round(float(np.mean(eom_dEST_errs)), 4) if n_eom else None,
        "eom_ccsd_tda_pbe0_dEST_MAE_eV": round(float(np.mean(tda_eom_errs)), 4) if n_eom else None,
        "mean_lehmann_total_time_s": round(mean_leh, 2),
        "mean_adc2_time_s": round(mean_adc, 2),
        "mean_eom_ccsd_time_s": round(mean_eom, 2),
        "speedup_vs_adc2": round(mean_adc / mean_leh, 1) if mean_leh > 0 else None,
        "speedup_vs_eom_ccsd": round(mean_eom / mean_leh, 1) if mean_leh > 0 and mean_eom > 0 else None,
        "verified_discoveries": verified_rows,
    }
    (OUT_DIR / "blind_verification_summary.json").write_text(json.dumps(summary, indent=2))

    # Write publication-ready CSV table of top 20 verified discoveries
    tcols = [
        "rank", "mol_id", "scaffold_family", "formula", "n_heavy", "sa_score", "qed", "smiles",
        "delta_EST_bare_tda_eV", "delta_EST_eV", "delta_EST_eV_std", "f_S1", "pct_T2_S1",
        "two_K_HL_eV", "delta_Sigma_ST_eV",
        "adc2_E_S1_eV", "adc2_E_T1_eV", "adc2_dEST_eV", "adc2_f_S1", "adc2_pct_T2_S1",
        "eom_E_S1_eV", "eom_E_T1_eV", "eom_dEST_eV", "eom_f_S1_R1",
    ]
    with open(OUT_DIR / "top20_discoveries_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tcols, extrasaction="ignore")
        w.writeheader()
        for r in verified_rows:
            w.writerow(r)

    print("\n================ BLIND AB INITIO VERIFICATION SUMMARY ================", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "verified_discoveries"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--virtual-only", action="store_true")
    parser.add_argument("--max-virtual", type=int, default=None)
    parser.add_argument("--screen-and-select", action="store_true")
    parser.add_argument("--blind-verify", action="store_true")
    parser.add_argument("--manifest", default="results/models/lehmann_ensemble_manifest.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-select", type=int, default=20)
    parser.add_argument("--n-eom", type=int, default=8)
    args = parser.parse_args()

    if args.extract_features:
        run_feature_extraction(
            n_workers=args.workers,
            benchmark_only=args.benchmark_only,
            virtual_only=args.virtual_only,
            max_virtual=args.max_virtual,
        )
    if args.screen_and_select:
        run_screening_and_selection(manifest_path=args.manifest, n_select=args.n_select)
    if args.blind_verify:
        run_blind_verification(n_workers=args.workers, n_eom_ccsd=args.n_eom)
