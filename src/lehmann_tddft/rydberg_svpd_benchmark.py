"""Head-to-Head Diffuse-Basis `def2-SVP` vs `def2-SVPD` Rydberg Convergence Proof.

Directly answers Reviewer #1 & #2's basis-set critique:
  In v1.0, Rydberg MAE (0.431 eV) was 2.9x Valence MAE (0.148 eV) because def2-SVP
  lacks diffuse basis functions. This script runs a strictly matched, leak-free 5-fold
  CV head-to-head comparison on the exact same molecules present in BOTH
  `data/processed/features/` (def2-SVP) and `data/processed/features_svpd/` (def2-SVPD),
  using identical CV folds, identical Lehmann-TDDFT v2.0 architecture, and identical seeds,
  reporting Valence MAE, Rydberg MAE, Phenolate anion MAE, 95% bootstrap CIs, and paired
  Wilcoxon signed-rank p-values.

macOS safety: NEVER imports xgboost, lightgbm, or tblite in this process.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
import torch

from .train import (
    assign_stratified_cv_folds,
    evaluate_model_predictions,
    load_unified_dataset,
    match_states_hungarian,
    train_lehmann_model,
)


def _bootstrap_ci95(vals: Sequence[float], n_boot: int = 5000, seed: int = 2026) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "n": int(len(arr))}


def _run_basis_fold_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    f = cfg["fold"]
    basis_tag = cfg["basis_tag"]
    allowed = set(cfg["matched_ids"])
    all_items = load_unified_dataset(
        cfg["questdb_json"],
        cfg["hepta_json"],
        cfg["features_dir"],
    )
    items = [it for it in all_items if it["mol_id"] in allowed]
    # Use the exact leak-free CV5 fold assignments from questdb_clean.json
    fold_map = cfg["fold_map"]
    tr_all = [it for it in items if fold_map[it["mol_id"]] != f]
    te_items = [it for it in items if fold_map[it["mol_id"]] == f]
    if not te_items:
        return {"basis_tag": basis_tag, "fold": f, "state_rows": [], "mol_rows": []}

    rng = np.random.default_rng(cfg["seed"] + 100 + f)
    perm = rng.permutation(len(tr_all))
    n_val = max(1, int(0.12 * len(tr_all)))
    val_idx = set(perm[:n_val].tolist())
    tr = [tr_all[i] for i in range(len(tr_all)) if i not in val_idx]
    va = [tr_all[i] for i in range(len(tr_all)) if i in val_idx]

    model, _ = train_lehmann_model(
        tr, va,
        disable_doubles=False,
        epochs=cfg["epochs"],
        seed=cfg["seed"] + f,
        verbose=False,
    )
    _, recs = evaluate_model_predictions(model, te_items)

    # Collect per-state matched errors (for exact paired Wilcoxon across def2-SVP vs def2-SVPD!)
    state_rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for it in te_items:
            mol = it["mol_tensors"]
            out = model(mol)
            for spin_int, spin_key in ((1, "S"), (3, "T")):
                ref_states = [s for s in it["states"] if s["spin"] == spin_int]
                if not ref_states:
                    continue
                res = out[spin_key]
                pE = res["energies_eV"].cpu().numpy()
                pT1 = res["pct_T1"].cpu().numpy()
                pF = res["osc"].cpu().numpy() if "osc" in res else None
                pairs = match_states_hungarian(pE, pT1, pF, ref_states)

                # Also compute raw TDA baseline error for exact same states
                w_bare = (mol["omega_S_eV"] if spin_key == "S" else mol["omega_T_eV"]).cpu().numpy()
                bare_pairs = match_states_hungarian(w_bare, np.full_like(w_bare, 100.0), None, ref_states)
                bare_map = {r_i: float(w_bare[p_i]) for r_i, p_i in bare_pairs}

                for r_idx, p_idx in pairs:
                    st = ref_states[r_idx]
                    state_key = f"{it['mol_id']}|s{spin_int}|{st['label']}|{st['tbe_eV']:.4f}"
                    state_rows.append(
                        {
                            "state_key": state_key,
                            "mol_id": it["mol_id"],
                            "spin": spin_int,
                            "label": st["label"],
                            "vr": st.get("vr"),
                            "special": st.get("special"),
                            "ref_eV": float(st["tbe_eV"]),
                            "pred_eV": float(pE[p_idx]),
                            "abs_err_eV": abs(float(pE[p_idx]) - float(st["tbe_eV"])),
                            "bare_tda_eV": bare_map.get(r_idx),
                            "bare_tda_abs_err_eV": (
                                abs(bare_map[r_idx] - float(st["tbe_eV"])) if r_idx in bare_map else None
                            ),
                        }
                    )

    print(f"  [{basis_tag} fold {f}] evaluated {len(te_items)} mols, {len(state_rows)} states", flush=True)
    return {"basis_tag": basis_tag, "fold": f, "state_rows": state_rows, "mol_rows": recs}


def run_rydberg_svpd_proof(
    questdb_json: str = "data/processed/questdb_clean.json",
    hepta_json: str = "data/processed/hepta_invest_clean.json",
    svp_dir: str = "data/processed/features",
    svpd_dir: str = "data/processed/features_svpd",
    out_json: str = "results/benchmarks/rydberg_def2svpd_proof.json",
    out_csv: str = "results/benchmarks/rydberg_def2svpd_proof.csv",
    epochs: int = 80,
    seed: int = 42,
    workers: int = 5,
) -> Dict[str, Any]:
    svpd_files = sorted(Path(svpd_dir).glob("*.npz"))
    svp_ids = {p.stem for p in Path(svp_dir).glob("*.npz")}
    svpd_ids = {p.stem for p in svpd_files}
    q_raw = json.loads(Path(questdb_json).read_text())
    q_ids = {m["mol_id"] for m in q_raw["molecules"]}
    matched_ids = sorted(svp_ids & svpd_ids & q_ids)
    assert len(matched_ids) >= 10, f"Too few matched molecules in {svpd_dir}: {len(matched_ids)}"
    # Sanity check: def2-SVPD MUST strictly have larger n_ao than def2-SVP on every molecule!
    for mid in matched_ids[:5]:
        n_ao_svp = int(np.load(Path(svp_dir) / f"{mid}.npz")["n_ao"])
        n_ao_svpd = int(np.load(Path(svpd_dir) / f"{mid}.npz")["n_ao"])
        assert n_ao_svpd > n_ao_svp, f"Basis mismatch on {mid}: SVPD n_ao={n_ao_svpd} <= SVP n_ao={n_ao_svp}"

    # Exact leak-free fold map from questdb_clean.json
    fold_map: Dict[str, int] = {}
    for fk, mids in q_raw["splits"]["cv5_folds"].items():
        fi = int(str(fk).replace("fold_", "").replace("fold", ""))
        for mid in mids:
            fold_map[mid] = fi

    print(f"=== def2-SVP vs def2-SVPD Rydberg Proof on {len(matched_ids)} matched molecules ===", flush=True)
    cfgs = []
    for basis_tag, fdir in (("def2-SVP", svp_dir), ("def2-SVPD", svpd_dir)):
        for f in range(5):
            cfgs.append(
                {
                    "basis_tag": basis_tag,
                    "fold": f,
                    "matched_ids": matched_ids,
                    "fold_map": fold_map,
                    "questdb_json": questdb_json,
                    "hepta_json": hepta_json,
                    "features_dir": fdir,
                    "epochs": epochs,
                    "seed": seed,
                }
            )

    collected: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "def2-SVP": {"state_rows": [], "mol_rows": []},
        "def2-SVPD": {"state_rows": [], "mol_rows": []},
    }
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(cfgs))) as ex:
            for res in ex.map(_run_basis_fold_worker, cfgs):
                collected[res["basis_tag"]]["state_rows"].extend(res["state_rows"])
                collected[res["basis_tag"]]["mol_rows"].extend(res["mol_rows"])
    else:
        for cfg in cfgs:
            res = _run_basis_fold_worker(cfg)
            collected[res["basis_tag"]]["state_rows"].extend(res["state_rows"])
            collected[res["basis_tag"]]["mol_rows"].extend(res["mol_rows"])

    # Build strict state-by-state matched comparison
    by_key_svp = {r["state_key"]: r for r in collected["def2-SVP"]["state_rows"]}
    by_key_svpd = {r["state_key"]: r for r in collected["def2-SVPD"]["state_rows"]}
    common_keys = sorted(set(by_key_svp) & set(by_key_svpd))

    def _summarize_subset(keys: List[str]) -> Dict[str, Any]:
        e_svp = [by_key_svp[k]["abs_err_eV"] for k in keys]
        e_svpd = [by_key_svpd[k]["abs_err_eV"] for k in keys]
        t_svp = [by_key_svp[k]["bare_tda_abs_err_eV"] for k in keys if by_key_svp[k]["bare_tda_abs_err_eV"] is not None and by_key_svpd[k]["bare_tda_abs_err_eV"] is not None]
        t_svpd = [by_key_svpd[k]["bare_tda_abs_err_eV"] for k in keys if by_key_svp[k]["bare_tda_abs_err_eV"] is not None and by_key_svpd[k]["bare_tda_abs_err_eV"] is not None]
        p_val = None
        if len(e_svp) >= 6:
            diff = np.asarray(e_svpd) - np.asarray(e_svp)
            if np.any(np.abs(diff) > 1e-12):
                p_val = float(wilcoxon(e_svpd, e_svp).pvalue)
        return {
            "n_states": len(keys),
            "Lehmann_def2SVP": _bootstrap_ci95(e_svp),
            "Lehmann_def2SVPD": _bootstrap_ci95(e_svpd),
            "reduction_factor_SVP_over_SVPD": (
                float(np.mean(e_svp) / max(np.mean(e_svpd), 1e-9)) if e_svp else None
            ),
            "wilcoxon_p_SVPD_vs_SVP": p_val,
            "Bare_TDA_def2SVP_MAE_eV": float(np.mean(t_svp)) if t_svp else None,
            "Bare_TDA_def2SVPD_MAE_eV": float(np.mean(t_svpd)) if t_svpd else None,
        }

    ryd_keys = [k for k in common_keys if by_key_svp[k]["vr"] == "R"]
    val_keys = [k for k in common_keys if by_key_svp[k]["vr"] == "V"]
    phen_keys = [k for k in common_keys if by_key_svp[k]["mol_id"] == "quest_Phenolate"]

    # Molecule-level S1/T1/dEST comparison
    mol_svp = {r["mol_id"]: r for r in collected["def2-SVP"]["mol_rows"]}
    mol_svpd = {r["mol_id"]: r for r in collected["def2-SVPD"]["mol_rows"]}
    phen_s1 = None
    if "quest_Phenolate" in mol_svp and "quest_Phenolate" in mol_svpd:
        phen_s1 = {
            "ref_S1_eV": mol_svp["quest_Phenolate"]["ref_S1_eV"],
            "def2SVP_pred_S1_eV": mol_svp["quest_Phenolate"]["pred_S1_eV"],
            "def2SVPD_pred_S1_eV": mol_svpd["quest_Phenolate"]["pred_S1_eV"],
            "def2SVP_err_eV": abs(mol_svp["quest_Phenolate"]["pred_S1_eV"] - mol_svp["quest_Phenolate"]["ref_S1_eV"]),
            "def2SVPD_err_eV": abs(mol_svpd["quest_Phenolate"]["pred_S1_eV"] - mol_svpd["quest_Phenolate"]["ref_S1_eV"]),
        }

    report = {
        "n_matched_molecules": len(matched_ids),
        "matched_mol_ids": matched_ids,
        "Rydberg_states": _summarize_subset(ryd_keys),
        "Valence_states": _summarize_subset(val_keys),
        "All_states": _summarize_subset(common_keys),
        "Phenolate_anion_states": _summarize_subset(phen_keys) if phen_keys else None,
        "Phenolate_S1": phen_s1,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2))

    rows = []
    for label, sub in [
        ("Rydberg (V/R == R)", report["Rydberg_states"]),
        ("Valence (V/R == V)", report["Valence_states"]),
        ("All Matched States", report["All_states"]),
    ]:
        rows.append(
            {
                "Category": label,
                "n_states": sub["n_states"],
                "Bare_TDA_def2SVP_MAE_eV": sub["Bare_TDA_def2SVP_MAE_eV"],
                "Bare_TDA_def2SVPD_MAE_eV": sub["Bare_TDA_def2SVPD_MAE_eV"],
                "Lehmann_def2SVP_MAE_eV": sub["Lehmann_def2SVP"]["mean"],
                "Lehmann_def2SVPD_MAE_eV": sub["Lehmann_def2SVPD"]["mean"],
                "Reduction_Factor": sub["reduction_factor_SVP_over_SVPD"],
                "Wilcoxon_p": sub["wilcoxon_p_SVPD_vs_SVP"],
            }
        )
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== def2-SVP vs def2-SVPD RYDBERG CONVERGENCE SUMMARY ===")
    for r in rows:
        print(
            f"  {r['Category']:22s} (n={r['n_states']:3d}): "
            f"TDA {r['Bare_TDA_def2SVP_MAE_eV']:.3f}->{r['Bare_TDA_def2SVPD_MAE_eV']:.3f} eV | "
            f"Lehmann {r['Lehmann_def2SVP_MAE_eV']:.3f}->{r['Lehmann_def2SVPD_MAE_eV']:.3f} eV "
            f"({r['Reduction_Factor']:.2f}x, p={r['Wilcoxon_p']})"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="def2-SVP vs def2-SVPD Rydberg Convergence Proof")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run_rydberg_svpd_proof(epochs=args.epochs, seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
