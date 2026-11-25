#!/usr/bin/env python3
"""Extract verified per-state CV5 arrays and real-molecule Sigma_xc(omega) curves
directly from saved checkpoints and processed datasets for publication figures.

Strictly read-only on models/results; outputs to figures/plot_cache.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

torch.set_num_threads(2)

from lehmann_tddft.model import load_checkpoint, npz_to_mol_tensors
from lehmann_tddft.train import (
    assign_stratified_cv_folds,
    load_unified_dataset,
    match_states_hungarian,
)


def main() -> None:
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract exact CV5 OOF per-state records (matching train.py float32)
    items = load_unified_dataset(
        questdb_json=str(ROOT / "data/processed/questdb_clean.json"),
        hepta_json=str(ROOT / "data/processed/hepta_invest_clean.json"),
        features_dir=str(ROOT / "data/processed/features"),
        dtype=torch.float32,
    )
    folds = assign_stratified_cv_folds(items, n_folds=5, seed=42)
    leh_models = [
        load_checkpoint(str(ROOT / f"results/models/lehmann_cv_fold{f}.pt"), dtype=torch.float32)[0]
        for f in range(5)
    ]
    sta_models = [
        load_checkpoint(str(ROOT / f"results/models/static_delta_cv_fold{f}.pt"), dtype=torch.float32)[0]
        for f in range(5)
    ]

    q_raw = {
        f"quest_{m['name']}": m
        for m in json.loads((ROOT / "data/processed/questdb_clean.json").read_text())["molecules"]
    }

    state_records = []
    sta_pct_t1_records = []
    with torch.no_grad():
        for it, fl in zip(items, folds):
            mid = it["mol_id"]
            out_leh = leh_models[fl](it["mol_tensors"])
            out_sta = sta_models[fl](it["mol_tensors"])
            safe_map = {}
            if mid in q_raw:
                for s in q_raw[mid]["states"]:
                    safe_map[(s["spin"], s["state_label"], round(s["tbe_avtz_eV"], 4))] = bool(s["safe"])

            for spin_int, spin_key in ((1, "S"), (3, "T")):
                ref_states = [s for s in it["states"] if s["spin"] == spin_int]
                if not ref_states:
                    continue
                res = out_leh[spin_key]
                pE = res["energies_eV"].cpu().numpy()
                pT1 = res["pct_T1"].cpu().numpy()
                pT2 = res["pct_T2"].cpu().numpy()
                pF = res["osc"].cpu().numpy() if "osc" in res else None
                pairs = match_states_hungarian(pE, pT1, pF, ref_states)

                for r_idx, p_idx in pairs:
                    st = ref_states[r_idx]
                    re = float(st["tbe_eV"])
                    pe = float(pE[p_idx])
                    sf = safe_map.get((spin_int, st["label"], round(re, 4)), True) if mid in q_raw else True
                    state_records.append(
                        {
                            "mol_id": mid,
                            "name": it["name"],
                            "source": it["source"],
                            "spin": spin_int,
                            "label": st["label"],
                            "vr": st.get("vr"),
                            "special": st.get("special"),
                            "safe": sf,
                            "ref_eV": re,
                            "pred_eV": pe,
                            "abs_err_eV": abs(pe - re),
                            "ref_pct_T1": float(st["pct_T1"]) if st.get("pct_T1") is not None else None,
                            "pred_pct_T1": float(pT1[p_idx]),
                            "pred_pct_T2": float(pT2[p_idx]),
                            "ref_osc": float(st["osc_ref"])
                            if (spin_key == "S" and st.get("osc_ref") is not None)
                            else None,
                            "pred_osc": float(pF[p_idx]) if (spin_key == "S" and pF is not None) else None,
                        }
                    )

                res_s = out_sta[spin_key]
                pE_s = res_s["energies_eV"].cpu().numpy()
                pT1_s = res_s["pct_T1"].cpu().numpy()
                pF_s = res_s["osc"].cpu().numpy() if "osc" in res_s else None
                pairs_s = match_states_hungarian(pE_s, pT1_s, pF_s, ref_states)
                for r_idx, p_idx in pairs_s:
                    st = ref_states[r_idx]
                    if st.get("pct_T1") is not None:
                        sta_pct_t1_records.append(
                            {
                                "ref_pct_T1": float(st["pct_T1"]),
                                "pred_pct_T1": float(pT1_s[p_idx]),
                                "special": st.get("special"),
                            }
                        )

    # Verify exact match with cv5_summary.json
    cv5_sum = json.loads((ROOT / "results/benchmarks/cv5_summary.json").read_text())["Lehmann-TDDFT"]["suite_C_full_spectrum_matched"]
    all_mae = float(np.mean([r["abs_err_eV"] for r in state_records]))
    val_mae = float(np.mean([r["abs_err_eV"] for r in state_records if r["vr"] == "V"]))
    ryd_mae = float(np.mean([r["abs_err_eV"] for r in state_records if r["vr"] == "R"]))
    gd_mae = float(np.mean([r["abs_err_eV"] for r in state_records if r["special"] == "GD"]))
    t1_recs = [r for r in state_records if r["ref_pct_T1"] is not None]
    t1_mae = float(np.mean([abs(r["pred_pct_T1"] - r["ref_pct_T1"]) for r in t1_recs]))
    osc_recs = [r for r in state_records if r["ref_osc"] is not None and r["pred_osc"] is not None]
    osc_mae = float(np.mean([abs(r["pred_osc"] - r["ref_osc"]) for r in osc_recs]))

    print(f"Verification vs cv5_summary.json:")
    print(f"  n_states: {len(state_records)} vs {cv5_sum['n_states']}")
    print(f"  all_MAE:  {all_mae:.6f} vs {cv5_sum['all_states_MAE_eV']:.6f}")
    print(f"  val_MAE:  {val_mae:.6f} vs {cv5_sum['valence_MAE_eV']:.6f}")
    print(f"  ryd_MAE:  {ryd_mae:.6f} vs {cv5_sum['rydberg_MAE_eV']:.6f}")
    print(f"  GD_MAE:   {gd_mae:.6f} vs {cv5_sum['GD_matched_MAE_eV']:.6f}")
    print(f"  %T1_MAE:  {t1_mae:.6f} vs {cv5_sum['pct_T1_MAE']:.6f}")
    print(f"  osc_MAE:  {osc_mae:.6f} vs {cv5_sum['osc_MAE']:.6f}")

    # 2. Extract exact real-molecule Sigma_xc(omega) curves on trans-Butadiene (float64)
    m0_64, _ = load_checkpoint(str(ROOT / "results/models/lehmann_cv_fold0.pt"), dtype=torch.float64)
    mol_but = npz_to_mol_tensors(str(ROOT / "data/processed/features/quest_Butadiene.npz"), dtype=torch.float64)
    out_but = m0_64(mol_but)
    res_S = out_but["S"]
    V = res_S["V"].detach().numpy()
    D = res_S["D"].detach().numpy()
    H_SS = res_S["H_SS"].detach().numpy()
    E_S = res_S["energies_eV"].detach().numpy()
    pct_T2 = res_S["pct_T2"].detach().numpy()
    c_S = res_S["c_S"].detach().numpy()
    osc_but = res_S["osc"].detach().numpy()

    # Identify S1 (bright ^1B_u) and lowest emergent GD (2^1A_g)
    # Also find the single-exciton state with strongest coupling to lowest poles (usually state 0 = 1^1B_u or 2^1A_g partner)
    # Project Sigma(omega) onto the lowest singlet eigenvector u_S1 (normalized singles component of S1)
    c_s1 = c_S[:, 0]
    u_s1 = c_s1 / np.linalg.norm(c_s1)
    e_s1_static = float(u_s1 @ H_SS @ u_s1)
    v_proj = V.T @ u_s1  # (N_D,)
    # Find dominant coupled poles
    pole_weights = v_proj * v_proj
    top_poles_idx = np.argsort(pole_weights)[::-1][:10]

    omega_grid = np.linspace(3.0, 12.5, 951)
    eta = 0.15
    diff = omega_grid[:, None] - D[None, :]
    sigma_re_eta = np.sum(pole_weights[None, :] * diff / (diff * diff + eta * eta), axis=1)
    sigma_im_eta = np.sum(pole_weights[None, :] * (-eta) / (diff * diff + eta * eta), axis=1)
    # Exact unbroadened Re Sigma and dSigma/domega away from poles (clip at poles for plotting)
    sigma_re_exact = np.sum(pole_weights[None, :] / np.where(np.abs(diff) < 1e-4, np.nan, diff), axis=1)
    dsigma_exact = -np.sum(pole_weights[None, :] / np.where(np.abs(diff) < 1e-4, np.nan, diff * diff), axis=1)

    se_but = m0_64.evaluate_self_energy(mol_but)
    # Find emergent 2^1A_g double state in Butadiene
    gd_idx = int(np.where(pct_T2 >= 40.0)[0][0])

    # Also compute on Cyclazine and D3h Heptazine (spin-complete v2.0 antisymmetric ^1(TT)_- inversion!)
    mol_cyc = npz_to_mol_tensors(str(ROOT / "data/processed/features/quest_Cyclazine.npz"), dtype=torch.float64)
    se_cyc = m0_64.evaluate_self_energy(mol_cyc)

    mol_hep = npz_to_mol_tensors(str(ROOT / "data/processed/features/quest_Heptazine.npz"), dtype=torch.float64)
    se_hep = m0_64.evaluate_self_energy(mol_hep)

    # Print top GD details for annotations in Fig 2b
    gd_list = [r for r in state_records if r["special"] == "GD"]
    print("\nAll 28 GD matched states (sorted by |err|):")
    for r in sorted(gd_list, key=lambda x: x["abs_err_eV"]):
        print(f"  {r['name']:24s} {r['label']:10s} ref={r['ref_eV']:6.3f} pred={r['pred_eV']:6.3f} err={r['abs_err_eV']:.3f} %T2={r['pred_pct_T2']:.1f}")

    cache = {
        "model_version": "v2.0",
        "state_records": state_records,
        "static_pct_t1_records": sta_pct_t1_records,
        "butadiene_spectral": {
            "omega_grid": omega_grid.tolist(),
            "eta_eV": eta,
            "sigma_re_eta": sigma_re_eta.tolist(),
            "sigma_im_eta": sigma_im_eta.tolist(),
            "sigma_re_exact": [None if np.isnan(x) else float(x) for x in sigma_re_exact],
            "dsigma_exact": [None if np.isnan(x) else float(x) for x in dsigma_exact],
            "E_S1_eV": float(E_S[0]),
            "E_S1_static_eV": e_s1_static,
            "Sigma_S1_eV": float(se_but["Sigma_S1_dyn_eV"].item()),
            "Z_S1": float(se_but["Z_S1"].item()),
            "Z_S1_residual": float(se_but["Z_S1_residual"].item()),
            "E_2Ag_eV": float(E_S[gd_idx]),
            "pct_T2_2Ag": float(pct_T2[gd_idx]),
            "poles_D_eV": D.tolist(),
            "pole_weights_v2": pole_weights.tolist(),
            "top_pole_energies": D[top_poles_idx].tolist(),
            "top_pole_weights": pole_weights[top_poles_idx].tolist(),
        },
        "cyclazine_self_energy": {
            "E_S1_eV": float(se_cyc["E_S1_eV"].item()),
            "E_T1_eV": float(se_cyc["E_T1_eV"].item()),
            "delta_EST_eV": float(se_cyc["delta_EST_eV"].item()),
            "delta_EST_static_eV": float(se_cyc["delta_EST_static_eV"].item()),
            "delta_Sigma_ST_eV": float(se_cyc["delta_Sigma_ST_eV"].item()),
            "Z_S1": float(se_cyc["Z_S1"].item()),
        },
        "heptazine_self_energy": {
            "E_S1_eV": float(se_hep["E_S1_eV"].item()),
            "E_T1_eV": float(se_hep["E_T1_eV"].item()),
            "delta_EST_eV": float(se_hep["delta_EST_eV"].item()),
            "delta_EST_static_eV": float(se_hep["delta_EST_static_eV"].item()),
            "delta_Sigma_ST_eV": float(se_hep["delta_Sigma_ST_eV"].item()),
            "Sigma_S1_TT_anti_eV": float(se_hep["Sigma_S1_TT_anti_eV"].item()) if "Sigma_S1_TT_anti_eV" in se_hep else None,
            "Z_S1": float(se_hep["Z_S1"].item()),
        },
    }
    out_path = out_dir / "plot_cache.json"
    out_path.write_text(json.dumps(cache, indent=2))
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
