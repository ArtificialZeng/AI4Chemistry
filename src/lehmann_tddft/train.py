"""Training, Leak-Free State-Matching Loss, CV/OOD Benchmarking & Ablations for Lehmann-TDDFT.

Executes the full experimental protocol:
  1. 5-Fold Molecule-Level Cross-Validation across QUESTDB + HEPTA-INVEST:
     - `Lehmann-TDDFT` (Full causal dynamical self-energy operator)
     - `Static-Delta-TDDFT` Ablation (identical architecture with V_theta == 0)
     - `SchNet3D` Geometric MPNN Baseline (3D coordinates only)
     - Published QM baselines (B3LYP, PBE0, CAM-B3LYP, M06-2X, DF-PBE0 RI-TDA,
       CIS(D), ADC(2), CC2, CCSD) on exact same splits.
  2. OOD-Large Chromophore Generalization:
     - Train ONLY on small/medium MAIN molecules (n_heavy <= 8); test blind on
       CHROM + BIO chromophores (n_heavy >= 9).
  3. OOD-Double Excitation Family Holdout:
     - Hold out polyene/conjugated-chain doubles OR cyclic antiaromatic/azine doubles.
  4. OOD-INVEST Zero-Shot Scaffold Transfer:
     - Train with ZERO heptazine / HEPTA-INVEST molecules; test zero-shot on
       all 44 HEPTA-INVEST + QUESTDB Heptazine.
     - Train without Cyclazine (and non-heptazine INVEST cores); test zero-shot.
  5. Mechanistic Quantum Phase Diagram export (2K_HL, dE_ST_static, dSigma_ST)
     and 3D GNN vs Lehmann-TDDFT Size-Intensivity contrast audit.

macOS safety: NEVER imports xgboost, lightgbm, or tblite in this process.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import (
    SchNet3DBaseline,
    audit_gnn_size_intensivity,
    evaluate_published_qm_baselines,
    predict_schnet_baseline,
    train_schnet_baseline,
)
from .model import (
    LehmannTDDFTOperator,
    load_checkpoint,
    npz_to_mol_tensors,
    safe_eigh,
    save_checkpoint,
)
from .theorems import make_synthetic_molecule


# ---------------------------------------------------------------------------
# Unified Dataset Record Loader (QUESTDB + HEPTA-INVEST + .npz Feature Cache)
# ---------------------------------------------------------------------------

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def _normalize_state_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single excited-state record from QUESTDB/HEPTA-INVEST JSON."""
    tbe = _safe_float(
        raw.get("tbe_avtz_eV", raw.get("tbe_eV", raw.get("ref_eV", raw.get("energy_eV"))))
    )
    pct_t1 = _safe_float(raw.get("pct_T1", raw.get("percent_T1", raw.get("percent_R1"))))
    osc = _safe_float(raw.get("osc_cc3", raw.get("osc", raw.get("f_osc"))))
    spin_raw = raw.get("spin", 1)
    if isinstance(spin_raw, str):
        spin = 3 if ("3" in spin_raw or spin_raw.upper().startswith("T")) else 1
    else:
        spin = int(spin_raw)
    special = raw.get("special")
    if isinstance(special, str):
        special = special.strip().upper()
        if special in ("", "NONE", "NULL"):
            special = None
    # Infer GD / PD cleanly from %T1 if not tagged explicitly
    if special is None and pct_t1 is not None:
        if pct_t1 < 50.0:
            special = "GD"
        elif pct_t1 < 85.0:
            special = "PD"

    baselines = dict(raw.get("baselines", {}))
    # Also pick up flat baseline columns if present directly on the state dict
    for m_key in ("B3LYP", "PBE0", "CAM-B3LYP", "M06-2X", "CIS(D)", "ADC(2)", "CC2", "CCSD", "CC3"):
        if m_key in raw and m_key not in baselines:
            v = _safe_float(raw[m_key])
            if v is not None:
                baselines[m_key] = v

    return {
        "label": str(raw.get("state_label", raw.get("label", ""))),
        "spin": spin,
        "vr": raw.get("vr", "V"),
        "type": raw.get("type", ""),
        "special": special,
        "tbe_eV": tbe,
        "pct_T1": pct_t1,
        "osc_ref": osc,
        "baselines": baselines,
    }


def load_unified_dataset(
    questdb_json: str = "data/processed/questdb_clean.json",
    hepta_json: str = "data/processed/hepta_invest_clean.json",
    features_dir: str = "data/processed/features",
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, Any]]:
    """Load all cleaned QUESTDB + HEPTA-INVEST molecules that have `.npz` caches."""
    feat_root = Path(features_dir)
    items: List[Dict[str, Any]] = []

    def _find_npz(mol_id: str, npz_hint: Optional[str] = None) -> Optional[Path]:
        candidates = [
            feat_root / f"{mol_id}.npz",
            feat_root / f"{mol_id.replace('/', '_')}.npz",
        ]
        if npz_hint:
            candidates.append(Path(npz_hint))
        for c in candidates:
            if c.exists():
                return c
        return None

    # 1. Load QUESTDB
    if os.path.exists(questdb_json):
        qdata = json.loads(Path(questdb_json).read_text())
        q_records = qdata.get("molecules", qdata) if isinstance(qdata, dict) else qdata
        if isinstance(q_records, dict):
            q_records = [{"mol_id": k, **v} for k, v in q_records.items()]

        for rec in q_records:
            mol_id = str(rec.get("mol_id", rec.get("name", "")))
            npz_path = _find_npz(mol_id, rec.get("feature_path", rec.get("npz_path")))
            if npz_path is None:
                continue
            raw_npz = dict(np.load(npz_path, allow_pickle=False))
            mol_t = npz_to_mol_tensors(raw_npz, dtype=dtype)
            states = [_normalize_state_record(s) for s in rec.get("states", [])]
            states = [s for s in states if s["tbe_eV"] is not None]
            if not states:
                continue

            s_states = sorted([s for s in states if s["spin"] == 1], key=lambda x: x["tbe_eV"])
            t_states = sorted([s for s in states if s["spin"] == 3], key=lambda x: x["tbe_eV"])

            # Check whether lowest QUESTDB S1 / T1 genuinely corresponds to the molecule's lowest
            # vertical state (and not a selective high-lying state benchmark that skipped S1/T1!)
            wS0 = float(raw_npz["omega_S_eV"][0]) if len(raw_npz["omega_S_eV"]) > 0 else 99.0
            wT0 = float(raw_npz["omega_T_eV"][0]) if len(raw_npz["omega_T_eV"]) > 0 else 99.0
            ref_S1 = s_states[0]["tbe_eV"] if s_states else None
            ref_T1 = t_states[0]["tbe_eV"] if t_states else None
            ref_f_S1 = s_states[0]["osc_ref"] if s_states else None
            ref_T1_pct_S1 = s_states[0]["pct_T1"] if s_states else None
            ref_T1_pct_T1 = t_states[0]["pct_T1"] if t_states else None

            # If QUESTDB skipped true S1 (lowest listed singlet is > 1.35 eV above TDA S1),
            # do not treat it as true lowest S1 in zero-matching S1/T1 loss
            true_S1 = (ref_S1 is not None) and (ref_S1 <= wS0 + 1.35)
            true_T1 = (ref_T1 is not None) and (ref_T1 <= wT0 + 1.35)

            items.append(
                {
                    "mol_id": mol_id,
                    "name": str(rec.get("name", mol_id)),
                    "source": "QUESTDB",
                    "subset": str(rec.get("subset", "MAIN")).upper(),
                    "family": str(rec.get("family", rec.get("chem_family", "other"))).lower(),
                    "n_atoms": int(raw_npz["n_atoms"]),
                    "n_heavy": int(raw_npz["n_heavy"]),
                    "fold": rec.get("cv5_fold", rec.get("fold", rec.get("cv_fold"))),
                    "npz_path": str(npz_path),
                    "mol_tensors": mol_t,
                    "omega_S_eV": raw_npz["omega_S_eV"],
                    "omega_T_eV": raw_npz["omega_T_eV"],
                    "osc_S": raw_npz["osc_S"],
                    "atomic_numbers": raw_npz["atomic_numbers"],
                    "coords_angstrom": raw_npz["coords_angstrom"],
                    "states": states,
                    "ref_S1_eV": ref_S1 if true_S1 else None,
                    "ref_T1_eV": ref_T1 if true_T1 else None,
                    "ref_f_S1": ref_f_S1 if true_S1 else None,
                    "ref_pct_T1_S1": ref_T1_pct_S1 if true_S1 else None,
                    "ref_pct_T1_T1": ref_T1_pct_T1 if true_T1 else None,
                    "is_invest_core": (
                        true_S1 and true_T1 and (ref_S1 - ref_T1 < 0.0)
                    ),
                }
            )

    # 2. Load HEPTA-INVEST44
    if os.path.exists(hepta_json):
        hdata = json.loads(Path(hepta_json).read_text())
        h_records = hdata.get("molecules", hdata) if isinstance(hdata, dict) else hdata
        if isinstance(h_records, dict):
            h_records = [{"mol_id": k, **v} for k, v in h_records.items()]

        for rec in h_records:
            mol_id = str(rec.get("mol_id", rec.get("molecule", "")))
            npz_path = _find_npz(mol_id, rec.get("feature_path", rec.get("npz_path")))
            if npz_path is None:
                continue
            raw_npz = dict(np.load(npz_path, allow_pickle=False))
            mol_t = npz_to_mol_tensors(raw_npz, dtype=dtype)

            ref_S1 = _safe_float(rec.get("E_S1_eV", rec.get("ref_S1_eV")))
            ref_T1 = _safe_float(rec.get("E_T1_eV", rec.get("ref_T1_eV")))
            ref_f_S1 = _safe_float(rec.get("f_S1_eomccsd", rec.get("f_S1", rec.get("osc_S1"))))
            pct_s1 = _safe_float(rec.get("percent_R1_S1", rec.get("pct_T1_S1")))
            pct_t1 = _safe_float(rec.get("percent_R1_T1", rec.get("pct_T1_T1")))
            if ref_S1 is None or ref_T1 is None:
                continue

            # Extract published method baselines on HEPTA-INVEST (e.g. CAM-B3LYP, ADC(2))
            b_s1, b_t1 = {}, {}
            for m_key in ("CAM-B3LYP", "B3LYP", "PBE0", "ADC(2)", "CC2", "EOM-CCSD", "CCSD"):
                b_dict = rec.get("baselines", {})
                if m_key in b_dict and isinstance(b_dict[m_key], dict):
                    vs = _safe_float(b_dict[m_key].get("E_S1_eV"))
                    vt = _safe_float(b_dict[m_key].get("E_T1_eV"))
                    if vs is not None:
                        b_s1[m_key] = vs
                    if vt is not None:
                        b_t1[m_key] = vt
            if "cam_b3lyp_S1_eV" in rec and _safe_float(rec["cam_b3lyp_S1_eV"]) is not None:
                b_s1["CAM-B3LYP"] = float(rec["cam_b3lyp_S1_eV"])
            if "cam_b3lyp_T1_eV" in rec and _safe_float(rec["cam_b3lyp_T1_eV"]) is not None:
                b_t1["CAM-B3LYP"] = float(rec["cam_b3lyp_T1_eV"])

            states = [
                {
                    "label": "S1",
                    "spin": 1,
                    "vr": "V",
                    "type": "ppi",
                    "special": None,
                    "tbe_eV": ref_S1,
                    "pct_T1": pct_s1,
                    "osc_ref": ref_f_S1,
                    "baselines": b_s1,
                },
                {
                    "label": "T1",
                    "spin": 3,
                    "vr": "V",
                    "type": "ppi",
                    "special": None,
                    "tbe_eV": ref_T1,
                    "pct_T1": pct_t1,
                    "osc_ref": None,
                    "baselines": b_t1,
                },
            ]
            items.append(
                {
                    "mol_id": mol_id,
                    "name": str(rec.get("name", rec.get("molecule", mol_id))),
                    "source": "HEPTA-INVEST",
                    "subset": "HEPTA-INVEST",
                    "family": "heptazine",
                    "n_atoms": int(raw_npz["n_atoms"]),
                    "n_heavy": int(raw_npz["n_heavy"]),
                    "fold": rec.get("cv5_fold", rec.get("fold", rec.get("cv_fold"))),
                    "npz_path": str(npz_path),
                    "mol_tensors": mol_t,
                    "omega_S_eV": raw_npz["omega_S_eV"],
                    "omega_T_eV": raw_npz["omega_T_eV"],
                    "osc_S": raw_npz["osc_S"],
                    "atomic_numbers": raw_npz["atomic_numbers"],
                    "coords_angstrom": raw_npz["coords_angstrom"],
                    "states": states,
                    "ref_S1_eV": ref_S1,
                    "ref_T1_eV": ref_T1,
                    "ref_f_S1": ref_f_S1,
                    "ref_pct_T1_S1": pct_s1,
                    "ref_pct_T1_T1": pct_t1,
                    "is_invest_core": (ref_S1 - ref_T1 < 0.0),
                }
            )

    # Deduplicate by mol_id if any overlap (keep QUESTDB TBE if exact duplicate)
    seen = set()
    deduped = []
    for it in items:
        if it["mol_id"] not in seen:
            seen.add(it["mol_id"])
            deduped.append(it)
    return deduped


def assign_stratified_cv_folds(items: List[Dict[str, Any]], n_folds: int = 5, seed: int = 42) -> List[int]:
    """Use conformer-grouped precomputed `cv5_fold` if present on all items,
    else assign deterministic molecule-level stratified 5-fold CV indices.
    """
    if items and all(it.get("fold") is not None for it in items):
        return [int(it["fold"]) % n_folds for it in items]
    rng = np.random.default_rng(seed)
    buckets: Dict[Tuple, List[int]] = {}
    for idx, it in enumerate(items):
        has_gd = any(s.get("special") == "GD" for s in it["states"])
        key = (it["source"], it["subset"], has_gd, bool(it.get("is_invest_core", False)))
        buckets.setdefault(key, []).append(idx)

    folds = [0] * len(items)
    # Deterministic sorted bucket order
    cursor = 0
    for key in sorted(buckets.keys()):
        idxs = list(buckets[key])
        rng.shuffle(idxs)
        for idx in idxs:
            folds[idx] = cursor % n_folds
            cursor += 1
    return folds


# ---------------------------------------------------------------------------
# Leak-Free Stop-Gradient Hungarian Matching & Multi-Task Physics Loss
# ---------------------------------------------------------------------------

def match_states_hungarian(
    pred_E: np.ndarray,
    pred_pct_T1: np.ndarray,
    pred_osc: Optional[np.ndarray],
    ref_states: Sequence[Dict[str, Any]],
    w_t1_cost: float = 3.0,
    w_osc_cost: float = 0.6,
    max_candidates: int = 64,
) -> List[Tuple[int, int]]:
    """Stop-gradient Hungarian assignment matching reference states -> predicted eigenstates.

    Returns list of (ref_idx, pred_idx) pairs.
    Prevents genuine doubles (%T1 < 50%) from ever falsely matching pure singlets
    by weighting character mismatch strongly in the assignment cost matrix.
    """
    m = len(ref_states)
    n_cand = min(len(pred_E), max_candidates)
    if m == 0 or n_cand == 0:
        return []

    cost = np.zeros((m, n_cand), dtype=np.float64)
    pE = pred_E[:n_cand]
    pT1 = pred_pct_T1[:n_cand]
    pF = pred_osc[:n_cand] if pred_osc is not None else None

    for i, st in enumerate(ref_states):
        ref_e = float(st["tbe_eV"])
        c = np.abs(pE - ref_e)
        ref_t1 = st.get("pct_T1")
        spec = st.get("special")
        if ref_t1 is not None:
            wt = w_t1_cost * (1.8 if spec == "GD" else 1.3 if spec == "PD" else 1.0)
            c = c + wt * np.abs(pT1 - float(ref_t1)) / 100.0
        elif spec == "GD":
            # Penalize matching pure singlets if %T1 numeric value missing
            c = c + 4.5 * np.clip((pT1 - 50.0) / 50.0, 0.0, 1.0)
        if pF is not None and st.get("osc_ref") is not None:
            ref_f = max(float(st["osc_ref"]), 0.0)
            c = c + w_osc_cost * np.abs(np.log1p(20.0 * np.clip(pF, 0.0, None)) - np.log1p(20.0 * ref_f))
        cost[i, :] = c

    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def compute_molecule_loss(
    model: LehmannTDDFTOperator,
    item: Dict[str, Any],
    w_energy: float = 1.0,
    w_gd_mult: float = 3.5,
    w_pd_mult: float = 2.0,
    w_gd_zo: float = 3.5,
    w_t1_char: float = 1.5,
    w_osc: float = 0.4,
    w_s1t1: float = 1.5,
    w_dest: float = 4.5,
    w_stat_pos: float = 5.0,
    w_reg_delta: float = 8e-3,
    w_reg_delta_diff: float = 4e-2,
    w_reg_D: float = 2e-3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable physics-informed loss for a single molecule (v2.0)."""
    mol = item["mol_tensors"]
    out = model(mol)
    device = out["E_S1_eV"].device
    dtype = out["E_S1_eV"].dtype

    loss_E = torch.tensor(0.0, dtype=dtype, device=device)
    loss_T1 = torch.tensor(0.0, dtype=dtype, device=device)
    loss_osc = torch.tensor(0.0, dtype=dtype, device=device)
    loss_gd_zo = torch.tensor(0.0, dtype=dtype, device=device)
    n_e_terms = 0.0
    n_t1_terms = 0.0
    n_osc_terms = 0.0
    n_gd_zo_terms = 0.0

    for spin_int, spin_key in ((1, "S"), (3, "T")):
        ref_states = [s for s in item["states"] if s["spin"] == spin_int]
        if not ref_states:
            continue
        res = out[spin_key]
        pred_E = res["energies_eV"]
        pred_T1 = res["pct_T1"]
        pred_T2 = res["pct_T2"]
        pred_osc = res.get("osc", None)

        with torch.no_grad():
            pairs = match_states_hungarian(
                pred_E.detach().cpu().numpy(),
                pred_T1.detach().cpu().numpy(),
                pred_osc.detach().cpu().numpy() if pred_osc is not None else None,
                ref_states,
            )

        for r_idx, p_idx in pairs:
            st = ref_states[r_idx]
            spec = st.get("special")
            mult = w_gd_mult if spec == "GD" else (w_pd_mult if spec == "PD" else 1.0)
            target_e = torch.tensor(float(st["tbe_eV"]), dtype=dtype, device=device)
            loss_E = loss_E + mult * F.smooth_l1_loss(pred_E[p_idx], target_e, beta=0.12)
            n_e_terms += mult

            if st.get("pct_T1") is not None:
                target_z = torch.tensor(float(st["pct_T1"]) / 100.0, dtype=dtype, device=device)
                # Supervising Z_n = ||c_{n,S}||^2 directly constrains dSigma/domega!
                loss_T1 = loss_T1 + mult * F.smooth_l1_loss(res["Z_qp"][p_idx], target_z, beta=0.05)
                n_t1_terms += mult

            if spin_key == "S" and pred_osc is not None and st.get("osc_ref") is not None:
                target_f = max(float(st["osc_ref"]), 0.0)
                pred_logf = torch.log1p(10.0 * pred_osc[p_idx].clamp_min(0.0))
                ref_logf = torch.tensor(math.log1p(10.0 * target_f), dtype=dtype, device=device)
                loss_osc = loss_osc + F.smooth_l1_loss(pred_logf, ref_logf, beta=0.05)
                n_osc_terms += 1.0

        # Direct ZERO-ORACLE Genuine Double-Excitation localization loss:
        # Supervises the lowest energy-ordered emergent double eigenstates (%T2 >= 40%)
        # directly against the sorted reference GD energies (eliminates unsupervised low ghost poles!).
        if not model.disable_doubles:
            gd_refs = sorted(
                [s for s in ref_states if s.get("special") == "GD"],
                key=lambda x: float(x["tbe_eV"]),
            )
            if gd_refs:
                em_idx = torch.where(pred_T2 >= 40.0)[0]
                if em_idx.numel() == 0:
                    # Fallback to top-%T2 poles if none yet exceed 40%
                    _, top_t2 = torch.topk(pred_T2, min(len(gd_refs), pred_T2.numel()))
                    em_idx = top_t2[torch.argsort(pred_E[top_t2])]
                for k_gd, st in enumerate(gd_refs):
                    if k_gd < em_idx.numel():
                        p_idx = em_idx[k_gd]
                        target_e = torch.tensor(float(st["tbe_eV"]), dtype=dtype, device=device)
                        loss_gd_zo = loss_gd_zo + F.smooth_l1_loss(pred_E[p_idx], target_e, beta=0.10)
                        if st.get("pct_T1") is not None:
                            target_z = torch.tensor(float(st["pct_T1"]) / 100.0, dtype=dtype, device=device)
                            loss_gd_zo = loss_gd_zo + 0.4 * F.smooth_l1_loss(res["Z_qp"][p_idx], target_z, beta=0.05)
                        n_gd_zo_terms += 1.0
                # Ghost-margin penalty: push ALL spurious double poles (%T2 >= 40%) strictly above ref_gd_min - 0.10 eV!
                ref_gd_min = float(gd_refs[0]["tbe_eV"])
                ghost_viol = F.relu((ref_gd_min - 0.10) - pred_E[em_idx])
                loss_gd_zo = loss_gd_zo + 2.0 * ghost_viol.pow(2).sum()
            elif spin_key == "S" and item.get("ref_S1_eV") is not None:
                # Non-GD molecule optical-window protection against spurious low %T2 >= 50% ghost poles
                em_ghost = torch.where(pred_T2 >= 50.0)[0]
                if em_ghost.numel() > 0:
                    low_viol = F.relu((float(item["ref_S1_eV"]) + 0.40) - pred_E[em_ghost])
                    loss_gd_zo = loss_gd_zo + 1.0 * low_viol.pow(2).sum()
                    n_gd_zo_terms += 0.25

    if n_e_terms > 0:
        loss_E = loss_E / n_e_terms
    if n_t1_terms > 0:
        loss_T1 = loss_T1 / n_t1_terms
    if n_osc_terms > 0:
        loss_osc = loss_osc / n_osc_terms
    if n_gd_zo_terms > 0:
        loss_gd_zo = loss_gd_zo / n_gd_zo_terms

    # Direct ZERO-MATCHING-AMBIGUITY lowest-state loss on S1, T1, and delta_EST!
    loss_lowest = torch.tensor(0.0, dtype=dtype, device=device)
    loss_dest = torch.tensor(0.0, dtype=dtype, device=device)
    has_s1 = item.get("ref_S1_eV") is not None and out["S"]["energies_eV"].numel() > 0
    has_t1 = item.get("ref_T1_eV") is not None and out["T"]["energies_eV"].numel() > 0

    if has_s1:
        t_s1 = torch.tensor(float(item["ref_S1_eV"]), dtype=dtype, device=device)
        loss_lowest = loss_lowest + F.smooth_l1_loss(out["E_S1_eV"], t_s1, beta=0.12)
        if item.get("ref_pct_T1_S1") is not None:
            t_z1 = torch.tensor(float(item["ref_pct_T1_S1"]) / 100.0, dtype=dtype, device=device)
            loss_lowest = loss_lowest + 0.5 * F.smooth_l1_loss(out["S"]["Z_qp"][0], t_z1, beta=0.05)
    if has_t1:
        t_t1 = torch.tensor(float(item["ref_T1_eV"]), dtype=dtype, device=device)
        loss_lowest = loss_lowest + F.smooth_l1_loss(out["E_T1_eV"], t_t1, beta=0.12)
        if item.get("ref_pct_T1_T1") is not None:
            t_zt1 = torch.tensor(float(item["ref_pct_T1_T1"]) / 100.0, dtype=dtype, device=device)
            loss_lowest = loss_lowest + 0.5 * F.smooth_l1_loss(out["T"]["Z_qp"][0], t_zt1, beta=0.05)
    if has_s1 and has_t1:
        ref_dest = float(item["ref_S1_eV"]) - float(item["ref_T1_eV"])
        t_dest = torch.tensor(ref_dest, dtype=dtype, device=device)
        # Slightly upweight negative/near-zero dEST regimes critical for Hund's-rule boundary
        dest_mult = 1.6 if ref_dest < 0.2 else 1.0
        loss_dest = dest_mult * F.smooth_l1_loss(out["delta_EST_eV"], t_dest, beta=0.05)

    # Physics-grounded static-dynamical separation (ONLY when doubles are active!):
    # Adiabatic static exchange 2K_HL >= 0 forbids static Hund's-rule inversion; genuine
    # dynamical self-energy Delta Sigma_ST(omega) > 0 MUST drive inversion (incl. D3h Heptazine!).
    loss_stat_pos = torch.tensor(0.0, dtype=dtype, device=device)
    reg_delta_diff = torch.tensor(0.0, dtype=dtype, device=device)
    if (not model.disable_doubles) and has_s1 and has_t1:
        e_s1_stat = safe_eigh(out["S"]["H_SS"])[0][0]
        e_t1_stat = safe_eigh(out["T"]["H_SS"])[0][0]
        dest_stat = e_s1_stat - e_t1_stat
        two_k_hl = 2.0 * mol["S_feats"][0, 1].clamp_min(0.0)
        m_stat = torch.clamp(0.45 * two_k_hl, min=0.06, max=0.14)
        loss_stat_pos = F.relu(m_stat - dest_stat).pow(2)
        d_s_diag = torch.diagonal(out["S"]["Delta"])
        d_t_diag = torch.diagonal(out["T"]["Delta"])
        reg_delta_diff = (d_s_diag[0] - d_t_diag[0]).pow(2)

    # Physical anti-collapse regularization:
    # Keep static diagonal shifts compact and anchor unobserved bi-exciton poles around D_phys!
    eff_w_reg_delta = 5e-3 if model.disable_doubles else w_reg_delta
    reg_delta = (
        torch.diagonal(out["S"]["Delta"]).pow(2).mean()
        + torch.diagonal(out["T"]["Delta"]).pow(2).mean()
        + 0.2 * ((out["S"]["Delta"] ** 2).mean() + (out["T"]["Delta"] ** 2).mean())
    )
    reg_D = torch.tensor(0.0, dtype=dtype, device=device)
    n_ds_base = mol["D_S_bare_eV"].shape[0]
    if n_ds_base > 0:
        d_ref_s = out["S"].get("D_phys", mol["D_S_bare_eV"])[:n_ds_base]
        reg_D = reg_D + ((out["S"]["D"][:n_ds_base] - d_ref_s) ** 2).mean()
    n_dt_base = mol["D_T_bare_eV"].shape[0]
    if n_dt_base > 0:
        d_ref_t = out["T"].get("D_phys", mol["D_T_bare_eV"])[:n_dt_base]
        reg_D = reg_D + ((out["T"]["D"][:n_dt_base] - d_ref_t) ** 2).mean()

    total = (
        w_energy * loss_E
        + w_gd_zo * loss_gd_zo
        + w_t1_char * loss_T1
        + w_osc * loss_osc
        + w_s1t1 * loss_lowest
        + w_dest * loss_dest
        + w_stat_pos * loss_stat_pos
        + w_reg_delta_diff * reg_delta_diff
        + eff_w_reg_delta * reg_delta
        + w_reg_D * reg_D
    )
    return total, {
        "total": float(total.item()),
        "E": float(loss_E.item()),
        "GD_zo": float(loss_gd_zo.item()),
        "T1_char": float(loss_T1.item()),
        "osc": float(loss_osc.item()),
        "lowest": float(loss_lowest.item()),
        "dest": float(loss_dest.item()),
        "stat_pos": float(loss_stat_pos.item()),
    }


# ---------------------------------------------------------------------------
# Comprehensive Leak-Free Evaluator (Suites A, B, C)
# ---------------------------------------------------------------------------

def _safe_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    if len(y_true) < 2:
        return None
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < 1e-12:
        return None
    return 1.0 - ss_res / ss_tot


def _safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    if len(y_true) < 3:
        return None
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return None
    rho, _ = spearmanr(y_true, y_pred)
    return None if math.isnan(float(rho)) else float(rho)


@torch.no_grad()
def evaluate_model_predictions(
    model: LehmannTDDFTOperator,
    items: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a trained Lehmann-TDDFT (or Static-Delta) model across Suites A, B, C.

    Suite A (Zero matching ambiguity): lowest S1, lowest T1, delta_EST = S1 - T1, INVEST classification.
    Suite B (Zero oracle): Genuine Double (GD, %T1_ref < 50%) & Partial Double (PD) emergence & MAE
            selecting strictly from emergent double eigenstates (%T2 >= 40% for GD, %T2 >= 12% for PD).
    Suite C (Full spectrum): Hungarian-matched MAE/RMSE/R2 across all states, %T1, and oscillator strengths.
    """
    model.eval()

    # Suite A accumulators
    s1_true, s1_pred = [], []
    t1_true, t1_pred = [], []
    dest_true, dest_pred = [], []
    f_s1_true, f_s1_pred = [], []
    tp = fp = fn = tn = 0

    # Suite B accumulators (Zero-oracle double excitation emergence!)
    gd_total = 0
    gd_detected = 0
    gd_errs_zero_oracle: List[float] = []
    gd_t1_errs: List[float] = []
    pd_total = 0
    pd_detected = 0
    pd_errs_zero_oracle: List[float] = []

    # Suite C accumulators (Full spectrum)
    all_e_true, all_e_pred = [], []
    s_e_true, s_e_pred = [], []
    t_e_true, t_e_pred = [], []
    v_e_errs, r_e_errs = [], []
    gd_matched_errs, pd_matched_errs = [], []
    t1_char_true, t1_char_pred = [], []
    osc_true, osc_pred = [], []

    per_mol_records: List[Dict[str, Any]] = []

    for it in items:
        mol = it["mol_tensors"]
        out = model(mol)
        se = model.evaluate_self_energy(mol)

        p_s1 = float(out["E_S1_eV"].item())
        p_t1 = float(out["E_T1_eV"].item())
        p_dest = float(out["delta_EST_eV"].item())
        p_fs1 = float(out["f_S1"].item())
        p_t1_s1 = float(out["pct_T1_S1"].item())
        p_t1_t1 = float(out["pct_T1_T1"].item())

        r_s1 = it.get("ref_S1_eV")
        r_t1 = it.get("ref_T1_eV")
        r_dest = (r_s1 - r_t1) if (r_s1 is not None and r_t1 is not None) else None

        if r_s1 is not None:
            s1_true.append(r_s1)
            s1_pred.append(p_s1)
        if r_t1 is not None:
            t1_true.append(r_t1)
            t1_pred.append(p_t1)
        if r_dest is not None:
            dest_true.append(r_dest)
            dest_pred.append(p_dest)
            if r_dest < 0.0 and p_dest < 0.0:
                tp += 1
            elif r_dest >= 0.0 and p_dest < 0.0:
                fp += 1
            elif r_dest < 0.0 and p_dest >= 0.0:
                fn += 1
            else:
                tn += 1
        if it.get("ref_f_S1") is not None:
            f_s1_true.append(float(it["ref_f_S1"]))
            f_s1_pred.append(p_fs1)

        # Suite B: Zero-Oracle Emergent Double-Excitation Evaluation
        mol_gd_details = []
        for spin_int, spin_key in ((1, "S"), (3, "T")):
            res = out[spin_key]
            pE = res["energies_eV"].cpu().numpy()
            pT1 = res["pct_T1"].cpu().numpy()
            pT2 = res["pct_T2"].cpu().numpy()

            # Genuine Doubles (GD): strictly require emergent double character %T2 >= 40%!
            gd_refs = sorted(
                [s for s in it["states"] if s["spin"] == spin_int and s.get("special") == "GD"],
                key=lambda x: x["tbe_eV"],
            )
            if gd_refs:
                emergent_gd_idx = np.where(pT2 >= 40.0)[0]  # already sorted by ascending energy!
                for k, st in enumerate(gd_refs):
                    gd_total += 1
                    if k < len(emergent_gd_idx):
                        idx = int(emergent_gd_idx[k])
                        gd_detected += 1
                        err = abs(float(pE[idx]) - float(st["tbe_eV"]))
                        gd_errs_zero_oracle.append(err)
                        if st.get("pct_T1") is not None:
                            gd_t1_errs.append(abs(float(pT1[idx]) - float(st["pct_T1"])))
                        mol_gd_details.append(
                            {
                                "label": st["label"],
                                "spin": spin_int,
                                "ref_eV": float(st["tbe_eV"]),
                                "pred_eV": float(pE[idx]),
                                "ref_pct_T1": st.get("pct_T1"),
                                "pred_pct_T1": float(pT1[idx]),
                                "pred_pct_T2": float(pT2[idx]),
                                "detected": True,
                            }
                        )
                    else:
                        mol_gd_details.append(
                            {
                                "label": st["label"],
                                "spin": spin_int,
                                "ref_eV": float(st["tbe_eV"]),
                                "pred_eV": None,
                                "detected": False,
                            }
                        )

            # Partial Doubles (PD): require emergent double admixture %T2 >= 12%
            pd_refs = sorted(
                [s for s in it["states"] if s["spin"] == spin_int and s.get("special") == "PD"],
                key=lambda x: x["tbe_eV"],
            )
            if pd_refs:
                emergent_pd_idx = np.where((pT2 >= 12.0) & (pT2 <= 75.0))[0]
                if len(emergent_pd_idx) == 0:
                    emergent_pd_idx = np.where(pT2 >= 12.0)[0]
                for k, st in enumerate(pd_refs):
                    pd_total += 1
                    if len(emergent_pd_idx) > 0:
                        # Zero-oracle energy-ordered pick among emergent partial doubles
                        pick = int(emergent_pd_idx[min(k, len(emergent_pd_idx) - 1)])
                        pd_detected += 1
                        pd_errs_zero_oracle.append(abs(float(pE[pick]) - float(st["tbe_eV"])))

        # Suite C: Full-Spectrum Hungarian-Matched Evaluation
        for spin_int, spin_key in ((1, "S"), (3, "T")):
            ref_states = [s for s in it["states"] if s["spin"] == spin_int]
            if not ref_states:
                continue
            res = out[spin_key]
            pE = res["energies_eV"].cpu().numpy()
            pT1 = res["pct_T1"].cpu().numpy()
            pF = res["osc"].cpu().numpy() if "osc" in res else None
            pairs = match_states_hungarian(pE, pT1, pF, ref_states)
            for r_idx, p_idx in pairs:
                st = ref_states[r_idx]
                re = float(st["tbe_eV"])
                pe = float(pE[p_idx])
                err = abs(pe - re)
                all_e_true.append(re)
                all_e_pred.append(pe)
                if spin_int == 1:
                    s_e_true.append(re)
                    s_e_pred.append(pe)
                else:
                    t_e_true.append(re)
                    t_e_pred.append(pe)
                if st.get("vr") == "V":
                    v_e_errs.append(err)
                elif st.get("vr") == "R":
                    r_e_errs.append(err)
                if st.get("special") == "GD":
                    gd_matched_errs.append(err)
                elif st.get("special") == "PD":
                    pd_matched_errs.append(err)
                if st.get("pct_T1") is not None:
                    t1_char_true.append(float(st["pct_T1"]))
                    t1_char_pred.append(float(pT1[p_idx]))
                if spin_key == "S" and pF is not None and st.get("osc_ref") is not None:
                    osc_true.append(float(st["osc_ref"]))
                    osc_pred.append(float(pF[p_idx]))

        per_mol_records.append(
            {
                "mol_id": it["mol_id"],
                "name": it["name"],
                "source": it["source"],
                "subset": it["subset"],
                "family": it["family"],
                "n_heavy": it["n_heavy"],
                "ref_S1_eV": r_s1,
                "pred_S1_eV": p_s1,
                "ref_T1_eV": r_t1,
                "pred_T1_eV": p_t1,
                "ref_delta_EST_eV": r_dest,
                "pred_delta_EST_eV": p_dest,
                "bare_tda_delta_EST_eV": float(se["delta_EST_bare_tda_eV"].item()),
                "two_K_HL_eV": float(se["two_K_HL_eV"].item()),
                "delta_EST_static_eV": float(se["delta_EST_static_eV"].item()),
                "delta_Sigma_ST_eV": float(se["delta_Sigma_ST_eV"].item()),
                "Sigma_S1_dyn_eV": float(se["Sigma_S1_dyn_eV"].item()),
                "Sigma_S1_TT_eV": float(se["Sigma_S1_TT_eV"].item()),
                "Sigma_S1_SS_eV": float(se["Sigma_S1_SS_eV"].item()),
                "Sigma_S1_TT_anti_eV": float(se["Sigma_S1_TT_anti_eV"].item()),
                "Sigma_T1_dyn_eV": float(se["Sigma_T1_dyn_eV"].item()),
                "Z_S1": float(se["Z_S1"].item()),
                "pred_f_S1": p_fs1,
                "ref_f_S1": it.get("ref_f_S1"),
                "pred_pct_T1_S1": p_t1_s1,
                "pred_pct_T1_T1": p_t1_t1,
                "gd_details": mol_gd_details,
            }
        )

    def _mae_rmse(yt, yp):
        if not yt:
            return None, None
        diff = np.asarray(yp) - np.asarray(yt)
        return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))

    s1_mae, s1_rmse = _mae_rmse(s1_true, s1_pred)
    t1_mae, t1_rmse = _mae_rmse(t1_true, t1_pred)
    dest_mae, dest_rmse = _mae_rmse(dest_true, dest_pred)
    all_mae, all_rmse = _mae_rmse(all_e_true, all_e_pred)
    s_mae, _ = _mae_rmse(s_e_true, s_e_pred)
    t_mae, _ = _mae_rmse(t_e_true, t_e_pred)
    t1c_mae, _ = _mae_rmse(t1_char_true, t1_char_pred)
    osc_mae, _ = _mae_rmse(osc_true, osc_pred)

    # MCC for INVEST classification
    denom_mcc = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = float((tp * tn - fp * fn) / denom_mcc) if denom_mcc > 0 else None
    n_cls = tp + fp + fn + tn
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None

    metrics = {
        "n_molecules": len(items),
        "suite_A_lowest_zero_matching": {
            "n_S1": len(s1_true),
            "S1_MAE_eV": s1_mae,
            "S1_RMSE_eV": s1_rmse,
            "S1_R2": _safe_r2(s1_true, s1_pred),
            "n_T1": len(t1_true),
            "T1_MAE_eV": t1_mae,
            "T1_RMSE_eV": t1_rmse,
            "T1_R2": _safe_r2(t1_true, t1_pred),
            "n_dEST": len(dest_true),
            "dEST_MAE_eV": dest_mae,
            "dEST_RMSE_eV": dest_rmse,
            "dEST_R2": _safe_r2(dest_true, dest_pred),
            "INVEST_TP": tp,
            "INVEST_FP": fp,
            "INVEST_FN": fn,
            "INVEST_TN": tn,
            "INVEST_accuracy": float((tp + tn) / n_cls) if n_cls > 0 else None,
            "INVEST_precision": prec,
            "INVEST_recall": rec,
            "INVEST_F1": f1,
            "INVEST_MCC": mcc,
            "f_S1_MAE": _mae_rmse(f_s1_true, f_s1_pred)[0],
            "f_S1_spearman": _safe_spearman(f_s1_true, f_s1_pred),
        },
        "suite_B_double_excitations_zero_oracle": {
            "GD_total": gd_total,
            "GD_detected": gd_detected,
            "GD_detection_rate": float(gd_detected / gd_total) if gd_total > 0 else None,
            "GD_zero_oracle_MAE_eV": float(np.mean(gd_errs_zero_oracle)) if gd_errs_zero_oracle else None,
            "GD_pct_T1_MAE": float(np.mean(gd_t1_errs)) if gd_t1_errs else None,
            "PD_total": pd_total,
            "PD_detected": pd_detected,
            "PD_detection_rate": float(pd_detected / pd_total) if pd_total > 0 else None,
            "PD_zero_oracle_MAE_eV": float(np.mean(pd_errs_zero_oracle)) if pd_errs_zero_oracle else None,
        },
        "suite_C_full_spectrum_matched": {
            "n_states": len(all_e_true),
            "all_states_MAE_eV": all_mae,
            "all_states_RMSE_eV": all_rmse,
            "all_states_R2": _safe_r2(all_e_true, all_e_pred),
            "singlet_MAE_eV": s_mae,
            "triplet_MAE_eV": t_mae,
            "valence_MAE_eV": float(np.mean(v_e_errs)) if v_e_errs else None,
            "rydberg_MAE_eV": float(np.mean(r_e_errs)) if r_e_errs else None,
            "GD_matched_MAE_eV": float(np.mean(gd_matched_errs)) if gd_matched_errs else None,
            "PD_matched_MAE_eV": float(np.mean(pd_matched_errs)) if pd_matched_errs else None,
            "pct_T1_n": len(t1_char_true),
            "pct_T1_MAE": t1c_mae,
            "pct_T1_spearman": _safe_spearman(t1_char_true, t1_char_pred),
            "osc_n": len(osc_true),
            "osc_MAE": osc_mae,
            "osc_R2": _safe_r2(osc_true, osc_pred),
            "osc_spearman": _safe_spearman(osc_true, osc_pred),
        },
    }
    return metrics, per_mol_records


# ---------------------------------------------------------------------------
# Training Loop for LehmannTDDFTOperator (Full or Static-Delta Ablation)
# ---------------------------------------------------------------------------

def train_lehmann_model(
    train_items: Sequence[Dict[str, Any]],
    val_items: Sequence[Dict[str, Any]],
    disable_doubles: bool = False,
    hidden_dim: int = 48,
    n_layers: int = 2,
    dropout: float = 0.05,
    epochs: int = 100,
    lr: float = 4e-3,
    weight_decay: float = 2e-4,
    batch_size: int = 12,
    seed: int = 42,
    verbose: bool = True,
    dtype: torch.dtype = torch.float32,
    use_antisym_singlet_channel: bool = True,
    antisym_version: int = 2,
    use_phys_D_anchor: bool = True,
    max_delta_D_eV: float = 6.5,
    loss_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[LehmannTDDFTOperator, Dict[str, Any]]:
    """Train a Lehmann-TDDFT v2.0 (or Static-Delta) operator cleanly with early-best selection."""
    # Avoid macOS OpenMP thread contention on small per-molecule eigensolves
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Infer dtype from train_items if present
    if train_items:
        dtype = train_items[0]["mol_tensors"]["omega_S_eV"].dtype

    model = LehmannTDDFTOperator(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        disable_doubles=disable_doubles,
        max_delta_D_eV=max_delta_D_eV,
        init_alpha_scale=0.22,
        use_antisym_singlet_channel=(use_antisym_singlet_channel and not disable_doubles),
        antisym_version=antisym_version,
        use_phys_D_anchor=(use_phys_D_anchor and not disable_doubles),
    ).to(dtype=dtype)

    # Fit strictly elementwise normalizers ONLY on train_items (ZERO leakage!)
    model.fit_normalizers([it["mol_tensors"] for it in train_items])
    model.precompute_raw_features([it["mol_tensors"] for it in val_items])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.02)

    best_score = float("inf")
    best_state = None
    best_epoch = 0
    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_items))
        opt.zero_grad()
        ep_losses = []

        for step_i, idx in enumerate(order):
            loss, info = compute_molecule_loss(model, train_items[idx], **(loss_kwargs or {}))
            if not torch.isfinite(loss):
                continue
            (loss / batch_size).backward()
            ep_losses.append(info["total"])

            if (step_i + 1) % batch_size == 0 or (step_i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                opt.zero_grad()

        sched.step()

        # Evaluate on val_items every 5 epochs (or last 10 epochs)
        if val_items and ((ep + 1) % 5 == 0 or ep >= epochs - 10):
            val_metrics, _ = evaluate_model_predictions(model, val_items)
            mA = val_metrics["suite_A_lowest_zero_matching"]
            mB = val_metrics["suite_B_double_excitations_zero_oracle"]
            mC = val_metrics["suite_C_full_spectrum_matched"]
            # Composite validation score (lower is better)
            parts = []
            if mC["all_states_MAE_eV"] is not None:
                parts.append(mC["all_states_MAE_eV"])
            if mA["dEST_MAE_eV"] is not None:
                parts.append(1.5 * mA["dEST_MAE_eV"])
            if mB["GD_zero_oracle_MAE_eV"] is not None:
                parts.append(0.4 * min(float(mB["GD_zero_oracle_MAE_eV"]), 1.5))
            elif (not disable_doubles) and mB["GD_total"] > 0 and mB["GD_detected"] == 0:
                parts.append(2.0)
            score = float(np.mean(parts)) if parts else float(np.mean(ep_losses))
            history.append(
                {
                    "epoch": ep + 1,
                    "train_loss": float(np.mean(ep_losses)) if ep_losses else float("nan"),
                    "val_score": score,
                    "val_all_MAE": mC["all_states_MAE_eV"] or float("nan"),
                    "val_dEST_MAE": mA["dEST_MAE_eV"] or float("nan"),
                }
            )
            min_best_ep = min(30, max(epochs // 2, 1))
            if ep + 1 >= min_best_ep and score < best_score:
                best_score = score
                best_epoch = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose and ((ep + 1) % 20 == 0 or ep == epochs - 1):
                tag = "Static-Delta" if disable_doubles else "Lehmann-TDDFT"
                print(
                    f"    [{tag} seed={seed}] ep {ep+1:3d}/{epochs} ({time.time()-t0:.0f}s) | "
                    f"tr_loss={np.mean(ep_losses):.4f} | val_all_MAE={mC['all_states_MAE_eV'] or 0:.4f} | "
                    f"val_dEST_MAE={mA['dEST_MAE_eV'] or 0:.4f} | best={best_score:.4f}",
                    flush=True,
                )
        elif not val_items:
            tr_l = float(np.mean(ep_losses)) if ep_losses else float("inf")
            if tr_l < best_score:
                best_score = tr_l
                best_epoch = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    wall_s = time.time() - t0
    if verbose:
        tag = "Static-Delta" if disable_doubles else "Lehmann-TDDFT"
        print(f"  [{tag}] trained {epochs} ep in {wall_s:.1f}s (best_ep={best_epoch}, best_val={best_score:.4f})", flush=True)

    return model, {"best_epoch": best_epoch, "best_val_score": best_score, "wall_time_s": wall_s, "history": history}


# ---------------------------------------------------------------------------
# Experiment Runners (CV5, OOD-Large, OOD-Doubles, OOD-INVEST, Ensemble)
# ---------------------------------------------------------------------------

def _aggregate_metrics_from_predictions(
    all_records: List[Dict[str, Any]],
    models_and_splits: List[Tuple[LehmannTDDFTOperator, List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    """Compute exact pooled out-of-fold metrics across all 5 test folds."""
    # Re-run exact pooled evaluation by collecting predictions fold-by-fold
    pooled_eval_metrics = []
    # Aggregate Suite A directly from `all_records` (exact pooled across all molecules!)
    s1_t, s1_p, t1_t, t1_p, d_t, d_p, f_t, f_p = [], [], [], [], [], [], [], []
    tp = fp = fn = tn = 0
    for r in all_records:
        if r["ref_S1_eV"] is not None:
            s1_t.append(r["ref_S1_eV"])
            s1_p.append(r["pred_S1_eV"])
        if r["ref_T1_eV"] is not None:
            t1_t.append(r["ref_T1_eV"])
            t1_p.append(r["pred_T1_eV"])
        if r["ref_delta_EST_eV"] is not None:
            d_t.append(r["ref_delta_EST_eV"])
            d_p.append(r["pred_delta_EST_eV"])
            if r["ref_delta_EST_eV"] < 0 and r["pred_delta_EST_eV"] < 0:
                tp += 1
            elif r["ref_delta_EST_eV"] >= 0 and r["pred_delta_EST_eV"] < 0:
                fp += 1
            elif r["ref_delta_EST_eV"] < 0 and r["pred_delta_EST_eV"] >= 0:
                fn += 1
            else:
                tn += 1
        if r["ref_f_S1"] is not None:
            f_t.append(r["ref_f_S1"])
            f_p.append(r["pred_f_S1"])

    # Also pool Suites B & C directly by invoking each fold model on its own test fold
    gd_tot = gd_det = pd_tot = pd_det = 0
    gd_zo_errs, gd_t1_errs, pd_zo_errs = [], [], []
    all_t, all_p, s_t, s_p, t_t, t_p = [], [], [], [], [], []
    v_errs, r_errs, gd_m_errs, pd_m_errs = [], [], [], []
    t1c_t, t1c_p, osc_t, osc_p = [], [], [], []
    fold_summaries = []

    for model, fold_items in models_and_splits:
        fm, _ = evaluate_model_predictions(model, fold_items)
        fold_summaries.append(fm)
        # Re-accumulate exact state-level pooled stats
        for it in fold_items:
            out = model(it["mol_tensors"])
            for spin_int, spin_key in ((1, "S"), (3, "T")):
                res = out[spin_key]
                pE = res["energies_eV"].detach().cpu().numpy()
                pT1 = res["pct_T1"].detach().cpu().numpy()
                pT2 = res["pct_T2"].detach().cpu().numpy()
                pF = res["osc"].detach().cpu().numpy() if "osc" in res else None

                gd_refs = sorted(
                    [s for s in it["states"] if s["spin"] == spin_int and s.get("special") == "GD"],
                    key=lambda x: x["tbe_eV"],
                )
                if gd_refs:
                    em = np.where(pT2 >= 40.0)[0]
                    for k, st in enumerate(gd_refs):
                        gd_tot += 1
                        if k < len(em):
                            idx = int(em[k])
                            gd_det += 1
                            gd_zo_errs.append(abs(float(pE[idx]) - float(st["tbe_eV"])))
                            if st.get("pct_T1") is not None:
                                gd_t1_errs.append(abs(float(pT1[idx]) - float(st["pct_T1"])))

                pd_refs = sorted(
                    [s for s in it["states"] if s["spin"] == spin_int and s.get("special") == "PD"],
                    key=lambda x: x["tbe_eV"],
                )
                if pd_refs:
                    em_pd = np.where((pT2 >= 12.0) & (pT2 <= 75.0))[0]
                    if len(em_pd) == 0:
                        em_pd = np.where(pT2 >= 12.0)[0]
                    for k, st in enumerate(pd_refs):
                        pd_tot += 1
                        if len(em_pd) > 0:
                            idx = int(em_pd[min(k, len(em_pd) - 1)])
                            pd_det += 1
                            pd_zo_errs.append(abs(float(pE[idx]) - float(st["tbe_eV"])))

                ref_states = [s for s in it["states"] if s["spin"] == spin_int]
                if not ref_states:
                    continue
                pairs = match_states_hungarian(pE, pT1, pF, ref_states)
                for r_idx, p_idx in pairs:
                    st = ref_states[r_idx]
                    re, pe = float(st["tbe_eV"]), float(pE[p_idx])
                    err = abs(pe - re)
                    all_t.append(re)
                    all_p.append(pe)
                    if spin_int == 1:
                        s_t.append(re)
                        s_p.append(pe)
                    else:
                        t_t.append(re)
                        t_p.append(pe)
                    if st.get("vr") == "V":
                        v_errs.append(err)
                    elif st.get("vr") == "R":
                        r_errs.append(err)
                    if st.get("special") == "GD":
                        gd_m_errs.append(err)
                    elif st.get("special") == "PD":
                        pd_m_errs.append(err)
                    if st.get("pct_T1") is not None:
                        t1c_t.append(float(st["pct_T1"]))
                        t1c_p.append(float(pT1[p_idx]))
                    if spin_key == "S" and pF is not None and st.get("osc_ref") is not None:
                        osc_t.append(float(st["osc_ref"]))
                        osc_p.append(float(pF[p_idx]))

    def _mae_rmse(yt, yp):
        if not yt:
            return None, None
        diff = np.asarray(yp) - np.asarray(yt)
        return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))

    s1_mae, s1_rmse = _mae_rmse(s1_t, s1_p)
    t1_mae, t1_rmse = _mae_rmse(t1_t, t1_p)
    d_mae, d_rmse = _mae_rmse(d_t, d_p)
    all_mae, all_rmse = _mae_rmse(all_t, all_p)
    denom_mcc = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = float((tp * tn - fp * fn) / denom_mcc) if denom_mcc > 0 else None
    n_cls = tp + fp + fn + tn
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else None

    return {
        "n_molecules": len(all_records),
        "suite_A_lowest_zero_matching": {
            "n_S1": len(s1_t),
            "S1_MAE_eV": s1_mae,
            "S1_RMSE_eV": s1_rmse,
            "S1_R2": _safe_r2(s1_t, s1_p),
            "n_T1": len(t1_t),
            "T1_MAE_eV": t1_mae,
            "T1_RMSE_eV": t1_rmse,
            "T1_R2": _safe_r2(t1_t, t1_p),
            "n_dEST": len(d_t),
            "dEST_MAE_eV": d_mae,
            "dEST_RMSE_eV": d_rmse,
            "dEST_R2": _safe_r2(d_t, d_p),
            "INVEST_TP": tp,
            "INVEST_FP": fp,
            "INVEST_FN": fn,
            "INVEST_TN": tn,
            "INVEST_accuracy": float((tp + tn) / n_cls) if n_cls > 0 else None,
            "INVEST_precision": prec,
            "INVEST_recall": rec,
            "INVEST_F1": f1,
            "INVEST_MCC": mcc,
            "f_S1_MAE": _mae_rmse(f_t, f_p)[0],
            "f_S1_spearman": _safe_spearman(f_t, f_p),
        },
        "suite_B_double_excitations_zero_oracle": {
            "GD_total": gd_tot,
            "GD_detected": gd_det,
            "GD_detection_rate": float(gd_det / gd_tot) if gd_tot > 0 else None,
            "GD_zero_oracle_MAE_eV": float(np.mean(gd_zo_errs)) if gd_zo_errs else None,
            "GD_pct_T1_MAE": float(np.mean(gd_t1_errs)) if gd_t1_errs else None,
            "PD_total": pd_tot,
            "PD_detected": pd_det,
            "PD_detection_rate": float(pd_det / pd_tot) if pd_tot > 0 else None,
            "PD_zero_oracle_MAE_eV": float(np.mean(pd_zo_errs)) if pd_zo_errs else None,
        },
        "suite_C_full_spectrum_matched": {
            "n_states": len(all_t),
            "all_states_MAE_eV": all_mae,
            "all_states_RMSE_eV": all_rmse,
            "all_states_R2": _safe_r2(all_t, all_p),
            "singlet_MAE_eV": _mae_rmse(s_t, s_p)[0],
            "triplet_MAE_eV": _mae_rmse(t_t, t_p)[0],
            "valence_MAE_eV": float(np.mean(v_errs)) if v_errs else None,
            "rydberg_MAE_eV": float(np.mean(r_errs)) if r_errs else None,
            "GD_matched_MAE_eV": float(np.mean(gd_m_errs)) if gd_m_errs else None,
            "PD_matched_MAE_eV": float(np.mean(pd_m_errs)) if pd_m_errs else None,
            "pct_T1_n": len(t1c_t),
            "pct_T1_MAE": _mae_rmse(t1c_t, t1c_p)[0],
            "pct_T1_spearman": _safe_spearman(t1c_t, t1c_p),
            "osc_n": len(osc_t),
            "osc_MAE": _mae_rmse(osc_t, osc_p)[0],
            "osc_R2": _safe_r2(osc_t, osc_p),
            "osc_spearman": _safe_spearman(osc_t, osc_p),
        },
        "per_fold_dEST_MAEs": [f["suite_A_lowest_zero_matching"]["dEST_MAE_eV"] for f in fold_summaries],
        "per_fold_all_MAEs": [f["suite_C_full_spectrum_matched"]["all_states_MAE_eV"] for f in fold_summaries],
    }


def _eval_schnet_preds(preds: List[Dict[str, Any]], items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pred_map = {p["mol_id"]: p for p in preds}
    s1_t, s1_p, t1_t, t1_p, d_t, d_p = [], [], [], [], [], []
    tp = fp = fn = tn = 0
    for it in items:
        p = pred_map.get(it["mol_id"])
        if p is None:
            continue
        if it.get("ref_S1_eV") is not None:
            s1_t.append(it["ref_S1_eV"])
            s1_p.append(p["E_S1_eV"])
        if it.get("ref_T1_eV") is not None:
            t1_t.append(it["ref_T1_eV"])
            t1_p.append(p["E_T1_eV"])
        if it.get("ref_S1_eV") is not None and it.get("ref_T1_eV") is not None:
            rd = it["ref_S1_eV"] - it["ref_T1_eV"]
            pd = p["delta_EST_eV"]
            d_t.append(rd)
            d_p.append(pd)
            if rd < 0 and pd < 0:
                tp += 1
            elif rd >= 0 and pd < 0:
                fp += 1
            elif rd < 0 and pd >= 0:
                fn += 1
            else:
                tn += 1

    def _mae(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) if a else None

    denom_mcc = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    return {
        "n_S1": len(s1_t),
        "S1_MAE_eV": _mae(s1_t, s1_p),
        "n_T1": len(t1_t),
        "T1_MAE_eV": _mae(t1_t, t1_p),
        "n_dEST": len(d_t),
        "dEST_MAE_eV": _mae(d_t, d_p),
        "dEST_R2": _safe_r2(d_t, d_p),
        "INVEST_TP": tp,
        "INVEST_FP": fp,
        "INVEST_FN": fn,
        "INVEST_TN": tn,
        "INVEST_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "INVEST_MCC": float((tp * tn - fp * fn) / denom_mcc) if denom_mcc > 0 else None,
        "GD_detection_rate": 0.0,
        "note": "3D GNN predicts scalar (S1, T1) only; cannot emit double excitations, %T1, or oscillator strengths.",
    }


def _run_single_cv_fold_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone worker process for 1 CV fold (macOS spawn-safe, 1 CPU thread)."""
    torch.set_num_threads(1)
    f = cfg["fold"]
    items = load_unified_dataset(cfg["questdb_json"], cfg["hepta_json"], cfg["features_dir"])
    folds = assign_stratified_cv_folds(items, n_folds=5, seed=cfg["seed"])
    models_dir = Path(cfg["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    tr_all = [it for it, fl in zip(items, folds) if fl != f]
    te_items = [it for it, fl in zip(items, folds) if fl == f]
    rng_val = np.random.default_rng(cfg["seed"] + 100 + f)
    perm = rng_val.permutation(len(tr_all))
    n_val = max(1, int(0.12 * len(tr_all)))
    val_idx = set(perm[:n_val].tolist())
    tr_items = [tr_all[i] for i in range(len(tr_all)) if i not in val_idx]
    va_items = [tr_all[i] for i in range(len(tr_all)) if i in val_idx]

    leh_ckpt = str(models_dir / f"lehmann_cv_fold{f}.pt")
    if cfg.get("resume") and os.path.exists(leh_ckpt):
        print(f"  [Fold {f}] Resuming existing {leh_ckpt}", flush=True)
        m_leh, _ = load_checkpoint(leh_ckpt, dtype=torch.float32)
    else:
        m_leh, info_leh = train_lehmann_model(
            tr_items, va_items, disable_doubles=False, epochs=cfg["epochs"], seed=cfg["seed"] + f
        )
        save_checkpoint(m_leh, leh_ckpt, extra={"fold": f, "train_info": info_leh})
    _, recs_l = evaluate_model_predictions(m_leh, te_items)
    for r in recs_l:
        r["cv_fold"] = f

    sta_ckpt = str(models_dir / f"static_delta_cv_fold{f}.pt")
    if cfg.get("resume") and os.path.exists(sta_ckpt):
        print(f"  [Fold {f}] Resuming existing {sta_ckpt}", flush=True)
        m_sta, _ = load_checkpoint(sta_ckpt, dtype=torch.float32)
    else:
        m_sta, info_sta = train_lehmann_model(
            tr_items, va_items, disable_doubles=True, epochs=cfg["epochs"], seed=cfg["seed"] + f
        )
        save_checkpoint(m_sta, sta_ckpt, extra={"fold": f, "train_info": info_sta})
    _, recs_s = evaluate_model_predictions(m_sta, te_items)
    for r in recs_s:
        r["cv_fold"] = f

    gnn_ckpt = str(models_dir / f"schnet3d_cv_fold{f}.pt")
    if cfg.get("resume") and os.path.exists(gnn_ckpt):
        print(f"  [Fold {f}] Resuming existing {gnn_ckpt}", flush=True)
        m_gnn = SchNet3DBaseline(hidden_dim=64, n_layers=4, n_rbf=32, r_cutoff=6.0, dropout=0.05)
        m_gnn.load_state_dict(torch.load(gnn_ckpt, map_location="cpu", weights_only=True))
        m_gnn.eval()
    else:
        m_gnn, _ = train_schnet_baseline(tr_items, va_items, epochs=cfg["schnet_epochs"], seed=cfg["seed"] + f)
        torch.save(m_gnn.state_dict(), gnn_ckpt)
    gnn_preds = predict_schnet_baseline(m_gnn, te_items)

    return {
        "fold": f,
        "leh_ckpt": leh_ckpt,
        "sta_ckpt": sta_ckpt,
        "gnn_ckpt": gnn_ckpt,
        "leh_recs": recs_l,
        "sta_recs": recs_s,
        "gnn_preds": gnn_preds,
    }


def _export_cv5_csv_table(summary: Dict[str, Any], out_csv: Path) -> None:
    cols = [
        "Method", "S1_MAE_eV", "T1_MAE_eV", "dEST_MAE_eV", "dEST_R2",
        "INVEST_recall", "INVEST_MCC", "GD_detection_rate", "GD_MAE_eV",
        "All_States_MAE_eV", "pct_T1_MAE", "osc_MAE", "osc_R2",
    ]
    rows = []
    for label in ("Lehmann-TDDFT", "Static-Delta-TDDFT (Ablation V=0)"):
        m = summary[label]
        A = m["suite_A_lowest_zero_matching"]
        B = m["suite_B_double_excitations_zero_oracle"]
        C = m["suite_C_full_spectrum_matched"]
        rows.append(
            {
                "Method": label,
                "S1_MAE_eV": A["S1_MAE_eV"],
                "T1_MAE_eV": A["T1_MAE_eV"],
                "dEST_MAE_eV": A["dEST_MAE_eV"],
                "dEST_R2": A["dEST_R2"],
                "INVEST_recall": A["INVEST_recall"],
                "INVEST_MCC": A["INVEST_MCC"],
                "GD_detection_rate": B["GD_detection_rate"],
                "GD_MAE_eV": B["GD_zero_oracle_MAE_eV"],
                "All_States_MAE_eV": C["all_states_MAE_eV"],
                "pct_T1_MAE": C["pct_T1_MAE"],
                "osc_MAE": C["osc_MAE"],
                "osc_R2": C["osc_R2"],
            }
        )
    gnn = summary["SchNet3D (Geometric GNN Baseline)"]
    rows.append(
        {
            "Method": "SchNet3D (3D GNN Baseline)",
            "S1_MAE_eV": gnn["S1_MAE_eV"],
            "T1_MAE_eV": gnn["T1_MAE_eV"],
            "dEST_MAE_eV": gnn["dEST_MAE_eV"],
            "dEST_R2": gnn["dEST_R2"],
            "INVEST_recall": gnn["INVEST_recall"],
            "INVEST_MCC": gnn["INVEST_MCC"],
            "GD_detection_rate": 0.0,
            "GD_MAE_eV": None,
            "All_States_MAE_eV": None,
            "pct_T1_MAE": None,
            "osc_MAE": None,
            "osc_R2": None,
        }
    )
    for m_name, qm in summary.get("QM_Baselines", {}).items():
        rows.append(
            {
                "Method": m_name,
                "S1_MAE_eV": qm.get("S1_MAE_eV"),
                "T1_MAE_eV": qm.get("T1_MAE_eV"),
                "dEST_MAE_eV": qm.get("dEST_MAE_eV"),
                "dEST_R2": None,
                "INVEST_recall": qm.get("INVEST_recall"),
                "INVEST_MCC": None,
                "GD_detection_rate": 0.0 if qm.get("GD_detectable") is False else None,
                "GD_MAE_eV": qm.get("GD_MAE_eV"),
                "All_States_MAE_eV": qm.get("all_states_MAE_eV"),
                "pct_T1_MAE": None,
                "osc_MAE": None,
                "osc_R2": None,
            }
        )
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_cv5_experiment(
    items: List[Dict[str, Any]],
    models_dir: Path,
    bench_dir: Path,
    epochs: int = 160,
    schnet_epochs: int = 200,
    seed: int = 42,
    workers: int = 1,
    resume: bool = False,
    questdb_json: str = "data/processed/questdb_clean.json",
    hepta_json: str = "data/processed/hepta_invest_clean.json",
    features_dir: str = "data/processed/features",
) -> Dict[str, Any]:
    """Run complete 5-Fold Molecule-Level CV for Lehmann-TDDFT, Static-Delta, SchNet3D & QM baselines."""
    print(f"\n=== EXPERIMENT 1: 5-Fold Molecule-Level CV ({len(items)} molecules, workers={workers}, resume={resume}) ===", flush=True)
    folds = assign_stratified_cv_folds(items, n_folds=5, seed=seed)

    lehmann_models_splits = []
    static_models_splits = []
    lehmann_oof_records: List[Dict[str, Any]] = []
    static_oof_records: List[Dict[str, Any]] = []
    schnet_oof_preds: List[Dict[str, Any]] = []
    schnet_fold0: Optional[SchNet3DBaseline] = None
    manifest_paths = []

    cfgs = [
        {
            "fold": f,
            "questdb_json": questdb_json,
            "hepta_json": hepta_json,
            "features_dir": features_dir,
            "models_dir": str(models_dir),
            "epochs": epochs,
            "schnet_epochs": schnet_epochs,
            "seed": seed,
            "resume": resume,
        }
        for f in range(5)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, 5)) as ex:
            fold_outs = list(ex.map(_run_single_cv_fold_worker, cfgs))
    else:
        fold_outs = [_run_single_cv_fold_worker(c) for c in cfgs]

    for fo in sorted(fold_outs, key=lambda x: x["fold"]):
        f = fo["fold"]
        te_items = [it for it, fl in zip(items, folds) if fl == f]
        manifest_paths.append(fo["leh_ckpt"])
        lehmann_oof_records.extend(fo["leh_recs"])
        static_oof_records.extend(fo["sta_recs"])
        schnet_oof_preds.extend(fo["gnn_preds"])
        m_leh, _ = load_checkpoint(fo["leh_ckpt"], dtype=torch.float32)
        m_sta, _ = load_checkpoint(fo["sta_ckpt"], dtype=torch.float32)
        lehmann_models_splits.append((m_leh, te_items))
        static_models_splits.append((m_sta, te_items))
        if f == 0:
            schnet_fold0 = SchNet3DBaseline(hidden_dim=64, n_layers=4, n_rbf=32, r_cutoff=6.0, dropout=0.05)
            schnet_fold0.load_state_dict(torch.load(fo["gnn_ckpt"], map_location="cpu", weights_only=True))
            schnet_fold0.eval()

    # Aggregate pooled out-of-fold metrics
    lehmann_cv = _aggregate_metrics_from_predictions(lehmann_oof_records, lehmann_models_splits)
    static_cv = _aggregate_metrics_from_predictions(static_oof_records, static_models_splits)
    schnet_cv = _eval_schnet_preds(schnet_oof_preds, items)
    qm_baselines = evaluate_published_qm_baselines(items)

    # Save 5-fold ensemble manifest for downstream discovery (`invest-scout-lib`) & CI (`ci-photochem`)
    manifest = {
        "ensemble_type": "5-fold_CV_Lehmann_TDDFT",
        "checkpoints": manifest_paths,
        "n_models": len(manifest_paths),
        "cv5_summary_S1_MAE_eV": lehmann_cv["suite_A_lowest_zero_matching"]["S1_MAE_eV"],
        "cv5_summary_dEST_MAE_eV": lehmann_cv["suite_A_lowest_zero_matching"]["dEST_MAE_eV"],
        "cv5_summary_all_states_MAE_eV": lehmann_cv["suite_C_full_spectrum_matched"]["all_states_MAE_eV"],
    }
    (models_dir / "lehmann_ensemble_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save out-of-fold predictions & Quantum Phase Diagram CSV
    (bench_dir / "cv5_lehmann_predictions.json").write_text(json.dumps(lehmann_oof_records, indent=2))
    (bench_dir / "cv5_static_delta_predictions.json").write_text(json.dumps(static_oof_records, indent=2))
    (bench_dir / "cv5_schnet3d_predictions.json").write_text(json.dumps(schnet_oof_preds, indent=2))

    phase_cols = [
        "mol_id", "name", "source", "subset", "family", "n_heavy", "cv_fold",
        "ref_S1_eV", "pred_S1_eV", "ref_T1_eV", "pred_T1_eV",
        "ref_delta_EST_eV", "pred_delta_EST_eV", "bare_tda_delta_EST_eV",
        "two_K_HL_eV", "delta_EST_static_eV", "delta_Sigma_ST_eV",
        "Sigma_S1_dyn_eV", "Sigma_S1_TT_eV", "Sigma_S1_SS_eV", "Sigma_T1_dyn_eV",
        "Z_S1", "pred_f_S1", "ref_f_S1", "pred_pct_T1_S1", "pred_pct_T1_T1",
    ]
    with open(bench_dir / "quantum_phase_diagram.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=phase_cols)
        w.writeheader()
        for r in lehmann_oof_records:
            w.writerow({k: r.get(k) for k in phase_cols})

    # Size-intensivity contrast audit: SchNet3D vs Lehmann-TDDFT
    size_audit = {}
    if schnet_fold0 is not None and len(items) >= 2:
        # Pick heptazine (or largest) and ethylene/smallest for dramatic real-molecule test
        by_id = {it["mol_id"].lower(): it for it in items}
        m_a = next((it for it in items if "heptazine" in it["name"].lower() or "2t-7n" == it["name"].lower()), items[0])
        m_b = next((it for it in items if "ethylene" in it["name"].lower() or "ethene" in it["name"].lower()), items[-1])
        if m_a["mol_id"] == m_b["mol_id"]:
            m_b = items[-1]
        size_audit["schnet3d_dimer_R100A"] = {
            "mol_A": m_a["mol_id"],
            "mol_B": m_b["mol_id"],
            **audit_gnn_size_intensivity(schnet_fold0, m_a, m_b, separation_angstrom=100.0),
        }
        (bench_dir / "size_intensivity_gnn_vs_lehmann.json").write_text(json.dumps(size_audit, indent=2))

    summary = {
        "Lehmann-TDDFT": lehmann_cv,
        "Static-Delta-TDDFT (Ablation V=0)": static_cv,
        "SchNet3D (Geometric GNN Baseline)": schnet_cv,
        "QM_Baselines": qm_baselines,
        "Size_Intensivity_Audit": size_audit,
    }
    (bench_dir / "cv5_summary.json").write_text(json.dumps(summary, indent=2))
    _export_cv5_csv_table(summary, bench_dir / "cv5_comparison_table.csv")
    return summary


def _run_ood_split_worker(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    torch.set_num_threads(1)
    name = cfg["name"]
    items = load_unified_dataset(cfg["questdb_json"], cfg["hepta_json"], cfg["features_dir"])
    by_id = {it["mol_id"]: it for it in items}
    train_all = [by_id[mid] for mid in cfg["train_ids"] if mid in by_id]
    test_set = [by_id[mid] for mid in cfg["test_ids"] if mid in by_id]
    models_dir = Path(cfg["models_dir"])
    bench_dir = Path(cfg["bench_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== OOD EXPERIMENT [{name}]: train={len(train_all)}, blind_test={len(test_set)} ===")
    if len(train_all) < 5 or len(test_set) == 0:
        return cfg["key"], {"skipped": True, "n_train": len(train_all), "n_test": len(test_set)}

    rng_v = np.random.default_rng(cfg["seed"] + abs(hash(name)) % 1000)
    perm = rng_v.permutation(len(train_all))
    n_val = max(1, int(0.12 * len(train_all)))
    val_idx = set(perm[:n_val].tolist())
    tr = [train_all[i] for i in range(len(train_all)) if i not in val_idx]
    va = [train_all[i] for i in range(len(train_all)) if i in val_idx]

    m_leh, _ = train_lehmann_model(tr, va, disable_doubles=False, epochs=cfg["epochs"], seed=cfg["seed"])
    save_checkpoint(m_leh, str(models_dir / f"lehmann_ood_{name}.pt"), extra={"ood_split": name})
    met_leh, recs_leh = evaluate_model_predictions(m_leh, test_set)

    m_sta, _ = train_lehmann_model(tr, va, disable_doubles=True, epochs=cfg["epochs"], seed=cfg["seed"])
    met_sta, _ = evaluate_model_predictions(m_sta, test_set)

    m_gnn, _ = train_schnet_baseline(tr, va, epochs=cfg["schnet_epochs"], seed=cfg["seed"])
    met_gnn = _eval_schnet_preds(predict_schnet_baseline(m_gnn, test_set), test_set)

    qm_b = evaluate_published_qm_baselines(test_set)

    (bench_dir / f"ood_{name}_predictions.json").write_text(json.dumps(recs_leh, indent=2))
    return cfg["key"], {
        "n_train": len(train_all),
        "n_test": len(test_set),
        "train_mol_ids": [it["mol_id"] for it in train_all],
        "test_mol_ids": [it["mol_id"] for it in test_set],
        "Lehmann-TDDFT": met_leh,
        "Static-Delta-TDDFT": met_sta,
        "SchNet3D": met_gnn,
        "QM_Baselines": qm_b,
    }


def _export_ood_csv_table(ood_results: Dict[str, Any], out_csv: Path) -> None:
    cols = [
        "OOD_Split", "n_train", "n_test", "Method",
        "S1_MAE_eV", "T1_MAE_eV", "dEST_MAE_eV", "INVEST_recall",
        "GD_detection_rate", "GD_MAE_eV", "All_States_MAE_eV", "pct_T1_MAE",
    ]
    rows = []
    for split_key, res in ood_results.items():
        if res.get("skipped"):
            continue
        for method_key in ("Lehmann-TDDFT", "Static-Delta-TDDFT"):
            m = res[method_key]
            A = m["suite_A_lowest_zero_matching"]
            B = m["suite_B_double_excitations_zero_oracle"]
            C = m["suite_C_full_spectrum_matched"]
            rows.append(
                {
                    "OOD_Split": split_key,
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "Method": method_key,
                    "S1_MAE_eV": A["S1_MAE_eV"],
                    "T1_MAE_eV": A["T1_MAE_eV"],
                    "dEST_MAE_eV": A["dEST_MAE_eV"],
                    "INVEST_recall": A["INVEST_recall"],
                    "GD_detection_rate": B["GD_detection_rate"],
                    "GD_MAE_eV": B["GD_zero_oracle_MAE_eV"],
                    "All_States_MAE_eV": C["all_states_MAE_eV"],
                    "pct_T1_MAE": C["pct_T1_MAE"],
                }
            )
        gnn = res["SchNet3D"]
        rows.append(
            {
                "OOD_Split": split_key,
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "Method": "SchNet3D",
                "S1_MAE_eV": gnn["S1_MAE_eV"],
                "T1_MAE_eV": gnn["T1_MAE_eV"],
                "dEST_MAE_eV": gnn["dEST_MAE_eV"],
                "INVEST_recall": gnn["INVEST_recall"],
                "GD_detection_rate": 0.0,
                "GD_MAE_eV": None,
                "All_States_MAE_eV": None,
                "pct_T1_MAE": None,
            }
        )
        tda = res.get("QM_Baselines", {}).get("DF-PBE0_RI-TDA (Adiabatic Base)", {})
        if tda:
            rows.append(
                {
                    "OOD_Split": split_key,
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "Method": "DF-PBE0_RI-TDA",
                    "S1_MAE_eV": tda.get("S1_MAE_eV"),
                    "T1_MAE_eV": tda.get("T1_MAE_eV"),
                    "dEST_MAE_eV": tda.get("dEST_MAE_eV"),
                    "INVEST_recall": tda.get("INVEST_recall"),
                    "GD_detection_rate": 0.0,
                    "GD_MAE_eV": None,
                    "All_States_MAE_eV": None,
                    "pct_T1_MAE": None,
                }
            )
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_ood_suite(
    items: List[Dict[str, Any]],
    models_dir: Path,
    bench_dir: Path,
    epochs: int = 160,
    schnet_epochs: int = 200,
    seed: int = 42,
    workers: int = 1,
    questdb_json: str = "data/processed/questdb_clean.json",
    hepta_json: str = "data/processed/hepta_invest_clean.json",
    features_dir: str = "data/processed/features",
) -> Dict[str, Any]:
    """Execute ALL strict Out-of-Distribution (OOD) Holdout Experiments with ZERO leakage."""
    by_id = {it["mol_id"]: it for it in items}
    q_ood = {}
    if os.path.exists(questdb_json):
        qdata = json.loads(Path(questdb_json).read_text())
        q_ood = qdata.get("splits", {}).get("ood", {})

    def _is_heptazine(it: Dict[str, Any]) -> bool:
        nm = (it["name"] + " " + it["mol_id"]).lower()
        return it["source"] == "HEPTA-INVEST" or "heptazine" in nm or "2t-7n" in nm

    # OOD-1: Large Chromophores (Train ONLY on small/medium MAIN <= 8 heavy atoms; test blind on CHROM + BIO)
    if "ood_large_chrom" in q_ood:
        tr_large = [by_id[m] for m in q_ood["ood_large_chrom"]["train"] if m in by_id]
        te_large = [by_id[m] for m in q_ood["ood_large_chrom"]["test"] if m in by_id]
    else:
        tr_large = [it for it in items if it["source"] == "QUESTDB" and it["subset"] == "MAIN" and it["n_heavy"] <= 8]
        te_large = [it for it in items if it["source"] == "QUESTDB" and it["subset"] in ("CHROM", "BIO") and it["n_heavy"] >= 9]

    # OOD-2A: Double Excitations — Hold out Acyclic Polyenes / Dienes with GD/PD
    if "ood_double_polyenes_dienes" in q_ood:
        te_poly_ids = set(q_ood["ood_double_polyenes_dienes"]["test"])
        te_poly = [it for it in items if it["mol_id"] in te_poly_ids]
        tr_poly = [it for it in items if it["mol_id"] not in te_poly_ids]
    else:
        polyene_kw = ("butadiene", "hexatriene", "octatetraene", "glyoxal", "acrolein", "ethylene", "ethene")
        te_poly = [
            it for it in items
            if any(k in it["name"].lower() or k in it["mol_id"].lower() for k in polyene_kw)
            and any(s.get("special") in ("GD", "PD") for s in it["states"])
        ]
        te_poly_ids = {it["mol_id"] for it in te_poly}
        tr_poly = [it for it in items if it["mol_id"] not in te_poly_ids]

    # OOD-2B: Double Excitations — Hold out Cyclic Antiaromatics / Azines / Heterocycles with GD/PD
    # Strictly exclude ALL heptazine derivatives from train whenever quest_Heptazine is in test (zero IK14/scaffold leak!)
    if "ood_double_azines_heterocycles" in q_ood:
        te_cyc_ids = set(q_ood["ood_double_azines_heterocycles"]["test"])
        te_cyc = [it for it in items if it["mol_id"] in te_cyc_ids]
        has_hep_in_te = any(_is_heptazine(it) for it in te_cyc)
        tr_cyc = [it for it in items if (it["mol_id"] not in te_cyc_ids) and not (has_hep_in_te and _is_heptazine(it))]
    else:
        cyclic_kw = ("cyclobutadiene", "pentalene", "tetrazine", "pyrazine", "benzoquinone", "cyclopentadienone")
        te_cyc = [
            it for it in items
            if any(k in it["name"].lower() or k in it["mol_id"].lower() for k in cyclic_kw)
            and any(s.get("special") in ("GD", "PD") for s in it["states"])
        ]
        te_cyc_ids = {it["mol_id"] for it in te_cyc}
        tr_cyc = [it for it in items if it["mol_id"] not in te_cyc_ids]

    # OOD-3A: Zero-Shot Heptazine / HEPTA-INVEST Transfer (Train on QUESTDB excl. Heptazine -> test ALL 44 HEPTA-INVEST + Heptazine!)
    te_hepta = [it for it in items if _is_heptazine(it)]
    tr_no_hepta = [it for it in items if not _is_heptazine(it)]

    # Conformer/InChIKey-14 siblings of INVEST cores that MUST also be excluded from train in OOD3B & OOD3C:
    # quest_Pentalene (C2h) shares IK14 GUVXZFRDPCKWEM with quest_Pentalene_D2h;
    # quest_Indacene_C2h shares IK14 WEMQMWWWCBYPOV with quest_Indacene_D2h.
    invest_conformer_siblings = {"quest_Pentalene", "quest_Indacene_C2h"}

    # OOD-3B: Zero-Shot EMERGENT Hund's-Rule Violation (Train ONLY on normal dE_ST > 0 molecules, ZERO INVEST & ZERO IK14 siblings in training!)
    te_all_inv = [it for it in items if _is_heptazine(it) or it.get("is_invest_core", False)]
    te_all_inv_ids = {it["mol_id"] for it in te_all_inv} | invest_conformer_siblings
    tr_normal_only = [it for it in items if it["mol_id"] not in te_all_inv_ids]

    # OOD-3C: Zero-Shot Non-Heptazine INVEST Cores (Train on HEPTA-INVEST + Heptazine + normal QUESTDB excl. IK14 conformer siblings -> test Cyclazine/Pentalene/Indacene!)
    te_non_hepta_inv = [it for it in items if it.get("is_invest_core", False) and not _is_heptazine(it)]
    te_nh_ids = {it["mol_id"] for it in te_non_hepta_inv} | invest_conformer_siblings
    tr_nh = [it for it in items if it["mol_id"] not in te_nh_ids]

    split_specs = [
        ("OOD1_large_chromophores", "large_chromophores", tr_large, te_large),
        ("OOD2A_polyene_doubles", "polyene_doubles", tr_poly, te_poly),
        ("OOD2B_cyclic_doubles", "cyclic_doubles", tr_cyc, te_cyc),
        ("OOD3A_zero_shot_hepta_invest", "zero_shot_hepta_invest", tr_no_hepta, te_hepta),
        ("OOD3B_zero_shot_emergent_invest_from_normal", "zero_shot_emergent_invest_from_normal", tr_normal_only, te_all_inv),
        ("OOD3C_zero_shot_non_heptazine_invest", "zero_shot_non_heptazine_invest", tr_nh, te_non_hepta_inv),
    ]

    cfgs = [
        {
            "key": key,
            "name": name,
            "train_ids": [it["mol_id"] for it in tr],
            "test_ids": [it["mol_id"] for it in te],
            "questdb_json": questdb_json,
            "hepta_json": hepta_json,
            "features_dir": features_dir,
            "models_dir": str(models_dir),
            "bench_dir": str(bench_dir),
            "epochs": epochs,
            "schnet_epochs": schnet_epochs,
            "seed": seed,
        }
        for key, name, tr, te in split_specs
    ]

    ood_results: Dict[str, Any] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(cfgs))) as ex:
            for k, v in ex.map(_run_ood_split_worker, cfgs):
                ood_results[k] = v
    else:
        for c in cfgs:
            k, v = _run_ood_split_worker(c)
            ood_results[k] = v

    (bench_dir / "ood_summary.json").write_text(json.dumps(ood_results, indent=2))
    _export_ood_csv_table(ood_results, bench_dir / "ood_comparison_table.csv")
    return ood_results


# ---------------------------------------------------------------------------
# End-to-End Smoke Test on Synthetic Data (verifies zero bugs before real data!)
# ---------------------------------------------------------------------------

def run_synthetic_smoke_test() -> None:
    print("Running synthetic end-to-end training & evaluation smoke test...")
    items = []
    rng = np.random.default_rng(7)
    for i in range(18):
        inv = (i % 4 == 0)
        raw = make_synthetic_molecule(f"smoke_{i}", n_S=6, n_T=6, seed=100 + i, invest_like=inv)
        mol_t = npz_to_mol_tensors(raw, dtype=torch.float64)
        wS0 = float(raw["omega_S_eV"][0])
        wT0 = float(raw["omega_T_eV"][0])
        # Synthetic ground-truth targets mimicking dynamical stabilization & emergent GD
        ref_S1 = (wT0 - 0.18) if inv else (wS0 - 0.15)
        ref_T1 = wT0 - 0.05
        gd_e = float(raw["D_S_bare_eV"][0]) - 0.4
        states = [
            {"label": "S1", "spin": 1, "vr": "V", "type": "ppi", "special": None, "tbe_eV": ref_S1, "pct_T1": 89.0 if inv else 94.0, "osc_ref": float(raw["osc_S"][0]), "baselines": {"B3LYP": wS0}},
            {"label": "T1", "spin": 3, "vr": "V", "type": "ppi", "special": None, "tbe_eV": ref_T1, "pct_T1": 95.0, "osc_ref": None, "baselines": {"B3LYP": wT0}},
        ]
        if i % 3 == 0:
            states.append({"label": "2_1Ag", "spin": 1, "vr": "V", "type": "ppi", "special": "GD", "tbe_eV": gd_e, "pct_T1": 15.0, "osc_ref": 0.0, "baselines": {}})
        items.append(
            {
                "mol_id": f"smoke_{i}",
                "name": f"smoke_{i}",
                "source": "HEPTA-INVEST" if inv else "QUESTDB",
                "subset": "HEPTA-INVEST" if inv else ("CHROM" if i % 2 == 0 else "MAIN"),
                "family": "heptazine" if inv else "polyene",
                "n_atoms": 10,
                "n_heavy": 12 if inv else 6,
                "mol_tensors": mol_t,
                "omega_S_eV": raw["omega_S_eV"],
                "omega_T_eV": raw["omega_T_eV"],
                "osc_S": raw["osc_S"],
                "atomic_numbers": raw["atomic_numbers"],
                "coords_angstrom": raw["coords_angstrom"],
                "states": states,
                "ref_S1_eV": ref_S1,
                "ref_T1_eV": ref_T1,
                "ref_f_S1": float(raw["osc_S"][0]),
                "ref_pct_T1_S1": 89.0 if inv else 94.0,
                "ref_pct_T1_T1": 95.0,
                "is_invest_core": inv,
            }
        )

    tr, va, te = items[:12], items[12:15], items[15:]
    m_leh, _ = train_lehmann_model(tr, va, disable_doubles=False, epochs=25, seed=0)
    met_leh, _ = evaluate_model_predictions(m_leh, te)
    m_sta, _ = train_lehmann_model(tr, va, disable_doubles=True, epochs=25, seed=0)
    met_sta, _ = evaluate_model_predictions(m_sta, te)
    m_gnn, _ = train_schnet_baseline(tr, va, epochs=25, seed=0)
    met_gnn = _eval_schnet_preds(predict_schnet_baseline(m_gnn, te), te)
    audit = audit_gnn_size_intensivity(m_gnn, items[0], items[1])

    print("Smoke Lehmann Suite A:", json.dumps(met_leh["suite_A_lowest_zero_matching"], indent=2))
    print("Smoke Lehmann Suite B:", json.dumps(met_leh["suite_B_double_excitations_zero_oracle"], indent=2))
    print("Smoke Static  Suite B:", json.dumps(met_sta["suite_B_double_excitations_zero_oracle"], indent=2))
    print("Smoke SchNet3D:", json.dumps(met_gnn, indent=2))
    print("Smoke SchNet3D Size-Intensivity Violation:", json.dumps(audit, indent=2))
    assert met_leh["suite_A_lowest_zero_matching"]["dEST_MAE_eV"] is not None
    assert met_sta["suite_B_double_excitations_zero_oracle"]["GD_detected"] == 0
    print("\nSMOKE TEST PASSED CLEANLY!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lehmann-TDDFT Training & Benchmark Suite")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--questdb-json", default="data/processed/questdb_clean.json")
    parser.add_argument("--hepta-json", default="data/processed/hepta_invest_clean.json")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--bench-dir", default="results/benchmarks")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--schnet-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", choices=["all", "cv5", "ood"], default="all")
    args = parser.parse_args()

    torch.set_num_threads(1)

    if args.smoke_test:
        run_synthetic_smoke_test()
        return

    models_dir = Path(args.models_dir)
    bench_dir = Path(args.bench_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    items = load_unified_dataset(args.questdb_json, args.hepta_json, args.features_dir)
    print(f"Loaded {len(items)} molecules with `.npz` feature caches.", flush=True)
    n_q = sum(1 for it in items if it["source"] == "QUESTDB")
    n_h = sum(1 for it in items if it["source"] == "HEPTA-INVEST")
    n_st = sum(len(it["states"]) for it in items)
    n_gd = sum(1 for it in items for s in it["states"] if s.get("special") == "GD")
    n_pd = sum(1 for it in items for s in it["states"] if s.get("special") == "PD")
    print(f"  QUESTDB: {n_q} mols | HEPTA-INVEST: {n_h} mols | Total states: {n_st} (GD={n_gd}, PD={n_pd})", flush=True)

    if args.only in ("all", "cv5"):
        run_cv5_experiment(
            items,
            models_dir,
            bench_dir,
            epochs=args.epochs,
            schnet_epochs=args.schnet_epochs,
            seed=args.seed,
            workers=args.workers,
            resume=args.resume,
            questdb_json=args.questdb_json,
            hepta_json=args.hepta_json,
            features_dir=args.features_dir,
        )
    if args.only in ("all", "ood"):
        run_ood_suite(
            items,
            models_dir,
            bench_dir,
            epochs=args.epochs,
            schnet_epochs=args.schnet_epochs,
            seed=args.seed,
            workers=args.workers,
            questdb_json=args.questdb_json,
            hepta_json=args.hepta_json,
            features_dir=args.features_dir,
        )


if __name__ == "__main__":
    main()
