"""Full-Dimensional (3N-6) On-the-Fly Analytical NAMD Benchmark Suite for Lehmann-NAMD.

Runs matched parallel ensembles (identical harmonic Wigner initial conditions, identical
Velocity-Verlet + Tully FSSH + EDC integrator) comparing:
  1. Lehmann-NAMD (exact analytical O(N^3) causal gradients & NACVs, <S^2> == 0)
  2. SA2-CASSCF   (exact analytical PySCF mc.nuc_grad + mc.nac_method reference)
  3. Adiabatic DF-PBE0 (d_01 == 0 identically; proves trapping / instability crash!)
  + Live Spin-Flip SF-TDA-BHHLYP <S^2>(t) probes along trajectories.

Systems:
  - ethylene    : 12 internal DOF (18 Cartesian), C=C twist/pyr ultrafast decay
  - methanimine :  9 internal DOF (15 Cartesian), C=N photoisomerization
  - fulvene     : 30 internal DOF (36 Cartesian), Ibele-Curchod benchmark (C=CH2 stretch+twist)
  - psb3        : 36 internal DOF (42 Cartesian), cis->trans retinal Schiff base photoisomerization
  - uracil      : 30 internal DOF (36 Cartesian), biological RNA/DNA nucleobase UV photoprotection
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from scipy.optimize import curve_fit

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lehmann_namd.causal_operator import HA_TO_EV, OperatorParams
from lehmann_namd.namd_engine import (
    ANG_TO_BOHR,
    BOHR_TO_ANG,
    AdiabaticPBE0Backend,
    CASSCFBackend,
    LehmannBackend,
    compute_harmonic_wigner_modes,
    optimize_s0_geometry_df_pbe0,
    run_fssh_trajectory_3n,
    sample_wigner_initial_conditions,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "namd_3n"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Geometric Observables (Dihedrals, Pyramidalization, Bond Stretches)
# ---------------------------------------------------------------------------

def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / max(np.linalg.norm(b1), 1e-12)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.rad2deg(np.arctan2(y, x)))


def _ethylene_obs(r_bohr: np.ndarray) -> Dict[str, float]:
    r = r_bohr * BOHR_TO_ANG
    r_cc = float(np.linalg.norm(r[1] - r[0]))
    # Twist angle between CH2 planes
    n1 = np.cross(r[2] - r[0], r[3] - r[0])
    n2 = np.cross(r[4] - r[1], r[5] - r[1])
    n1 /= max(np.linalg.norm(n1), 1e-12)
    n2 /= max(np.linalg.norm(n2), 1e-12)
    twist = float(np.rad2deg(np.arccos(np.clip(abs(np.dot(n1, n2)), 0.0, 1.0))))
    # Max pyramidalization angle of the two CH2 groups
    cc_u = (r[1] - r[0]) / max(r_cc, 1e-12)
    pyr1 = float(np.rad2deg(np.arcsin(np.clip(abs(np.dot(n1, cc_u)), 0.0, 1.0))))
    pyr2 = float(np.rad2deg(np.arcsin(np.clip(abs(np.dot(n2, cc_u)), 0.0, 1.0))))
    return {"r_cc_ang": r_cc, "twist_deg": twist, "pyr_max_deg": max(pyr1, pyr2)}


def _methanimine_obs(r_bohr: np.ndarray) -> Dict[str, float]:
    r = r_bohr * BOHR_TO_ANG
    r_cn = float(np.linalg.norm(r[1] - r[0]))
    # Twist of H4-N1-C0-H2 dihedral
    d1 = abs(_dihedral_deg(r[2], r[0], r[1], r[4]))
    d2 = abs(_dihedral_deg(r[3], r[0], r[1], r[4]))
    twist = min(d1, abs(180.0 - d1))
    # C-N-H bend angle
    v1 = r[0] - r[1]; v2 = r[4] - r[1]
    bend = float(np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))))
    return {"r_cn_ang": r_cn, "twist_deg": float(twist), "cnh_bend_deg": bend, "dihedral_deg": float(d1)}


def _fulvene_obs(r_bohr: np.ndarray) -> Dict[str, float]:
    r = r_bohr * BOHR_TO_ANG
    r_exo = float(np.linalg.norm(r[5] - r[0]))
    # Twist of exocyclic CH2 (atoms 10,11 on C5) relative to ring plane (C1,C4 on C0)
    n_ring = np.cross(r[1] - r[0], r[4] - r[0])
    n_ch2 = np.cross(r[10] - r[5], r[11] - r[5])
    n_ring /= max(np.linalg.norm(n_ring), 1e-12)
    n_ch2 /= max(np.linalg.norm(n_ch2), 1e-12)
    twist = float(np.rad2deg(np.arccos(np.clip(abs(np.dot(n_ring, n_ch2)), 0.0, 1.0))))
    return {"r_exo_ang": r_exo, "twist_deg": twist}


def _psb3_obs(r_bohr: np.ndarray) -> Dict[str, float]:
    r = r_bohr * BOHR_TO_ANG
    # Central C2=C3 (atoms 1-2-3-4) dihedral: cis ~ 0 deg, trans ~ 180 deg!
    dih = abs(_dihedral_deg(r[1], r[2], r[3], r[4]))
    r_23 = float(np.linalg.norm(r[3] - r[2]))
    # Bond-length alternation (BLA)
    bla = (np.linalg.norm(r[1] - r[0]) + np.linalg.norm(r[3] - r[2]) + np.linalg.norm(r[5] - r[4])) / 3.0 - (
        np.linalg.norm(r[2] - r[1]) + np.linalg.norm(r[4] - r[3])
    ) / 2.0
    return {"dih_c2c3_deg": float(dih), "r_c2c3_ang": r_23, "bla_ang": float(bla), "is_trans": float(dih > 90.0)}


def _uracil_obs(r_bohr: np.ndarray) -> Dict[str, float]:
    r = r_bohr * BOHR_TO_ANG
    # C5=C6 (atoms 4-5) bond length and C4-C5-C6-N1 puckering dihedral (3-4-5-0)
    r_c5c6 = float(np.linalg.norm(r[5] - r[4]))
    pucker = abs(_dihedral_deg(r[3], r[4], r[5], r[0]))
    h5_oop = abs(_dihedral_deg(r[10], r[4], r[5], r[11]))
    return {"r_c5c6_ang": r_c5c6, "ring_pucker_deg": float(pucker), "h5c5c6h6_dih_deg": float(h5_oop)}


def _select_pi_mos_fulvene(mol, mf):
    ao_labels = mol.ao_labels()
    pz_idx = [i for i, lbl in enumerate(ao_labels) if "C 2pz" in lbl]
    S = mol.intor("int1e_ovlp")
    pz_w = np.sum((S[pz_idx, :] @ mf.mo_coeff) ** 2, axis=0)
    nocc = mol.nelectron // 2
    occ_pi = (np.argsort(pz_w[:nocc])[-3:] + 1).tolist()
    vir_pi = (np.argsort(pz_w[nocc:])[-3:] + nocc + 1).tolist()
    return sorted(occ_pi + vir_pi)


def _select_pi_mos_psb3(mol, mf):
    """Select 6 pi orbitals (3 occ + 3 vir) for PSB3+ SA2-CASSCF(6,6)."""
    # Align by largest out-of-plane / HOMO-2..LUMO+2 heuristic around frontier
    nocc = mol.nelectron // 2
    return list(range(nocc - 2, nocc + 4))


def get_system_config(name: str) -> Dict[str, Any]:
    if name == "ethylene":
        atoms = ["C", "C", "H", "H", "H", "H"]
        seed_ang = np.array([
            [-0.67, 0.0, 0.0], [0.67, 0.0, 0.0],
            [-1.24, 0.93, 0.0], [-1.24, -0.93, 0.0],
            [1.24, 0.93, 0.0], [1.24, -0.93, 0.0],
        ])
        return {
            "name": "ethylene", "atoms": atoms, "charge": 0, "seed_bohr": seed_ang * ANG_TO_BOHR,
            "basis": "6-31g*", "auxbasis": "weigend", "grids_level": 1, "ncas": 2, "nelecas": 2,
            "select_pi": None, "obs_fn": _ethylene_obs, "n_steps": 140, "dt_fs": 0.5,
            "default_ntraj": 16, "probe_sf_every": 16,
        }
    if name == "methanimine":
        atoms = ["C", "N", "H", "H", "H"]
        seed_ang = np.array([
            [0.00, 0.00, 0.00], [0.00, 0.00, 1.27],
            [0.00, 0.94, -0.56], [0.00, -0.94, -0.56],
            [0.00, 0.88, 1.76],
        ])
        return {
            "name": "methanimine", "atoms": atoms, "charge": 0, "seed_bohr": seed_ang * ANG_TO_BOHR,
            "basis": "6-31g*", "auxbasis": "weigend", "grids_level": 1, "ncas": 3, "nelecas": 4,
            "select_pi": None, "obs_fn": _methanimine_obs, "n_steps": 120, "dt_fs": 0.5,
            "default_ntraj": 16, "probe_sf_every": 15,
        }
    if name == "fulvene":
        atoms = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
        seed_ang = np.array([
            [ 0.00,  0.95, 0.0], [ 1.18,  0.15, 0.0], [ 0.68, -1.18, 0.0],
            [-0.68, -1.18, 0.0], [-1.18,  0.15, 0.0], [ 0.00,  2.31, 0.0],
            [ 2.21,  0.42, 0.0], [ 1.35, -2.05, 0.0], [-1.35, -2.05, 0.0],
            [-2.21,  0.42, 0.0], [ 0.93,  2.87, 0.0], [-0.93,  2.87, 0.0],
        ])
        return {
            "name": "fulvene", "atoms": atoms, "charge": 0, "seed_bohr": seed_ang * ANG_TO_BOHR,
            "basis": "6-31g", "auxbasis": "weigend", "grids_level": 0, "ncas": 6, "nelecas": 6,
            "select_pi": _select_pi_mos_fulvene, "obs_fn": _fulvene_obs, "n_steps": 100, "dt_fs": 0.5,
            "default_ntraj": 16, "probe_sf_every": 15,
        }
    if name == "psb3":
        # Canonical 11-cis PSB3+ (s-trans C0=C1-C2, cis C2=C3, s-trans C3-C4=NH2+)
        atoms = ["C", "C", "C", "C", "C", "N", "H", "H", "H", "H", "H", "H", "H", "H"]
        seed_ang = np.array([
            [-2.76,  1.23,  0.00],  # C0
            [-1.40,  1.23,  0.00],  # C1
            [-0.69,  0.00,  0.00],  # C2
            [ 0.69,  0.00,  0.00],  # C3 (central cis C2=C3!)
            [ 1.40,  1.23,  0.00],  # C4
            [ 2.72,  1.23,  0.00],  # N5 (s-trans outward!)
            [-3.30,  2.16,  0.00],  # H6 on C0
            [-3.30,  0.30,  0.00],  # H7 on C0
            [-0.86,  2.16,  0.00],  # H8 on C1
            [-1.23, -0.94,  0.00],  # H9 on C2
            [ 1.23, -0.94,  0.00],  # H10 on C3
            [ 0.86,  2.16,  0.00],  # H11 on C4
            [ 3.22,  2.10,  0.00],  # H12 on N5
            [ 3.22,  0.36,  0.00],  # H13 on N5
        ])
        return {
            "name": "psb3", "atoms": atoms, "charge": 1, "seed_bohr": seed_ang * ANG_TO_BOHR,
            "basis": "6-31g", "auxbasis": "weigend", "grids_level": 0, "ncas": 6, "nelecas": 6,
            "select_pi": _select_pi_mos_psb3, "obs_fn": _psb3_obs, "n_steps": 180, "dt_fs": 0.5,
            "default_ntraj": 8, "probe_sf_every": 25,
        }
    if name == "uracil":
        atoms = ["N", "C", "N", "C", "C", "C", "O", "O", "H", "H", "H", "H"]
        seed_ang = np.array([
            [-1.19, -0.68, 0.0], [ 0.00, -1.35, 0.0], [ 1.19, -0.66, 0.0],
            [ 1.24,  0.73, 0.0], [-0.02,  1.38, 0.0], [-1.22,  0.68, 0.0],
            [ 0.01, -2.57, 0.0], [ 2.33,  1.28, 0.0], [-2.07, -1.17, 0.0],
            [ 2.06, -1.16, 0.0], [-0.04,  2.46, 0.0], [-2.18,  1.17, 0.0],
        ])
        return {
            "name": "uracil", "atoms": atoms, "charge": 0, "seed_bohr": seed_ang * ANG_TO_BOHR,
            "basis": "6-31g", "auxbasis": "weigend", "grids_level": 0, "ncas": 6, "nelecas": 6,
            "select_pi": None, "obs_fn": _uracil_obs, "n_steps": 120, "dt_fs": 0.5,
            "default_ntraj": 10, "probe_sf_every": 20,
        }
    raise ValueError(f"Unknown system: {name}")


def _worker_run_traj(task: Tuple[Any, ...]) -> Dict[str, Any]:
    sys_name, method_name, traj_id, r0, v0, n_steps, dt_fs, probe_sf = task
    cfg = get_system_config(sys_name)
    gl = cfg.get("grids_level", 1)
    if method_name == "lehmann":
        backend = LehmannBackend(
            charge=cfg["charge"], basis=cfg["basis"], auxbasis=cfg["auxbasis"],
            xc="pbe0", grids_level=gl, params=OperatorParams(), nacv_gap_cutoff_ev=4.5,
        )
    elif method_name == "casscf":
        backend = CASSCFBackend(
            ncas=cfg["ncas"], nelecas=cfg["nelecas"], charge=cfg["charge"],
            basis=cfg["basis"], select_pi_fn=cfg["select_pi"],
        )
    elif method_name == "adiabatic_pbe0":
        backend = AdiabaticPBE0Backend(
            charge=cfg["charge"], basis=cfg["basis"], auxbasis=cfg["auxbasis"],
            xc="pbe0", grids_level=gl,
        )
    else:
        raise ValueError(method_name)

    sf_kwargs = {"charge": cfg["charge"], "basis": cfg["basis"], "auxbasis": cfg["auxbasis"]} if probe_sf > 0 else None
    res = run_fssh_trajectory_3n(
        atoms=cfg["atoms"],
        r0_bohr=r0,
        v0_au=v0,
        backend=backend,
        n_steps=n_steps,
        dt_fs=dt_fs,
        init_state=1,
        seed=10000 + traj_id * 37,
        probe_sf_every=probe_sf,
        sf_probe_kwargs=sf_kwargs,
        coord_obs_fn=cfg["obs_fn"],
    )
    res["traj_id"] = traj_id
    return res


def _fit_lifetime_fs(times: np.ndarray, pop_s1: np.ndarray) -> Dict[str, float]:
    """Fit delayed exponential P_S1(t) = exp(-max(t - t0, 0) / tau) for excited-state lifetime."""
    try:
        def _model(t, t0, tau):
            return np.where(t <= t0, 1.0, np.exp(-(t - t0) / tau))
        popt, _ = curve_fit(_model, times, pop_s1, p0=[5.0, 35.0], bounds=([0.0, 2.0], [100.0, 5000.0]), maxfev=5000)
        return {"t_delay_fs": float(popt[0]), "tau_s1_fs": float(popt[1]), "tau_total_fs": float(popt[0] + popt[1])}
    except Exception:
        return {"t_delay_fs": 0.0, "tau_s1_fs": float("nan"), "tau_total_fs": float("nan")}


def run_benchmark_system(
    sys_name: str,
    methods: List[str],
    n_traj: Optional[int] = None,
    n_steps: Optional[int] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    cfg = get_system_config(sys_name)
    n_traj = n_traj or cfg["default_ntraj"]
    n_steps = n_steps or cfg["n_steps"]
    dt_fs = cfg["dt_fs"]

    sys_dir = RESULTS_DIR / sys_name
    sys_dir.mkdir(parents=True, exist_ok=True)
    modes_cache = sys_dir / "wigner_modes.npz"

    if modes_cache.exists():
        npz = np.load(modes_cache, allow_pickle=True)
        modes = {k: npz[k] for k in npz.files}
        modes["atoms"] = list(cfg["atoms"])
        modes["charge"] = cfg["charge"]
        print(f"[{sys_name}] Loaded cached harmonic Wigner modes ({len(modes['freq_cm1'])} modes).", flush=True)
    else:
        print(f"[{sys_name}] Optimizing S0 geometry and computing 3N DF-PBE0 harmonic Hessian...", flush=True)
        c_eq = optimize_s0_geometry_df_pbe0(
            cfg["atoms"], cfg["seed_bohr"], charge=cfg["charge"], basis=cfg["basis"], auxbasis=cfg["auxbasis"]
        )
        modes = compute_harmonic_wigner_modes(
            cfg["atoms"], c_eq, charge=cfg["charge"], basis=cfg["basis"], auxbasis=cfg["auxbasis"]
        )
        np.savez(
            modes_cache,
            coords_eq_bohr=modes["coords_eq_bohr"],
            masses_au=modes["masses_au"],
            omega_au=modes["omega_au"],
            freq_cm1=modes["freq_cm1"],
            vib_vecs_mw=modes["vib_vecs_mw"],
        )
        print(f"[{sys_name}] Saved {len(modes['freq_cm1'])} harmonic modes (min={modes['freq_cm1'].min():.1f}, max={modes['freq_cm1'].max():.1f} cm^-1).", flush=True)

    # Generate IDENTICAL deterministic Wigner initial conditions shared across ALL methods & processes!
    sys_seed = 20260 + sum((k + 1) * ord(ch) for k, ch in enumerate(sys_name)) % 1000
    ics = sample_wigner_initial_conditions(modes, n_traj=n_traj, seed=sys_seed)
    summary_path = sys_dir / "ensemble_summary.json"
    if summary_path.exists():
        with open(summary_path) as fh:
            summary = json.load(fh)
    else:
        summary = {"methods": {}}
    summary.update({
        "system": sys_name,
        "atoms": cfg["atoms"],
        "dof_internal": 3 * len(cfg["atoms"]) - 6,
        "basis": cfg["basis"],
        "n_traj": n_traj,
        "n_steps": n_steps,
        "dt_fs": dt_fs,
        "t_max_fs": n_steps * dt_fs,
    })

    ctx = mp.get_context("spawn")
    ckpt_dir = sys_dir / "traj_ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    for m_name in methods:
        t0 = time.perf_counter()
        trajs: List[Dict[str, Any]] = []
        pending_tasks = []
        for i in range(n_traj):
            cp = ckpt_dir / f"{m_name}_traj_{i:02d}.json"
            if cp.exists():
                with open(cp) as fh:
                    tr_old = json.load(fh)
                if tr_old.get("n_steps_target") == n_steps:
                    trajs.append(tr_old)
                    continue
            pending_tasks.append((
                sys_name, m_name, i, ics[i][0], ics[i][1], n_steps, dt_fs,
                cfg["probe_sf_every"] if (m_name == "lehmann" and i < 5) else 0,
            ))

        if pending_tasks:
            print(
                f"[{sys_name}:{m_name}] Launching {len(pending_tasks)}/{n_traj} full-{3*len(cfg['atoms'])-6}D trajectories "
                f"({n_steps*dt_fs:.0f} fs, {len(trajs)} cached) on {min(workers, len(pending_tasks))} workers...",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=min(workers, len(pending_tasks)), mp_context=ctx) as pool:
                futs = [pool.submit(_worker_run_traj, t) for t in pending_tasks]
                for fut in as_completed(futs):
                    tr = fut.result()
                    tr["n_steps_target"] = n_steps
                    with open(ckpt_dir / f"{m_name}_traj_{tr['traj_id']:02d}.json", "w") as fh:
                        json.dump(tr, fh)
                    trajs.append(tr)
                    n_h = len([h for h in tr["hops"] if "to" in h])
                    print(
                        f"  [{sys_name}:{m_name}] traj {tr['traj_id']:02d} done in {tr['wall_s']:.1f}s | "
                        f"steps={tr['n_steps_completed']}/{n_steps} | final_S={tr['active_state'][-1]} | "
                        f"hops={n_h} | dE_max={tr['e_drift_max_mev']:.2f} meV | instab={tr['instabilities']}",
                        flush=True,
                    )
        else:
            print(f"[{sys_name}:{m_name}] All {n_traj} trajectories already cached!", flush=True)
        trajs.sort(key=lambda x: x["traj_id"])

        # Aggregate ensemble observables
        times = np.arange(n_steps + 1) * dt_fs
        pop_act = np.zeros(n_steps + 1)
        pop_wf = np.zeros(n_steps + 1)
        valid_counts = np.zeros(n_steps + 1)
        e_drifts = []
        e_stds = []
        n_crashed = 0
        n_instab_trajs = 0
        total_hops = 0
        hop_gaps = []
        sf_s2_s0_vals = []
        sf_s2_s1_vals = []

        for tr in trajs:
            L = len(tr["active_state"])
            pop_act[:L] += np.asarray(tr["active_state"], dtype=np.float64)
            pop_wf[:L] += np.asarray(tr["pop_s1_wf"], dtype=np.float64)
            valid_counts[:L] += 1.0
            e_drifts.append(tr["e_drift_max_mev"])
            e_stds.append(tr["e_drift_std_mev"])
            if tr["crashed"]:
                n_crashed += 1
            if tr["instabilities"] > 0:
                n_instab_trajs += 1
            for h in tr["hops"]:
                if "to" in h:
                    total_hops += 1
                    hop_gaps.append(h["gap_ev"])
            for sp in tr["sf_s2_probes"]:
                sf_s2_s0_vals.append(sp["s2_s0"])
                sf_s2_s1_vals.append(sp["s2_s1"])

        pop_act_mean = pop_act / np.maximum(valid_counts, 1.0)
        pop_wf_mean = pop_wf / np.maximum(valid_counts, 1.0)
        tau_fit = _fit_lifetime_fs(times, pop_act_mean)

        # System-specific photochemical yield (e.g. PSB3 cis->trans quantum yield!)
        extra_metrics: Dict[str, Any] = {}
        if sys_name == "psb3":
            s0_trajs = [tr for tr in trajs if tr["active_state"][-1] == 0]
            trans_trajs = [tr for tr in s0_trajs if tr["obs"][-1].get("is_trans", 0.0) > 0.5]
            extra_metrics["n_decayed_to_s0"] = len(s0_trajs)
            extra_metrics["n_trans_product"] = len(trans_trajs)
            extra_metrics["phi_trans_given_s0"] = float(len(trans_trajs) / max(len(s0_trajs), 1))
            extra_metrics["phi_trans_total"] = float(len(trans_trajs) / max(len(trajs), 1))

        m_summary = {
            "n_traj": len(trajs),
            "n_crashed": n_crashed,
            "n_instability_trajs": n_instab_trajs,
            "total_hops": total_hops,
            "mean_hop_gap_ev": float(np.mean(hop_gaps)) if hop_gaps else None,
            "e_drift_median_mev": float(np.median(e_drifts)),
            "e_drift_mean_mev": float(np.mean(e_drifts)),
            "e_std_median_mev": float(np.median(e_stds)),
            "final_s1_pop_active": float(pop_act_mean[-1]),
            "final_s1_pop_wf": float(pop_wf_mean[-1]),
            "lifetime_fit": tau_fit,
            "times_fs": times.tolist(),
            "pop_s1_active": pop_act_mean.tolist(),
            "pop_s1_wf": pop_wf_mean.tolist(),
            "sf_tda_s2_s0_mean": float(np.mean(sf_s2_s0_vals)) if sf_s2_s0_vals else None,
            "sf_tda_s2_s1_mean": float(np.mean(sf_s2_s1_vals)) if sf_s2_s1_vals else None,
            "sf_tda_s2_s1_max": float(np.max(sf_s2_s1_vals)) if sf_s2_s1_vals else None,
            "wall_s": time.perf_counter() - t0,
            **extra_metrics,
        }
        summary["methods"][m_name] = m_summary

        # Save detailed per-method trajectory archive
        with open(sys_dir / f"trajs_{m_name}.json", "w") as fh:
            json.dump({"summary": m_summary, "trajectories": trajs}, fh)
        print(
            f"[{sys_name}:{m_name}] ENSEMBLE DONE in {m_summary['wall_s']:.1f}s | "
            f"P_S1(end)={m_summary['final_s1_pop_active']:.3f} | tau={tau_fit['tau_total_fs']:.1f} fs | "
            f"dE_med={m_summary['e_drift_median_mev']:.2f} meV | hops={total_hops} | instab_trajs={n_instab_trajs}",
            flush=True,
        )

    with open(sys_dir / "ensemble_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=["ethylene", "methanimine", "fulvene", "psb3", "uracil"])
    parser.add_argument("--methods", nargs="+", default=["lehmann", "casscf", "adiabatic_pbe0"])
    parser.add_argument("--ntraj", type=int, default=None)
    parser.add_argument("--nsteps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_benchmark_system(args.system, args.methods, n_traj=args.ntraj, n_steps=args.nsteps, workers=args.workers)


if __name__ == "__main__":
    main()
