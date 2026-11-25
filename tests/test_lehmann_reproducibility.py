"""Automated Reproducibility & Physical-Law Test Suite for Lehmann-TDDFT v2.0.

Run via:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from lehmann_tddft.model import load_checkpoint, npz_to_mol_tensors
from lehmann_tddft.theorems import make_exact_direct_sum_dimer

ROOT = Path(__file__).resolve().parents[1]


class TestLehmannPhysicalLaws(unittest.TestCase):
    """Live machine-precision checks on trained v2.0 checkpoint (Theorems 1-4)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, _ = load_checkpoint(
            str(ROOT / "results/models/lehmann_cv_fold0.pt"), map_location="cpu"
        )
        cls.model.eval()
        cls.but_npz = str(ROOT / "data/processed/features/quest_Butadiene.npz")
        cls.hep_npz = str(ROOT / "data/processed/features/quest_Heptazine.npz")

    def test_zero_coordinate_and_Z_shortcut(self) -> None:
        """Model output must be bit-identical under arbitrary coordinate/Z scrambling."""
        m1 = npz_to_mol_tensors(self.hep_npz)
        m2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in m1.items()}
        rng = torch.Generator().manual_seed(42)
        m2["coords_angstrom"] = m2["coords_angstrom"] + 50.0 * torch.randn(
            m2["coords_angstrom"].shape, generator=rng, dtype=m2["coords_angstrom"].dtype
        )
        m2["atomic_numbers"] = torch.ones_like(m2["atomic_numbers"]) * 99
        with torch.no_grad():
            o1 = self.model(m1)
            o2 = self.model(m2)
        self.assertEqual(float((o1["S"]["energies_eV"] - o2["S"]["energies_eV"]).abs().max()), 0.0)
        self.assertEqual(float((o1["T"]["energies_eV"] - o2["T"]["energies_eV"]).abs().max()), 0.0)
        self.assertEqual(float((o1["S"]["osc"] - o2["S"]["osc"]).abs().max()), 0.0)

    def test_theorem1_parseval_dipole_norm_conservation(self) -> None:
        """Sum of dressed transition-dipole-squared norms must equal bare TDA sum to 1e-12."""
        mol = npz_to_mol_tensors(self.but_npz)
        with torch.no_grad():
            out = self.model(mol)
        bare_norm2 = float((mol["dip_S_au"].double() ** 2).sum())
        dressed_norm2 = float((out["S"]["dip_au"].double() ** 2).sum())
        self.assertLess(abs(dressed_norm2 - bare_norm2), 1e-12)

    def test_theorem1_kramers_kronig_and_schur_identity(self) -> None:
        """Verify direct Schur complement residual and dissipative Im(Sigma^R) <= 0."""
        mol = npz_to_mol_tensors(self.but_npz)
        with torch.no_grad():
            out = self.model(mol)
        S = out["S"]
        H_SS = S["H_SS"].double()
        V = S["V"].double()
        D = S["D"].double()
        e0 = S["energies_eV"][0].double()
        c0 = S["c_S"][:, 0].double()
        sigma_e0 = V @ torch.diag(1.0 / (e0 - D)) @ V.T
        res = float(((H_SS + sigma_e0) @ c0 - e0 * c0).norm())
        self.assertLess(res, 1e-11)
        # Retarded imaginary part at omega = 5.0 eV, eta = 0.05 eV must be negative semidefinite
        eta = 0.05
        omega = 5.0
        im_sig = -eta * (V @ torch.diag(1.0 / ((omega - D) ** 2 + eta ** 2)) @ V.T)
        max_eig = float(torch.linalg.eigvalsh(im_sig).max())
        self.assertLessEqual(max_eig, 1e-14)

    def test_theorem2_exact_phase_flip_equivariance(self) -> None:
        """Flipping arbitrary single-exciton phases leaves eigenvalues and f invariant to 1e-12."""
        mol = npz_to_mol_tensors(self.but_npz)
        with torch.no_grad():
            o_ref = self.model(mol)
        rng = np.random.default_rng(7)
        mol_f = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in mol.items()}
        ns = mol["omega_S_eV"].shape[0]
        nt = mol["omega_T_eV"].shape[0]
        ps = torch.tensor(rng.choice([-1.0, 1.0], size=ns), dtype=torch.float64)
        pt = torch.tensor(rng.choice([-1.0, 1.0], size=nt), dtype=torch.float64)
        mol_f["dip_S_au"] = mol["dip_S_au"] * ps[:, None]
        mol_f["SS_couplings"] = mol["SS_couplings"] * (ps[:, None, None] * ps[None, :, None])
        mol_f["TT_couplings"] = mol["TT_couplings"] * (pt[:, None, None] * pt[None, :, None])

        def _pair_signs(pairs, chans, spin):
            out_s = torch.ones(pairs.shape[0], dtype=torch.float64)
            for idx in range(pairs.shape[0]):
                i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
                ch = int(chans[idx])
                if spin == "S":
                    # 0=TT+, 1=SS+, 2=TT-
                    out_s[idx] = (ps[i] * ps[j]) if ch == 1 else (pt[i] * pt[j])
                else:
                    # 0=TT, 1=ST
                    out_s[idx] = (pt[i] * pt[j]) if ch == 0 else (ps[i] * pt[j])
            return out_s

        p_ds = _pair_signs(mol["D_S_pairs"], mol["D_S_channel"], "S")
        p_dt = _pair_signs(mol["D_T_pairs"], mol["D_T_channel"], "T")
        mol_f["W_SD_S"] = mol["W_SD_S"] * (ps[:, None, None] * p_ds[None, :, None])
        mol_f["W_SD_T"] = mol["W_SD_T"] * (pt[:, None, None] * p_dt[None, :, None])
        with torch.no_grad():
            o_flip = self.model(mol_f)
        self.assertLess(float((o_ref["S"]["energies_eV"] - o_flip["S"]["energies_eV"]).abs().max()), 1e-12)
        self.assertLess(float((o_ref["T"]["energies_eV"] - o_flip["T"]["energies_eV"]).abs().max()), 1e-12)
        self.assertLess(float((o_ref["S"]["osc"] - o_flip["S"]["osc"]).abs().max()), 1e-12)

    def test_theorem3_exact_size_intensivity(self) -> None:
        """Exact direct-sum dimer Heptazine (+) Butadiene must shift local energies by < 1e-12 eV."""
        raw_a = {k: v for k, v in np.load(self.hep_npz, allow_pickle=True).items()}
        raw_b = {k: v for k, v in np.load(self.but_npz, allow_pickle=True).items()}
        mol_a = npz_to_mol_tensors(raw_a)
        mol_ab = npz_to_mol_tensors(make_exact_direct_sum_dimer(raw_a, raw_b))
        with torch.no_grad():
            out_a = self.model(mol_a)
            out_ab = self.model(mol_ab)
        for sp in ("S", "T"):
            ea = out_a[sp]["energies_eV"][:6].double()
            eab = out_ab[sp]["energies_eV"].double()
            min_diffs = torch.stack([(eab - e).abs().min() for e in ea])
            self.assertLess(float(min_diffs.max()), 1e-12, f"Theorem 3 size-intensivity drift in {sp}")
        dest_a = float(out_a["S"]["energies_eV"][0] - out_a["T"]["energies_eV"][0])
        s1_ab = float(out_ab["S"]["energies_eV"][torch.argmin((out_ab["S"]["energies_eV"] - out_a["S"]["energies_eV"][0]).abs())])
        t1_ab = float(out_ab["T"]["energies_eV"][torch.argmin((out_ab["T"]["energies_eV"] - out_a["T"]["energies_eV"][0]).abs())])
        self.assertLess(abs((s1_ab - t1_ab) - dest_a), 1e-12)

    def test_theorem4_exact_dynamical_inversion_identity(self) -> None:
        """Verify Theorem 4 exact identity dEST == dEST_stat - dSigma_ST and dynamical inversion on Heptazine."""
        mol = npz_to_mol_tensors(self.hep_npz)
        with torch.no_grad():
            out = self.model(mol)
        e_s1 = float(out["S"]["energies_eV"][0].double())
        e_t1 = float(out["T"]["energies_eV"][0].double())
        u_s1 = out["S"]["c_S"][:, 0].double()
        u_s1 = u_s1 / u_s1.norm()
        u_t1 = out["T"]["c_S"][:, 0].double()
        u_t1 = u_t1 / u_t1.norm()
        e_s1_stat = float(u_s1 @ out["S"]["H_SS"].double() @ u_s1)
        e_t1_stat = float(u_t1 @ out["T"]["H_SS"].double() @ u_t1)
        vs, ds = out["S"]["V"].double(), out["S"]["D"].double()
        vt, dt = out["T"]["V"].double(), out["T"]["D"].double()
        sig_s1 = float(u_s1 @ (vs @ torch.diag(1.0 / (e_s1 - ds)) @ vs.T) @ u_s1)
        sig_t1 = float(u_t1 @ (vt @ torch.diag(1.0 / (e_t1 - dt)) @ vt.T) @ u_t1)
        dest_stat = e_s1_stat - e_t1_stat
        dsig_st = sig_t1 - sig_s1
        self.assertLess(abs((e_s1 - e_t1) - (dest_stat - dsig_st)), 1e-12)
        self.assertGreater(dest_stat, 0.0, "Heptazine must have positive static splitting in spin-complete v2.0")
        self.assertLess(e_s1 - e_t1, 0.0, "Heptazine must dynamically invert (dEST < 0)")
        self.assertLess(sig_s1, -0.1, "Antisymmetric ^1(TT)_- channel must strongly stabilize S1")


class TestAuditedResultsAndZeroLeakage(unittest.TestCase):
    """Verify audited JSON invariants, zero leakage, and deliverable completeness."""

    def test_audit_report_zero_leaks_and_bit_exact(self) -> None:
        audit = json.loads((ROOT / "results/audit_report.json").read_text())
        w1 = audit["wave1_data_and_splits"]
        self.assertEqual(w1["cv5_leakage_audit"]["n_exact_cross_fold_leaks"], 0)
        self.assertTrue(w1["cv5_leakage_audit"]["strict_zero_leakage_passed"])
        for sp_name, sp_val in w1["ood_leakage_audit"].items():
            self.assertEqual(sp_val["n_ik14_leaks"], 0, f"Leak detected in {sp_name}")
        self.assertEqual(w1["novel_23_multi_scaffold_audit"]["n_overlaps_with_187_benchmark"], 0)
        p3 = audit["wave2_blind_invest_discovery"]
        self.assertEqual(len(p3["top20_overlap_with_train_ik14"]), 0)
        self.assertEqual(len(p3["top20_overlap_with_bench34_ik14"]), 0)
        self.assertTrue(p3["strict_novelty_passed"])
        self.assertEqual(p3["summary_cross_check"]["n_mismatches_vs_raw_cache"], 0)
        self.assertTrue(p3["summary_cross_check"]["exact_cache_match"])
        self.assertTrue(p3["live_leak_free_ensemble_sync_check"]["synced_with_leak_free_manifest"])
        # Regression: basis_and_adiabatic_audit must confirm 25/25 strictly inverted
        ba_aud = p3["basis_and_adiabatic_audit"]
        self.assertEqual(ba_aud["n_molecules"], 5)
        self.assertEqual(ba_aud["n_checks"], 25)
        self.assertEqual(ba_aud["n_inverted"], 25)
        self.assertTrue(ba_aud["all_strictly_inverted"])
        # Regression: NEVPT2 triple-confirmed IDs must include all 5/8 EOM hits & exact 2 boron cores
        nev_aud = p3["nevpt2_triple_verification_audit"]
        self.assertEqual(len(nev_aud["triple_method_confirmed_ids"]), 5)
        self.assertEqual(
            nev_aud["triple_method_confirmed_boron_core_ids"],
            ["VBI_0159_B0N_m", "VBI_0213_B0N_pp"],
        )
        # Regression: audit speedup table must use matched def2-SVP RI-ADC(2) times (not def2-TZVP!)
        st_rows = {
            r["mol_id"]: r
            for r in audit["wave1_benchmarks_and_referee_audit"]["timing_and_scaling_audit"]["direct_measured_speedup_table"]
        }
        for mid in ("5AP", "VBI_0069_3N_13_pp", "VBI_0089_3N_13_pp", "VBI_0210_3N_13_m", "VBI_0236_6AP_tri"):
            self.assertIn(mid, st_rows)
            self.assertLess(
                st_rows[mid]["ri_adc2_2roots_s"],
                500.0,
                f"Audit timing table mixed larger-basis ADC(2) time into def2-SVP row for {mid}",
            )
        bm_audit = audit["phase_v2_reviewer_upgrades"]["pillar4_v2_upgrades"]["blind_meci_search"]
        self.assertEqual(bm_audit["status"], "AUDITED")
        self.assertEqual(len(bm_audit.get("blockers", [])), 0)

    def test_cv5_and_rydberg_proof_metrics(self) -> None:
        cv5 = json.loads((ROOT / "results/benchmarks/cv5_summary.json").read_text())
        sa = cv5["Lehmann-TDDFT"]["suite_A_lowest_zero_matching"]
        self.assertLess(sa["dEST_MAE_eV"], 0.090)
        self.assertGreater(sa["dEST_R2"], 0.96)
        ryd = json.loads((ROOT / "results/benchmarks/rydberg_def2svpd_proof.json").read_text())
        self.assertGreater(ryd["Rydberg_states"]["reduction_factor_SVP_over_SVPD"], 2.2)

    def test_matched_root_scaling_complete(self) -> None:
        mrs = json.loads((ROOT / "results/benchmarks/matched_root_scaling.json").read_text())
        self.assertEqual(mrs["status"], "complete")
        self.assertEqual(len(mrs["records"]), 9)
        for rec in mrs["records"]:
            self.assertGreater(rec["lehmann_tddft_8S8T"]["nn_forward_cpu_ms"], 0.0)
        self.assertLess(mrs["power_law_fits_cpu"]["Lehmann-TDDFT_8S8T"]["exponent_alpha"], 2.3)

    def test_basis_and_t1_relaxation_all_inverted(self) -> None:
        ba = json.loads((ROOT / "results/invest_discovery/basis_and_adiabatic_check.json").read_text())
        n_checked = 0
        for mid in ("5AP", "VBI_0089_3N_13_pp", "VBI_0210_3N_13_m", "VBI_0069_3N_13_pp", "VBI_0236_6AP_tri"):
            t1r = ba["t1_relaxation_and_dft_geom_check"][mid]
            self.assertTrue(t1r["ok"])
            self.assertLess(t1r["adc2_at_pbe0_S0_geom"]["dEST_eV"], 0.0)
            self.assertLess(t1r["adc2_at_pbe0_T1_geom"]["dEST_at_T1_geom_eV"], 0.0)
            n_checked += 2
            for basis in ("def2-svp", "aug-cc-pvdz", "def2-tzvp"):
                self.assertLess(ba["larger_basis_convergence"][mid][basis]["dEST_eV"], 0.0)
                n_checked += 1
        self.assertEqual(n_checked, 25)

    def test_fssh_all_three_ci_systems(self) -> None:
        """Verify 3/3 LOMO branching-plane Tully FSSH systems (Ethylene, Methanimine, Fulvene)."""
        fssh = json.loads((ROOT / "results/conical_intersections/fssh_dynamics.json").read_text())
        for sys_name in ("ethylene", "methanimine", "fulvene"):
            self.assertIn(sys_name, fssh, f"Missing FSSH system: {sys_name}")
            ms = fssh[sys_name]["methods"]
            self.assertEqual(ms["Adiabatic_PBE0_TDA"]["final_S0_yield"], 0.0)
            self.assertEqual(ms["Lehmann_TDDFT"]["ever_entered_neg_gap_frac"], 0.0)
            self.assertGreater(ms["Lehmann_TDDFT"]["final_S0_yield"], 0.30)
            self.assertLess(
                ms["Lehmann_TDDFT"]["pop_S1_MAE_vs_NEVPT2"],
                ms["Adiabatic_PBE0_TDA"]["pop_S1_MAE_vs_NEVPT2"],
            )

    def test_readme_referenced_files_exist(self) -> None:
        """Verify all Python source files listed in README.md exist on disk."""
        import re
        readme_txt = (ROOT / "README.md").read_text()
        py_refs = set(re.findall(r"([a-zA-Z0-9_]+\.py)", readme_txt))
        self.assertGreaterEqual(len(py_refs), 15)
        all_py = {p.name for p in (ROOT / "src").rglob("*.py")} | {p.name for p in (ROOT / "scripts").rglob("*.py")} | {p.name for p in (ROOT / "tests").rglob("*.py")}
        missing = sorted(py_refs - all_py)
        self.assertEqual(missing, [], f"README references nonexistent scripts: {missing}")

    def test_publication_deliverables_exist(self) -> None:
        expected = [
            "manuscript/lehmann_tddft_nature.pdf",
            "manuscript/lehmann_tddft_nature.tex",
            "manuscript/lehmann_tddft_SI.pdf",
            "manuscript/lehmann_tddft_SI.tex",
            "manuscript/references.bib",
            "manuscript/EXECUTIVE_SUMMARY.md",
            "manuscript/NATCOMPUTSCI-26-5124/cover_letter.pdf",
            "manuscript/NATCOMPUTSCI-26-5124/lehmann_tddft_nature_submission.pdf",
            "manuscript/NATCOMPUTSCI-26-5124/lehmann_tddft_SI.pdf",
            "manuscript/NATCOMPUTSCI-26-5124/lehmann_tddft_source_for_review.zip",
            "results/audit_report.md",
            "figures/fig1_framework_theorems.pdf",
            "figures/fig2_questdb_doubles.pdf",
            "figures/fig3_conical_intersections.pdf",
            "figures/fig4_invest_discovery.pdf",
            "figures/fig5_scaling_diagnostics.pdf",
            "figures/fig6_extended_ci_ablations.pdf",
        ]
        for rel in expected:
            p = ROOT / rel
            self.assertTrue(p.exists() and p.stat().st_size > 1000, f"Missing/empty deliverable: {rel}")
        self.assertEqual(
            (ROOT / "manuscript/lehmann_tddft_SI.pdf").read_bytes(),
            (ROOT / "manuscript/NATCOMPUTSCI-26-5124/lehmann_tddft_SI.pdf").read_bytes(),
            "Submission SI PDF is out of sync with manuscript/lehmann_tddft_SI.pdf",
        )
        import zipfile
        with zipfile.ZipFile(ROOT / "manuscript/NATCOMPUTSCI-26-5124/lehmann_tddft_source_for_review.zip") as zf:
            names = zf.namelist()
            self.assertGreaterEqual(len(names), 30)
            for name in names:
                self.assertEqual(
                    zf.read(name),
                    (ROOT / name).read_bytes(),
                    f"Source review zip out of sync with disk for {name}",
                )

    def test_freeze_manifest_sha256_integrity(self) -> None:
        """Every deterministic deliverable in FREEZE_MANIFEST.json must match exact bytes & SHA-256."""
        import hashlib
        manifest_path = ROOT / "FREEZE_MANIFEST.json"
        self.assertTrue(manifest_path.exists(), "Missing FREEZE_MANIFEST.json")
        manifest = json.loads(manifest_path.read_text())
        deliverables = manifest.get("deliverables", {})
        self.assertGreaterEqual(len(deliverables), 30)
        for rel, meta in deliverables.items():
            p = ROOT / rel
            self.assertTrue(p.exists(), f"Manifest file missing: {rel}")
            data = p.read_bytes()
            self.assertEqual(len(data), meta["bytes"], f"Byte-size drift in {rel}")
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(digest, meta["sha256"], f"SHA-256 drift in {rel}")


if __name__ == "__main__":
    unittest.main()
