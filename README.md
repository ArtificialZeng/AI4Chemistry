# `Lehmann-TDDFT v2.0`: Causal Neural Self-Energy Lifting Beyond the Adiabatic Approximation in TD-DFT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Zero-Leak Audit](https://img.shields.io/badge/Adversarial_Audit-0_Leaks-brightgreen.svg)](results/audit_report.md)

Official code, audited benchmark results, *Nature Computational Science* manuscript, Supplementary Information, and zero-leak verification suite for:

> **Causal Neural Self-Energy Lifting Overcomes the Adiabatic Approximation in Time-Dependent Density Functional Theory**

---

## Scientific Overview

Production linear-response time-dependent density functional theory (TD-DFT) relies on the frequency-independent adiabatic approximation ($f_{xc}(\mathbf{r},\mathbf{r}',\omega) \equiv f_{xc}(0)$), creating a shared mathematical barrier across three central frontiers of excited-state chemistry:

1. **Absence of genuine double excitations** (`0/28` detectable on QUESTDB 2025 for adiabatic TD-DFT and perturbative double-hybrid CIS(D) TD-DFT).
2. **Inability to capture Hund's-rule-violating inverted singlet–triplet (INVEST) gaps** ($\Delta E_{ST} = E_{S_1} - E_{T_1} < 0$).
3. **Topological collapse of photochemical $S_1/S_0$ conical intersections** from codimension-2 $(3N-8)$-dimensional seams to $(3N-7)$-dimensional crossings with imaginary RPA instabilities and trivial Berry phase ($\gamma = 0$).

**Spin-complete `Lehmann-TDDFT v2.0`** resolves all three barriers within a single $O(N^3)$ density-fitted operator framework by lifting the static single-exciton manifold into a **gauge-equivariant, size-intensive Hermitian Feshbach–Löwdin Hamiltonian** $\mathcal{H}_\theta^{(\sigma)}$ spanning both symmetric $^1(TT)_+$ and group-theory-required antisymmetric $^1(TT)_-$ bi-exciton channels with physically anchored diagonal poles $D_\mu^{\mathrm{phys}}$. Exact Schur complementation yields a strictly Kramers–Kronig causal dynamical self-energy $\mathbf{\Sigma}_{xc}^\theta(\omega) = \mathbf{V}_\theta(\omega\mathbf{I}-\mathbf{D}_\theta)^{-1}\mathbf{V}_\theta^\top$ with **zero global pooling** and **zero atomic-coordinate shortcuts**.

---

## Repository Layout

```text
AI_Chemistry/
├── src/lehmann_tddft/               # Core library, operator, QM engine, benchmarks & audit
│   ├── model.py                     # Gauge-equivariant spin-complete Lehmann-TDDFT v2.0 operator
│   ├── qm_engine.py                 # O(N^3) DF-PBE0 RI-TDA + spin-complete RI bi-exciton tensors
│   ├── theorems.py                  # Formal verification of Theorems 1-5 & D3h selection rules
│   ├── train.py                     # Zero-leak 5-fold CV & OOD1-3C training harness
│   ├── rydberg_svpd_benchmark.py    # Direct 64-mol (643-state) def2-SVP -> def2-SVPD proof
│   ├── blind_doubles_benchmark.py   # 3-fold leave-molecule-family-out zero-oracle doubles
│   ├── double_hybrid_baseline.py    # Spin-adapted RI-CIS(D) O(N^5) double-hybrid baselines
│   ├── matched_root_scaling.py      # Single-core matched-root (8S+8T) CPU/wall scaling ladder
│   ├── conical_intersections.py     # SA2-CASSCF+SC-NEVPT2 grids & LOMO conical intersections
│   ├── ci_lomo_transfer.py          # Zero-manual-beta LOMO transfer & single-mol cross-transfer
│   ├── sf_tddft_baseline.py         # Spin-flip TD-DFT (BHHLYP/PBE0) <S^2> spin-purity audit
│   ├── blind_meci_search.py         # Autonomous blind MECI search (Ethylene, Methanimine, PSB3)
│   ├── fssh_dynamics.py             # 300-trajectory 2D branching-plane Tully FSSH dynamics
│   ├── finalize_discovery.py        # Pillar 3 virtual screening & blind verification synchronizer
│   ├── v2_triple_verification.py    # SA-CASSCF+SC-NEVPT2, larger-basis ADC(2), & T1 relaxation
│   ├── ab_initio_verify.py          # Spin-adapted RI-EE-ADC(2) & PySCF EOM-EE-CCSD verifiers
│   └── audit.py                     # Independent zero-leak adversarial & statistical audit suite
├── scripts/
│   ├── make_figures.py              # Generates all 6 publication PDF + 300 DPI PNG figures
│   └── build_supplementary_info.py  # Generates Supplementary Information TeX directly from JSONs
├── tests/
│   └── test_lehmann_reproducibility.py  # Automated unittest/pytest verification suite
├── manuscript/
│   ├── lehmann_tddft_nature.tex     # Main Nature Computational Science manuscript source
│   ├── lehmann_tddft_nature.pdf     # Compiled 17-page publication-ready PDF (0 warnings)
│   ├── lehmann_tddft_SI.tex         # Supplementary Information source (Proofs + Tables S1-S5)
│   ├── lehmann_tddft_SI.pdf         # Compiled Supplementary Information PDF
│   ├── references.bib               # 113 DOI-verified BibTeX references
│   └── EXECUTIVE_SUMMARY.md         # Concise audited scorecard
├── figures/                         # Figs 1-5 + Extended Data Fig 1 (PDF + 300 DPI PNG)
├── results/                         # All audited JSON/CSV/NPZ outputs & v2.0 checkpoints
└── data/                            # Curated QUESTDB 2025, HEPTA-INVEST, & novel XYZ/features
```

---

## Reproducibility & Environment Setup

### 1. Environment & macOS Runtime Advisory
Tested on **Apple M3 Max (macOS, Python 3.12)** with PySCF 2.x, PyTorch 2.x, RDKit, SciPy, geomeTRIC, and Matplotlib.

> **Important macOS OpenMP Safety Note:** Never import `torch` in the same Python process as `tblite`/`xgboost`/`lightgbm`, and set single-thread BLAS/OMP variables when running PyTorch scripts on Apple Silicon to prevent `libomp` deadlock:
> ```bash
> export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
> ```

### 2. Fast Automated Verification (~15 seconds)
Run the automated reproducibility test suite covering **Theorems 1–4** to machine precision, **zero InChIKey-14 leakage**, and exact on-disk artifact consistency:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

### 3. Full Independent Zero-Leak Adversarial Audit (~20 seconds)
Re-verifies all 187 CV5 molecules, 6 OOD splits, 23 external cores, 520/320/Top-20 virtual candidates, Theorems 1–5, matched QM baselines, diffuse `def2-SVPD` proof, blind doubles, double-hybrid baselines, matched-root scaling, LOMO CIs, blind MECIs, FSSH dynamics, and three-tier blind INVEST caches:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=src .venv/bin/python src/lehmann_tddft/audit.py
```

### 4. Re-generating Publication Figures, Main PDF & Supplementary Information PDF
```bash
# Re-render all 6 publication figures (vector PDF + 300 DPI PNG)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/make_figures.py

# Regenerate Supplementary Information TeX directly from audited JSONs
.venv/bin/python scripts/build_supplementary_info.py

# Compile Main Manuscript and Supplementary Information PDFs
cd manuscript
latexmk -pdf -interaction=nonstopmode lehmann_tddft_nature.tex
latexmk -pdf -interaction=nonstopmode lehmann_tddft_SI.tex
```

---

## License
Released under the MIT License for academic and scientific reproducibility.
