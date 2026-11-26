# Lehmann-NAMD (AI4Chemistry Project #2)

**Exact analytical differentiable causal self-energy gradients and non-adiabatic couplings for full-dimensional ($3N{-}6$) on-the-fly photodynamics beyond adiabatic TD-DFT.**

Developed cleanly on branch `feat/lehmann-namd` without modifying Project #1 (`Lehmann-TDDFT v2.0`, frozen on `main@e17a474` / tag `v2.0-submission-freeze`).

## Highlights & Verified Results

- **Theorem 1 (Generalized Causal Hellmann--Feynman & Dyson Residue Identity):** Verified to float64 precision on twisted-pyramidalized ethylene (`max |Z_dir - Z_dyson| = 2.51e-14`, `max |Causal HF err| = 4.65e-13`).
- **Theorem 2 (Exact $3N$ Analytical Gradients, NACVs & Invariances):** Single-pass analytical $(\nabla E_0, \nabla E_1, \mathbf{h}_{01})$ matches all-$3N$ central finite differences to `1.94e-9`--`4.96e-5` Ha/Bohr across $\mathrm{H}_2\mathrm{O}$ and 18D $\mathrm{C}_2\mathrm{H}_4$, preserving $O(n_{\mathrm{occ}})\times O(n_{\mathrm{vir}})$, $\mathbb{Z}_2$, translational/rotational invariance, and exact size-intensivity (`||d01(He@50 Bohr)|| = 4.15e-10`).
- **Theorem 3 (Direct $3N$ Branching-Plane Alignment & Quantized Berry Circulation):** Matches exact analytical PySCF SA2-CASSCF branching planes to $|\cos(\hat{\mathbf{g}})|=0.983$--$0.998$ and $|\cos(\hat{\mathbf{h}})|=0.994$--$0.999$ across ethylene (18D), methanimine (15D), and fulvene (36D), recovering $1/r$ NACV divergence and quantized 3N Berry circulation $\gamma/\pi=+1.0043$ ($-0.00004$ shifted control).
- **Theorem 4 (Single-Core $O(N^3)$ Scaling to 24-Atom Azobenzene):** Empirical $t_{\mathrm{anal}}\propto N_{\mathrm{AO}}^{2.58}$ from 5 to 24 atoms (15 to 72 DOF), achieving **$47.4\times$ speedup** over $6N$ finite differences on 24-atom azobenzene ($N_{\mathrm{AO}}=246$, `2,048.5 s` vs `27.0 h`).
- **Unconstrained Full-$3N$ Blind Analytical MECI Discovery:** Cold-SCF Trust-Radius BFGS blindly discovers all 5 benchmark $S_1/S_0$ MECIs (15D--42D) in 48--79 evaluations: ethylene (`0.0016 eV`, `0.0010 Å` heavy RMSD), methanimine (`0.0051 eV`, `0.0044 Å`), fulvene (`0.0099 eV`, `0.082 eV` CASSCF), PSB3$^+$ (`0.0039 eV`), and uracil (`0.035 eV` independent SC-NEVPT2), with $5/5$ certified topological Berry holonomy $W=-1$ vs $+1$ controls.
- **Full-($3N{-}6$) On-the-Fly Tully FSSH Photodynamics (5 Systems, 60 Trajectories):**
  - **Methanimine (9D, 16 trajs $\times$ 60 fs):** $P_{S_1}=0.313$ ($\tau=64.2$ fs, 13 hops, 14% faster than SA2-CASSCF(4,3)), vs Adiabatic PBE0 `0` hops ($2/16$ instab.).
  - **Ethylene (12D, 16 trajs $\times$ 60 fs):** $P_{S_1}=0.250$ ($\tau=44.5$ fs, 18 hops, 0 instab.), vs SA2-CASSCF(2,2) orbital flips up to `9.00 eV` and Adiabatic PBE0 `0` hops ($8/16$ instab., up to `11.65 eV`).
  - **Fulvene (30D, 12 trajs $\times$ 40 fs):** $P_{S_1}=0.333$ ($|c_1|^2=0.333$, $\tau=31.6$ fs, **$4.94\times$ faster per step** than SA2-CASSCF(6,6)), vs Adiabatic PBE0 `0` hops.
  - **11-\emph{cis} PSB3$^+$ (36D, 8 trajs $\times$ 90 fs) & Uracil (30D, 8 trajs $\times$ 50 fs):** Active-space-free biological/retinal dynamics matching adiabatic PBE0 energy conservation (`103.2` & `120.5 meV`, 0 crashes) while eliminating live SF-TDA-BHHLYP spin contamination ($\langle\hat{S}^2\rangle_{S_1}$ up to `1.54` / mean `1.06` on uracil).

## Key Artifacts

- Manuscript PDF: [`manuscript/lehmann_namd_nature.pdf`](manuscript/lehmann_namd_nature.pdf) (10 pages, 0 warnings / 0 errors)
- Supplementary Information PDF: [`manuscript/lehmann_namd_SI.pdf`](manuscript/lehmann_namd_SI.pdf) (6 pages, 0 warnings / 0 errors)
- Publication Figures: [`figures/fig1_theorems_and_scaling.pdf`](figures/fig1_theorems_and_scaling.pdf) through [`figures/fig5_conservation_spin_isom.pdf`](figures/fig5_conservation_spin_isom.pdf)
- Independent Audit & Freeze Manifest: [`results/audit_namd_report.json`](results/audit_namd_report.json) (`all_passed: true`) and [`FREEZE_MANIFEST_NAMD.json`](FREEZE_MANIFEST_NAMD.json)

## Quick Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH=projects/lehmann_namd/src .venv/bin/python -m unittest discover -s projects/lehmann_namd/tests -v

PYTHONPATH=projects/lehmann_namd/src .venv/bin/python -m lehmann_namd.audit_namd
.venv/bin/python projects/lehmann_namd/scripts/make_figures.py
```
