"""Shared data & tensor schema for Lehmann-TDDFT across all agents.

Every processed molecule is stored as a compressed `.npz` file with exact,
gauge-invariant / gauge-equivariant quantum tensors computed at O(N^3) cost
from DF-PBE0/def2-SVP + 1-pass BLAS RI-TDA + RI 3-center integrals.

Units:
  - Energies: Hartree (Ha) in raw arrays unless suffixed `_eV` (1 Ha = 27.211386245988 eV).
  - Lengths / Dipoles: atomic units (Bohr, e*Bohr).
  - Oscillator strengths: dimensionless f = (2/3) * omega_Ha * |mu_au|^2.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

HA_TO_EV = 27.211386245988

# Exact keys required in each molecule's `.npz` cache file:
NPZ_REQUIRED_KEYS: List[str] = [
    # Metadata (scalar / string)
    "mol_id",              # str: unique identifier
    "n_atoms",             # int
    "n_heavy",             # int
    "n_ao",                # int
    "n_aux",               # int
    "n_occ_act",           # int: active valence occupied orbitals
    "n_vir_act",           # int: active virtual orbitals
    "homo_lumo_gap_eV",    # float: KS HOMO-LUMO gap (eV)
    "e_scf_ha",            # float: DF-PBE0 SCF total energy (Ha)
    "wall_time_s",         # float: total O(N^3) QM feature extraction wall time (s)

    # Adiabatic TDA Singlet eigenstates (N_S <= 16)
    "omega_S_eV",          # (N_S,) float64: adiabatic TDA singlet energies (eV)
    "dip_S_au",            # (N_S, 3) float64: transition dipole vectors mu_{0I} (a.u.)
    "osc_S",               # (N_S,) float64: adiabatic TDA oscillator strengths
    "S_feats",             # (N_S, F_single) float64: strictly gauge-invariant single-exciton scalar features
                           #   [omega_eV, K_II_eV (self-Coulomb Q_I.Q_I), X_II_eV (self-exchange),
                           #    Lambda_spatial (hole-particle spatial overlap), D_CT_bohr,
                           #    nto_entropy (NTO entanglement), nto_inv_pr (sum lambda_k^2, MO-rotation invariant!),
                           #    eps_diff_mean_eV, eps_diff_std_eV, dip_norm_au,
                           #    hole_part_quad_aniso, fxc_diag_eV]
    "SS_pair_feats",       # (N_S, N_S, F_ss) float64: strictly gauge-invariant pair features between singlets I,J
                           #   [abs(Q_I.Q_J)_eV, Tr(Ph_I Ph_J), Tr(Pp_I Pp_J),
                           #    abs(Tr(X_I^T X_J)), spatial_density_overlap_IJ]
    "SS_couplings",        # (N_S, N_S, 2) float64: strictly gauge-equivariant static S-S couplings
                           #   [Q_I.Q_J_eV (signed Coulomb), signed_fxc_or_exch_eV]

    # Adiabatic TDA Triplet eigenstates (N_T <= 16)
    "omega_T_eV",          # (N_T,) float64: adiabatic TDA triplet energies (eV)
    "T_feats",             # (N_T, F_single) float64: strictly gauge-invariant triplet-exciton scalar features
    "TT_pair_feats",       # (N_T, N_T, F_ss) float64: gauge-invariant pair features between triplets I,J
    "TT_couplings",        # (N_T, N_T, 2) float64: gauge-equivariant static T-T couplings

    # Cross singlet-triplet overlap (key physical invariant for INVEST!)
    "ST_overlap_abs",      # (N_S, N_T) float64: |Tr(X_S,I^T X_T,J)|
    "ST_hole_ov",          # (N_S, N_T) float64: Tr(Ph_S,I Ph_T,J)
    "ST_part_ov",          # (N_S, N_T) float64: Tr(Pp_S,I Pp_T,J)

    # Singlet-manifold bi-exciton (doubles) basis D^{(S)} of size N_DS:
    # Constructed from:
    #   channel 0: triplet-triplet spin-coupled to singlet ^1(T_I T_J) for I <= J (drives 2^1Ag & INVEST!)
    #   channel 1: singlet-singlet ^1(S_I S_J) for I <= J
    "D_S_bare_eV",         # (N_DS,) float64: bare bi-exciton energy Omega_I + Omega_J (eV)
    "D_S_channel",         # (N_DS,) int32: 0 for ^1(TT), 1 for ^1(SS)
    "D_S_pairs",           # (N_DS, 2) int32: indices (I, J) of constituent excitons
    "D_S_feats",           # (N_DS, F_double) float64: strictly gauge-invariant bi-exciton features
                           #   [bare_eV, omega_I, omega_J, abs(omega_I-omega_J), is_diag(I==J),
                           #    channel_TT, Tr(Ph_I Ph_J), Tr(Pp_I Pp_J), abs(Tr(X_I^T X_J)),
                           #    J_IJ_eV (Coulomb Q_I.Q_J), W_ph_corr_eV (particle-hole interaction),
                           #    spatial_overlap_IJ, K_I_eV, K_J_eV, v0D_abs_eV]
    "W_SD_S",              # (N_S, N_DS, 4) float64: EXACT 4-channel gauge-equivariant RI Slater-Condon
                           #   couplings W^{(r)}_{K,(IJ)} (eV) between singlet K and singlet bi-exciton (I,J):
                           #   r=0: particle-Coulomb C^{(p)}_{K;IJ}
                           #   r=1: hole-Coulomb C^{(h)}_{K;IJ}
                           #   r=2: particle-Exchange X^{(p)}_{K;IJ}
                           #   r=3: hole-Exchange X^{(h)}_{K;IJ}
                           #   Guarantees: (i) exact phase equivariance s_K s_I s_J,
                           #   (ii) exact point-group selection rules (0 if Gamma_K not in Gamma_I x Gamma_J),
                           #   (iii) exact size-intensivity (0 identically for separated fragments!).
    "v0D_S",               # (N_DS, 2) float64: EXACT gauge-equivariant ground-to-doubles couplings
                           #   <0|V_ee|IJ> (Coulomb & Exchange channels, eV) for S0<->D CI topology!

    # Triplet-manifold bi-exciton (doubles) basis D^{(T)} of size N_DT:
    # Constructed from:
    #   channel 0: triplet-triplet spin-coupled to triplet ^3(T_I T_J) for I < J
    #   channel 1: singlet-triplet ^3(S_I T_J)
    "D_T_bare_eV",         # (N_DT,) float64
    "D_T_channel",         # (N_DT,) int32: 0 for ^3(TT), 1 for ^3(ST)
    "D_T_pairs",           # (N_DT, 2) int32
    "D_T_feats",           # (N_DT, F_double) float64
    "W_SD_T",              # (N_T, N_DT, 4) float64: EXACT 4-channel gauge-equivariant RI couplings (eV)

    # Atomic coordinates & atomic numbers (ONLY for 3D GNN baseline comparison; NOT used by Lehmann-TDDFT!)
    "atomic_numbers",      # (n_atoms,) int32
    "coords_angstrom",     # (n_atoms, 3) float64
]

F_SINGLE = 12
F_SS = 5
F_DOUBLE = 15
N_W_CHANNELS = 4
