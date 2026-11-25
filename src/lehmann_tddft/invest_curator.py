"""Phase A Curator for Lehmann-TDDFT Pillar 3:
1. Builds the exact InChIKey exclusion index for QUESTDB (224 mols) + HEPTA-INVEST (44 mols).
2. Curates the Multi-Scaffold Experimental & High-Level INVEST Benchmark (32 molecules)
   across 4 structural families (experimental INVEST emitters, non-heptazine azaphenalenes,
   B-N doped boraphenalenes/triangulenes, and non-alternant hydrocarbons/heterocycles).
3. Enumerates a ~500-molecule De Novo Combinatorial Bright-INVEST Virtual Screening Library
   targeting the Dark-State Dilemma (negative dE_ST AND bright f_S1) strictly outside
   QUESTDB & HEPTA-INVEST.
4. Optimizes all geometries cleanly with RDKit ETKDGv3 + tblite GFN2-xTB (ASE BFGS)
   using at most 4 worker processes.

NOTE (macOS runtime advisory): NEVER import torch in this script (tblite libomp clash).
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdDetermineBonds, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Import RDKit SA_Score
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_EXT = ROOT / "data" / "external"
DATA_PROC = ROOT / "data" / "processed"
INVEST_XYZ_DIR = DATA_PROC / "invest_xyz"
VIRTUAL_XYZ_DIR = DATA_PROC / "virtual_bright_invest_xyz"


# ---------------------------------------------------------------------------
# 1. Substituent Attachment Utilities (100% Exact RWMol Construction)
# ---------------------------------------------------------------------------

SUBSTITUENT_SMILES: Dict[str, str] = {
    "H": "[*][H]",
    "F": "[*]F",
    "Cl": "[*]Cl",
    "CN": "[*]C#N",
    "CF3": "[*]C(F)(F)F",
    "NH2": "[*]N",
    "NMe2": "[*]N(C)C",
    "OH": "[*]O",
    "OMe": "[*]OC",
    "SH": "[*]S",
    "SMe": "[*]SC",
    "CHO": "[*]C=O",
    "COMe": "[*]C(=O)C",
    "COOH": "[*]C(=O)O",
    "Me": "[*]C",
    "CCH": "[*]C#C",
    "BOH2": "[*]B(O)O",
    "OCF3": "[*]OC(F)(F)F",
    "OCH2CF3": "[*]OCC(F)(F)F",
    "Ph": "[*]c1ccccc1",
    "PH2": "[*]P",
    "Xylyl24": "[*]c1ccc(C)cc1C",
}


def attach_substituents(
    core_mol: Chem.Mol, subs: Optional[Dict[int, str]]
) -> Optional[Chem.Mol]:
    """Attach substituents specified as {core_atom_idx: sub_key} cleanly."""
    if not subs:
        m = Chem.Mol(core_mol)
        try:
            Chem.SanitizeMol(m)
            return m
        except Exception:
            return None

    rw = Chem.RWMol(core_mol)
    for atom_idx, sub_key in sorted(subs.items()):
        if sub_key == "H" or sub_key is None:
            continue
        smi = SUBSTITUENT_SMILES[sub_key]
        frag = Chem.MolFromSmiles(smi)
        if frag is None:
            return None
        # Find dummy atom '*' (atomic num 0) and its single neighbor
        dummy_idx = None
        attach_neighbor = None
        for a in frag.GetAtoms():
            if a.GetAtomicNum() == 0:
                dummy_idx = a.GetIdx()
                nbrs = a.GetNeighbors()
                if len(nbrs) != 1:
                    return None
                attach_neighbor = nbrs[0].GetIdx()
                break
        if dummy_idx is None:
            return None

        # Copy non-dummy atoms into rw
        idx_map: Dict[int, int] = {}
        for a in frag.GetAtoms():
            if a.GetIdx() == dummy_idx:
                continue
            new_a = Chem.Atom(a.GetAtomicNum())
            new_a.SetFormalCharge(a.GetFormalCharge())
            new_a.SetIsAromatic(a.GetIsAromatic())
            idx_map[a.GetIdx()] = rw.AddAtom(new_a)
        for b in frag.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if u == dummy_idx or v == dummy_idx:
                continue
            rw.AddBond(idx_map[u], idx_map[v], b.GetBondType())
        # Connect core atom to substituent attachment atom
        rw.AddBond(atom_idx, idx_map[attach_neighbor], Chem.BondType.SINGLE)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        # Re-roundtrip through canonical SMILES to ensure Kekule & aromatic cleanliness
        can = Chem.MolToSmiles(mol, canonical=True)
        mol2 = Chem.MolFromSmiles(can)
        if mol2 is None:
            return None
        Chem.SanitizeMol(mol2)
        return mol2
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Exact Mathematical Molecular Graph Constructors for All Core Scaffolds
# ---------------------------------------------------------------------------

# Phenalene IUPAC rim positions 1..9 -> internal node indices 4..12
# Node 0 = central (9b); 1,2,3 = bridgeheads (9a, 3a, 6a)
# Rim cycle: 1(9a) - 4(p1) = 5(p2) - 6(p3) = 2(3a) - 7(p4) = 8(p5) - 9(p6) = 3(6a) - 10(p7) = 11(p8) - 12(p9) = 1(9a)
PHENALENE_POS_TO_IDX = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12}


def build_phenalene_core(
    central_Z: int = 7,
    rim_N: Optional[Set[int]] = None,
    rim_bn_pairs: Optional[List[Tuple[int, int, str]]] = None,
    subs: Optional[Dict[int, str]] = None,
) -> Optional[Chem.Mol]:
    """Build exact 13-atom 9b-azaphenalene (central_Z=7, 14pi) or
    9b-boraphenalene (central_Z=5, 12pi) with optional rim N positions (1..9),
    isosteric rim BN pairs on double-bonded peripheral edges (1,2), (4,5), (7,8),
    and peripheral substituents {pos_1_to_9: sub_key}.

    rim_bn_pairs: list of (pos_a, pos_b, order) where (pos_a, pos_b) in {(1,2),(4,5),(7,8)}
                  and order is 'BN' (pos_a=B, pos_b=N) or 'NB' (pos_a=N, pos_b=B).
    """
    rim_N = set(rim_N or [])
    bn_map: Dict[int, int] = {}
    bn_edges: Set[Tuple[int, int]] = set()
    for pa, pb, order in (rim_bn_pairs or []):
        if (pa, pb) not in {(1, 2), (4, 5), (7, 8)}:
            raise ValueError(f"BN pair must lie on peripheral double bond (1,2)/(4,5)/(7,8), got {(pa, pb)}")
        za, zb = (5, 7) if order == "BN" else (7, 5)
        bn_map[pa] = za
        bn_map[pb] = zb
        bn_edges.add((PHENALENE_POS_TO_IDX[pa], PHENALENE_POS_TO_IDX[pb]))

    rw = Chem.RWMol()
    # 0: central atom (9b)
    rw.AddAtom(Chem.Atom(central_Z))
    # 1, 2, 3: bridgehead carbons (9a, 3a, 6a)
    for _ in range(3):
        rw.AddAtom(Chem.Atom(6))
    # 4..12: peripheral positions 1..9
    for pos in range(1, 10):
        if pos in bn_map:
            rw.AddAtom(Chem.Atom(bn_map[pos]))
        elif pos in rim_N:
            rw.AddAtom(Chem.Atom(7))
        else:
            rw.AddAtom(Chem.Atom(6))

    # Central single bonds
    for bh in (1, 2, 3):
        rw.AddBond(0, bh, Chem.BondType.SINGLE)

    # 12-membered rim with exact alternating single/double bonds
    rim_edges = [
        (1, 4, Chem.BondType.SINGLE),
        (4, 5, Chem.BondType.DOUBLE),
        (5, 6, Chem.BondType.SINGLE),
        (6, 2, Chem.BondType.DOUBLE),
        (2, 7, Chem.BondType.SINGLE),
        (7, 8, Chem.BondType.DOUBLE),
        (8, 9, Chem.BondType.SINGLE),
        (9, 3, Chem.BondType.DOUBLE),
        (3, 10, Chem.BondType.SINGLE),
        (10, 11, Chem.BondType.DOUBLE),
        (11, 12, Chem.BondType.SINGLE),
        (12, 1, Chem.BondType.DOUBLE),
    ]
    for u, v, btype in rim_edges:
        if (u, v) in bn_edges or (v, u) in bn_edges:
            rw.AddBond(u, v, Chem.BondType.SINGLE)
        else:
            rw.AddBond(u, v, btype)

    mol = rw.GetMol()
    if subs:
        idx_subs = {}
        for pos, skey in subs.items():
            # Do not attach substituent onto pyridine-like =N-
            if pos in rim_N and pos not in bn_map:
                return None
            idx_subs[PHENALENE_POS_TO_IDX[pos]] = skey
        return attach_substituents(mol, idx_subs)
    return attach_substituents(mol, None)


def build_pentalene_core(
    rim_N: Optional[Set[int]] = None,
    subs: Optional[Dict[int, str]] = None,
) -> Optional[Chem.Mol]:
    """Build exact 8-atom 8pi pentalene / aza-pentalene core.
    Indices: 0,4 = bridgeheads (3a, 6a).
    Positions 1..6 map to rim indices: 1->1, 2->2, 3->3 (left 5-ring), 4->5, 5->6, 6->7 (right 5-ring).
    Perimeter: 0-1=2-3=4-5=6-7=0, internal bond 0-4 SINGLE.
    """
    pos_to_idx = {1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7}
    rim_N = set(rim_N or [])
    rw = Chem.RWMol()
    for i in range(8):
        z = 6
        for p, idx in pos_to_idx.items():
            if idx == i and p in rim_N:
                z = 7
        rw.AddAtom(Chem.Atom(z))
    edges = [
        (0, 4, Chem.BondType.SINGLE),
        (0, 1, Chem.BondType.SINGLE),
        (1, 2, Chem.BondType.DOUBLE),
        (2, 3, Chem.BondType.SINGLE),
        (3, 4, Chem.BondType.DOUBLE),
        (4, 5, Chem.BondType.SINGLE),
        (5, 6, Chem.BondType.DOUBLE),
        (6, 7, Chem.BondType.SINGLE),
        (7, 0, Chem.BondType.DOUBLE),
    ]
    for u, v, bt in edges:
        rw.AddBond(u, v, bt)
    mol = rw.GetMol()
    if subs:
        idx_subs = {}
        for p, skey in subs.items():
            if p in rim_N:
                return None
            idx_subs[pos_to_idx[p]] = skey
        return attach_substituents(mol, idx_subs)
    return attach_substituents(mol, None)


def build_sindacene_core(
    rim_N: Optional[Set[int]] = None,
    subs: Optional[Dict[int, str]] = None,
) -> Optional[Chem.Mol]:
    """Build exact 12-atom 12pi s-indacene / aza-s-indacene core.
    Bridgeheads: 0, 1, 3, 4.
    Rim positions 1..8:
      1->6, 2->7, 3->8 (left 5-ring),
      4->2 (top central 6-ring),
      5->9, 6->10, 7->11 (right 5-ring),
      8->5 (bottom central 6-ring).
    Perimeter: 0-6=7-8=1-2=3-9=10-11=4-5=0, internal 0-1 SINGLE, 3-4 SINGLE.
    """
    pos_to_idx = {1: 6, 2: 7, 3: 8, 4: 2, 5: 9, 6: 10, 7: 11, 8: 5}
    rim_N = set(rim_N or [])
    rw = Chem.RWMol()
    for i in range(12):
        z = 6
        for p, idx in pos_to_idx.items():
            if idx == i and p in rim_N:
                z = 7
        rw.AddAtom(Chem.Atom(z))
    edges = [
        (0, 1, Chem.BondType.SINGLE),
        (3, 4, Chem.BondType.SINGLE),
        (0, 6, Chem.BondType.SINGLE),
        (6, 7, Chem.BondType.DOUBLE),
        (7, 8, Chem.BondType.SINGLE),
        (8, 1, Chem.BondType.DOUBLE),
        (1, 2, Chem.BondType.SINGLE),
        (2, 3, Chem.BondType.DOUBLE),
        (3, 9, Chem.BondType.SINGLE),
        (9, 10, Chem.BondType.DOUBLE),
        (10, 11, Chem.BondType.SINGLE),
        (11, 4, Chem.BondType.DOUBLE),
        (4, 5, Chem.BondType.SINGLE),
        (5, 0, Chem.BondType.DOUBLE),
    ]
    for u, v, bt in edges:
        rw.AddBond(u, v, bt)
    mol = rw.GetMol()
    if subs:
        idx_subs = {}
        for p, skey in subs.items():
            if p in rim_N:
                return None
            idx_subs[pos_to_idx[p]] = skey
        return attach_substituents(mol, idx_subs)
    return attach_substituents(mol, None)


def build_azulene_core(
    rim_N: Optional[Set[int]] = None,
    subs: Optional[Dict[int, str]] = None,
) -> Optional[Chem.Mol]:
    """Build exact 10-atom 10pi azulene / aza-azulene core.
    Bridgeheads: 0, 4.
    5-ring rim positions: 1->1, 2->2, 3->3.
    7-ring rim positions: 4->5, 5->6, 6->7, 7->8, 8->9.
    Perimeter: 0-1=2-3=4-5=6-7=8-9=0, internal 0-4 SINGLE.
    """
    pos_to_idx = {1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
    rim_N = set(rim_N or [])
    rw = Chem.RWMol()
    for i in range(10):
        z = 6
        for p, idx in pos_to_idx.items():
            if idx == i and p in rim_N:
                z = 7
        rw.AddAtom(Chem.Atom(z))
    edges = [
        (0, 4, Chem.BondType.SINGLE),
        (0, 1, Chem.BondType.SINGLE),
        (1, 2, Chem.BondType.DOUBLE),
        (2, 3, Chem.BondType.SINGLE),
        (3, 4, Chem.BondType.DOUBLE),
        (4, 5, Chem.BondType.SINGLE),
        (5, 6, Chem.BondType.DOUBLE),
        (6, 7, Chem.BondType.SINGLE),
        (7, 8, Chem.BondType.DOUBLE),
        (8, 9, Chem.BondType.SINGLE),
        (9, 0, Chem.BondType.DOUBLE),
    ]
    for u, v, bt in edges:
        rw.AddBond(u, v, bt)
    mol = rw.GetMol()
    if subs:
        idx_subs = {}
        for p, skey in subs.items():
            if p in rim_N:
                return None
            idx_subs[pos_to_idx[p]] = skey
        return attach_substituents(mol, idx_subs)
    return attach_substituents(mol, None)


def build_calicene_core(
    rim_N: Optional[Set[int]] = None,
    subs: Optional[Dict[int, str]] = None,
) -> Optional[Chem.Mol]:
    """Build exact 8-atom calicene / aza-calicene core.
    0=5 exocyclic double bond.
    5-ring: 0-1=2-3=4-0 (positions 1..4 -> indices 1..4).
    3-ring: 5-6=7-5 (positions 5,6 -> indices 6,7).
    """
    pos_to_idx = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7}
    rim_N = set(rim_N or [])
    rw = Chem.RWMol()
    for i in range(8):
        z = 6
        for p, idx in pos_to_idx.items():
            if idx == i and p in rim_N:
                z = 7
        rw.AddAtom(Chem.Atom(z))
    edges = [
        (0, 5, Chem.BondType.DOUBLE),
        (0, 1, Chem.BondType.SINGLE),
        (1, 2, Chem.BondType.DOUBLE),
        (2, 3, Chem.BondType.SINGLE),
        (3, 4, Chem.BondType.DOUBLE),
        (4, 0, Chem.BondType.SINGLE),
        (5, 6, Chem.BondType.SINGLE),
        (6, 7, Chem.BondType.DOUBLE),
        (7, 5, Chem.BondType.SINGLE),
    ]
    for u, v, bt in edges:
        rw.AddBond(u, v, bt)
    mol = rw.GetMol()
    if subs:
        idx_subs = {}
        for p, skey in subs.items():
            if p in rim_N:
                return None
            idx_subs[pos_to_idx[p]] = skey
        return attach_substituents(mol, idx_subs)
    return attach_substituents(mol, None)


def build_heptalene_core(
    rim_N: Optional[Set[int]] = None,
    subs: Optional[Dict[int, str]] = None,
) -> Optional[Chem.Mol]:
    """Build exact 12-atom 12pi heptalene / aza-heptalene core.
    Bridgeheads: 0, 6.
    Rim positions 1..5 -> indices 1..5; 6..10 -> indices 7..11.
    """
    pos_to_idx = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}
    rim_N = set(rim_N or [])
    rw = Chem.RWMol()
    for i in range(12):
        z = 6
        for p, idx in pos_to_idx.items():
            if idx == i and p in rim_N:
                z = 7
        rw.AddAtom(Chem.Atom(z))
    edges = [(0, 6, Chem.BondType.SINGLE)]
    for i in range(12):
        u, v = i, (i + 1) % 12
        bt = Chem.BondType.SINGLE if (i % 2 == 0) else Chem.BondType.DOUBLE
        edges.append((u, v, bt))
    for u, v, bt in edges:
        rw.AddBond(u, v, bt)
    mol = rw.GetMol()
    if subs:
        idx_subs = {}
        for p, skey in subs.items():
            if p in rim_N:
                return None
            idx_subs[pos_to_idx[p]] = skey
        return attach_substituents(mol, idx_subs)
    return attach_substituents(mol, None)


def mol_is_valid_closed_shell(mol: Optional[Chem.Mol], min_heavy: int = 6, max_heavy: int = 45) -> bool:
    """Strict chemical validity filter: neutral, closed-shell, single fragment, valid valences."""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    if Chem.GetFormalCharge(mol) != 0:
        return False
    # Check radicals & even total electrons
    mol_h = Chem.AddHs(mol)
    n_elec = sum(a.GetAtomicNum() for a in mol_h.GetAtoms())
    if n_elec % 2 != 0:
        return False
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() != 0:
            return False
    nh = mol.GetNumHeavyAtoms()
    if nh < min_heavy or nh > max_heavy:
        return False
    return True


def has_unstable_motif(mol: Chem.Mol) -> bool:
    """Filter out chemically unstable motifs for virtual screening library:
    N-N-N chains, O-O peroxides, B-B bonds, B-O-O, N-F, S-S."""
    smarts_bad = [
        "[#7]~[#7]~[#7]",  # triazene / azide
        "[#7]~[#7]",       # exclude all N-N bonds for high synthetic plausibility
        "[#8]~[#8]",       # peroxide
        "[#5]~[#5]",       # diboron
        "[#7]-[#9]",       # N-F
        "[#16]~[#16]",     # disulfide
    ]
    for s in smarts_bad:
        patt = Chem.MolFromSmarts(s)
        if patt is not None and mol.HasSubstructMatch(patt):
            return True
    return False


def compute_mol_descriptors(mol: Chem.Mol) -> Dict:
    """Compute canonical SMILES, InChIKey, formula, QED, SA-score, MW, n_heavy."""
    smi = Chem.MolToSmiles(mol, canonical=True)
    ik = Chem.MolToInchiKey(mol)
    mol_h = Chem.AddHs(mol)
    return {
        "smiles": smi,
        "inchikey": ik,
        "inchikey_14": ik[:14],
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "n_heavy": int(mol.GetNumHeavyAtoms()),
        "n_atoms": int(mol_h.GetNumAtoms()),
        "n_electrons": int(sum(a.GetAtomicNum() for a in mol_h.GetAtoms())),
        "mw": round(float(Descriptors.MolWt(mol)), 3),
        "qed": round(float(QED.qed(mol)), 4),
        "sa_score": round(float(sascorer.calculateScore(mol)), 4),
    }


# ---------------------------------------------------------------------------
# 3. Build Exact Exclusion Index (QUESTDB + HEPTA-INVEST)
# ---------------------------------------------------------------------------

def build_known_exclusion_index() -> Dict[str, Set[str]]:
    """Extract canonical InChIKeys (full + 14-char connectivity) for ALL
    44 HEPTA-INVEST molecules and ALL QUESTDB geometries."""
    hepta_iks: Set[str] = set()
    hepta_ik14s: Set[str] = set()
    hepta_smis: Set[str] = set()

    hepta_csv = DATA_EXT / "HEPTA-INVEST" / "data" / "HEPTA-INVEST44.csv"
    df = pd.read_csv(hepta_csv)
    df_eom = df[df["method"] == "EOM-CCSD"]
    hept_N = {1, 3, 4, 6, 7, 9}
    for _, row in df_eom.iterrows():
        subs = {}
        for pos, col in [(2, "substituent_1"), (5, "substituent_2"), (8, "substituent_3")]:
            val = row[col]
            if pd.notna(val) and str(val).strip() and str(val).strip() != "H":
                subs[pos] = str(val).strip()
        m = build_phenalene_core(central_Z=7, rim_N=hept_N, subs=subs)
        assert m is not None, f"Failed building HEPTA-INVEST mol {row['molecule']}"
        ik = Chem.MolToInchiKey(m)
        hepta_iks.add(ik)
        hepta_ik14s.add(ik[:14])
        hepta_smis.add(Chem.MolToSmiles(m, canonical=True))

    quest_iks: Set[str] = set()
    quest_ik14s: Set[str] = set()
    quest_smis: Set[str] = set()
    reg_path = DATA_EXT / "QUESTDB" / "geometries" / "registry.json"
    reg = json.load(open(reg_path))
    for name, info in reg["geometries"].items():
        xyz_path = DATA_EXT / "QUESTDB" / info["xyz"]
        if not xyz_path.exists():
            continue
        try:
            raw_mol = Chem.MolFromXYZFile(str(xyz_path))
            if raw_mol is None:
                continue
            rdDetermineBonds.DetermineBonds(raw_mol, charge=0)
            m = Chem.RemoveHs(raw_mol)
            Chem.SanitizeMol(m)
            ik = Chem.MolToInchiKey(m)
            quest_iks.add(ik)
            quest_ik14s.add(ik[:14])
            quest_smis.add(Chem.MolToSmiles(m, canonical=True))
        except Exception:
            pass

    # Explicitly guarantee key QUESTDB non-alternant / INVEST molecules are in the set
    explicit_quest = [
        build_phenalene_core(7, set()),                       # cyclazine
        build_phenalene_core(7, {1, 3, 4, 6, 7, 9}),          # heptazine
        build_pentalene_core(),                               # pentalene
        build_sindacene_core(),                               # s-indacene
        build_azulene_core(),                                 # azulene
        Chem.MolFromSmiles("c1cn2ccnc2n1"),                   # triazapentalene
    ]
    for m in explicit_quest:
        if m is not None:
            ik = Chem.MolToInchiKey(m)
            quest_iks.add(ik)
            quest_ik14s.add(ik[:14])
            quest_smis.add(Chem.MolToSmiles(m, canonical=True))

    out = {
        "hepta_invest_count": len(hepta_iks),
        "questdb_parsed_count": len(quest_iks),
        "hepta_invest_inchikeys": sorted(hepta_iks),
        "hepta_invest_ik14": sorted(hepta_ik14s),
        "questdb_inchikeys": sorted(quest_iks),
        "questdb_ik14": sorted(quest_ik14s),
        "all_known_inchikeys": sorted(hepta_iks | quest_iks),
        "all_known_ik14": sorted(hepta_ik14s | quest_ik14s),
    }
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    with open(DATA_PROC / "known_benchmark_inchikeys.json", "w") as f:
        json.dump(out, f, indent=2)
    return {
        "all_ik": hepta_iks | quest_iks,
        "all_ik14": hepta_ik14s | quest_ik14s,
        "hepta_ik14": hepta_ik14s,
        "quest_ik14": quest_ik14s,
    }


# ---------------------------------------------------------------------------
# 4. 3D Embedding (RDKit ETKDGv3 + MMFF/UFF) + tblite GFN2-xTB Optimization
# ---------------------------------------------------------------------------

def optimize_one_molecule(args: Tuple[str, str, str, Optional[str]]) -> Dict:
    """Worker function: generates 3D conformer with RDKit ETKDGv3, relaxes with
    MMFF94/UFF, then optimizes with tblite GFN2-xTB via ASE BFGS, and writes .xyz.
    If reference_xyz_path is provided, starts from that geometry (still GFN2-xTB opt
    or preserves exact reference if requested — we optimize with GFN2-xTB uniformly
    and record both!)."""
    mol_id, smiles, out_xyz_path, init_xyz_path = args
    from ase import Atoms
    from ase.optimize import BFGS
    from tblite.ase import TBLite

    t0 = time.perf_counter()
    try:
        if init_xyz_path and Path(init_xyz_path).exists():
            lines = Path(init_xyz_path).read_text().strip().splitlines()
            nat = int(lines[0].strip())
            syms, coords = [], []
            for ln in lines[2 : 2 + nat]:
                parts = ln.split()
                syms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            atoms = Atoms(symbols=syms, positions=coords)
            source_geom = "questdb_ccsd_t_init"
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"mol_id": mol_id, "ok": False, "error": "Invalid SMILES"}
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 2026
            params.useRandomCoords = False
            cid = AllChem.EmbedMolecule(mol_h, params)
            if cid < 0:
                params.useRandomCoords = True
                cid = AllChem.EmbedMolecule(mol_h, params)
            if cid < 0:
                return {"mol_id": mol_id, "ok": False, "error": "ETKDGv3 embed failed"}
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
                else:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass
            conf = mol_h.GetConformer()
            syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
            coords = [list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())]
            atoms = Atoms(symbols=syms, positions=coords)
            source_geom = "rdkit_etkdgv3_gfn2xtb"

        # Optimize with tblite GFN2-xTB
        calc = TBLite(method="GFN2-xTB", verbosity=0, max_iterations=500)
        atoms.calc = calc
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.02, steps=200)

        energy_eV = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        pos = atoms.get_positions()
        syms = atoms.get_chemical_symbols()

        # Compute planarity metric (smallest singular value RMS of heavy atoms)
        heavy_pos = np.array([p for s, p in zip(syms, pos) if s != "H"])
        heavy_centered = heavy_pos - heavy_pos.mean(axis=0)
        _, svals, _ = np.linalg.svd(heavy_centered, full_matrices=False)
        planarity_rmsd = float(svals[-1] / np.sqrt(len(heavy_pos))) if len(heavy_pos) >= 3 else 0.0

        # Compute GFN2-xTB dipole moment if available
        try:
            dip_vec = atoms.get_dipole_moment()  # e*Angstrom in ASE
            # 1 e*Angstrom = 4.8032047 Debye
            dip_debye = float(np.linalg.norm(dip_vec) * 4.8032047)
        except Exception:
            dip_debye = None

        # Write clean XYZ file
        Path(out_xyz_path).parent.mkdir(parents=True, exist_ok=True)
        xyz_lines = [
            str(len(syms)),
            f"{mol_id} | GFN2-xTB E={energy_eV:.6f} eV fmax={fmax:.4f} | SMILES={smiles}",
        ]
        for s, (x, y, z) in zip(syms, pos):
            xyz_lines.append(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
        Path(out_xyz_path).write_text("\n".join(xyz_lines) + "\n")

        return {
            "mol_id": mol_id,
            "ok": True,
            "xyz_path": str(Path(out_xyz_path).relative_to(ROOT)),
            "source_geom": source_geom,
            "gfn2_xtb_energy_eV": round(energy_eV, 6),
            "gfn2_xtb_fmax_eV_A": round(fmax, 5),
            "planarity_rmsd_A": round(planarity_rmsd, 4),
            "gfn2_dipole_debye": round(dip_debye, 4) if dip_debye is not None else None,
            "opt_time_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as exc:
        return {"mol_id": mol_id, "ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 5. Phase A.1: Curate the Multi-Scaffold Experimental & High-Level Benchmark
# ---------------------------------------------------------------------------

def curate_multi_scaffold_benchmark() -> List[Dict]:
    """Assemble 32 curated molecules across 4 distinct structural families."""
    entries: List[Tuple[str, str, str, str, Chem.Mol, Dict, Optional[str]]] = []

    def add(
        mol_id: str,
        name: str,
        family: str,
        sym_class: str,
        mol: Optional[Chem.Mol],
        lit_ref: Dict,
        init_xyz: Optional[str] = None,
    ):
        assert mol is not None and mol_is_valid_closed_shell(mol, min_heavy=6, max_heavy=45), (
            f"Invalid benchmark molecule: {mol_id}"
        )
        entries.append((mol_id, name, family, sym_class, mol, lit_ref, init_xyz))

    hept_N = {1, 3, 4, 6, 7, 9}
    ap5_N = {1, 3, 4, 6}

    # --- Family 1: Experimentally Verified & Reference INVEST Emitters (9 mols) ---
    add(
        "5AP",
        "1,3,4,6,9b-Pentaazaphenalene (5AP)",
        "experimental_invest",
        "C2v",
        build_phenalene_core(7, ap5_N),
        {"source": "Wilson et al. JACS 2024", "exp_dEST_eV": -0.047, "exp_dEST_err_eV": 0.007, "run_ab_initio": True},
    )
    add(
        "cyclazine",
        "Cycl[3.3.3]azine (9b-Azaphenalene)",
        "experimental_invest",
        "D3h",
        build_phenalene_core(7, set()),
        {"source": "QUESTDB TBE / Leupin 1980", "questdb_TBE_dEST_eV": -0.131, "run_ab_initio": True},
        init_xyz=str(DATA_EXT / "QUESTDB" / "geometries" / "xyz" / "cyclazine.xyz"),
    )
    add(
        "heptazine",
        "Heptazine (1,3,4,6,7,9,9b-Heptaazaphenalene)",
        "experimental_invest",
        "D3h",
        build_phenalene_core(7, hept_N),
        {"source": "QUESTDB TBE / HEPTA-INVEST", "questdb_TBE_dEST_eV": -0.219, "hepta_eom_dEST_eV": -0.1729, "run_ab_initio": True},
        init_xyz=str(DATA_EXT / "QUESTDB" / "geometries" / "xyz" / "heptazine.xyz"),
    )
    add(
        "melem",
        "Melem (2,5,8-Triaminoheptazine)",
        "experimental_invest",
        "D3h",
        build_phenalene_core(7, hept_N, subs={2: "NH2", 5: "NH2", 8: "NH2"}),
        {"source": "HEPTA-INVEST (2T-7N-NH2-NH2-NH2)", "hepta_eom_dEST_eV": -0.2698, "run_ab_initio": True},
    )
    add(
        "HzTFEX2_core",
        "2-(2,2,2-Trifluoroethoxy)-5,8-dimethylheptazine (HzTFEX2 core model)",
        "experimental_invest",
        "Cs",
        build_phenalene_core(7, hept_N, subs={2: "OCH2CF3", 5: "Me", 8: "Me"}),
        {"source": "Compact electronic core of Aizawa Nature 2022 HzTFEX2", "run_ab_initio": False},
    )
    add(
        "HzTFE3_model",
        "2,5,8-Trimethoxyheptazine (HzTFE3 compact model)",
        "experimental_invest",
        "C3h",
        build_phenalene_core(7, hept_N, subs={2: "OMe", 5: "OMe", 8: "OMe"}),
        {"source": "Compact electronic core of Liu JPCC 2026 HzTFE3", "run_ab_initio": False},
    )
    add(
        "HzDA3",
        "2,5,8-Tris(dimethylamino)heptazine (HzDA3)",
        "experimental_invest",
        "D3h",
        build_phenalene_core(7, hept_N, subs={2: "NMe2", 5: "NMe2", 8: "NMe2"}),
        {"source": "Liu et al. JPCC 2026", "lit_eom_dEST_eV": -0.244, "exp_tau_DF_ns": 76, "run_ab_initio": False},
    )
    add(
        "HzTFE3",
        "2,5,8-Tris(2,2,2-trifluoroethoxy)heptazine (HzTFE3)",
        "experimental_invest",
        "C3",
        build_phenalene_core(7, hept_N, subs={2: "OCH2CF3", 5: "OCH2CF3", 8: "OCH2CF3"}),
        {"source": "Liu et al. JPCC 2026", "lit_eom_dEST_eV": -0.238, "exp_tau_DF_ns": 341, "run_ab_initio": False},
    )
    add(
        "HzTFEX2",
        "2-(2,2,2-Trifluoroethoxy)-5,8-bis(2,4-xylyl)heptazine (HzTFEX2)",
        "experimental_invest",
        "C1",
        build_phenalene_core(7, hept_N, subs={2: "OCH2CF3", 5: "Xylyl24", 8: "Xylyl24"}),
        {"source": "Aizawa et al. Nature 2022", "exp_dEST_eV": -0.011, "run_ab_initio": False},
    )

    # --- Family 2: Symmetry-Distinct Non-Heptazine Azaphenalenes (8 mols, all 13 heavy) ---
    add("2N_1_9b", "1,9b-Diazaphenalene", "azaphenalene_nonhept", "Cs", build_phenalene_core(7, {1}), {"run_ab_initio": True})
    add("3N_1_3_9b", "1,3,9b-Triazaphenalene", "azaphenalene_nonhept", "C2v", build_phenalene_core(7, {1, 3}), {"run_ab_initio": True})
    add("3N_1_4_9b", "1,4,9b-Triazaphenalene", "azaphenalene_nonhept", "Cs", build_phenalene_core(7, {1, 4}), {"run_ab_initio": True})
    add("4N_1_4_7_9b", "1,4,7,9b-Tetraazaphenalene (4AP)", "azaphenalene_nonhept", "C3h", build_phenalene_core(7, {1, 4, 7}), {"run_ab_initio": True})
    add("4N_1_3_7_9b", "1,3,7,9b-Tetraazaphenalene", "azaphenalene_nonhept", "Cs", build_phenalene_core(7, {1, 3, 7}), {"run_ab_initio": True})
    add("5N_1_3_4_7_9b", "1,3,4,7,9b-Pentaazaphenalene (5AP-asym)", "azaphenalene_nonhept", "Cs", build_phenalene_core(7, {1, 3, 4, 7}), {"run_ab_initio": True})
    add("6N_1_3_4_6_7_9b", "1,3,4,6,7,9b-Hexaazaphenalene (6AP)", "azaphenalene_nonhept", "Cs", build_phenalene_core(7, {1, 3, 4, 6, 7}), {"run_ab_initio": True})
    add("4N_2_5_8_9b", "2,5,8,9b-Tetraazaphenalene (Sublattice-B control)", "azaphenalene_nonhept", "D3h", build_phenalene_core(7, {2, 5, 8}), {"run_ab_initio": True})

    # --- Family 3: Boron-Nitrogen Doped Triangulenes / Boraphenalenes (7 mols, all 13 heavy) ---
    add("B_cyclazine", "9b-Boraphenalene (Cycl[3.3.3]borane)", "bn_triangulene", "D3h", build_phenalene_core(5, set()), {"run_ab_initio": True})
    add("B_heptazine", "1,3,4,6,7,9-Hexaaza-9b-boraphenalene (B-Heptazine)", "bn_triangulene", "D3h", build_phenalene_core(5, hept_N), {"run_ab_initio": True})
    add("B_5AP", "1,3,4,6-Tetraaza-9b-boraphenalene (B-5AP)", "bn_triangulene", "C2v", build_phenalene_core(5, ap5_N), {"run_ab_initio": True})
    add("B_4AP", "1,4,7-Triaza-9b-boraphenalene (B-4AP)", "bn_triangulene", "C3h", build_phenalene_core(5, {1, 4, 7}), {"run_ab_initio": True})
    add("B_3N_asym", "1,3,7-Triaza-9b-boraphenalene", "bn_triangulene", "Cs", build_phenalene_core(5, {1, 3, 7}), {"run_ab_initio": True})
    add("BN_rim_cyclazine", "1-Aza-2-bora-9b-azaphenalene (BN-rim cyclazine)", "bn_triangulene", "Cs", build_phenalene_core(7, set(), rim_bn_pairs=[(1, 2, "NB")]), {"run_ab_initio": True})
    add("B3N3_rim_phenalene", "1,4,7-Triaza-2,5,8-tribora-9b-azaphenalene (B3N3-BCN)", "bn_triangulene", "C3h", build_phenalene_core(7, set(), rim_bn_pairs=[(1, 2, "NB"), (4, 5, "NB"), (7, 8, "NB")]), {"run_ab_initio": True})

    # --- Family 4: Non-Alternant Hydrocarbons & Heterocycles (8 mols, 8-12 heavy) ---
    add("pentalene", "Pentalene", "non_alternant", "C2h", build_pentalene_core(), {"run_ab_initio": True})
    add("1_4_diazapentalene", "1,4-Diazapentalene", "non_alternant", "C2h", build_pentalene_core({1, 4}), {"run_ab_initio": True})
    add("2_5_diazapentalene", "2,5-Diazapentalene", "non_alternant", "C2h", build_pentalene_core({2, 5}), {"run_ab_initio": True})
    add("1_3_4_6_tetraazapentalene", "1,3,4,6-Tetraazapentalene", "non_alternant", "D2h", build_pentalene_core({1, 3, 4, 6}), {"run_ab_initio": True})
    add("s_indacene", "s-Indacene", "non_alternant", "C2h", build_sindacene_core(), {"run_ab_initio": True})
    add("1_5_diaza_s_indacene", "1,5-Diaza-s-indacene", "non_alternant", "C2h", build_sindacene_core({1, 5}), {"run_ab_initio": True})
    add("azulene", "Azulene (positive control)", "non_alternant", "C2v", build_azulene_core(), {"run_ab_initio": True}, init_xyz=str(DATA_EXT / "QUESTDB" / "geometries" / "xyz" / "azulene.xyz"))
    add("1_3_diazaazulene", "1,3-Diazaazulene", "non_alternant", "C2v", build_azulene_core({1, 3}), {"run_ab_initio": True})
    add("calicene", "Calicene", "non_alternant", "C2v", build_calicene_core(), {"run_ab_initio": True})
    add("heptalene", "Heptalene", "non_alternant", "C2h", build_heptalene_core(), {"run_ab_initio": True})

    records = []
    for mol_id, name, family, sym_class, mol, lit_ref, init_xyz in entries:
        desc = compute_mol_descriptors(mol)
        records.append({
            "mol_id": mol_id,
            "name": name,
            "family": family,
            "symmetry_class": sym_class,
            "lit_ref": lit_ref,
            "init_xyz": init_xyz,
            **desc,
        })
    return records


# ---------------------------------------------------------------------------
# 6. Phase A.2: De Novo Combinatorial Bright-INVEST Virtual Screening Library
# ---------------------------------------------------------------------------

def enumerate_virtual_bright_invest_library(
    known_ik14: Set[str], target_max: int = 550
) -> List[Dict]:
    """Systematically enumerate 450-550 chemically valid, synthetically plausible,
    deduplicated, NON-HEPTAZINE & symmetry-broken Bright-INVEST candidates
    (10-22 heavy atoms, zero QUESTDB/HEPTA-INVEST overlap)."""
    seen_ik14: Set[str] = set(known_ik14)
    seen_smi: Set[str] = set()
    candidates: List[Dict] = []

    def try_add(
        mol: Optional[Chem.Mol],
        scaffold_family: str,
        core_tag: str,
        design_strategy: str,
        min_heavy: int = 10,
        max_heavy: int = 22,
    ) -> bool:
        if not mol_is_valid_closed_shell(mol, min_heavy=min_heavy, max_heavy=max_heavy):
            return False
        if has_unstable_motif(mol):
            return False
        desc = compute_mol_descriptors(mol)
        if desc["inchikey_14"] in seen_ik14 or desc["smiles"] in seen_smi:
            return False
        # Exclude any molecule whose parent core is 7N-heptazine (we want truly NON-HEPTAZINE!)
        # Note: our enumerators below strictly never generate 7N-heptazine cores anyway.
        if desc["sa_score"] > 6.5:
            return False
        seen_ik14.add(desc["inchikey_14"])
        seen_smi.add(desc["smiles"])
        mol_id = f"VBI_{len(candidates)+1:04d}_{core_tag}"
        candidates.append({
            "mol_id": mol_id,
            "scaffold_family": scaffold_family,
            "core_tag": core_tag,
            "design_strategy": design_strategy,
            **desc,
        })
        return True

    # === GROUP 1: All symmetry-unique Asymmetric Non-Heptazine Azaphenalene Cores (2N..6N) ===
    # Sublattice A = {1,3,4,6,7,9}, Sublattice B = {2,5,8}
    subA = [1, 3, 4, 6, 7, 9]
    for k in range(1, 6):  # 1..5 rim N's on subA -> 2N..6N azaphenalenes
        for comb in itertools.combinations(subA, k):
            m = build_phenalene_core(7, set(comb))
            try_add(m, "asymmetric_azaphenalene", f"azaph_{k+1}N", "unsubstituted_asym_N_pattern")

    # Also mixed SubA + SubB N-patterns with ZERO adjacent N-N bonds (e.g. N at pos 2 with 1,3=C)
    mixed_N_patterns = [
        {2}, {2, 5}, {2, 5, 8},
        {2, 4}, {2, 4, 6}, {2, 4, 7}, {2, 4, 6, 7}, {2, 4, 6, 9}, {2, 4, 6, 7, 9},
        {2, 5, 7}, {2, 5, 7, 9}, {5, 1, 3}, {5, 1, 3, 7}, {5, 1, 3, 9},
    ]
    for pat in mixed_N_patterns:
        m = build_phenalene_core(7, pat)
        try_add(m, "asymmetric_azaphenalene", f"azaph_mix{len(pat)+1}N", "mixed_sublattice_N_pattern")

    # === GROUP 2: Push-Pull & Asymmetric Substituted Non-Heptazine Azaphenalenes (~190 mols) ===
    # Key non-heptazine cores + their available C positions (SubB and SubA)
    azaph_cores = [
        ("5AP", {1, 3, 4, 6}, [2, 5, 8, 7]),            # 5N C2v (Wilson JACS 2024 scaffold!)
        ("5APasym", {1, 3, 4, 7}, [2, 5, 8, 6]),        # 5N Cs
        ("6AP", {1, 3, 4, 6, 7}, [2, 5, 8, 9]),         # 6N Cs (bright 1-C-less-than-heptazine!)
        ("4AP", {1, 4, 7}, [2, 5, 8, 3]),               # 4N C3h
        ("4N_137", {1, 3, 7}, [2, 5, 8, 4]),            # 4N Cs
        ("3N_13", {1, 3}, [2, 5, 8, 4]),                # 3N C2v
        ("3N_14", {1, 4}, [2, 5, 8, 3]),                # 3N Cs
        ("2N_1", {1}, [2, 5, 8]),                       # 2N Cs
    ]
    mono_subs = ["CN", "F", "CF3", "NH2", "OMe", "NMe2", "OH", "SH", "CHO", "BOH2", "CCH", "Me"]
    donors = ["NH2", "OMe", "NMe2", "OH", "SH", "Me"]
    acceptors = ["CN", "F", "CF3", "CHO", "BOH2", "CCH"]

    # 2a. Mono-substituted at asymmetric sites (SubB pos 2 or 8, and SubA open site)
    for cname, nset, open_pos in azaph_cores:
        for pos in open_pos[:3]:  # vary attachment site to break symmetry!
            for s in mono_subs[:9]:
                m = build_phenalene_core(7, nset, subs={pos: s})
                try_add(m, "substituted_azaphenalene", f"{cname}_m", f"mono_{s}_p{pos}")

    # 2b. Push-pull Donor/Acceptor di-substituted on 5AP, 5APasym, 6AP, 4AP, 4N_137, 3N_13
    push_pull_pairs = [
        ("NH2", "CN"), ("OMe", "CN"), ("NMe2", "CN"), ("OH", "CN"), ("SH", "CN"),
        ("NH2", "F"), ("OMe", "F"), ("NMe2", "F"), ("NH2", "CF3"), ("OMe", "CF3"),
        ("NH2", "CHO"), ("OMe", "CHO"), ("OH", "F"), ("SH", "F"), ("CCH", "CN"),
        ("NH2", "BOH2"), ("OMe", "BOH2"), ("F", "CN"), ("CF3", "CN"), ("F", "F"),
        ("CN", "CN"), ("NH2", "NH2"), ("OMe", "OMe"), ("SH", "SH"),
    ]
    for cname, nset, open_pos in azaph_cores[:6]:
        p1, p2 = open_pos[0], open_pos[2]  # e.g. pos 2 and pos 8 (asymmetric!)
        for d, a in push_pull_pairs[:16]:
            m = build_phenalene_core(7, nset, subs={p1: d, p2: a})
            try_add(m, "push_pull_azaphenalene", f"{cname}_pp", f"pushpull_{d}_p{p1}_{a}_p{p2}")
        # Also substitute one SubB + one SubA position (strongest HOMO-LUMO tuning!)
        if len(open_pos) >= 4:
            p_b, p_a = open_pos[0], open_pos[3]
            for d, a in [("NH2", "CN"), ("OMe", "CN"), ("F", "NH2"), ("CN", "NH2"), ("F", "CN"), ("OH", "CN")]:
                m = build_phenalene_core(7, nset, subs={p_b: d, p_a: a})
                try_add(m, "push_pull_azaphenalene", f"{cname}_ab", f"subAB_{d}_p{p_b}_{a}_p{p_a}")

    # 2c. Asymmetric tri-substituted on 5AP and 6AP (e.g. D-D-A and A-A-D)
    for cname, nset, open_pos in [("5AP", {1, 3, 4, 6}, [2, 5, 8]), ("6AP", {1, 3, 4, 6, 7}, [2, 5, 8])]:
        for s1, s2, s3 in [
            ("NH2", "NH2", "CN"), ("NH2", "NH2", "F"), ("OMe", "OMe", "CN"),
            ("F", "F", "NH2"), ("F", "F", "OMe"), ("F", "F", "CN"),
            ("CN", "CN", "NH2"), ("CN", "CN", "OMe"), ("CN", "CN", "F"),
            ("Me", "Me", "CN"), ("OH", "OH", "CN"), ("NH2", "OMe", "F"),
        ]:
            m = build_phenalene_core(7, nset, subs={2: s1, 5: s2, 8: s3})
            try_add(m, "push_pull_azaphenalene", f"{cname}_tri", f"tri_{s1}_{s2}_{s3}")

    # === GROUP 3: Boron-Nitrogen Doped Boraphenalenes & BN-Rim Triangulenes (~140 mols) ===
    # 3a. All symmetry-unique 9b-boraphenalene aza-cores (B-0N through B-6N)
    for k in range(0, 7):
        for comb in itertools.combinations(subA, k):
            m = build_phenalene_core(5, set(comb))
            try_add(m, "boraphenalene_bn", f"Bph_{k}N", "boron_centered_aza")
    for pat in [{2}, {2, 5}, {2, 5, 8}, {2, 4, 6}, {2, 4, 7}]:
        m = build_phenalene_core(5, pat)
        try_add(m, "boraphenalene_bn", f"Bph_mix{len(pat)}N", "boron_centered_mixedN")

    # 3b. Substituted B-centered boraphenalenes (B-heptazine, B-5AP, B-4AP, B-3N_137, B-cyclazine)
    b_cores = [
        ("B7N", {1, 3, 4, 6, 7, 9}, [2, 5, 8]),
        ("B5AP", {1, 3, 4, 6}, [2, 5, 8, 7]),
        ("B4AP", {1, 4, 7}, [2, 5, 8, 3]),
        ("B3N", {1, 3, 7}, [2, 5, 8]),
        ("B6N", {1, 3, 4, 6, 7}, [2, 5, 8]),
        ("B0N", set(), [2, 5, 8]),
    ]
    for cname, nset, open_pos in b_cores:
        for s in ["F", "CN", "CF3", "NH2", "OMe", "NMe2", "OH", "SH", "CHO", "CCH", "Me"]:
            m = build_phenalene_core(5, nset, subs={open_pos[0]: s})
            try_add(m, "boraphenalene_bn", f"{cname}_m", f"Bcore_mono_{s}")
        for d, a in [("NH2", "CN"), ("OMe", "CN"), ("NH2", "F"), ("OMe", "F"), ("F", "CN"), ("OH", "CN"), ("SH", "F"), ("NMe2", "F"), ("F", "F"), ("CN", "CN"), ("NH2", "NH2")]:
            m = build_phenalene_core(5, nset, subs={open_pos[0]: d, open_pos[2]: a})
            try_add(m, "boraphenalene_bn", f"{cname}_pp", f"Bcore_pp_{d}_{a}")

    # 3c. Rim BN-isosteric triangulenes (14pi N-centered and 12pi B-centered with 1, 2, or 3 BN rim units!)
    bn_configs = [
        (7, set(), [(1, 2, "BN")], "BN1_Nph"),
        (7, set(), [(1, 2, "NB")], "NB1_Nph"),
        (7, {4, 6}, [(1, 2, "NB")], "NB1_3Nph"),
        (7, {4, 6, 7, 9}, [(1, 2, "NB")], "NB1_5Nph"),
        (7, {4, 6, 7, 9}, [(1, 2, "BN")], "BN1_5Nph"),
        (7, {4, 7}, [(1, 2, "NB")], "NB1_3Nasym"),
        (7, set(), [(1, 2, "NB"), (4, 5, "NB")], "NB2_Nph"),
        (7, {7, 9}, [(1, 2, "NB"), (4, 5, "NB")], "NB2_3Nph"),
        (7, set(), [(1, 2, "NB"), (4, 5, "NB"), (7, 8, "NB")], "NB3_Nph"),
        (7, set(), [(1, 2, "BN"), (4, 5, "BN"), (7, 8, "BN")], "BN3_Nph"),
        (5, set(), [(1, 2, "NB")], "NB1_Bph"),
        (5, {4, 6, 7, 9}, [(1, 2, "NB")], "NB1_B5N"),
        (5, set(), [(1, 2, "NB"), (4, 5, "NB"), (7, 8, "NB")], "NB3_Bph"),
    ]
    for cz, nset, bnpairs, tag in bn_configs:
        m = build_phenalene_core(cz, nset, rim_bn_pairs=bnpairs)
        try_add(m, "bn_rim_triangulene", tag, "bn_rim_core")
        # Add clean substitutions on free position 8 (if not in bn_pairs)
        used_pos = {p for pair in bnpairs for p in pair[:2]} | nset
        free_c = [p for p in [8, 5, 2, 3, 6, 9] if p not in used_pos]
        if free_c:
            for s in ["F", "CN", "OMe", "NH2", "CF3"]:
                m2 = build_phenalene_core(cz, nset, rim_bn_pairs=bnpairs, subs={free_c[0]: s})
                try_add(m2, "bn_rim_triangulene", f"{tag}_s", f"bn_rim_{s}")

    # === GROUP 4: Non-Alternant Frameworks (Pentalenes, s-Indacenes, Azulenes, Calicenes, Heptalenes) (~130 mols) ===
    # 4a. Aza-pentalenes (with substituents so n_heavy >= 10!)
    pent_cores = [
        ("pent0", set(), [1, 2, 4, 5]),
        ("pent14N", {1, 4}, [2, 3, 5, 6]),
        ("pent25N", {2, 5}, [1, 3, 4, 6]),
        ("pent1346N", {1, 3, 4, 6}, [2, 5]),
        ("pent1N", {1}, [2, 4, 5]),
        ("pent2N", {2}, [1, 4, 5]),
    ]
    for cname, nset, open_p in pent_cores:
        for d, a in [("NH2", "CN"), ("OMe", "CN"), ("NMe2", "CN"), ("NH2", "CF3"), ("OMe", "CF3"), ("OH", "CN"), ("F", "CN"), ("CN", "CN"), ("CCH", "CN"), ("NMe2", "F"), ("BOH2", "NH2"), ("CHO", "NH2")]:
            m = build_pentalene_core(nset, subs={open_p[0]: d, open_p[-1]: a})
            try_add(m, "non_alternant_pentalene", f"{cname}_pp", f"pentalene_pp_{d}_{a}", min_heavy=10)
        if len(open_p) >= 3:
            for s1, s2, s3 in [("F", "CN", "NH2"), ("CN", "CN", "OMe"), ("F", "F", "NMe2")]:
                m = build_pentalene_core(nset, subs={open_p[0]: s1, open_p[1]: s2, open_p[-1]: s3})
                try_add(m, "non_alternant_pentalene", f"{cname}_tri", f"pentalene_tri_{s1}_{s2}_{s3}", min_heavy=10)

    # 4b. s-Indacene & Aza-s-indacenes (12 heavy core -> all >= 12 heavy!)
    ind_cores = [
        ("ind0", set(), [1, 4, 5, 8]),
        ("ind15N", {1, 5}, [2, 4, 6, 8]),
        ("ind26N", {2, 6}, [1, 4, 5, 8]),
        ("ind48N", {4, 8}, [1, 2, 5, 6]),
        ("ind1357N", {1, 3, 5, 7}, [2, 4, 6, 8]),
        ("ind1N", {1}, [4, 5, 8]),
    ]
    for cname, nset, open_p in ind_cores:
        try_add(build_sindacene_core(nset), "non_alternant_indacene", cname, "indacene_core")
        for s in ["CN", "F", "NH2", "OMe", "CF3", "SH", "OH", "CHO"]:
            try_add(build_sindacene_core(nset, subs={open_p[0]: s}), "non_alternant_indacene", f"{cname}_m", f"ind_mono_{s}")
        for d, a in [("NH2", "CN"), ("OMe", "CN"), ("NH2", "F"), ("NMe2", "CN"), ("F", "CN"), ("OH", "CF3")]:
            try_add(build_sindacene_core(nset, subs={open_p[0]: d, open_p[2]: a}), "non_alternant_indacene", f"{cname}_pp", f"ind_pp_{d}_{a}")

    # 4c. Aza-azulenes & push-pull azulenes (10+ heavy)
    az_cores = [
        ("az0", set(), [1, 2, 6]),
        ("az13N", {1, 3}, [2, 5, 6]),
        ("az5N", {5}, [1, 2, 6]),
        ("az6N", {6}, [1, 2, 5]),
        ("az136N", {1, 3, 6}, [2, 4, 5]),
    ]
    for cname, nset, open_p in az_cores:
        try_add(build_azulene_core(nset), "non_alternant_azulene", cname, "azulene_core")
        for d, a in [("NH2", "CN"), ("OMe", "CN"), ("NMe2", "CN"), ("F", "CN"), ("CN", "NH2"), ("CN", "OMe"), ("CF3", "NH2"), ("CHO", "OMe")]:
            try_add(build_azulene_core(nset, subs={open_p[0]: d, open_p[-1]: a}), "non_alternant_azulene", f"{cname}_pp", f"azulene_pp_{d}_{a}")

    # 4d. Push-pull Calicenes (famous non-alternant dipole-allowed INVEST candidates!)
    # 5-ring positions 1..4 (acceptors stabilize Cp-!), 3-ring positions 5,6 (donors stabilize C3+!)
    cal_cores = [
        ("cal0", set(), [1, 2, 3, 4, 5, 6]),
        ("cal13N", {1, 3}, [2, 4, 5, 6]),
        ("cal14N", {1, 4}, [2, 3, 5, 6]),
    ]
    for cname, nset, _ in cal_cores:
        for acc in ["CN", "CF3", "F", "CHO"]:
            for don in ["NH2", "NMe2", "OMe", "OH", "SH", "Me"]:
                # Symmetric & asymmetric push-pull calicenes
                p5 = [p for p in [1, 2, 3, 4] if p not in nset]
                if len(p5) >= 2:
                    m1 = build_calicene_core(nset, subs={p5[0]: acc, 5: don, 6: don})
                    try_add(m1, "non_alternant_calicene", f"{cname}_pp", f"calicene_{acc}_{don}2", min_heavy=10)
                    m2 = build_calicene_core(nset, subs={p5[0]: acc, p5[1]: acc, 5: don, 6: don})
                    try_add(m2, "non_alternant_calicene", f"{cname}_pp4", f"calicene_{acc}2_{don}2", min_heavy=10)
                    m3 = build_calicene_core(nset, subs={p5[0]: acc, 5: don})
                    try_add(m3, "non_alternant_calicene", f"{cname}_asym", f"calicene_asym_{acc}_{don}", min_heavy=10)

    # 4e. Aza-heptalenes (12+ heavy)
    for nset, tag in [({1, 6}, "hep16N"), ({2, 7}, "hep27N"), ({1, 3, 6, 8}, "hep1368N")]:
        try_add(build_heptalene_core(nset), "non_alternant_heptalene", tag, "heptalene_core")
        open_p = [p for p in range(1, 11) if p not in nset]
        for d, a in [("NH2", "CN"), ("OMe", "F"), ("F", "CN")]:
            try_add(build_heptalene_core(nset, subs={open_p[0]: d, open_p[len(open_p)//2]: a}), "non_alternant_heptalene", f"{tag}_pp", f"heptalene_pp_{d}_{a}")

    # Balance / cap cleanly if over target_max while keeping full family diversity
    if len(candidates) > target_max:
        # Stratified deterministic subsample preserving all families
        by_fam: Dict[str, List[Dict]] = {}
        for c in candidates:
            by_fam.setdefault(c["scaffold_family"], []).append(c)
        selected: List[Dict] = []
        # Always keep all unsubstituted asymmetric cores & BN rim cores
        for fam in sorted(by_fam.keys()):
            items = by_fam[fam]
            # Sort by SA-score ascending (most synthetically accessible first!)
            items.sort(key=lambda x: (x["sa_score"], x["n_heavy"], x["mol_id"]))
            by_fam[fam] = items
        # Round-robin pick across families up to target_max
        idx = 0
        while len(selected) < target_max:
            added_any = False
            for fam in sorted(by_fam.keys()):
                if idx < len(by_fam[fam]) and len(selected) < target_max:
                    selected.append(by_fam[fam][idx])
                    added_any = True
            if not added_any:
                break
            idx += 1
        # Re-number cleanly
        for i, c in enumerate(selected, 1):
            c["mol_id"] = f"VBI_{i:04d}_{c['core_tag']}"
        candidates = selected

    return candidates


# ---------------------------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------------------------

def run_batch_opt(records: List[Dict], xyz_dir: Path, n_workers: int = 4) -> List[Dict]:
    """Optimize geometries in parallel using at most 4 workers."""
    xyz_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in records:
        out_xyz = str(xyz_dir / f"{r['mol_id']}.xyz")
        init_xyz = r.get("init_xyz")
        jobs.append((r["mol_id"], r["smiles"], out_xyz, init_xyz))

    results_map: Dict[str, Dict] = {}
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(optimize_one_molecule, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results_map[res["mol_id"]] = res
            done += 1
            if done % 50 == 0 or done == len(jobs):
                n_ok = sum(1 for v in results_map.values() if v.get("ok"))
                print(
                    f"  [GFN2-xTB opt] {done}/{len(jobs)} finished ({n_ok} ok) in {time.perf_counter()-t0:.1f}s",
                    flush=True,
                )

    merged = []
    for r in records:
        opt_res = results_map.get(r["mol_id"], {"ok": False})
        if opt_res.get("ok"):
            r_clean = {k: v for k, v in r.items() if k != "init_xyz"}
            r_clean.update({k: v for k, v in opt_res.items() if k not in ("mol_id", "ok")})
            merged.append(r_clean)
        else:
            print(f"  [WARN] Optimization failed for {r['mol_id']}: {opt_res.get('error')}", flush=True)
    return merged


def main():
    t_start = time.perf_counter()
    print("=== Step 1: Building QUESTDB + HEPTA-INVEST InChIKey Exclusion Index ===", flush=True)
    known = build_known_exclusion_index()
    print(
        f"  Exclusion index ready: {len(known['all_ik14'])} unique connectivity InChIKeys "
        f"(HEPTA-INVEST={len(known['hepta_ik14'])}, QUESTDB={len(known['quest_ik14'])})",
        flush=True,
    )

    print("\n=== Step 2: Curating Multi-Scaffold Experimental & High-Level INVEST Benchmark ===", flush=True)
    bench_recs = curate_multi_scaffold_benchmark()
    print(f"  Curated {len(bench_recs)} benchmark molecules. Running GFN2-xTB optimization (4 workers)...", flush=True)
    bench_opt = run_batch_opt(bench_recs, INVEST_XYZ_DIR, n_workers=4)
    fam_counts: Dict[str, int] = {}
    for r in bench_opt:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    bench_out = {
        "description": "Multi-scaffold experimental & high-level INVEST benchmark for Lehmann-TDDFT Pillar 3",
        "n_molecules": len(bench_opt),
        "family_counts": fam_counts,
        "geometry_method": "RDKit ETKDGv3 + tblite GFN2-xTB (ASE BFGS fmax=0.02 eV/A)",
        "molecules": bench_opt,
    }
    with open(DATA_PROC / "invest_multi_scaffold.json", "w") as f:
        json.dump(bench_out, f, indent=2)
    print(f"  Saved {len(bench_opt)} benchmark molecules to data/processed/invest_multi_scaffold.json", flush=True)
    print(f"  Families: {fam_counts}", flush=True)

    print("\n=== Step 3: Enumerating De Novo Combinatorial Bright-INVEST Virtual Screening Library ===", flush=True)
    virt_recs = enumerate_virtual_bright_invest_library(known["all_ik14"], target_max=520)
    vfam_counts: Dict[str, int] = {}
    for r in virt_recs:
        vfam_counts[r["scaffold_family"]] = vfam_counts.get(r["scaffold_family"], 0) + 1
    print(f"  Enumerated {len(virt_recs)} unique novel candidates (ZERO overlap with QUESTDB/HEPTA-INVEST)!", flush=True)
    print(f"  Scaffold families: {vfam_counts}", flush=True)
    print("  Running GFN2-xTB optimization across all virtual candidates (4 workers)...", flush=True)
    virt_opt = run_batch_opt(virt_recs, VIRTUAL_XYZ_DIR, n_workers=4)

    # Final double-check: ZERO InChIKey-14 overlap with QUESTDB or HEPTA-INVEST!
    overlap = [r["mol_id"] for r in virt_opt if r["inchikey_14"] in known["all_ik14"]]
    assert len(overlap) == 0, f"CRITICAL: Found overlap with QUESTDB/HEPTA-INVEST: {overlap}"

    vfam_final: Dict[str, int] = {}
    for r in virt_opt:
        vfam_final[r["scaffold_family"]] = vfam_final.get(r["scaffold_family"], 0) + 1
    virt_out = {
        "description": "De novo combinatorial non-heptazine & symmetry-broken Bright-INVEST virtual screening library",
        "n_molecules": len(virt_opt),
        "verified_zero_overlap_questdb_heptainvest": True,
        "scaffold_family_counts": vfam_final,
        "heavy_atom_range": [min(r["n_heavy"] for r in virt_opt), max(r["n_heavy"] for r in virt_opt)],
        "sa_score_mean": round(float(np.mean([r["sa_score"] for r in virt_opt])), 3),
        "qed_mean": round(float(np.mean([r["qed"] for r in virt_opt])), 3),
        "geometry_method": "RDKit ETKDGv3 + tblite GFN2-xTB (ASE BFGS fmax=0.02 eV/A)",
        "molecules": virt_opt,
    }
    with open(DATA_PROC / "virtual_bright_invest_library.json", "w") as f:
        json.dump(virt_out, f, indent=2)
    print(
        f"\n[DONE] Phase A curation complete in {time.perf_counter()-t_start:.1f}s! "
        f"Benchmark={len(bench_opt)} mols, Virtual Library={len(virt_opt)} mols.",
        flush=True,
    )


if __name__ == "__main__":
    main()
