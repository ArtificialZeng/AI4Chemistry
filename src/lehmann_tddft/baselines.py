"""Baselines & Ablation Models for Lehmann-TDDFT Benchmarking.

Includes:
  1. `SchNet3DBaseline`: Modern 3D continuous-filter radial-basis geometric MPNN
     (SchNet-style, pure PyTorch — zero libomp clash on macOS) trained on exact
     same molecule-level splits to predict (E_S1, E_T1, delta_EST = E_S1 - E_T1).
  2. `audit_gnn_size_intensivity`: Quantifies how global-pooled 3D GNNs fail
     size-intensivity on separated dimers A + B at R = 100 Angstrom.
  3. `evaluate_published_qm_baselines`: Computes strictly matched metrics for
     published adiabatic TD-DFT (B3LYP, PBE0, CAM-B3LYP, M06-2X), our DF-PBE0
     RI-TDA, and correlated wavefunction methods (CIS(D), ADC(2), CC2, CCSD)
     on any evaluation molecule split.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Geometric MPNN Baseline (SchNet-style continuous-filter radial GNN)
# ---------------------------------------------------------------------------

class GaussianRBF(nn.Module):
    def __init__(self, r_min: float = 0.0, r_max: float = 6.0, n_rbf: int = 32):
        super().__init__()
        centers = torch.linspace(r_min, r_max, n_rbf)
        width = (r_max - r_min) / max(n_rbf - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (width * width)
        self.r_max = r_max

    def forward(self, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # dist: (E,)
        rbf = torch.exp(-self.gamma * (dist[:, None] - self.centers[None, :]) ** 2)
        cutoff = 0.5 * (torch.cos(math.pi * (dist / self.r_max).clamp(0.0, 1.0)) + 1.0)
        cutoff = cutoff * (dist < self.r_max).to(dist.dtype)
        return rbf, cutoff


class CFConvInteraction(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int, dropout: float = 0.05):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lin_in = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rbf: torch.Tensor,
        cutoff: torch.Tensor,
    ) -> torch.Tensor:
        # h: (N, D)
        W = self.filter_net(rbf) * cutoff[:, None]  # (E, D)
        x = self.lin_in(h)
        if src.numel() > 0:
            msg = x[src] * W  # (E, D)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
        else:
            agg = torch.zeros_like(h)
        return self.norm(h + self.lin_out(agg))


class SchNet3DBaseline(nn.Module):
    """Strong 3D geometric GNN predicting (E_S1, E_T1, delta_EST = E_S1 - E_T1)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_rbf: int = 32,
        r_cutoff: float = 6.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.r_cutoff = r_cutoff
        self.embed = nn.Embedding(100, hidden_dim)
        self.rbf = GaussianRBF(0.0, r_cutoff, n_rbf)
        self.layers = nn.ModuleList(
            [CFConvInteraction(hidden_dim, n_rbf, dropout=dropout) for _ in range(n_layers)]
        )
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [E_S1_eV, E_T1_eV]
        )
        self.register_buffer("target_mean", torch.tensor([4.5, 3.5], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor([2.0, 2.0], dtype=torch.float32))

    @torch.no_grad()
    def fit_target_stats(self, targets_st: np.ndarray) -> None:
        if len(targets_st) == 0:
            return
        t = torch.as_tensor(targets_st, dtype=torch.float32)
        self.target_mean.copy_(t.mean(dim=0))
        self.target_std.copy_(t.std(dim=0, unbiased=False).clamp_min(0.25))

    def forward_single(self, atomic_numbers: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = atomic_numbers.long().clamp(0, 99)
        pos = coords.float()
        n = z.shape[0]
        h = self.embed(z)

        # Pairwise distances within r_cutoff
        diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        dist_mat = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        eye_mask = ~torch.eye(n, dtype=torch.bool, device=pos.device)
        mask = eye_mask & (dist_mat < self.r_cutoff)
        dst, src = torch.where(mask)
        dist = dist_mat[dst, src]
        rbf, cutoff = self.rbf(dist)

        for layer in self.layers:
            h = layer(h, src, dst, rbf, cutoff)

        # Standard global mean + sum/sqrt(N) + attention readout
        h_mean = h.mean(dim=0)
        h_sum = h.sum(dim=0) / math.sqrt(max(n, 1))
        attn_w = torch.softmax(self.attn_gate(h).squeeze(-1), dim=0)
        h_attn = (attn_w[:, None] * h).sum(dim=0)
        g = torch.cat([h_mean, h_sum, h_attn], dim=-1)
        raw = self.head(g)
        pred = raw * self.target_std + self.target_mean
        E_S1 = pred[0]
        E_T1 = pred[1]
        return {
            "E_S1_eV": E_S1,
            "E_T1_eV": E_T1,
            "delta_EST_eV": E_S1 - E_T1,
        }


def train_schnet_baseline(
    train_items: Sequence[Dict[str, Any]],
    val_items: Sequence[Dict[str, Any]],
    epochs: int = 220,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[SchNet3DBaseline, Dict[str, float]]:
    """Train `SchNet3DBaseline` on molecules that have ground-truth (E_S1, E_T1)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Filter to molecules having at least S1 or T1 ground truth
    def _valid(items):
        out = []
        for it in items:
            if it.get("ref_S1_eV") is not None or it.get("ref_T1_eV") is not None:
                out.append(it)
        return out

    tr = _valid(train_items)
    va = _valid(val_items)

    model = SchNet3DBaseline(hidden_dim=64, n_layers=4, n_rbf=32, r_cutoff=6.0, dropout=0.05).to(device)
    s1_vals = [it["ref_S1_eV"] for it in tr if it.get("ref_S1_eV") is not None]
    t1_vals = [it["ref_T1_eV"] for it in tr if it.get("ref_T1_eV") is not None]
    if s1_vals and t1_vals:
        model.target_mean.copy_(torch.tensor([np.mean(s1_vals), np.mean(t1_vals)], dtype=torch.float32, device=device))
        model.target_std.copy_(
            torch.tensor([max(np.std(s1_vals), 0.25), max(np.std(t1_vals), 0.25)], dtype=torch.float32, device=device)
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-5)

    best_state = None
    best_val_loss = float("inf")
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        # Mini-batch gradient accumulation (batch size 16)
        opt.zero_grad()
        bs = 16
        for step_idx, idx in enumerate(order):
            it = tr[idx]
            z = torch.as_tensor(it["atomic_numbers"], dtype=torch.long, device=device)
            pos = torch.as_tensor(it["coords_angstrom"], dtype=torch.float32, device=device)
            pred = model.forward_single(z, pos)
            loss = torch.tensor(0.0, device=device)
            has_s1 = it.get("ref_S1_eV") is not None
            has_t1 = it.get("ref_T1_eV") is not None
            if has_s1:
                loss = loss + F.smooth_l1_loss(pred["E_S1_eV"], torch.tensor(it["ref_S1_eV"], dtype=torch.float32, device=device), beta=0.2)
            if has_t1:
                loss = loss + F.smooth_l1_loss(pred["E_T1_eV"], torch.tensor(it["ref_T1_eV"], dtype=torch.float32, device=device), beta=0.2)
            if has_s1 and has_t1:
                ref_d = it["ref_S1_eV"] - it["ref_T1_eV"]
                loss = loss + 2.0 * F.smooth_l1_loss(pred["delta_EST_eV"], torch.tensor(ref_d, dtype=torch.float32, device=device), beta=0.1)
            (loss / bs).backward()
            if (step_idx + 1) % bs == 0 or (step_idx + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                opt.zero_grad()
        sched.step()

        # Validation check every 5 epochs
        if va and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            model.eval()
            v_errs = []
            with torch.no_grad():
                for it in va:
                    z = torch.as_tensor(it["atomic_numbers"], dtype=torch.long, device=device)
                    pos = torch.as_tensor(it["coords_angstrom"], dtype=torch.float32, device=device)
                    pred = model.forward_single(z, pos)
                    e = 0.0
                    c = 0
                    if it.get("ref_S1_eV") is not None:
                        e += abs(float(pred["E_S1_eV"].item()) - it["ref_S1_eV"])
                        c += 1
                    if it.get("ref_T1_eV") is not None:
                        e += abs(float(pred["E_T1_eV"].item()) - it["ref_T1_eV"])
                        c += 1
                    if it.get("ref_S1_eV") is not None and it.get("ref_T1_eV") is not None:
                        e += 2.0 * abs(float(pred["delta_EST_eV"].item()) - (it["ref_S1_eV"] - it["ref_T1_eV"]))
                        c += 2
                    if c > 0:
                        v_errs.append(e / c)
            v_score = float(np.mean(v_errs)) if v_errs else float("inf")
            if v_score < best_val_loss:
                best_val_loss = v_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_score": best_val_loss}


@torch.no_grad()
def predict_schnet_baseline(
    model: SchNet3DBaseline,
    items: Sequence[Dict[str, Any]],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    model.eval()
    preds = []
    for it in items:
        z = torch.as_tensor(it["atomic_numbers"], dtype=torch.long, device=device)
        pos = torch.as_tensor(it["coords_angstrom"], dtype=torch.float32, device=device)
        out = model.forward_single(z, pos)
        preds.append(
            {
                "mol_id": it["mol_id"],
                "E_S1_eV": float(out["E_S1_eV"].item()),
                "E_T1_eV": float(out["E_T1_eV"].item()),
                "delta_EST_eV": float(out["delta_EST_eV"].item()),
            }
        )
    return preds


@torch.no_grad()
def audit_gnn_size_intensivity(
    model: SchNet3DBaseline,
    mol_A: Dict[str, Any],
    mol_B: Dict[str, Any],
    separation_angstrom: float = 100.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """Demonstrate that global-pooled 3D GNNs fail exact size-intensivity on A + B."""
    model.eval()
    zA = torch.as_tensor(mol_A["atomic_numbers"], dtype=torch.long, device=device)
    pA = torch.as_tensor(mol_A["coords_angstrom"], dtype=torch.float32, device=device)
    zB = torch.as_tensor(mol_B["atomic_numbers"], dtype=torch.long, device=device)
    pB = torch.as_tensor(mol_B["coords_angstrom"], dtype=torch.float32, device=device) + torch.tensor(
        [0.0, 0.0, separation_angstrom], dtype=torch.float32, device=device
    )

    outA = model.forward_single(zA, pA)
    outAB = model.forward_single(torch.cat([zA, zB]), torch.cat([pA, pB], dim=0))
    return {
        "E_S1_A_eV": float(outA["E_S1_eV"].item()),
        "E_S1_AB_eV": float(outAB["E_S1_eV"].item()),
        "abs_err_S1_eV": abs(float(outAB["E_S1_eV"].item()) - float(outA["E_S1_eV"].item())),
        "E_T1_A_eV": float(outA["E_T1_eV"].item()),
        "E_T1_AB_eV": float(outAB["E_T1_eV"].item()),
        "abs_err_T1_eV": abs(float(outAB["E_T1_eV"].item()) - float(outA["E_T1_eV"].item())),
        "dEST_A_eV": float(outA["delta_EST_eV"].item()),
        "dEST_AB_eV": float(outAB["delta_EST_eV"].item()),
        "abs_err_dEST_eV": abs(float(outAB["delta_EST_eV"].item()) - float(outA["delta_EST_eV"].item())),
        "separation_angstrom": separation_angstrom,
    }


# ---------------------------------------------------------------------------
# Published QM & Adiabatic TD-DFT Baseline Evaluator
# ---------------------------------------------------------------------------

def evaluate_published_qm_baselines(
    eval_items: Sequence[Dict[str, Any]],
    methods: Sequence[str] = (
        "B3LYP", "PBE0", "CAM-B3LYP", "M06-2X", "wB97X-D",
        "BSE/evGW@PBE0", "CIS(D)", "ADC(2)", "ADC(3)", "CC2", "CCSD", "CASPT2", "SC-NEVPT2",
    ),
) -> Dict[str, Dict[str, Any]]:
    """Aggregate strict state-level and lowest-state (S1, T1, dEST, GD/PD) metrics
    for published QM methods stored inside the cleaned dataset records, PLUS our
    raw adiabatic DF-PBE0/def2-SVP RI-TDA baseline.
    """
    summary: Dict[str, Dict[str, Any]] = {}

    # 1. Our computed adiabatic DF-PBE0/def2-SVP RI-TDA (S1, T1, dEST)
    s1_errs, t1_errs, dest_errs = [], [], []
    invest_tp, invest_fp, invest_fn, invest_tn = 0, 0, 0, 0
    for it in eval_items:
        wS = it.get("omega_S_eV")
        wT = it.get("omega_T_eV")
        if wS is None or len(wS) == 0 or wT is None or len(wT) == 0:
            continue
        p_s1 = float(wS[0])
        p_t1 = float(wT[0])
        p_dest = p_s1 - p_t1
        r_s1 = it.get("ref_S1_eV")
        r_t1 = it.get("ref_T1_eV")
        if r_s1 is not None:
            s1_errs.append(abs(p_s1 - r_s1))
        if r_t1 is not None:
            t1_errs.append(abs(p_t1 - r_t1))
        if r_s1 is not None and r_t1 is not None:
            r_dest = r_s1 - r_t1
            dest_errs.append(abs(p_dest - r_dest))
            ref_inv = r_dest < 0.0
            pred_inv = p_dest < 0.0
            if ref_inv and pred_inv:
                invest_tp += 1
            elif (not ref_inv) and pred_inv:
                invest_fp += 1
            elif ref_inv and (not pred_inv):
                invest_fn += 1
            else:
                invest_tn += 1

    summary["DF-PBE0_RI-TDA (Adiabatic Base)"] = {
        "n_S1": len(s1_errs),
        "S1_MAE_eV": float(np.mean(s1_errs)) if s1_errs else None,
        "n_T1": len(t1_errs),
        "T1_MAE_eV": float(np.mean(t1_errs)) if t1_errs else None,
        "n_dEST": len(dest_errs),
        "dEST_MAE_eV": float(np.mean(dest_errs)) if dest_errs else None,
        "INVEST_TP": invest_tp,
        "INVEST_FP": invest_fp,
        "INVEST_FN": invest_fn,
        "INVEST_TN": invest_tn,
        "INVEST_recall": float(invest_tp / max(invest_tp + invest_fn, 1)) if (invest_tp + invest_fn) > 0 else None,
        "GD_detectable": False,
        "note": "Adiabatic TD-DFT cannot emit genuine double excitations or Hund's-rule inversion (recall=0%).",
    }

    # 2. Published QUESTDB / HEPTA-INVEST methods
    for m_name in methods:
        all_errs, s_errs, t_errs, gd_errs, pd_errs = [], [], [], [], []
        s1_e, t1_e, dest_e = [], [], []
        tp = fp = fn = tn = 0

        for it in eval_items:
            states = it.get("states", [])
            # Collect per-state method predictions
            m_sing = []
            m_trip = []
            for st in states:
                ref_e = st.get("tbe_eV")
                if ref_e is None:
                    continue
                bdict = st.get("baselines", {})
                val = bdict.get(m_name)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                err = abs(float(val) - float(ref_e))
                all_errs.append(err)
                spin = int(st.get("spin", 1))
                spec = st.get("special")
                if spin == 1:
                    s_errs.append(err)
                    m_sing.append((float(ref_e), float(val)))
                elif spin == 3:
                    t_errs.append(err)
                    m_trip.append((float(ref_e), float(val)))
                if spec == "GD":
                    gd_errs.append(err)
                elif spec == "PD":
                    pd_errs.append(err)

            # Lowest S1 / T1 / dEST comparison if both exist for this method
            if m_sing:
                m_sing.sort(key=lambda x: x[0])
                s1_e.append(abs(m_sing[0][1] - m_sing[0][0]))
            if m_trip:
                m_trip.sort(key=lambda x: x[0])
                t1_e.append(abs(m_trip[0][1] - m_trip[0][0]))
            if m_sing and m_trip:
                ref_d = m_sing[0][0] - m_trip[0][0]
                pred_d = m_sing[0][1] - m_trip[0][1]
                dest_e.append(abs(pred_d - ref_d))
                if ref_d < 0 and pred_d < 0:
                    tp += 1
                elif ref_d >= 0 and pred_d < 0:
                    fp += 1
                elif ref_d < 0 and pred_d >= 0:
                    fn += 1
                else:
                    tn += 1

        if all_errs or dest_e:
            summary[m_name] = {
                "n_states": len(all_errs),
                "all_states_MAE_eV": float(np.mean(all_errs)) if all_errs else None,
                "singlet_MAE_eV": float(np.mean(s_errs)) if s_errs else None,
                "triplet_MAE_eV": float(np.mean(t_errs)) if t_errs else None,
                "n_GD_reported": len(gd_errs),
                "GD_MAE_eV": float(np.mean(gd_errs)) if gd_errs else None,
                "n_PD_reported": len(pd_errs),
                "PD_MAE_eV": float(np.mean(pd_errs)) if pd_errs else None,
                "n_S1": len(s1_e),
                "S1_MAE_eV": float(np.mean(s1_e)) if s1_e else None,
                "n_T1": len(t1_e),
                "T1_MAE_eV": float(np.mean(t1_e)) if t1_e else None,
                "n_dEST": len(dest_e),
                "dEST_MAE_eV": float(np.mean(dest_e)) if dest_e else None,
                "INVEST_TP": tp,
                "INVEST_FP": fp,
                "INVEST_FN": fn,
                "INVEST_TN": tn,
                "INVEST_recall": float(tp / max(tp + fn, 1)) if (tp + fn) > 0 else None,
            }

    return summary
