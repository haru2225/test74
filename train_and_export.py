"""test74: a single fixed-sigma denoising force field (same architecture and
method as test70/test71) trained on BOTH the 8-atom SiO2 crystal AND the
1000-atom SiO2 glass, by alternating which dataset each training step draws
from. The network (DenoiserMPNN) has no atom-count-dependent parameters --
it processes an arbitrary N via a per-step periodic radius graph -- so the
SAME weights can in principle represent both regimes, if trained on both.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import os

# test74 (base): BOTH datasets are now N=1000 (CG beads), matched in size --
# crystal is a real Vashishta NVT 300K trajectory of a 5x5x5 tiling of the
# 8-atom primitive cell (see make_crystal_1000_data.py), replacing test72's
# N=8 crystal data. Paths are relative to this file so the repo can be
# cloned and run as-is on a supercomputer.
REPO_DIR = Path(__file__).resolve().parent
CRYSTAL_POSITIONS_PATH = REPO_DIR / "data_crystal_1000" / "positions.npy"
CRYSTAL_MANIFEST_PATH = CRYSTAL_POSITIONS_PATH.parent / "manifest.npz"
GLASS_POSITIONS_PATH = REPO_DIR / "data_glass_1000" / "positions.npy"
GLASS_MANIFEST_PATH = GLASS_POSITIONS_PATH.parent / "manifest.npz"
DEVICE = torch.device(os.environ.get("TEST73_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
OUT_DIR = Path(__file__).resolve().parent / "output"

# (P, T) conditions for each currently-available dataset, in physical units
# (T in K, P in GPa) -- both existing datasets were generated at ambient
# pressure, 300 K, so the conditioning mechanism is architecturally wired
# up but not yet exercised across a real (P, T) range. Adding more
# conditions (other polymorphs, melt, other pressures) only requires: (1)
# generating that trajectory (see README.md's data plan), (2) adding its
# (P, T) here, (3) adding it to the per-step dataset-choice logic below.
T_NORM_SCALE = 1000.0  # K
P_NORM_SCALE = 10.0  # GPa
CRYSTAL_CONDITION = (0.0 / P_NORM_SCALE, 300.0 / T_NORM_SCALE)
GLASS_CONDITION = (0.0 / P_NORM_SCALE, 300.0 / T_NORM_SCALE)

SIGMA_ANGSTROM = 0.15
CUTOFF_ANGSTROM = 6.0  # both datasets are now N=1000 (cell ~34-36 A), so a
# wider physical cutoff (1st+2nd Si-Si shells) fits safely under half the
# smaller cell length
HIDDEN_DIM = int(os.environ.get("TEST73_HIDDEN_DIM", 128))
N_LAYERS = int(os.environ.get("TEST73_N_LAYERS", 4))
CRYSTAL_BATCH_SIZE = int(os.environ.get("TEST73_BATCH_SIZE", 32))
GLASS_BATCH_SIZE = int(os.environ.get("TEST73_BATCH_SIZE", 32))
N_STEPS = int(os.environ.get("TEST73_N_STEPS", 50000))  # GPU on a supercomputer
# affords both a bigger network and far more steps than the CPU runs in
# test70-72; more steps than a single-dataset run: the network must
# now fit two regimes with one set of weights
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 1.0
SPIKE_ROLLBACK_FACTOR = 3.0
GLASS_STEP_PROBABILITY = 0.5  # fraction of steps drawn from the glass dataset


def load_positions_angstrom(path, manifest_path):
    positions_nm = np.load(path)
    manifest = np.load(manifest_path)
    cell_angstrom = manifest["cell_nm"].astype(np.float32) * 10.0
    return positions_nm.astype(np.float32) * 10.0, cell_angstrom


def periodic_radius_graph(pos: torch.Tensor, cell: torch.Tensor, cutoff: float):
    diff = pos[None, :, :] - pos[:, None, :]
    diff = diff - cell * torch.round(diff / cell)
    dist = diff.norm(dim=-1)
    mask = (dist < cutoff) & (dist > 1e-6)
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_vec = diff[src, dst]
    return torch.stack([src, dst], dim=0), edge_vec


class DenoiserMPNN(nn.Module):
    """test74: extends test70-73's DenoiserMPNN with (P, T) conditioning.
    Every atom's initial embedding is now h0 + condition_mlp([P_norm,
    T_norm]) instead of a single fixed learned vector -- this is what lets
    ONE model represent crystal polymorphs, melt, and glass as different
    points along a continuous (P, T) axis, rather than a handful of
    disconnected named regimes. P and T should be normalized (e.g. T/1000K,
    P/10GPa) before being passed in, so the conditioning MLP's input scale
    is well-behaved regardless of the raw units used when the training
    data was generated.
    """

    def __init__(self, hidden_dim: int, n_layers: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.h0 = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.condition_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            ) for _ in range(n_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])
        self.force_head_scalar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, pos: torch.Tensor, cell: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """condition: (B, 2) tensor of (P_norm, T_norm), one (P, T) pair
        per sample in the batch (shared across all atoms within that
        sample, since P and T are macroscopic/global conditions)."""
        batch_size, n_atoms, _ = pos.shape
        cond_embed = self.condition_mlp(condition)  # (B, hidden_dim)
        outputs = []
        for b in range(batch_size):
            edge_index, edge_vec = periodic_radius_graph(pos[b], cell, self.cutoff)
            src, dst = edge_index
            edge_len = edge_vec.norm(dim=-1, keepdim=True)
            edge_dir = edge_vec / edge_len.clamp_min(1e-6)

            h = (self.h0 + cond_embed[b]).unsqueeze(0).expand(n_atoms, -1).contiguous()
            for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
                m_input = torch.cat([h[src], h[dst], edge_len], dim=-1)
                m = message_mlp(m_input)
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, m)
                h = h + update_mlp(torch.cat([h, agg], dim=-1))

            m_input = torch.cat([h[src], h[dst], edge_len], dim=-1)
            edge_scalar = self.message_mlps[-1](m_input)
            edge_force_scalar = self.force_head_scalar(edge_scalar)
            per_atom_force = torch.zeros(n_atoms, 3, device=pos.device, dtype=pos.dtype)
            per_atom_force.index_add_(0, dst, edge_force_scalar * edge_dir)
            outputs.append(per_atom_force)
        return torch.stack(outputs, dim=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crystal_positions, crystal_cell = load_positions_angstrom(CRYSTAL_POSITIONS_PATH, CRYSTAL_MANIFEST_PATH)
    glass_positions, glass_cell = load_positions_angstrom(GLASS_POSITIONS_PATH, GLASS_MANIFEST_PATH)
    print(f"Crystal: {crystal_positions.shape[0]} frames, {crystal_positions.shape[1]} atoms, cell {crystal_cell}")
    print(f"Glass:   {glass_positions.shape[0]} frames, {glass_positions.shape[1]} atoms, cell {glass_cell}")

    print(f"Using device: {DEVICE}")
    crystal_t = torch.tensor(crystal_positions, dtype=torch.float32, device=DEVICE)
    glass_t = torch.tensor(glass_positions, dtype=torch.float32, device=DEVICE)
    crystal_cell_t = torch.tensor(crystal_cell, dtype=torch.float32, device=DEVICE)
    glass_cell_t = torch.tensor(glass_cell, dtype=torch.float32, device=DEVICE)

    torch.manual_seed(0)
    model = DenoiserMPNN(HIDDEN_DIM, N_LAYERS, CUTOFF_ANGSTROM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    losses_by_kind = {"crystal": [], "glass": []}
    n_rollbacks = 0
    last_good_state = None
    t_start = time.time()
    step = 1
    while step <= N_STEPS:
        is_glass = np.random.rand() < GLASS_STEP_PROBABILITY
        if is_glass:
            frame_idx = torch.randint(0, glass_t.shape[0], (GLASS_BATCH_SIZE,))
            r = glass_t[frame_idx]
            cell_t = glass_cell_t
            condition = torch.tensor([GLASS_CONDITION] * GLASS_BATCH_SIZE, dtype=torch.float32, device=DEVICE)
        else:
            frame_idx = torch.randint(0, crystal_t.shape[0], (CRYSTAL_BATCH_SIZE,))
            r = crystal_t[frame_idx]
            cell_t = crystal_cell_t
            condition = torch.tensor([CRYSTAL_CONDITION] * CRYSTAL_BATCH_SIZE, dtype=torch.float32, device=DEVICE)

        noise = torch.randn_like(r) * SIGMA_ANGSTROM
        r_noisy = r + noise
        target = -noise

        prediction = model(r_noisy, cell_t, condition)
        loss = ((prediction - target) ** 2).mean()
        loss_value = float(loss.detach())

        if len(losses) >= 20:
            baseline = np.mean(losses[-20:])
            if loss_value > SPIKE_ROLLBACK_FACTOR * baseline and last_good_state is not None:
                model.load_state_dict(last_good_state[0])
                optimizer.load_state_dict(last_good_state[1])
                n_rollbacks += 1
                print(f"step {step:5d}: loss {loss_value:.6f} > {SPIKE_ROLLBACK_FACTOR}x "
                      f"baseline {baseline:.6f} -- rolling back (rollback #{n_rollbacks})", flush=True)
                continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        losses.append(loss_value)
        losses_by_kind["glass" if is_glass else "crystal"].append(loss_value)
        if step % 20 == 0:
            last_good_state = (
                {k: v.clone() for k, v in model.state_dict().items()},
                optimizer.state_dict(), step,
            )
        if step % 200 == 0 or step == 1:
            recent_c = np.mean(losses_by_kind["crystal"][-50:]) if losses_by_kind["crystal"] else float("nan")
            recent_g = np.mean(losses_by_kind["glass"][-50:]) if losses_by_kind["glass"] else float("nan")
            elapsed = time.time() - t_start
            print(f"step {step:6d}/{N_STEPS}  loss_crystal={recent_c:.6f}  loss_glass={recent_g:.6f}  "
                  f"elapsed={elapsed:.1f}s  rollbacks={n_rollbacks}", flush=True)
            torch.save(
                {"model_state": model.state_dict(), "sigma_angstrom": SIGMA_ANGSTROM,
                 "cutoff_angstrom": CUTOFF_ANGSTROM, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
                 "crystal_cell_angstrom": crystal_cell, "glass_cell_angstrom": glass_cell,
                 "step": step},
                OUT_DIR / "checkpoint.pt",
            )
        step += 1

    with open(OUT_DIR / "loss_history.json", "w") as f:
        json.dump(losses_by_kind, f)
    print(f"Done. Final checkpoint: {OUT_DIR / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
