"""(P, T)-conditioned CGMD for test74. Unlike test70-73's validate_md.py,
the force evaluation now also depends on a `condition` tensor (P_norm,
T_norm), which can be held FIXED (equilibrium check at one condition) or
RAMPED over the course of the simulation (e.g. slowly cooling from a
melt-like condition toward a crystal-like one), to look for phase
transitions driven purely by changing the conditioning input -- not by
switching between differently-trained models.

IMPORTANT CAVEAT: this mechanism only works once the model has actually
been trained on multiple distinct (P, T) conditions (see README.md's data
plan -- not done yet). Right now both of test74's training datasets share
the SAME (P, T) label (300 K, ~0 GPa), so the conditioning input currently
has nothing informative to condition on; ramping it with the CURRENT
checkpoint will not show a real phase transition, only whatever
(likely near-constant) response the model happens to have learned for an
input range it never saw vary during training. Re-run this once test74 is
retrained on a genuinely (P, T)-diverse dataset.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_and_export import DenoiserMPNN, P_NORM_SCALE, T_NORM_SCALE  # noqa: E402

CKPT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"
KB_EV_PER_K = 8.617333e-5
MASS_AMU = 60.0843
DT_PS = 0.001
GAMMA_INV_PS = 1.0
METAL_UNITS_CONVERSION = 9648.533
N_STEPS = 2000
STRIDE = 50


def nn_dist_stats(pos, cell):
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= np.round(diff / cell) * cell
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return nn.mean(), nn.min(), nn.std()


def run_conditioned_md(model, sigma, cell_np, x0_np, p_gpa_schedule, t_k_schedule, n_steps=N_STEPS, stride=STRIDE):
    """p_gpa_schedule, t_k_schedule: either a constant (float) or a 1D
    array of length n_steps (a ramp), in physical units (GPa, K)."""
    cell = torch.tensor(cell_np, dtype=torch.float32)
    n_atoms = x0_np.shape[0]
    x = torch.tensor(x0_np, dtype=torch.float32)
    v = torch.zeros_like(x)

    def p_at(step):
        return p_gpa_schedule if np.isscalar(p_gpa_schedule) else p_gpa_schedule[step - 1]

    def t_at(step):
        return t_k_schedule if np.isscalar(t_k_schedule) else t_k_schedule[step - 1]

    torch.manual_seed(0)
    for step in range(1, n_steps + 1):
        t_now_k = t_at(step)
        p_now_gpa = p_at(step)
        kbT_ev = KB_EV_PER_K * t_now_k
        gamma_per_ps = 1.0 / GAMMA_INV_PS
        alpha = np.exp(-gamma_per_ps * DT_PS)
        f_scale = (1 - alpha) / gamma_per_ps
        condition = torch.tensor([[p_now_gpa / P_NORM_SCALE, t_now_k / T_NORM_SCALE]], dtype=torch.float32)

        with torch.no_grad():
            raw = model(x.unsqueeze(0), cell, condition)[0]
        f = kbT_ev * raw / (sigma ** 2)
        noise = torch.randn_like(x)
        accel = f / MASS_AMU * METAL_UNITS_CONVERSION
        thermal_var = kbT_ev * (1 - alpha ** 2) / MASS_AMU * METAL_UNITS_CONVERSION
        v_new = alpha * v + f_scale * accel + np.sqrt(thermal_var) * noise
        x = x + DT_PS * v_new
        x = x - cell * torch.round(x / cell)
        v = v_new
        if step % stride == 0:
            nn_mean, nn_min, nn_std = nn_dist_stats(x.numpy(), cell_np)
            ke_ev = 0.5 * MASS_AMU * (v ** 2).sum() / METAL_UNITS_CONVERSION
            temp_kinetic = float(2.0 * ke_ev / (3 * n_atoms * KB_EV_PER_K))
            if step % 200 == 0 or step == n_steps:
                print(f"  step {step:5d}  P_set={p_now_gpa:.2f} GPa  T_set={t_now_k:.1f} K  "
                      f"T_kinetic={temp_kinetic:.1f} K  nn_mean={nn_mean:.4f} A  "
                      f"nn_min={nn_min:.4f} A  nn_std={nn_std:.4f} A")


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = DenoiserMPNN(ckpt["hidden_dim"], ckpt["n_layers"], ckpt["cutoff_angstrom"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sigma = ckpt["sigma_angstrom"]

    glass_positions = np.load(Path(__file__).resolve().parent / "data_glass_1000" / "positions.npy").astype(np.float32) * 10.0
    cell_np = ckpt["glass_cell_angstrom"]

    print("=== Fixed-condition CGMD (300 K, 0 GPa) ===")
    run_conditioned_md(model, sigma, cell_np, glass_positions[0], p_gpa_schedule=0.0, t_k_schedule=300.0)

    print("\n=== Ramped-condition CGMD: 2500K -> 300K over 2000 steps (melt->glass/crystal cooling) ===")
    t_ramp = np.linspace(2500.0, 300.0, N_STEPS)
    run_conditioned_md(model, sigma, cell_np, glass_positions[0], p_gpa_schedule=0.0, t_k_schedule=t_ramp)


if __name__ == "__main__":
    main()
