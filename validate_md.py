"""Quick sanity check for test74's T-conditioning: run CGMD on the SAME
starting structure (the 300K crystal) twice, once with the model told
T_condition=300K and once with T_condition=3000K (as if it were the
melt), using the SAME physical thermostat temperature (300K) for the
actual Langevin dynamics both times. If conditioning is doing something
real (not being ignored), the two runs should behave differently -- e.g.
the T=3000K-conditioned force should be "softer"/less crystal-like, since
that's what it saw during training (melt data), even though the ACTUAL
simulated temperature is held at 300K in both cases.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_and_export import DenoiserMPNN  # noqa: E402

CKPT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"
KB_EV_PER_K = 8.617333e-5
TEMP_K_ACTUAL = 300.0  # the real thermostat temperature, held fixed in both runs
MASS_AMU = 60.0843
DT_PS = 0.001
GAMMA_INV_PS = 1.0
METAL_UNITS_CONVERSION = 9648.533
N_STEPS = 1500
STRIDE = 50


def nn_dist_stats(pos, cell):
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= np.round(diff / cell) * cell
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return nn.mean(), nn.min(), nn.std()


def run(model, sigma, cell_np, x0_np, t_norm_scale, condition_t_k, label):
    cell = torch.tensor(cell_np, dtype=torch.float32)
    n_atoms = x0_np.shape[0]
    x = torch.tensor(x0_np, dtype=torch.float32)
    v = torch.zeros_like(x)
    kbT_ev = KB_EV_PER_K * TEMP_K_ACTUAL
    gamma_per_ps = 1.0 / GAMMA_INV_PS
    alpha = np.exp(-gamma_per_ps * DT_PS)
    f_scale = (1 - alpha) / gamma_per_ps
    condition = torch.tensor([[condition_t_k / t_norm_scale]], dtype=torch.float32)
    torch.manual_seed(0)
    print(f"\n=== {label} (condition T={condition_t_k} K, actual thermostat T={TEMP_K_ACTUAL} K) ===")
    nn0 = nn_dist_stats(x0_np, cell_np)
    print(f"  initial  nn_mean={nn0[0]:.4f} A  nn_min={nn0[1]:.4f} A  nn_std={nn0[2]:.4f} A")
    for step in range(1, N_STEPS + 1):
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
        if step % STRIDE == 0 and (step % 300 == 0 or step == N_STEPS):
            nn_mean, nn_min, nn_std = nn_dist_stats(x.numpy(), cell_np)
            print(f"  step {step:5d}  nn_mean={nn_mean:.4f} A  nn_min={nn_min:.4f} A  nn_std={nn_std:.4f} A")


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = DenoiserMPNN(ckpt["hidden_dim"], ckpt["n_layers"], ckpt["cutoff_angstrom"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sigma = ckpt["sigma_angstrom"]
    t_norm_scale = ckpt["t_norm_scale"]
    cell_np = ckpt["cell_by_name"]["crystal_300K"]

    x0 = np.load(Path(__file__).resolve().parent / "data_crystal_300K" / "positions.npy").astype(np.float32)[0] * 10.0

    run(model, sigma, cell_np, x0, t_norm_scale, condition_t_k=300.0, label="Told T=300K (matches crystal training)")
    run(model, sigma, cell_np, x0, t_norm_scale, condition_t_k=3000.0, label="Told T=3000K (matches melt training) -- same start structure")


if __name__ == "__main__":
    main()
