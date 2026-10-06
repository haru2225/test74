"""Generate 3 real Vashishta MD datasets at N~64 Si (192 atoms total,
md/silica_beta_cristobalite_init.data, no tiling needed) covering distinct
(P, T) conditions, so test74's (P,T) conditioning can actually be trained
on something non-trivial, locally, without needing the supercomputer:

1. CRYSTAL: NVT at 300 K, starting from the real crystal structure.
2. MELT: NVT at 3000 K (well above SiO2's ~1986 K melting point), starting
   from the same crystal structure -- gives a genuinely liquid state.
3. GLASS: a quench from 3000 K down to 300 K (continues from the melt),
   giving a disordered-but-frozen state at N=64, distinct from test70-73's
   N=1000 glass (useful for checking the conditioning mechanism without
   needing matched atom counts across conditions -- test74's DenoiserMPNN
   has no atom-count-dependent parameters, same as test70-73).

All three share type1=Si (28.0855), type2=O (15.999), per
md/silica_beta_cristobalite_init.data's own convention.
"""
import subprocess
from pathlib import Path

import numpy as np

SOURCE_DATA = Path("/Users/harutokono/ScoreMD/md/silica_beta_cristobalite_init.data")
WORK_DIR = Path(__file__).resolve().parent / "small_multi_work"
POTENTIALS_DIR = Path(
    "/Users/harutokono/ScoreMD/toy-model/SiO2-CG/diffusion_for_multi_scale_molecular_dynamics"
    "/.venv/lib/python3.10/site-packages/lammps/share/lammps/potentials"
)
OUT_ROOT = Path(__file__).resolve().parent

EQ_STEPS = 3000
PROD_STEPS = 10000
DUMP_EVERY = 50  # -> 200 frames


def run_lammps(in_text: str, work_subdir: Path):
    work_subdir.mkdir(parents=True, exist_ok=True)
    in_path = work_subdir / "run.in"
    in_path.write_text(in_text)
    result = subprocess.run(["lmp_serial", "-in", str(in_path)], cwd=str(work_subdir),
                             capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        print(f"LAMMPS FAILED in {work_subdir}:", result.stderr[-1500:])
        return False
    print(f"{work_subdir.name}: {result.stdout.splitlines()[-1] if result.stdout else '(no output)'}")
    return True


def parse_and_extract_si(dump_path: Path):
    text = dump_path.read_text()
    blocks = text.split("ITEM: TIMESTEP")[1:]
    frames, cell = [], None
    for block in blocks:
        lines = block.strip().splitlines()
        n_atoms = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
        bounds_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: BOX BOUNDS"))
        bounds = [lines[bounds_idx + 1 + i].split() for i in range(3)]
        lengths = np.array([float(hi) - float(lo) for lo, hi in bounds])
        if cell is None:
            cell = lengths
        atoms_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
        header = lines[atoms_idx].split()[2:]
        id_col, type_col = header.index("id"), header.index("type")
        x_col, y_col, z_col = header.index("xu"), header.index("yu"), header.index("zu")
        rows = [lines[atoms_idx + 1 + i].split() for i in range(n_atoms)]
        rows.sort(key=lambda r: int(r[id_col]))
        types = np.array([int(r[type_col]) for r in rows])
        pos = np.array([[float(r[x_col]), float(r[y_col]), float(r[z_col])] for r in rows])
        frames.append(pos[types == 1])  # Si (type 1)
    return np.stack(frames, axis=0), cell


def save_dataset(name, si_frames, cell_angstrom):
    out_dir = OUT_ROOT / f"data_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "positions.npy", (si_frames / 10.0).astype(np.float32))
    np.savez(out_dir / "manifest.npz", cell_nm=(cell_angstrom / 10.0).astype(np.float32), source=name)
    print(f"Wrote {out_dir}: {si_frames.shape[0]} frames, {si_frames.shape[1]} Si atoms, "
          f"cell {cell_angstrom / 10.0} nm")


def main():
    pair_block = f"""
pair_style vashishta
pair_coeff * * {POTENTIALS_DIR}/SiO.1990.vashishta Si O
"""

    # 1) CRYSTAL at 300 K
    crystal_in = f"""
units metal
boundary p p p
atom_style atomic
read_data {SOURCE_DATA}
{pair_block}
velocity all create 300.0 42 mom yes rot yes dist gaussian
fix eqnvt all nvt temp 300.0 300.0 0.1
run {EQ_STEPS}
unfix eqnvt
fix prodnvt all nvt temp 300.0 300.0 0.1
dump traj all custom {DUMP_EVERY} crystal_300K.lammpstrj id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
write_data crystal_final.data
"""
    crystal_dir = WORK_DIR / "crystal_300K"
    if run_lammps(crystal_in, crystal_dir):
        frames, cell = parse_and_extract_si(crystal_dir / "crystal_300K.lammpstrj")
        save_dataset("crystal_300K", frames, cell)

    # 2) MELT at 3000 K, continuing from the crystal's final (equilibrated) structure
    melt_in = f"""
units metal
boundary p p p
atom_style atomic
read_data {crystal_dir / 'crystal_final.data'}
{pair_block}
velocity all create 3000.0 43 mom yes rot yes dist gaussian
fix eqnvt all nvt temp 3000.0 3000.0 0.1
run {EQ_STEPS}
unfix eqnvt
fix prodnvt all nvt temp 3000.0 3000.0 0.1
dump traj all custom {DUMP_EVERY} melt_3000K.lammpstrj id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
write_data melt_final.data
"""
    melt_dir = WORK_DIR / "melt_3000K"
    if run_lammps(melt_in, melt_dir):
        frames, cell = parse_and_extract_si(melt_dir / "melt_3000K.lammpstrj")
        save_dataset("melt_3000K", frames, cell)

    # 3) GLASS: quench the melt's final structure from 3000K -> 300K, then hold
    glass_in = f"""
units metal
boundary p p p
atom_style atomic
read_data {melt_dir / 'melt_final.data'}
{pair_block}
velocity all create 3000.0 44 mom yes rot yes dist gaussian
fix quench all nvt temp 3000.0 300.0 0.1
run {EQ_STEPS}
unfix quench
fix prodnvt all nvt temp 300.0 300.0 0.1
dump traj all custom {DUMP_EVERY} glass_300K.lammpstrj id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
"""
    glass_dir = WORK_DIR / "glass_300K"
    if run_lammps(glass_in, glass_dir):
        frames, cell = parse_and_extract_si(glass_dir / "glass_300K.lammpstrj")
        save_dataset("glass_300K", frames, cell)


if __name__ == "__main__":
    main()
