"""Build stishovite (rutile-type SiO2, high-pressure polymorph, 6-coordinate
Si) from known crystallographic parameters (a=b=4.1772 A, c=2.6651 A,
P4_2/mnm, O at Wyckoff 4f x=0.3053), tile to ~N=64+ Si atoms, and run a
real Vashishta NVT at 300 K -- a quick "casual" (per the user's own
phrasing) extra crystal polymorph so test74's conditioning isn't trained
on a single crystal structure only.
"""
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.lammpsdata import write_lammps_data
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

WORK_DIR = Path(__file__).resolve().parent / "stishovite_work"
POTENTIALS_DIR = Path(
    "/Users/harutokono/ScoreMD/toy-model/SiO2-CG/diffusion_for_multi_scale_molecular_dynamics"
    "/.venv/lib/python3.10/site-packages/lammps/share/lammps/potentials"
)
OUT_ROOT = Path(__file__).resolve().parent
TILE = (4, 4, 6)  # 2 Si/cell * 4*4*6 = 192 Si (576 atoms total)
EQ_STEPS = 3000
PROD_STEPS = 10000
DUMP_EVERY = 50


def main():
    a, c = 4.1772, 2.6651
    x_O = 0.3053
    lattice = Lattice.from_parameters(a, a, c, 90, 90, 90)
    species = ["Si", "Si", "O", "O", "O", "O"]
    frac = [
        [0, 0, 0], [0.5, 0.5, 0.5],
        [x_O, x_O, 0], [-x_O, -x_O, 0],
        [0.5 + x_O, 0.5 - x_O, 0.5], [0.5 - x_O, 0.5 + x_O, 0.5],
    ]
    structure = Structure(lattice, species, frac, coords_are_cartesian=False)
    structure.make_supercell(TILE)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    n_si = sum(1 for s in atoms.get_chemical_symbols() if s == "Si")
    print(f"Tiled stishovite {TILE}: {len(atoms)} atoms ({n_si} Si), cell {atoms.cell.lengths()}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    data_path = WORK_DIR / "stishovite_init.data"
    with open(data_path, "w") as f:
        write_lammps_data(f, atoms, specorder=["Si", "O"], units="metal", atom_style="atomic")

    dump_path = WORK_DIR / "stishovite_300K.lammpstrj"
    in_path = WORK_DIR / "run.in"
    in_path.write_text(f"""
units metal
boundary p p p
atom_style atomic
read_data {data_path.name}
mass 1 28.0855
mass 2 15.999
pair_style vashishta
pair_coeff * * {POTENTIALS_DIR}/SiO.1990.vashishta Si O
velocity all create 300.0 42 mom yes rot yes dist gaussian
fix eqnvt all nvt temp 300.0 300.0 0.1
run {EQ_STEPS}
unfix eqnvt
fix prodnvt all nvt temp 300.0 300.0 0.1
dump traj all custom {DUMP_EVERY} {dump_path.name} id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
""")
    result = subprocess.run(["lmp_serial", "-in", str(in_path)], cwd=str(WORK_DIR),
                             capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print("LAMMPS FAILED:", result.stderr[-1500:])
        return
    print(result.stdout.splitlines()[-1])

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
        frames.append(pos[types == 1])

    si_frames = np.stack(frames, axis=0)
    out_dir = OUT_ROOT / "data_stishovite_300K"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "positions.npy", (si_frames / 10.0).astype(np.float32))
    np.savez(out_dir / "manifest.npz", cell_nm=(cell / 10.0).astype(np.float32), source="stishovite_300K")
    print(f"Wrote {out_dir}: {si_frames.shape[0]} frames, {si_frames.shape[1]} Si atoms")


if __name__ == "__main__":
    main()
