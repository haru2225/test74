# test74: (P, T)-conditioned SiO2 CG force field (crystal polymorphs + melt + glass, N=1000)

Extends test73 (crystal + glass, both at N=1000, unconditioned) with explicit
**pressure (P) and temperature (T) conditioning**, so a single model can in
principle represent the whole phase landscape -- crystalline polymorphs,
the melt, and glass -- as different points along a continuous (P, T) axis,
rather than a handful of disconnected named regimes.

(Config env vars are still named `TEST73_*` for now -- same meaning,
renaming was skipped to save time; this is purely cosmetic.)

## What changed from test73 (implemented)

`DenoiserMPNN` now takes a third input, `condition: (B, 2)` = normalized
`(P/10 GPa, T/1000 K)` per sample, and uses `h0 + condition_mlp(condition)`
as every atom's initial embedding instead of a single fixed learned vector.
This is a real architectural change (verified with a smoke test), not just
a TODO -- see `CRYSTAL_CONDITION` / `GLASS_CONDITION` near the top of
`train_and_export.py` for where a dataset's (P, T) label is declared.

## What is NOT done yet: the data

Both datasets currently plugged in (`data_crystal_1000/`, `data_glass_1000/`,
copy from test73) are at the SAME condition (300 K, ~0 GPa), so the
conditioning mechanism is wired up but has nothing to condition ON yet --
training it now would just learn a constant offset. To make this
meaningful, more real MD trajectories are needed, spanning:

- **Multiple crystalline polymorphs** at 1000 Si atoms each (quartz,
  cristobalite, coesite, stishovite, tridymite, moganite, keatite, ...).
  `toy-model/SiO2-CG/diffcsp-sio2/data/sio2_mp20.pkl` already has 38 real
  SiO2 polymorph structures (from MP-20) that could be tiled up to N~1000
  and run through Vashishta NVT, the same way
  `test73/make_crystal_1000_data.py` did for the cristobalite primitive
  cell -- this is the most direct way to get polymorph diversity without
  needing to search for new structures.
- **Melt**: Vashishta NPT/NVT runs above SiO2's melting point (~1986 K;
  the real melt is likely best sampled around 2500-4000 K to get a
  genuinely liquid, not just "hot solid", state) at a few pressures.
- **Glass**: already have one topology/condition (300 K, ambient P) from
  test70-73; more quench rates/pressures would help (DM2's
  `sio2_3000_glass_*_sample*.dat` files include several quench rates not
  yet used -- see `toy-model/DM2/demo/demo_training/simu_data/`).
- **Pressure variation**: needs NPT (not just NVT) MD, i.e. a barostat, to
  actually sample different densities/pressures -- none of the MD scripts
  in this project (test69-73) have used NPT so far; this is new
  LAMMPS-script work, not just a parameter change.

Each additional (polymorph or phase, P, T) combination is one more
Vashishta MD run (~15-20 min each based on test73's timing at N=3000
atoms) plus one line added to the condition table in
`train_and_export.py`. Given the combinatorics (dozens of polymorphs x
several P x several T), this data-generation phase is substantial and is
the natural next thing to run on the supercomputer, likely in parallel
across many independent LAMMPS jobs before the diffusion-model training
step itself.

## Running

Same as test73 (see its README) -- smoke test locally:

```bash
export TEST73_HIDDEN_DIM=32 TEST73_N_LAYERS=2 TEST73_BATCH_SIZE=4 TEST73_N_STEPS=10
python train_and_export.py   # needs data_crystal_1000/ and data_glass_1000/
                              # copied in from test73 first
```

Supercomputer: `singularity build ... Singularity.test74.def` then
`qsub run_test74.pbs`, same pattern as test73.
