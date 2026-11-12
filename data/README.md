# Simulated HAADF-STEM data

Both specimens were simulated with abTEM 1.0.10 on the GPU (cupy). The microscope settings were:
80 kV, 25 mrad probe convergence, no aberrations, HAADF detector 70–200 mrad, and a Gaussian
source size of σ = 0.3 Å. The numerics were a Lobato potential at 0.06 Å sampling, 1 Å slices,
a 0.35 Å scan step, and 4 frozen-phonon configurations (σ 0.06 Å for Mo/W, 0.08 Å for S).
Intensities are fractions of the incident probe on the detector.

Each script simulates one whole periodic specimen:
1. Build the atomic model.
2. Displace all atoms at random (one frozen-phonon configuration).
3. Run the multislice in scan windows of ~50 Å, each with the atoms within 8 Å of it. Every window
   uses the same displaced atoms, so the windows join without seams.
4. Average the configurations, Fourier-resample the periodic image to ~0.125 Å/px, and blur it by the
   source size.

The images are periodic, so the specimens (`src/simulator/generation/`) can roll them. `meta.json` records
the geometry, the parameters and the run time.

Regenerate (GPU; times on one RTX PRO 5000 Blackwell):
```bash
uv run --group sim python -m simulator.generation.simulate_dopant   # ~1 h
uv run --group sim python -m simulator.generation.simulate_gb       # ~1.3 h
```

## `dopant/` — single W dopant in monolayer MoS₂ (task `dopant_zoom`)

- **Specimen:** 80 × 48 rectangular cells of 2H-MoS₂ (a = 3.16 Å), 252.8 × 262.7 Å, 23,040 atoms.
  One Mo is replaced by W.
- **W position:** the W sits in the middle of a scan window. Near a window edge a heavy column comes
  out a few % too dark.
- **Use in the environment:** each episode rolls the image so that the W lands at a random position
  away from the edges. The specimen is finite (stage limits).
- **Files:**
  - `haadf.npy`: the image, 2102 × 2022 px (rows = y).
  - `meta.json`: includes the W position (`dopant.pixel_row`, `dopant.pixel_col`).

## `grain_boundary/` — Σ7 symmetric tilt grain boundary in monolayer MoS₂ (task `grain_boundary`)

- **Atomic model:** two grains misoriented by 21.8°, with the lattice vector (2, 1) along the boundary.
  - It is an unrelaxed cut-and-mirror model. Overlapping atoms are merged or removed, which leaves
    open ~8-membered rings along the boundary.
  - The periodic cell is 637.2 × 133.8 Å and holds 29,472 atoms.
  - It contains two boundaries of different core structure: "main" in the middle and "wrap" at the edge.
- **Use in the environment:** each episode
  1. picks one of the two boundaries;
  2. repeats the image along the boundary and offsets it at random;
  3. rotates it by a uniform random angle (cubic-spline interpolation) and mirrors it at random;
  4. crops a 2140 × 2560 px specimen (~27 × 32 nm) in which the line runs in any direction.

  The specimen is finite (stage limits).
- **Files:**
  - `haadf.npy`: the image, 1070 × 5097 px. The boundaries run along the rows.
  - `bicrystal.xyz`: the atomic model.
  - `meta.json`: includes the boundary columns (`boundary_cols`).

The model is not relaxed, so its dislocation core is not the relaxed 5|7 core seen in experiments.
