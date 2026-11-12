"""HAADF-STEM image of monolayer 2H-MoS2 with one W atom substituting Mo.

1. Build one periodic monolayer of 80 x 48 rectangular MoS2 cells (~253 x 263 A). The scan is split
   into windows of ~50 A; the Mo atom nearest the middle of the central window is replaced by W.
2. Displace all atoms at random (frozen phonons) and run the abTEM multislice on the GPU, window by
   window: each window is simulated with the atoms within PAD of it only. Every window uses the same
   displaced atoms, so the windows join without seams.
3. Average the frozen-phonon configurations, resample the periodic image to PIXEL_SIZE, blur it by the
   effective source size, and save it to data/dopant/ with meta.json.

The image is periodic, so the specimen (specimens/dopant.py) can roll it to put the W anywhere.

Run (GPU, ~1 h on an RTX PRO 5000):  uv run --group sim python -m simulator.generation.simulate_dopant
"""

from __future__ import annotations

import json
import time

import abtem
import numpy as np
from ase import Atoms
from ase.build import mx2
from scipy.ndimage import gaussian_filter

from simulator.paths import DATA_DIR

OUT = DATA_DIR / "dopant"

# specimen
A_LAT = 3.16            # MoS2 in-plane lattice constant [A]
THICKNESS = 3.17        # S-S vertical distance [A]
CELLS = (80, 48)        # rectangular cells (a x sqrt(3) a) along x, y

# microscope
ENERGY_EV = 80e3
SEMIANGLE_MRAD = 25.0
HAADF_MRAD = (70.0, 200.0)
SOURCE_SIZE_A = 0.3     # Gaussian sigma of the effective source

# numerics
POTENTIAL_SAMPLING_A = 0.06
SLICE_THICKNESS_A = 1.0
SCAN_STEP_A = 0.35      # probe step before resampling (Nyquist for 25 mrad at 80 kV)
PIXEL_SIZE_A = 0.125    # output pixel size
FROZEN_PHONONS = 4
FP_SIGMA_A = {"Mo": 0.06, "W": 0.06, "S": 0.08}
WINDOW_POINTS = 144     # scan points per window side (~50 A)
PAD_A = 8.0             # atoms kept around each window [A]
SEED = 0


def build_monolayer() -> Atoms:
    """The periodic MoS2 monolayer."""
    unit = mx2("MoS2", "2H", a=A_LAT, thickness=THICKNESS, vacuum=2.0)
    return abtem.orthogonalize_cell(unit) * (CELLS[0], CELLS[1], 1)


def substitute_w(atoms: Atoms, near: tuple[float, float]) -> int:
    """Replace the Mo atom nearest `near` (x, y) by W; returns its index."""
    is_mo = np.array(atoms.get_chemical_symbols()) == "Mo"
    distance = np.linalg.norm(atoms.positions[:, :2] - near, axis=1)
    w_index = int(np.flatnonzero(is_mo)[np.argmin(distance[is_mo])])
    atoms.numbers[w_index] = 74
    return w_index


def displace(atoms: Atoms, rng: np.random.Generator) -> Atoms:
    """One frozen-phonon configuration: every atom moved by a Gaussian of its element's sigma."""
    sigma = np.array([FP_SIGMA_A[s] for s in atoms.get_chemical_symbols()])
    moved = atoms.copy()
    moved.positions += rng.normal(size=(len(atoms), 3)) * sigma[:, None]
    return moved


def scan_window(atoms: Atoms, x0: float, y0: float, nx: int, ny: int, step: tuple[float, float],
                probe, detector) -> np.ndarray:
    """HAADF signal of nx x ny scan points starting at (x0, y0), from the atoms within PAD_A of them.
    Returns [row = y, col = x]."""
    L = atoms.cell.lengths()
    width = np.array([nx * step[0], ny * step[1]]) + 2 * PAD_A

    # positions relative to the window corner, over the periodic images of the specimen that it can reach
    rel = (atoms.positions[:, :2] - [x0 - PAD_A, y0 - PAD_A]) % L[:2]
    numbers, positions = [], []
    for i in range(int(np.ceil(width[0] / L[0]))):
        for j in range(int(np.ceil(width[1] / L[1]))):
            shifted = rel + [i * L[0], j * L[1]]
            inside = (shifted[:, 0] < width[0]) & (shifted[:, 1] < width[1])
            numbers.append(atoms.numbers[inside])
            positions.append(np.c_[shifted[inside], atoms.positions[inside, 2]])
    window = Atoms(numbers=np.concatenate(numbers), positions=np.concatenate(positions),
                   cell=[width[0], width[1], L[2]], pbc=True)

    potential = abtem.Potential(window, sampling=POTENTIAL_SAMPLING_A, slice_thickness=SLICE_THICKNESS_A,
                                projection="infinite", parametrization="lobato")
    scan = abtem.GridScan(start=(PAD_A, PAD_A), end=(PAD_A + nx * step[0], PAD_A + ny * step[1]),
                          gpts=(nx, ny), endpoint=False)
    signal = probe.scan(potential, scan=scan, detectors=detector).compute(progress_bar=False)
    return np.asarray(signal.array).T


def fourier_resample(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Band-limited resampling of a periodic image to `shape` (zero padding in Fourier space)."""
    ny_in, nx_in = img.shape
    ny, nx = shape
    hy, hx = min(ny_in, ny) // 2, min(nx_in, nx) // 2

    F = np.fft.fftshift(np.fft.fft2(img))
    out = np.zeros(shape, dtype=complex)
    out[ny // 2 - hy:ny // 2 + hy, nx // 2 - hx:nx // 2 + hx] = \
        F[ny_in // 2 - hy:ny_in // 2 + hy, nx_in // 2 - hx:nx_in // 2 + hx]
    return np.fft.ifft2(np.fft.ifftshift(out)).real * (ny * nx) / (ny_in * nx_in)


def main() -> None:
    t0 = time.time()
    abtem.config.set({"device": "gpu"})
    rng = np.random.default_rng(SEED)

    atoms = build_monolayer()
    Lx, Ly = atoms.cell.lengths()[:2]

    # scan grid over exactly one period of the specimen, split into windows
    nx, ny = int(np.ceil(Lx / SCAN_STEP_A)), int(np.ceil(Ly / SCAN_STEP_A))
    step = (Lx / nx, Ly / ny)
    cols = np.array_split(np.arange(nx), int(np.ceil(nx / WINDOW_POINTS)))
    rows = np.array_split(np.arange(ny), int(np.ceil(ny / WINDOW_POINTS)))

    # the W sits in the middle of the window that holds the specimen centre, away from window edges
    # (near an edge a heavy column comes out a few % too dark); the specimen rolls it anywhere later
    c = next(c for c in cols if nx // 2 in c)
    r = next(r for r in rows if ny // 2 in r)
    w_index = substitute_w(atoms, ((c[0] + len(c) / 2) * step[0], (r[0] + len(r) / 2) * step[1]))
    print(f"specimen {Lx:.1f} x {Ly:.1f} A, {len(atoms)} atoms, W at {atoms.positions[w_index, :2]}")

    probe = abtem.Probe(energy=ENERGY_EV, semiangle_cutoff=SEMIANGLE_MRAD)
    detector = abtem.AnnularDetector(inner=HAADF_MRAD[0], outer=HAADF_MRAD[1])

    scan_image = np.zeros((ny, nx))
    for k in range(FROZEN_PHONONS):
        moved = displace(atoms, rng)
        for r in rows:
            for c in cols:
                scan_image[r[0]:r[-1] + 1, c[0]:c[-1] + 1] += scan_window(
                    moved, c[0] * step[0], r[0] * step[1], len(c), len(r), step, probe, detector)
        print(f"frozen phonon {k + 1}/{FROZEN_PHONONS} done, {time.time() - t0:.0f} s")
    scan_image /= FROZEN_PHONONS

    # output pixels, then the effective source size
    shape = (int(round(Ly / PIXEL_SIZE_A)), int(round(Lx / PIXEL_SIZE_A)))
    pixel = (Ly / shape[0], Lx / shape[1])
    image = fourier_resample(scan_image, shape)
    image = gaussian_filter(image, sigma=(SOURCE_SIZE_A / pixel[0], SOURCE_SIZE_A / pixel[1]), mode="wrap")
    image = np.clip(image, 0, None).astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "haadf.npy", image)

    w_x, w_y = atoms.positions[w_index, :2]
    meta = {
        "description": "HAADF-STEM image of monolayer 2H-MoS2 with one W substituting Mo; periodic",
        "image_file": "haadf.npy",
        "shape_rows_cols": list(image.shape),
        "pixel_size_A": {"x": pixel[1], "y": pixel[0]},
        "dopant": {"element": "W", "replaces": "Mo", "x_A": float(w_x), "y_A": float(w_y),
                   "pixel_row": float(w_y / pixel[0]), "pixel_col": float(w_x / pixel[1])},
        "specimen": {"material": "2H-MoS2 monolayer", "a_A": A_LAT, "S_S_thickness_A": THICKNESS,
                     "rectangular_cells": list(CELLS), "size_A": [float(Lx), float(Ly)], "atoms": len(atoms)},
        "microscope": {"energy_keV": ENERGY_EV / 1e3, "convergence_semiangle_mrad": SEMIANGLE_MRAD,
                       "haadf_mrad": list(HAADF_MRAD), "aberrations": "none", "source_size_sigma_A": SOURCE_SIZE_A},
        "simulation": {"method": "abTEM multislice (GPU) of the whole periodic specimen, in scan windows",
                       "potential_sampling_A": POTENTIAL_SAMPLING_A, "slice_thickness_A": SLICE_THICKNESS_A,
                       "projection": "infinite", "parametrization": "lobato",
                       "scan_step_A": {"x": step[0], "y": step[1]}, "window_points": WINDOW_POINTS, "pad_A": PAD_A,
                       "frozen_phonons": FROZEN_PHONONS, "fp_sigma_A": FP_SIGMA_A, "seed": SEED,
                       "intensity_unit": "fraction of the incident probe on the HAADF detector"},
        "versions": {"abtem": abtem.__version__, "numpy": np.__version__},
        "wall_time_s": round(time.time() - t0, 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {OUT / 'haadf.npy'} {image.shape}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
