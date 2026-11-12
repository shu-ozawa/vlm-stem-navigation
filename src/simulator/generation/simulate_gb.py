"""HAADF-STEM image of a Sigma7 symmetric tilt grain boundary in monolayer MoS2 (21.8 deg).

1. Build one periodic bicrystal by cut-and-mirror: grain 0 is the MoS2 lattice rotated so that the
   lattice vector (2, 1) points along y, and grain 1 is its mirror image. The cell holds two
   boundaries: "main" at x = Lx / 2 and "wrap" at x = 0. Atoms of the two grains that come too close
   are merged or removed (an unrelaxed model).
2. Displace all atoms at random (frozen phonons) and run the abTEM multislice on the GPU in scan
   windows of ~50 A, each with the atoms within PAD of it (the same displaced atoms for every window).
3. Average the configurations, resample the periodic image to PIXEL_SIZE, blur it by the effective
   source size, and save it to data/grain_boundary/ with meta.json and the atomic model.

The boundaries run along the image rows (vertical) and the image is periodic, so the specimen
(specimens/grain_boundary.py) can repeat it along the boundary and rotate it.

Run (GPU, ~1.3 h on an RTX PRO 5000):  uv run --group sim python -m simulator.generation.simulate_gb
"""

from __future__ import annotations

import json
import math
import time

import abtem
import numpy as np
from ase import Atoms
from scipy.ndimage import gaussian_filter

from simulator.paths import DATA_DIR

OUT = DATA_DIR / "grain_boundary"

# specimen
A_LAT = 3.16            # MoS2 in-plane lattice constant [A]
THICKNESS = 3.17        # S-S vertical distance [A]
BOUNDARY_VECTOR = (2, 1)  # lattice vector along the boundary: Sigma7, 21.8 deg
SHIFT_X_A = 0.5         # shift of grain 0 before cutting (selects the core structure)
GRAIN_PERIODS = 22      # grain periods |u| per grain across the boundary (Lx = 2 * 22 * 14.48 A)
BOUNDARY_PERIODS = 16   # boundary periods |v| along the boundary (Ly = 16 * 8.36 A)
MERGE_A = {"Mo": 1.6, "S": 1.2}                             # same-species atoms closer than this: merged
MIN_DISTANCE_A = {("Mo", "Mo"): 2.6, ("Mo", "S"): 2.1, ("S", "S"): 2.6}  # closer than this: one removed

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
FP_SIGMA_A = {"Mo": 0.06, "S": 0.08}
WINDOW_POINTS = 144     # scan points per window side (~50 A)
PAD_A = 8.0             # atoms kept around each window [A]
SEED = 0

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, math.sqrt(3) / 2])
S_SITE = np.array([1 / 3, 1 / 3])  # S2 column relative to Mo, in fractional coordinates of (a1, a2)


# ---------------------------------------------------------------- atomic model
def lattice_vector(m: int, n: int) -> np.ndarray:
    return A_LAT * (m * A1 + n * A2)


def lattice_in_box(rotation: np.ndarray, shift: np.ndarray, Lx: float, Ly: float) -> tuple[list[str], np.ndarray]:
    """The rotated and shifted MoS2 lattice inside [0, Lx) x [0, Ly): symbols and positions (z: Mo 0, S +-t/2)."""
    basis = rotation @ (A_LAT * np.stack([A1, A2], axis=1))
    corners = np.array([[0, 0], [Lx, 0], [0, Ly], [Lx, Ly]]) - shift
    frac = np.linalg.solve(basis, corners.T).T
    lo, hi = np.floor(frac.min(0)).astype(int) - 2, np.ceil(frac.max(0)).astype(int) + 2
    I, J = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    cells = np.stack([I.ravel(), J.ravel()], axis=1).astype(float)

    symbols, positions = [], []
    for symbol, site, heights in (("Mo", np.zeros(2), (0.0,)), ("S", S_SITE, (THICKNESS / 2, -THICKNESS / 2))):
        xy = (basis @ (cells + site).T).T + shift
        inside = (xy[:, 0] >= -1e-6) & (xy[:, 0] < Lx - 1e-6) & (xy[:, 1] >= -1e-6) & (xy[:, 1] < Ly - 1e-6)
        for z in heights:
            symbols += [symbol] * int(inside.sum())
            positions.append(np.c_[xy[inside], np.full(inside.sum(), z)])
    return symbols, np.concatenate(positions)


def resolve_overlaps(symbols: np.ndarray, positions: np.ndarray, grain: np.ndarray, L: np.ndarray):
    """Merge, then remove, atoms of different grains that come too close (periodic in x and y).
    Pairs are handled from the closest one; only atoms within 4 A of a boundary can take part.

    1. Mo-Mo, then S-S on the same layer, closer than MERGE_A in-plane: one atom at the midpoint (grain 2)
    2. repeat until nothing changes: Mo-Mo, Mo-S, S-S pairs closer than MIN_DISTANCE_A (3D);
       Mo-Mo and S-S remove the grain-1 atom, Mo-S the S
    """
    alive = np.ones(len(symbols), bool)
    log = {"merged_Mo": 0, "merged_S": 0, "removed_Mo": 0, "removed_S": 0}

    # atoms of different grains can only meet near a boundary (x = 0 and x = Lx / 2)
    x = positions[:, 0] % L[0]
    near = np.minimum.reduce([x, L[0] - x, np.abs(x - L[0] / 2)]) < 4.0

    def close_pairs(first: str, second: str, cutoff: float, planar: bool) -> list[tuple[int, int]]:
        """Alive pairs (i of `first`, j of `second`) of different grains closer than cutoff, closest first."""
        pairs = []
        for i in np.flatnonzero((symbols == first) & alive & near):
            for j in np.flatnonzero((symbols == second) & alive & near):
                if grain[i] == grain[j]:
                    continue
                d = positions[j] - positions[i]
                d[:2] -= L * np.round(d[:2] / L)
                if planar and abs(d[2]) > 0.1:
                    continue
                r = np.hypot(d[0], d[1]) if planar else np.linalg.norm(d)
                if r < cutoff:
                    pairs.append((r, i, j))
        return [(i, j) for _, i, j in sorted(pairs)]

    for species in ("Mo", "S"):
        for i, j in close_pairs(species, species, MERGE_A[species], planar=True):
            if alive[i] and alive[j] and grain[i] == 0 and grain[j] == 1:
                d = positions[j, :2] - positions[i, :2]
                positions[i, :2] += (d - L * np.round(d / L)) / 2
                grain[i] = 2
                alive[j] = False
                log[f"merged_{species}"] += 1

    changed = True
    while changed:
        changed = False
        for first, second in (("Mo", "Mo"), ("Mo", "S"), ("S", "S")):
            for i, j in close_pairs(first, second, MIN_DISTANCE_A[(first, second)], planar=False):
                if alive[i] and alive[j]:
                    removed = j if (second == "S" and first == "Mo") or grain[j] == 1 else i
                    alive[removed] = False
                    log[f"removed_{symbols[removed]}"] += 1
                    changed = True

    return symbols[alive], positions[alive], grain[alive], log


def build_bicrystal() -> tuple[Atoms, dict]:
    """The periodic bicrystal (boundaries at x = 0 and x = Lx / 2) and its geometry."""
    m, n = BOUNDARY_VECTOR
    v = lattice_vector(m, n)                          # along the boundary (y)
    p, q = m + 2 * n, -(2 * m + n)                    # perpendicular lattice vector (x)
    g = math.gcd(p, q)
    u = lattice_vector(p // g, q // g)
    phi = math.atan2(v[1], v[0])
    c, s = math.cos(math.pi / 2 - phi), math.sin(math.pi / 2 - phi)
    rotation = np.array([[c, -s], [s, c]])

    Lx = 2 * GRAIN_PERIODS * np.linalg.norm(u)
    Ly = BOUNDARY_PERIODS * np.linalg.norm(v)

    # grain 0 on the left half, grain 1 its mirror image on the right half
    symbols, pos0 = lattice_in_box(rotation, np.array([SHIFT_X_A, 0.0]), Lx / 2, Ly)
    pos1 = pos0.copy()
    pos1[:, 0] = (Lx - pos0[:, 0]) % Lx
    symbols = np.array(symbols + symbols)
    positions = np.concatenate([pos0, pos1])
    grain = np.r_[np.zeros(len(pos0), int), np.ones(len(pos1), int)]

    symbols, positions, grain, log = resolve_overlaps(symbols, positions, grain, np.array([Lx, Ly]))
    positions[:, 2] += THICKNESS / 2 + 2.0
    atoms = Atoms(list(symbols), positions=positions, cell=[Lx, Ly, THICKNESS + 4.0], pbc=True)

    misorientation = math.degrees(2 * phi) % 60
    info = {"boundary_vector_mn": [m, n], "period_along_boundary_A": float(np.linalg.norm(v)),
            "grain_period_A": float(np.linalg.norm(u)), "misorientation_deg": min(misorientation, 60 - misorientation),
            "shift_x_A": SHIFT_X_A, "size_A": [float(Lx), float(Ly)], "boundaries_x_A": {"main": Lx / 2, "wrap": 0.0},
            "overlaps": log, "atoms": len(atoms)}
    return atoms, info


# ---------------------------------------------------------------- HAADF simulation
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

    atoms, model = build_bicrystal()
    Lx, Ly = atoms.cell.lengths()[:2]
    print(f"bicrystal {Lx:.1f} x {Ly:.1f} A, {len(atoms)} atoms, overlaps {model['overlaps']}")

    # scan grid over exactly one period of the specimen, split into windows
    nx, ny = int(np.ceil(Lx / SCAN_STEP_A)), int(np.ceil(Ly / SCAN_STEP_A))
    step = (Lx / nx, Ly / ny)
    cols = np.array_split(np.arange(nx), int(np.ceil(nx / WINDOW_POINTS)))
    rows = np.array_split(np.arange(ny), int(np.ceil(ny / WINDOW_POINTS)))

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
    atoms.write(OUT / "bicrystal.xyz")

    meta = {
        "description": "HAADF-STEM image of a Sigma7 symmetric tilt grain boundary in monolayer MoS2 "
                       "(unrelaxed cut-and-mirror model); periodic, boundaries along the rows",
        "image_file": "haadf.npy",
        "model_file": "bicrystal.xyz",
        "shape_rows_cols": list(image.shape),
        "pixel_size_A": {"x": pixel[1], "y": pixel[0]},
        "boundary_cols": {"main": float(Lx / 2 / pixel[1]), "wrap": 0.0},
        "model": model,
        "microscope": {"energy_keV": ENERGY_EV / 1e3, "convergence_semiangle_mrad": SEMIANGLE_MRAD,
                       "haadf_mrad": list(HAADF_MRAD), "aberrations": "none", "source_size_sigma_A": SOURCE_SIZE_A},
        "simulation": {"method": "abTEM multislice (GPU) of the whole periodic bicrystal, in scan windows",
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
