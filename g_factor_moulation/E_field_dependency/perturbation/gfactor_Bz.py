import time
from multiprocessing import Pool, cpu_count

import kwant
import numpy as np
from pymablock import block_diagonalize
from scipy.linalg import eigh, qr
from scipy.sparse import bmat, block_diag, csr_matrix, identity
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm

from Hamiltonian_mathematica_v2 import psi_new_basis, gs_shifted_bz


# ---------------------------------
# Parameters
# ---------------------------------
N = 50
Nband = 10
L = 300
B = 0.01
gspin = 2
muB = 5.788e-5
flag = [1, 0]
nef = 20
# emax_V_um = 0.06
# emax_V_angstrom = emax_V_um * 1e-4
# efield_list = np.linspace(0, emax_V_angstrom, nef)

# Set this to an integer (for example, 30) to select the number of workers.
# None uses all available CPUs, capped by the number of electric-field points.
N_PROCESSES = None

Ny = Nz = N
lat = kwant.lattice.square(norbs=Nband)
lat_E = kwant.lattice.square(norbs=Nband)


# ==================================================
# General Kwant system builder
# ==================================================
def make_system_bz(kx, Bz):
    """Build the 2H-Ge nanowire Hamiltonian for B = (0, 0, Bz)."""
    syst = kwant.Builder()
    a = L / (Ny + 1)
    y_center = (Ny - 1) / 2.0

    def onsite(site):
        y, _ = site.tag
        y_mid_centered = y - y_center
        return gs_shifted_bz(
            a=a,
            kx_base=kx,
            ry=0,
            rz=0,
            Bz=Bz,
            y_mid_centered=y_mid_centered,
        )

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        ry = int(y1 - y2)
        rz = int(z1 - z2)
        y_mid_centered = 0.5 * (y1 + y2) - y_center
        return gs_shifted_bz(
            a=a,
            kx_base=kx,
            ry=ry,
            rz=rz,
            Bz=Bz,
            y_mid_centered=y_mid_centered,
        )

    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite

    # Use only one representative from each Hermitian hopping pair.
    for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(direction, lat, lat)] = hopping

    return syst.finalized()


def make_system_E(flagy):
    """Build the position operator for Ey (flagy=1) or Ez (flagy=0)."""
    syst = kwant.Builder()
    a = L / (Ny + 1)

    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                potential = -((y + 1) * a - L / 2)
            else:
                potential = -((z + 1) * a - L / 2)
            syst[lat_E(y, z)] = potential * np.eye(Nband)

    return syst.finalized()


# Worker-local constants are assigned once per process by the Pool initializer,
# instead of being serialized again for every electric-field point.
_worker_H00 = None
_worker_H10 = None
_worker_H12 = None
_worker_H14 = None
_worker_H16 = None
_worker_B = None
_worker_muB = None


def init_gfactor_worker(H00, H10, H12, H14, H16, magnetic_field, bohr_magneton):
    """Initialize constant effective-Hamiltonian data in each worker."""
    global _worker_H00, _worker_H10, _worker_H12, _worker_H14,_worker_H16, _worker_B, _worker_muB
    _worker_H00 = H00
    _worker_H10 = H10
    _worker_H12 = H12
    _worker_H14 = H14
    _worker_H16 = H16
    _worker_B = magnetic_field
    _worker_muB = bohr_magneton


def calculate_gfactor(Ef):
    """Calculate the Bz splitting and g-factor at one electric field."""
    H_eff = (
        _worker_H00
        + _worker_H10 * _worker_B
        + _worker_H12 * _worker_B * Ef**2
        + _worker_H14 * _worker_B * Ef**4
        + _worker_H16 * _worker_B * Ef**6
    )
    eigenvalues = eigh(H_eff, eigvals_only=True)
    gap = np.abs(eigenvalues[1] - eigenvalues[0])
    gfactor = gap / (_worker_muB * _worker_B)
    return Ef, eigenvalues, gap, gfactor


def build_spin_matrix():
    """Construct the full spin-z matrix in the real-space basis."""
    zero = csr_matrix((5, 5))
    sz_up = identity(5, format="csr")
    sz_down = -identity(5, format="csr")
    smatrix_sparse = 0.5 * bmat(
        [[sz_up, zero], [zero, sz_down]],
        format="csr",
    )
    return block_diag([smatrix_sparse] * N**2, format="csr")


def run_direction(flagy, H0_sparse, V_Bz, psi_new,efield_list):
    """Calculate and save the Bz g-factor modulation for Ey or Ez."""
    direction = "Ey" if flagy == 1 else "Ez"
    print(f"\nCalculating Bz g-factor modulation for {direction}...")

    V_E = make_system_E(flagy).hamiltonian_submatrix(sparse=True)
    H_list_sparse = [H0_sparse, V_Bz, V_E]

    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse,
        subspace_eigenvectors=[psi_new],
    )

    # Materialize the 2x2 matrices before sending them to worker processes.
    H00 = np.asarray(Heff_coeffs[(0, 0, 0, 0)])
    H10 = np.asarray(Heff_coeffs[(0, 0, 1, 0)])
    H12 = np.asarray(Heff_coeffs[(0, 0, 1, 2)])
    H14 = np.asarray(Heff_coeffs[(0, 0, 1, 4)])
    H16 = np.asarray(Heff_coeffs[(0, 0, 1, 6)])

    # available_workers = cpu_count() if N_PROCESSES is None else N_PROCESSES
    n_workers = 8 #max(1, min(int(available_workers), len(efield_list)))
    print(
        f"Calculating {len(efield_list)} electric-field points "
        f"with {n_workers} workers..."
    )

    with Pool(
        processes=n_workers,
        initializer=init_gfactor_worker,
        initargs=(H00, H10, H12, H14, H16, B, muB),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(calculate_gfactor, efield_list, chunksize=1),
                total=len(efield_list),
                desc=f"{direction} sweep",
                unit="field",
            )
        )

    # Pool.imap preserves the order of efield_list.
    for Ef, eigenvalues, gap, gfactor in results:
        print(
            f"L = {L}, E = {Ef * 1e4:.3f}, "
            # f"ev1 = {eigenvalues[0]:.6f}, ev2 = {eigenvalues[1]:.6f}, "
            f"gap = {gap * 1e6:.4f}, g = {gfactor:.3f}"
        )

    # Columns: L [Angstrom], gap [micro-eV], E [V/micrometre], g.
    output_filename = f"gz_moulation_{direction}.dat"
    output_data = np.asarray(
        [
            [L, gap * 1e6, Ef * 1e4, gfactor]
            for Ef, _, gap, gfactor in results
        ]
    )
    np.savetxt(
        output_filename,
        output_data,
        fmt=["%.8f", "%.10e", "%.10e", "%.10e"],
        header="L_Angstrom gap_microeV E_V_per_um g",
    )
    print(f"Saved data to {output_filename}")


def main():
    t0 = time.time()
    print(f"\nBuilding system for N = {N}")
    print("\nBuilding H(+Bz) and H(-Bz)...")

    H_Bp = make_system_bz(0.0, B).hamiltonian_submatrix(sparse=True)
    H_Bm = make_system_bz(0.0, -B).hamiltonian_submatrix(sparse=True)

    # Linear magnetic perturbation, in eV/T, including orbital and spin terms.
    H_Bz_linear = (H_Bp - H_Bm) / (2.0 * B)
    V_Bz = H_Bz_linear + gspin * muB * build_spin_matrix()

    H0_sparse = make_system_bz(0.0, 0.0).hamiltonian_submatrix(sparse=True)
    _, evecs = eigsh(H0_sparse, k=2, sigma=0.00)
    evecs_ortho, _ = qr(evecs, mode="economic")
    psi_new = np.asarray(
        psi_new_basis(evecs_ortho[:, 0], evecs_ortho[:, 1], N)
    ).T

    for flagy in flag:
        if flagy == 1:
            emax_V_um = 1.0
        if flagy == 0:
            emax_V_um = 0.075
        emax_V_angstrom = emax_V_um * 1e-4
        efield_list = np.linspace(0, emax_V_angstrom, nef) 
        run_direction(flagy, H0_sparse, V_Bz, psi_new, efield_list)

    print(f"\nTotal runtime: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
