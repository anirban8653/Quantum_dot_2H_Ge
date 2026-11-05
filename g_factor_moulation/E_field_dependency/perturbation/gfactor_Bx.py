import os
# import time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import kwant
import numpy as np
from scipy.sparse import csr_matrix, bmat, block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
from pymablock import block_diagonalize
from multiprocessing import Pool
from tqdm import tqdm
# import time

from Hamiltonian_mathematica_v2 import gso, psi_new_basis
np.set_printoptions(linewidth=150)


_H00 = None
_H10 = None
_H12 = None
_H14 = None


# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N= 50
Nband = 10
kx = 0.0
hbar = 1.05457e-34
q = 1.602e-19
muB = 5.788e-5
gspin = 2.0
Bf = 0.01 #applied magnetic field in T
Ny = Nz = N
L = 300
flag = [1, 0]
nef = 20


# ---------------------------------------------
#               SPIN MATRIX
# ---------------------------------------------
def build_spin_matrix():
    sminus = 1

    sud = np.zeros((5, 5), dtype=complex)
    sud[0, 0] = sminus
    sud[1, 1] = sminus
    sud[4, 4] = sminus
    sud[2, 3] = sminus
    sud[3, 2] = sminus

    sud_sparse = csr_matrix(sud)
    sdu_sparse = sud_sparse.conjugate()
    zero = csr_matrix((5, 5))

    smatrix = 0.5 * bmat([
        [zero, sud_sparse],
        [sdu_sparse, zero]
    ], format='csr')

    return smatrix


lat = kwant.lattice.square(norbs=Nband)

def make_system_bx(L, Bx):
    syst = kwant.Builder()

    a = L / (N + 1)
    a_m = a * 1e-10

    phase_per_tesla = q * a_m**2 / hbar

    gso_cache = {}

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(
                a, kx, dy, dz
            )

    # Onsite terms
    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache[(0, 0)]

    def hopping(site1, site2):
        yi, zi = site1.tag
        yj, zj = site2.tag

        # Keep the same convention for every value of Bx
        dy = yj - yi
        dz = zj - zi

        val = gso_cache[(dy, dz)]

        y_mid = 0.5 * (yi + yj)

        phi = (
            phase_per_tesla
            * y_mid
            * dz
        )

        phase = np.exp(-1j * Bx * phi)

        return val * phase

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue

            syst[
                kwant.builder.HoppingKind(
                    (dy, dz), lat, lat
                )
            ] = hopping

    return syst.finalized()


lat_E = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system H_E
# ---------------------------------
def make_system_E():

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                V = - ((y + 1) * a - L / 2)
            else:
                V = - ((z + 1) * a - L / 2)
            syst[lat_E(y, z)] = V * np.eye(Nband) 

    return syst.finalized()






def build_effective_hamiltonian_coefficients():
    """Compute the Ef-independent effective-Hamiltonian coefficients once."""
    dB = Bf  # Tesla

    syst_0 = make_system_bx(L, 0.0)
    syst_p = make_system_bx(L, +dB)
    syst_m = make_system_bx(L, -dB)

    H0_sparse = syst_0.hamiltonian_submatrix(sparse=True)
    Horb_p = syst_p.hamiltonian_submatrix(sparse=True)
    Horb_m = syst_m.hamiltonian_submatrix(sparse=True)

    # Orbital coefficient in eV/T
    V_orbital = (Horb_p - Horb_m) / (2.0 * dB)

    # Spin coefficient in eV/T
    smatrix_sparse = build_spin_matrix()
    smatrixfull = block_diag([smatrix_sparse] * (N**2), format='csr')
    V_spin = gspin * muB * smatrixfull

    # Total first-order magnetic perturbation
    V_Bx = V_orbital + V_spin

    syst2 = make_system_E()
    V_E = syst2.hamiltonian_submatrix(sparse=True)


    H_list_sparse = [H0_sparse, V_Bx, V_E]

    num_eigenvectors = 2
    evals, evecs = eigsh(H0_sparse, k=num_eigenvectors, sigma=0.00)

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    psi_new = psi_new_basis(psi1, psi2, N)
    psi_new = np.array(psi_new).T

    Heff_coeffs, *_ = block_diagonalize(H_list_sparse, subspace_eigenvectors=[psi_new])



    H00 = np.asarray(Heff_coeffs[(0,0,0,0)])
    H10 = np.asarray(Heff_coeffs[(0,0,1,0)])
    H12 = np.asarray(Heff_coeffs[(0,0,1,2)])
    H14 = np.asarray(Heff_coeffs[(0,0,1,4)])
    H16 = np.asarray(Heff_coeffs[(0,0,1,6)])

    return H00, H10, H12, H14, H16


def init_worker(H00, H10, H12, H14, H16):
    """Store the small coefficient matrices once in each worker process."""
    global _H00, _H10, _H12, _H14, _H16
    _H00 = H00
    _H10 = H10
    _H12 = H12
    _H14 = H14
    _H16 = H16



def calculate_efield(Ef):
    """Calculate the gap and g factor for one electric-field value."""
    H_eff = _H00 + _H10 * Bf + _H12 * Bf * Ef**2 + _H14 * Bf * Ef**4+ _H16 * Bf * Ef**6
    w = eigh(H_eff, eigvals_only=True)
    gap = np.abs(w[1] - w[0])
    g = gap / (muB * Bf)
    return [L, gap, Ef, g], w


def main():
    H00, H10, H12, H14, H16 = build_effective_hamiltonian_coefficients()

    # imap preserves efield_list ordering while yielding completed results so
    # tqdm can update continuously.  Limit workers to the number of tasks.
    # nworkers = min(len(efield_list), os.cpu_count() or 1)
    nworkers = 8
    with Pool(
        processes=nworkers,
        initializer=init_worker,
        initargs=(H00, H10, H12, H14, H16),
    ) as pool:
        calculated = list(tqdm(
            pool.imap(calculate_efield, efield_list),
            total=len(efield_list),
            desc="Electric fields",
            unit="field",
        ))

    results = [result for result, _ in calculated]
    for result, w in calculated:
        _, _, Ef, g = result
        print(
            f"L = {L}, E = {Ef*1e4:.3f}, ev1 = {w[0]:.6f}, "
            f"ev2 = {w[1]:.6f}, g = {g:.3f}"
        )

    output_filename = "gx_moulation_Ey.dat" if flagy == 1 else "gx_moulation_Ez.dat"
    np.savetxt(
        output_filename,
        np.asarray(results),
        header="L gap E g",
        fmt="%.12e",
    )
    print(f"Saved results to {output_filename}")


if __name__ == "__main__":
    for flagy in flag:
        if flagy == 1:
            emax_V_um = 1.0
        if flagy == 0:
            emax_V_um = 0.075
        emax_V_angstrom = emax_V_um * 1e-4
        efield_list = np.linspace(0, emax_V_angstrom, nef) 
        main()

    