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
flagy = 1
nef = 20
# efield_list = [0]#np.linspace(0, emax_V_angstrom, nef)
Ef =0.1 *1e-4
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


H00, H10, H12, H14, H16 = build_effective_hamiltonian_coefficients()


correction = (H10 + H12 * Ef**2 + H14 * Ef**4+ H16 * Ef**6)
H_eff = H00 + correction * Bf
w = eigh(H_eff, eigvals_only=True)
gap = np.abs(w[1] - w[0])
g = gap / (muB * Bf)

print(g.round(6))

print(correction.round(8))

w = eigh(correction*Bf, eigvals_only=True)
gap = np.abs(w[1] - w[0])
g = gap / (muB * Bf)

print(g.round(6))

